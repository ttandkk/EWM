"""Rejudge saved physical terminal states or replay every recorded simulator action.

Rejudging needs only CPU task predicates. Action replay starts Isaac Sim, restores
the recorded scene and initial state, and records new RGB/events and terminal truth.
It never contacts a policy server or replaces the original evaluation artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


RESULT_NAME = re.compile(r'trial_(\d+)_result\.json\Z')


def _read(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rejudge_trial(result_file):
    """Recompute one recorded terminal predicate, without running simulation."""
    from piper_bench.collection.runtime import terminal_valid
    result_file = Path(result_file).resolve()
    value = _read(result_file)
    if not isinstance(value.get('success'), bool):
        raise ValueError(f'{result_file}: saved success must be a boolean')
    spec = value['spec']
    passed = bool(terminal_valid(spec['task'], spec['config'], value['terminal']))
    return dict(result_file=str(result_file), result_sha256=_sha256(result_file),
                task=spec['task'], trial=value['trial'], recorded_success=value['success'],
                terminal_success=passed, agrees=passed == value['success'],
                simulation_executed=False, mode='terminal_rejudge')


def rejudge_result(result_dir, trial=None):
    """Rejudge a task directory or a suite recursively; preserve errors per file."""
    from piper_bench import SOURCE_LAYOUT_VERSION
    from piper_bench.collection.protocol import source_hashes
    root = Path(result_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if trial is not None and trial < 0:
        raise ValueError('trial must be nonnegative')
    paths = sorted(path for path in root.rglob('trial_*_result.json')
                   if RESULT_NAME.fullmatch(path.name)
                   and (trial is None or int(RESULT_NAME.fullmatch(path.name)[1]) == trial))
    if not paths:
        raise FileNotFoundError(f'No saved trial results under {root}')
    rows, errors = [], []
    for path in paths:
        try:
            rows.append(rejudge_trial(path))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(dict(result_file=str(path), error_type=type(exc).__name__, error=str(exc)))
    mismatches = [row for row in rows if not row['agrees']]
    return dict(mode='terminal_rejudge', simulation_executed=False,
                source_layout=SOURCE_LAYOUT_VERSION, source_sha256=source_hashes(), result_dir=str(root),
                trials=len(paths), rejudged=len(rows), recorded_successes=sum(row['recorded_success'] for row in rows),
                terminal_successes=sum(row['terminal_success'] for row in rows),
                passed=not errors and not mismatches, mismatches=mismatches, errors=errors, results=rows)


def load_replay(result_dir, trial):
    """Validate a complete recorded action stream before allocating a simulator."""
    directory = Path(result_dir).resolve()
    if trial < 0:
        raise ValueError('trial must be nonnegative')
    stem = f'trial_{trial:03d}'
    files = {name: directory / filename for name, filename in (
        ('manifest', 'manifest.json'), ('scenes', 'scene_manifest.json'),
        ('result', stem + '_result.json'), ('initial', stem + '_initial_state.json'),
        ('steps', stem + '_steps.json'))}
    values = {name: _read(path) for name, path in files.items()}
    result, manifest, rows = values['result'], values['manifest'], values['steps']
    if not isinstance(result.get('success'), bool):
        raise ValueError('saved success must be a boolean')
    spec = result['spec']
    if result['trial'] != trial or spec['task'] != manifest['task']:
        raise ValueError('trial/task identifiers disagree with the saved manifest')
    if values['initial']['spec'] != spec or values['scenes']['episodes'][trial] != spec:
        raise ValueError('initial state, scene manifest and trial result specify different scenes')
    if manifest.get('control_hz') != 30:
        raise ValueError('recorded control clock must be 30 Hz')
    if not isinstance(rows, list) or not rows or len(rows) != result['steps']:
        raise ValueError('recorded actions must contain every executed simulator step')
    for index, row in enumerate(rows, 1):
        action = row.get('action')
        if row.get('step') != index:
            raise ValueError('recorded action steps must be contiguous starting at 1')
        if (not isinstance(action, list) or len(action) != 7
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in action)):
            raise ValueError(f'action {index} must contain seven finite numbers')
        if 'simulation_time_s' in row and not math.isclose(row['simulation_time_s'], index / 30, abs_tol=1e-7):
            raise ValueError(f'action {index} does not follow the recorded 30 Hz clock')
    values.update(directory=directory, trial=trial,
                  input_sha256={name: _sha256(path) for name, path in files.items()})
    return values


def _new_output(source, output_dir):
    source, output = Path(source).resolve(), Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError('output-dir must be outside the source result directory')
    output.mkdir(parents=True, exist_ok=False)
    return output


def _initial_differences(saved, actual, tolerance=1e-5):
    import numpy as np
    differences = {}
    for key in ('object_poses', 'object_velocities', 'robot_joints', 'cabinet_pose',
                'container_pose', 'container_velocity'):
        if key not in saved or saved[key] is None:
            continue
        before, after = np.asarray(saved[key]), np.asarray(actual.get(key))
        if before.shape != after.shape or not np.allclose(before, after, rtol=1e-5, atol=tolerance):
            differences[key] = dict(recorded=saved[key], replayed=actual.get(key))
    return differences


def replay_trial(result_dir, trial, output_dir, *, device='cpu'):
    """Run fixed actions in an already-started Isaac application (no policy call).

    The command-line main owns AppLauncher. Python callers must start it before
    calling this function. GPU rendering/event generation is still required when
    physics runs on CPU, exactly as in the evaluation runner.
    """
    if device != 'cpu':
        raise ValueError('Piper action replay requires CPU physics')
    saved = load_replay(result_dir, trial)
    output = _new_output(result_dir, output_dir)
    from piper_bench.collection.protocol import write_json
    try:
        return _run_replay(saved, output)
    except BaseException as exc:
        write_json(output / 'failure.json', dict(mode='fixed_action_replay', complete=False,
                   type=type(exc).__name__, error=str(exc), input_sha256=saved['input_sha256']))
        raise


def _run_replay(saved, output):
    import cv2
    import gymnasium as gym
    import numpy as np
    import subprocess
    import torch
    from collections import deque
    from piper_bench import SOURCE_LAYOUT_VERSION
    from piper_bench.collection.protocol import source_hashes, write_json
    from piper_bench.collection.runtime import initialize, terminal_state, terminal_valid
    from piper_bench.evaluation.constants import CAMERAS, RESET_WARMUP_STEPS
    from piper_bench.evaluation.recording import TrialVideo, preview
    from piper_bench.evaluation.runner import capture_step
    from piper_bench.evaluation.runtime import observation
    from piper_bench.protocols.evaluation_protocol import stable_steps
    from piper_bench.protocols.dynamic_evaluation import numerical_equal
    from piper_bench.sensors.capture import configure_renderer, snapshot
    from piper_bench.sensors.preview import create_event_camera
    from piper_bench.tasks import load_task, register_tasks

    manifest, recorded, trial = saved['manifest'], saved['result'], saved['trial']
    spec, name = recorded['spec'], manifest['task']
    module = load_task(name)
    config = module.Config(**manifest['scene_config'])
    calibration = module.camera_metadata()
    if name.startswith('tennis_'):
        from piper_bench.tasks.tennis.camera import camera_metadata_for_config
        calibration = camera_metadata_for_config(config)
    if not numerical_equal(calibration, manifest['camera_calibration']):
        raise ValueError('recorded camera calibration differs from the reconstructed scene')
    configure_renderer(manifest['pathtracing_spp'])
    register_tasks()
    env_cfg = module.make_env_cfg(config, device='cpu', seed=manifest['scene_seed'])
    background = None
    if manifest['condition'] == 'background_ood':
        from piper_bench.protocols.ood_protocol import configure_background
        version = manifest['imaging']['version']
        background = configure_background(env_cfg.scene, version=version)
        if not numerical_equal(background, manifest['background']):
            raise ValueError('recorded background parameters differ from the selected version')
    env = gym.make(module.TASK, cfg=env_cfg).unwrapped
    try:
        if round(1 / env.physics_dt) != manifest['physics_hz']:
            raise ValueError('replay physics clock differs from the recorded evaluation')
        runtime = module.Runtime(env, config)
        runtime._eval_task = name
        initial_grip = getattr(runtime, 'initial_gripper_action', 1.)
        zero = torch.tensor([[0., 0., 0., 0., 0., 0., initial_grip]], device=env.device)
        light_state, lights = None, None
        if 'imaging' in spec:
            import omni.usd
            from piper_bench.sensors.lighting import FixedSceneLighting
            lights = FixedSceneLighting(omni.usd.get_context().get_stage())
            light_state = lights.apply(spec['imaging'].get('intensity_multiplier', 1.))
        dvs = create_event_camera(env, output / 'dvs_tmp')
        if hasattr(runtime, 'motion_enabled'):
            runtime.motion_enabled = False
        env.reset(seed=spec['seed'])
        settle = getattr(runtime, 'settle_action', lambda: zero)
        for _ in range(45 if name.startswith('tennis_') else RESET_WARMUP_STEPS):
            module.set_views(env)
            env.step(settle())
        initial = initialize(runtime, spec)
        differences = _initial_differences(saved['initial']['initial'], initial)
        write_json(output / 'initial_state.json', dict(spec=spec, initial=initial, differences=differences))
        if differences:
            raise ValueError('replayed initial physical state differs; see initial_state.json')
        runtime._eval_opened = initial_grip > 0
        if name.startswith('tennis_'):
            runtime.collection_catch_window = deque(maxlen=runtime.tracker.window_samples)
        from piper_bench.protocols.evaluation_imaging import EvaluationEvents, TrialImaging
        runtime._eval_event_recorder = EvaluationEvents()
        if lights is not None:
            lights.verify(spec['imaging'].get('intensity_multiplier', 1.))
            runtime._eval_rgb = TrialImaging(runtime, dvs, spec['imaging'], light_state, background)
        else:
            from piper_bench.evaluation.rgb_ood import RGBOODConfig, TrialRGBOOD
            rgb = manifest.get('rgb_ood') or {}
            mode = rgb.get('mode', 'legacy')
            params = dict(mode=mode, seed=rgb.get('seed', manifest['scene_seed']))
            if mode == 'legacy':
                params.update(lighting=rgb.get('lighting', False), motion_blur=rgb.get('motion_blur', False),
                              severity=rgb.get('severity', 'moderate'))
            runtime._eval_rgb = TrialRGBOOD(RGBOODConfig(**params), trial)
        dvs.reset(torch.tensor([0], dtype=torch.long, device='cuda:0'))
        dvs.recorder._events.pop(0, None)
        dvs._conveyor_snapshot = snapshot(dvs, env)
        obs = observation(runtime, dvs, CAMERAS, rgb_transform=runtime._eval_rgb.apply_rgb)
        if np.any(obs['obs_voxel']):
            raise ValueError('replay initial event frame is not empty')
        cv2.imwrite(str(output / 'initial.png'), preview(obs, 0))
        video = TrialVideo(output / 'replay.mp4', CAMERAS, scene_label=f'{name} fixed recorded actions')
        rows, first_terminal = [], None
        try:
            video.write(obs, 0, 0)
            for row in saved['steps']:
                action = torch.tensor([row['action']], dtype=torch.float32, device=env.device)
                obs, success, reason = capture_step(runtime, dvs, action, CAMERAS)
                step = row['step']
                if first_terminal is None and (success or reason):
                    first_terminal = dict(step=step, success=bool(success), reason=reason)
                video.write(obs, step, stable_steps(runtime))
                rows.append(dict(step=step, action=row['action'], success=bool(success), reason=reason,
                                 simulation_time_s=runtime.time, event_count=obs['event_counts']))
        finally:
            video.close()
        subprocess.run([video.ffmpeg, '-v', 'error', '-i', str(video.path), '-f', 'null', '-'],
                       check=True, timeout=120)
        terminal = terminal_state(runtime, name, runtime._eval_opened)
        valid = bool(terminal_valid(name, spec['config'], terminal))
        deltas = {}
        for key in ('poses', 'velocities', 'tcp', 'time', 'stable_steps'):
            previous, current = np.asarray(recorded['terminal'][key]), np.asarray(terminal[key])
            deltas[key] = float(np.max(np.abs(current - previous))) if previous.shape == current.shape else None
        event_archive = runtime._eval_event_recorder.save(output / 'events')
        cv2.imwrite(str(output / 'terminal.png'), preview(obs, len(rows)))
        write_json(output / 'steps.json', rows)
        result = dict(mode='fixed_action_replay', simulation_executed=True, complete=True,
                      source_layout=SOURCE_LAYOUT_VERSION, source_sha256=source_hashes(),
                      input_sha256=saved['input_sha256'], source_result_dir=str(saved['directory']),
                      task=name, trial=trial, actions_replayed=len(rows), policy_requests=0,
                      recorded_success=recorded['success'], terminal_success=valid,
                      runtime_success=rows[-1]['success'],
                      success_matches=valid == recorded['success'] == rows[-1]['success'],
                      first_terminal=first_terminal, terminal_max_abs_delta=deltas,
                      terminal=terminal, spec=spec, video=str(video.path), video_decoded=True,
                      event_archive=event_archive, physics_hz=round(1 / env.physics_dt), control_hz=30)
        write_json(output / 'replay_result.json', result)
        return result
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--trial', type=int)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rejudge-only', action='store_true',
                        help='CPU terminal predicate audit; does not execute simulator actions')
    return parser


def main(argv=None):
    parser = build_parser()
    early, _ = parser.parse_known_args(argv)
    if early.rejudge_only:
        args = parser.parse_args(argv)
        report = rejudge_result(args.result_dir, args.trial)
        output = _new_output(args.result_dir, args.output_dir)
        from piper_bench.collection.protocol import write_json
        write_json(output / 'rejudge_report.json', report)
        print(json.dumps({key: report[key] for key in (
            'mode', 'simulation_executed', 'trials', 'rejudged', 'recorded_successes',
            'terminal_successes', 'passed', 'mismatches', 'errors')}, indent=2))
        return 0 if report['passed'] else 1
    if early.trial is None:
        parser.error('--trial is required for fixed action replay')
    load_replay(early.result_dir, early.trial)
    from piper_bench.backend.isaac import launcher_class
    AppLauncher = launcher_class()
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device='cpu', headless=True, enable_cameras=True)
    args = parser.parse_args(argv)
    from piper_bench.backend.isaac import launch
    app = launch(args)
    exit_code = 1
    try:
        result = replay_trial(args.result_dir, args.trial, args.output_dir, device=args.device)
        exit_code = 0 if result['success_matches'] else 1
        print(json.dumps({key: result[key] for key in ('mode', 'simulation_executed', 'actions_replayed',
                                                      'recorded_success', 'terminal_success', 'success_matches')}))
        return exit_code
    finally:
        app.close(exit_code=exit_code)


if __name__ == '__main__':
    raise SystemExit(main())
