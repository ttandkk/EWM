"""Shared RGB/EVIS physical-oracle validation, with unlabelled dual-view videos."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main(task_name):
    from piper_bench.backend.isaac import launcher_class
    AppLauncher = launcher_class()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=['preview', 'oracle'], default='oracle')
    parser.add_argument('--trials', type=int)
    parser.add_argument('--initial-manifest', type=Path,
                        help='Replay a fixed Push-T pose-v2 initial-state manifest')
    parser.add_argument('--max-steps', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--events', action='store_true')
    parser.add_argument('--deployment-audit', action='store_true',
                        help='Record physical wrist attachment, RGB motion and USD composition evidence')
    parser.add_argument('--pathtracing-spp', type=int, default=16)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device='cpu')
    args = parser.parse_args()
    if args.device != 'cpu' or args.max_steps < 1 or (args.trials is not None and args.trials < 1):
        parser.error('use CPU physics and positive trials/max-steps; rendering still uses the GPU')
    if args.initial_manifest is not None and task_name != 'push_t':
        parser.error('--initial-manifest currently supports Push-T only')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from piper_bench.backend.video import find_ffmpeg
    ffmpeg = find_ffmpeg()
    from piper_bench.backend.isaac import launch
    app = launch(args)
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        from PIL import Image
        from piper_bench.tasks import load_task
        from piper_bench.tasks import register_tasks
        from piper_bench.tasks.conveyor.core import CAMERAS
        from piper_bench.sensors.capture import configure_renderer
        task = load_task(task_name); config = task.Config()
        initial_manifest = None
        if args.initial_manifest is not None:
            from piper_bench.tasks.push_t.protocol import load_manifest
            from piper_bench.collection.runtime import initialize
            initial_manifest = load_manifest(args.initial_manifest)
            config = task.Config(**initial_manifest['episodes'][0]['config'])
            if args.trials is None:
                args.trials = len(initial_manifest['episodes'])
            if args.trials > len(initial_manifest['episodes']):
                raise ValueError('trials exceed the fixed initial-state manifest')
            shutil.copy2(args.initial_manifest, out/'initial_manifest.json')
        args.trials = args.trials or 1
        if task_name.startswith('tennis_'):
            # Preview the same versioned distribution as new training data.
            from piper_bench.collection.protocol import make_plan
            from piper_bench.collection.protocol import base_config
            from piper_bench.collection.protocol import episode_spec
            from piper_bench.collection.runtime import initialize
            plan = make_plan(task_name, args.trials, args.seed, 'validation')
            config = base_config(task_name, plan)
            initial_manifest = dict(episodes=[episode_spec(plan, i) for i in range(args.trials)])
            (out/'sampling_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        register_tasks(); configure_renderer(args.pathtracing_spp)
        env = gym.make(task.TASK, cfg=task.make_env_cfg(config, device='cpu', seed=args.seed)).unwrapped
        runtime = task.Runtime(env, config)
        dvs, event_protocol = None, None
        if args.events:
            from piper_bench.sensors.preview import create_event_camera
            from piper_bench.sensors.preview import event_metadata
            from piper_bench.sensors.preview import EventPreview
            dvs = create_event_camera(env, out/'dvs_tmp'); event_protocol = event_metadata(dvs)
        calibration = task.camera_metadata()
        if task_name.startswith('tennis_'):
            from piper_bench.tasks.tennis.camera import camera_metadata_for_config
            calibration = camera_metadata_for_config(config)
        (out/'protocol.json').write_text(json.dumps(dict(task=task.TASK, config=config.metadata(),
            camera=calibration, control_hz=30, physics_hz=round(1/env.physics_dt),
            oracle=task.Oracle.PROFILE, events=event_protocol, text_overlays=False,
            training_data=False), indent=2)+'\n')

        def panel():
            images = [env.scene[name].data.output['rgb'][0, ..., :3].detach().cpu().numpy().astype(np.uint8) for name in CAMERAS]
            return Image.fromarray(np.concatenate(images, axis=1))

        results = []
        zero = torch.tensor([[0., 0., 0., 0., 0., 0., getattr(runtime, 'initial_gripper_action', 1.)]], device=env.device)
        settle_action = getattr(runtime, 'settle_action', lambda: zero)
        for trial in range(args.trials):
            directory = out/f'{args.mode}_{trial:03d}'; directory.mkdir()
            env.reset()
            for _ in range(60 if initial_manifest else 45):
                task.set_views(env); env.step(settle_action())
            if initial_manifest:
                spec = initial_manifest['episodes'][trial]
                initial = initialize(runtime, spec)
                (directory/'initial_spec.json').write_text(json.dumps(spec, indent=2)+'\n')
                (directory/'initial_verified.json').write_text(json.dumps(initial, indent=2)+'\n')
            else:
                runtime.reset(args.seed+trial); runtime.supply()
                for _ in range(getattr(runtime, 'settle_steps', 15)):
                    task.set_views(env); env.step(settle_action())
            task.set_views(env); env.sim.forward(); env.sim.render()
            for name in CAMERAS:
                env.scene[name].update(0., force_recompute=True)
            panel().save(directory/'initial.png')
            (directory/'initial_state.json').write_text(json.dumps(runtime.metadata(), indent=2)+'\n')
            events = EventPreview(dvs, runtime, directory, ffmpeg, labels=False) if dvs is not None else None
            video = subprocess.Popen([ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', '512x256', '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(directory/'preview.mp4')], stdin=subprocess.PIPE)
            oracle = task.Oracle(runtime); rows = []; seen = set(); success = False; reason = None
            audit = None
            if args.deployment_audit:
                from piper_bench.backend.diagnostics import RuntimeAudit
                audit = RuntimeAudit(env)
            try:
                limit = min(args.max_steps, getattr(runtime, 'preview_steps', 120)) if args.mode == 'preview' else args.max_steps
                for step in range(limit):
                    action = oracle.action() if args.mode == 'oracle' and not (initial_manifest and step == 0) else zero
                    if hasattr(runtime, 'before_step'):
                        runtime.before_step(action)
                    task.set_views(env); env.step(action)
                    joints = env.scene['robot'].data.joint_pos[0].detach().cpu().numpy()
                    # A held 52 mm cup also keeps the fingers apart: require the
                    # release command as well as aperture before marking placed.
                    gripper_open = bool(action[0, -1] > 0 and abs(joints[-2]-joints[-1]) > .045)
                    success, reason = runtime.after_step(gripper_open)
                    reason = oracle.failure or reason
                    poses, velocities, tcp = runtime.state()
                    row = dict(step=step+1, time=runtime.time, stage=oracle.stage, poses=poses.tolist(),
                        velocities=velocities.tolist(), tcp=tcp.tolist(), joints=joints.tolist(),
                        tcp_quaternion=env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().tolist(),
                        target=None if oracle.target is None else oracle.target.tolist(),
                        action=action[0].detach().cpu().tolist(), gripper_open=gripper_open,
                        metrics=runtime.metrics(gripper_open), success=success, reason=reason)
                    rows.append(row); frame = panel(); video.stdin.write(frame.tobytes())
                    if audit is not None:
                        audit.capture()
                    if events is not None:
                        row['events'] = events.step(frame)
                    if oracle.stage not in seen:
                        seen.add(oracle.stage); frame.save(directory/f'step_{step+1:04d}_{oracle.stage}.png')
                    if step % 30 == 0 or success or reason:
                        (directory/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
                        print(f'[{task_name}] trial={trial} step={step+1} stage={oracle.stage} '
                              f'metrics={row["metrics"]} reason={reason}', flush=True)
                    if success or reason:
                        break
            finally:
                video.stdin.close()
                code = video.wait(timeout=60)
                (directory/'steps.json').write_text(json.dumps(rows, indent=2)+'\n')
                if events is not None:
                    events.close()
                if code:
                    raise RuntimeError('video encoding failed')
            subprocess.run([ffmpeg, '-v', 'error', '-i', str(directory/'preview.mp4'), '-f', 'null', '-'], check=True)
            panel().save(directory/'terminal.png')
            result = dict(seed=spec['seed'] if initial_manifest else args.seed+trial,
                          frames=len(rows), task_success=bool(success), reason=reason,
                          passed=bool(success) if args.mode == 'oracle' else reason is None,
                          mode=args.mode, video_decoded=True, metrics=rows[-1]['metrics'])
            if events is not None:
                result['events'] = events.report; result['passed'] &= events.report['passed']
            if audit is not None:
                evidence = audit.finish(directory, events=events is not None)
                result['deployment_audit_passed'] = evidence['passed']
            (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            results.append(result)
        complete = dict(complete=True, passed=all(r['passed'] for r in results), results=results)
        (out/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
        if not complete['passed']:
            raise RuntimeError('physical task validation failed; inspect result.json and steps.json')
    finally:
        if env is not None:
            env.close()
        app.close()
