"""Closed-loop benchmark evaluation against an independent policy service.

Task definitions are shipped with the suite. A training export may be supplied
for additional provenance checks, but no model configuration is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import runtime as shared
from .rgb_ood import FIXED_MODES, RGBOODConfig, TrialRGBOOD
from piper_bench.tasks import load_task, register_tasks
from piper_bench.collection.protocol import COLLECTION_TASKS as TASKS, source_hashes, write_json
from piper_bench.collection.sampling import VERSION as DYNAMIC_VERSION, VERSIONS as DYNAMIC_VERSIONS
from piper_bench.protocols.dynamic_evaluation import step_budget, condition_breakdown, numerical_equal
from piper_bench.protocols.evaluation_protocol import (
    CONDITIONS, digest, load_scene_manifest, scene_manifest, stable_steps, validate_collection,
)
from piper_bench.protocols.execution import DEFAULT_PROFILE, PROFILES, load_execution
from piper_bench.protocols.suite import DEFAULT_SUITE, load_suite, load_collection as suite_collection
from piper_bench.protocols.ood_protocol import BACKGROUND_VERSION, BACKGROUND_PROFILES


def build_parser(app_launcher=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--execution-profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--background-profile", choices=BACKGROUND_PROFILES, default=BACKGROUND_VERSION)
    parser.add_argument("--policy-url", help="WebSocket policy endpoint; model runs in its own environment")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scene-manifest", type=Path, help="Reuse an exact --plan-only JSON scene list")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--condition", choices=CONDITIONS, default="id")
    parser.add_argument("--policy-mode", choices=("joint", "action_only", "rgb_action"), default="joint",
                        help="Legacy server mode; versioned services declare their mode in hello")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-steps", type=int, help="Fixed task limit from the selected suite; override only with --interface-only")
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--execution-horizon", "--action-chunk-size", type=int, default=10)
    parser.add_argument("--min-execution-steps", type=int, default=None,
                        help="Minimum actions per active continuous plan; 1 restores immediate switching")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--scene-seed", type=int, default=20280002)
    parser.add_argument("--decode-predicted-events", action="store_true")
    parser.add_argument("--record-predictions", action=argparse.BooleanOptionalAction, default=None,
                        help="Record RGB/event latents and decode used predictions after each trial (default)")
    parser.add_argument("--predicted-event-threshold", type=float, default=1/30)
    parser.add_argument("--predicted-event-min-neighbors", type=int, default=2)
    parser.add_argument("--profile-timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latency-aware", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latency-controller", choices=("serial", "buffered", "continuous"), default=None,
                        help="continuous keeps inferring and switches after the minimum actions; execution-horizon only controls serial/buffered")
    parser.add_argument("--pathtracing-spp", type=int)
    parser.add_argument("--interface-only", action="store_true")
    parser.add_argument("--rgb-ood-mode", choices=tuple(FIXED_MODES), default="clean",
                        help="Existing stack RGB-only ablations; not physical lighting/exposure OOD")
    if app_launcher is None:
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--enable_cameras", action="store_true")
    else:
        app_launcher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu", speed_ood="none", conveyor_speed=None, conveyor_speed_range=None)
    return parser


def load_settings(args):
    protocol = load_execution(args.execution_profile)
    if args.latency_controller is None:
        args.latency_controller = protocol.latency_controller
    if args.min_execution_steps is None:
        args.min_execution_steps = protocol.min_execution_steps
    if not args.interface_only and (
        args.latency_controller != protocol.latency_controller
        or args.min_execution_steps != protocol.min_execution_steps
        or args.latency_aware != protocol.latency_aware
        or args.scene_seed != protocol.scene_seed
        or args.seed != protocol.policy_seed
        or args.execution_horizon != protocol.execution_horizon
    ):
        raise ValueError("Execution settings differ from the selected profile; use --interface-only for diagnostics")
    if not args.interface_only and args.sampling_steps != 4:
        raise ValueError("The formal inference configuration uses 4 flow steps; use --interface-only for diagnostics")
    dataset = args.dataset_dir.expanduser().resolve() if args.dataset_dir else None
    collection = validate_collection(args.task, dataset) if dataset else suite_collection(args.task, args.suite)
    cfg = load_task(args.task).Config(**collection["base_config"])
    # Preserve the fixed limit recorded by already-created legacy scene manifests.
    # The complete manifest, including data provenance, is still validated below.
    legacy_max_steps = None
    if args.scene_manifest is not None:
        saved_budget = json.loads(args.scene_manifest.read_text()).get("step_budget", {})
        if saved_budget.get("rule") == "strictly_next_multiple_of_50":
            legacy_max_steps = saved_budget["max_steps"]
    budget = (step_budget(dataset, legacy_max_steps=legacy_max_steps)
              if dataset and collection["version"] in DYNAMIC_VERSIONS else None)
    if not dataset:
        budget = dict(load_suite(args.suite)["tasks"][args.task]["step_budget"])
    limit = budget["max_steps"] if budget else round(cfg.max_seconds * shared.CONTROL_HZ)
    if args.max_steps is None:
        args.max_steps = limit
    elif args.max_steps != limit and not args.interface_only:
        raise ValueError("task time limit must match the evaluation protocol; use --interface-only for a short diagnostic")
    shared.validate_args(args)
    if args.pathtracing_spp is not None and args.pathtracing_spp != collection["pathtracing_spp"]:
        raise ValueError("pathtracing-spp must match the training export")
    if (args.condition != "id" or collection["version"] in DYNAMIC_VERSIONS) and args.rgb_ood_mode != "clean":
        raise ValueError("evaluate speed and RGB perturbations separately")
    scenes = scene_manifest(collection, args.trials, args.scene_seed, args.condition,
                            background_version=args.background_profile)
    if scenes.get("success_config_overrides"):
        from dataclasses import replace
        cfg = replace(cfg, **scenes["success_config_overrides"])
    if budget:
        scenes["step_budget"] = dict(budget, effective_max_steps=args.max_steps)
    if args.scene_manifest is not None:
        scenes = load_scene_manifest(args.scene_manifest, scenes)
    if not args.plan_only and (not (args.policy_url or args.ready_file) or args.output_dir is None):
        raise ValueError("evaluation requires policy-url (or ready-file) and output-dir")
    if not args.plan_only and not args.enable_cameras:
        raise ValueError("pass --enable_cameras for RGB/HDR/EVIS observations")
    return dataset, collection, cfg, scenes


def capture_step(runtime, dvs, action, event_cameras):
    """Only execute policy actions; ground truth never enters the wire request."""
    from piper_bench.collection.runtime import step_result
    name = runtime._eval_task
    module = load_task(name)
    previous, previous_time = dvs._conveyor_snapshot, runtime.time
    if name == "turntable_cups":
        runtime.before_step()
    elif name.startswith("tennis_"):
        runtime.before_step(action)
    module.set_views(runtime.env)
    runtime.env.step(action)
    success, reason, runtime._eval_opened = step_result(runtime, name, action)
    if name.startswith("tennis_"):
        from piper_bench.tasks.tennis.collection import success_sample
        runtime.collection_catch_window.append(success_sample(runtime))
    current = shared.snapshot(dvs, runtime.env)
    dvs.warp_and_process(previous, current, shared.WARP, previous_time, 1/240)
    dvs._conveyor_snapshot = current
    obs = shared.observation(runtime, dvs, event_cameras, rgb_transform=runtime._eval_rgb.apply_rgb)
    return obs, success, reason


def run_paused_loop(args, runtime, dvs, conn, hello, checkpoint_step, event_cameras,
                    obs, video, rows, chunks, timings, heartbeat, output, trial):
    import torch
    module = load_task(args.task)
    step, replans, success, reason = 0, 0, False, None
    inference_times, round_trip_times = [], []
    while step < args.max_steps:
        horizon = min(args.execution_horizon, args.max_steps-step)
        simulation_time = runtime.time
        started = time.perf_counter()
        reply, values = shared.policy_request(conn, args, dict(
            op="predict", step=checkpoint_step, seed=args.seed+trial*args.max_steps+step,
            sampling_steps=args.sampling_steps, action_chunk_size=horizon, prompt=module.PROMPT,
            event_time_contract=shared.EVENT_TIME_CONTRACT, observation_row_offset=0,
            raw_observation=True, timing=obs["timing"], decode_future_events=args.decode_predicted_events,
        ), {k: obs[k] for k in ("obs_video", "obs_voxel", "proprio")})
        round_trip = time.perf_counter()-started
        if runtime.time != simulation_time:
            raise RuntimeError("simulation advanced during paused inference")
        actions, predicted, inference = shared.validate_reply(
            reply, values, horizon, event_cameras, args, int(hello.get("count_clip", 15)))
        inference_times.append(inference)
        round_trip_times.append(round_trip)
        replans += 1
        for index, action in enumerate(actions):
            started = time.perf_counter()
            command = torch.from_numpy(action.copy())[None].to(runtime.env.device)
            obs, success, reason = capture_step(runtime, dvs, command, event_cameras)
            capture_seconds = time.perf_counter()-started
            step += 1
            prediction = None if predicted is None or index >= len(predicted) else predicted[index]
            end = ("success" if success else reason if reason else "step_budget" if step >= args.max_steps
                   else "max_horizon" if index+1 == horizon else None)
            started = time.perf_counter()
            video.write(obs, step, stable_steps(runtime), predicted_counts=prediction)
            row = dict(step=step, simulation_time_s=runtime.time, replan=replans-1, chunk_index=index,
                       action=action.tolist(), event_count=obs["event_counts"], timing=obs["timing"],
                       stable_success_steps=stable_steps(runtime), success=bool(success), failure_reason=reason,
                       chunk_end_reason=end, inference_seconds=inference, round_trip_seconds=round_trip)
            diagnostics = getattr(getattr(runtime, "tracker", None), "last_diagnostics", None)
            if diagnostics is not None:
                row["success_diagnostics"] = diagnostics
            if args.profile_timing:
                row["timing_seconds"] = dict(capture_total=capture_seconds, video_write=time.perf_counter()-started)
                timings.append(row["timing_seconds"])
            rows.append(row)
            if step % 10 == 0 or end is not None:
                write_json(output / f"trial_{trial:03d}_steps.json", rows)
                heartbeat.touch()
                print(f"[ewm-tasks] {args.task} trial={trial+1}/{args.trials} step={step} end={end}", flush=True)
            if end is not None:
                chunks.append(dict(replan=replans-1, executed_horizon=index+1, end_reason=end))
                break
        if success or reason:
            break
    return obs, success, reason, step, replans, inference_times, round_trip_times, None


def prepare_results(output, manifest, scenes, args):
    path = output / "manifest.json"
    if args.resume:
        saved = json.loads(path.read_text())
        if not numerical_equal({k:v for k,v in saved.items() if k not in ("server_optimization", "dataset_dir")},
                               {k:v for k,v in manifest.items() if k not in ("server_optimization", "dataset_dir")}):
            raise ValueError("resume model, scene, code or evaluation settings differ")
        load_scene_manifest(output / "scene_manifest.json", scenes)
        results = shared.load_completed_results(output, args.trials)
        shared.archive_interrupted_trial(output, len(results))
        (output / "failure.json").unlink(missing_ok=True)
        return results
    if path.exists() or list(output.glob("trial_*")):
        raise FileExistsError("existing evaluation requires --resume")
    write_json(output / "scene_manifest.json", scenes)
    write_json(path, manifest)
    return []


def verify_policy_data(descriptor, task, dataset, suite):
    """Compare immutable metadata identities, never training-machine paths."""
    from piper_bench.protocols.data import dataset_identity, verify_identity
    training = descriptor.get("training_data", {})
    if "manifest" in training:
        manifest = training["manifest"]
        if manifest.get("version") != "piper_training_data_v1":
            raise ValueError("Unsupported policy training-data manifest version")
        training = manifest["tasks"]
    expected = training.get(task)
    if training and expected is None:
        raise ValueError(f"Policy training-data manifest does not contain task {task}")
    raw_hash = (hashlib.sha256((dataset / "meta/piper_collection_config.json").read_bytes()).hexdigest()
                if dataset else load_suite(suite)["tasks"][task]["collection_file_sha256"])
    result = dict(scope="collection_metadata", collection_file_sha256=raw_hash,
                  policy_reference_present=bool(expected), status="not_provided")
    if expected:
        if expected.get("collection_sha256") != raw_hash:
            raise ValueError("Policy training collection metadata differs from the selected task suite/export")
        result["status"] = "verified"
        if "metadata_identity" in expected:
            if dataset is None:
                raise ValueError("Policy requires dataset metadata verification; supply --dataset-dir")
            identity = dataset_identity(dataset)
            verify_identity(identity, expected["metadata_identity"])
            result.update(scope="collection_camera_temporal_metadata", metadata_identity=identity)
    return result


def formal_inference_settings(hello):
    """EWM formal timing uses the recorded compilation/warmup configuration."""
    descriptor = hello.get('policy_descriptor', {})
    if descriptor.get('family') in ('openwam', 'fastwam'):
        return True  # Baselines retain their native inference implementation.
    settings = hello.get('optimization', hello)
    expected = dict(compile_core_active=True, compile_mode='default', cuda_graphs_enabled=False,
                    cache_context_kv=True, warmup_iters=3)
    if any(settings.get(key) != value for key, value in expected.items()):
        return False
    benchmark = settings.get('startup_benchmark') or {}
    return benchmark.get('iterations') == 3


def run(args, dataset, collection, cfg, scenes):
    import cv2
    import gymnasium as gym
    import torch
    from websockets.sync.client import connect
    from piper_bench.collection.runtime import initialize, terminal_state, terminal_valid
    from piper_bench.sensors.preview import create_event_camera

    output, module = args.output_dir.resolve(), load_task(args.task)
    if (output / "complete.json").exists():
        raise FileExistsError(f"refusing to overwrite completed evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    heartbeat = output / "progress.heartbeat"
    heartbeat.touch()
    shared.configure_renderer(collection["pathtracing_spp"])
    register_tasks()
    dynamic = collection["version"] in DYNAMIC_VERSIONS
    calibration = collection["camera"]
    if args.task.startswith("tennis_"):
        from piper_bench.tasks.tennis.camera import camera_metadata_for_config
        calibration = camera_metadata_for_config(cfg)
    env_cfg = module.make_env_cfg(cfg, device="cpu", seed=args.scene_seed)
    background = None
    if dynamic and args.condition == "background_ood":
        from piper_bench.protocols.ood_protocol import configure_background
        background = configure_background(env_cfg.scene, version=args.background_profile)
    env = gym.make(module.TASK, cfg=env_cfg).unwrapped
    results, attempted = [], 0

    def summarize(error=None):
        shared.write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                             requested_trials=args.trials, attempted_trials=attempted, results=results,
                             task_name=module.TASK,
                             decode_predicted_events=args.decode_predicted_events or (shared.recording_enabled(args) and args.policy_mode == "joint"),
                             error=error)
        with (output / "summary.txt").open("a") as f:
            for key, value in shared.recording_metadata(args).items():
                f.write(f"{key}: {str(value).lower() if isinstance(value, bool) else value}\n")
            f.write(f"condition: {args.condition}\ntraining_mode: {args.policy_mode}\n")
            f.write(f"simulation_clock: {shared.simulation_clock(args)}\n")
            f.write(f"interface_only: {str(args.interface_only).lower()}\n")
            f.write(f"success_version: {scenes['success_version']}\n")
            if args.condition == 'speed_ood':
                planned = condition_breakdown([dict(spec=s, success=False) for s in scenes['episodes']])
                observed = condition_breakdown(results)
                for group, expected in planned.items():
                    actual = observed.get(group, dict(trials=0, successes=0))
                    rate = actual['successes']/expected['trials'] if not args.interface_only else None
                    f.write(f"speed_ood_{group}: successes={actual['successes']}, completed={actual['trials']}, "
                            f"requested={expected['trials']}, success_rate={rate}\n")
            if args.interface_only:
                f.write("formal_success_rate: not_applicable\n")

    try:
        runtime = module.Runtime(env, cfg)
        runtime._eval_task, runtime._eval_opened = args.task, True
        initial_grip = getattr(runtime, "initial_gripper_action", 1.)
        zero = torch.tensor([[0., 0., 0., 0., 0., 0., initial_grip]], device=env.device)
        lights = None
        if dynamic:
            import omni.usd
            from piper_bench.sensors.lighting import FixedSceneLighting
            lights = FixedSceneLighting(omni.usd.get_context().get_stage())
        dvs = create_event_camera(env, output / "dvs_tmp")
        endpoint = args.policy_url or json.loads(args.ready_file.read_text())["url"]
        with connect(endpoint, open_timeout=30, max_size=shared.MAX_MESSAGE, compression=None) as conn:
            hello, _ = shared.request(conn, dict(op="hello"))
            checkpoint_step, cameras = shared.validate_hello(hello, args)
            if args.record_predictions is None:
                args.record_predictions = bool(hello.get("post_trial_visualization_supported"))
            descriptor = hello.get("policy_descriptor", {})
            data_check = verify_policy_data(descriptor, args.task, dataset, args.suite)
            checkpoint_hash = descriptor.get("checkpoint_sha256", hello.get("checkpoint_sha256"))
            if isinstance(checkpoint_hash, dict):
                checkpoint_hash = checkpoint_hash.get(str(checkpoint_step))
            if not checkpoint_hash:
                raise ValueError("Policy service must declare the selected checkpoint SHA256")
            rgb_config = RGBOODConfig(mode=args.rgb_ood_mode, seed=args.scene_seed)
            manifest = dict(version=scenes["version"], task=args.task, sim_task=module.TASK,
                **shared.recording_metadata(args),
                prompt=module.PROMPT, dataset_dir=str(dataset) if dataset else None, collection_sha256=digest(collection),
                suite=args.suite, execution_profile=load_execution(args.execution_profile).metadata(),
                policy_descriptor=descriptor, training_data_validation=data_check,
                scene_manifest_sha256=digest(scenes), scene_config=cfg.metadata(),
                camera_calibration=calibration, condition=args.condition,
                **(dict(collection_camera_calibration=collection["camera"])
                   if calibration != collection["camera"] else {}),
                checkpoint_step=checkpoint_step, checkpoint_sha256=checkpoint_hash,
                config_sha256=hello["config_sha256"], normalization_sha256=hello["normalization_sha256"],
                trials=args.trials, max_steps=args.max_steps, execution_horizon=args.execution_horizon,
                sampling_steps=args.sampling_steps, seed=args.seed, scene_seed=args.scene_seed,
                inference_settings=dict(sampling_steps=args.sampling_steps,
                                        optimization=hello.get("optimization", {}),
                                        formal_configuration_matches=formal_inference_settings(hello)),
                training_mode=args.policy_mode, event_input=args.policy_mode == "joint",
                event_time_contract=shared.EVENT_TIME_CONTRACT, observation_row_offset=0,
                latency_aware=args.latency_aware, latency_controller=args.latency_controller,
                **({"min_execution_steps": args.min_execution_steps}
                   if args.latency_aware and args.latency_controller == "continuous" else {}),
                simulation_clock=shared.simulation_clock(args),
                decode_predicted_events=args.decode_predicted_events or (shared.recording_enabled(args) and args.policy_mode == "joint"),
                predicted_event_threshold=args.predicted_event_threshold,
                predicted_event_min_neighbors=args.predicted_event_min_neighbors,
                profile_timing=args.profile_timing, rgb_ood=rgb_config.metadata(), interface_only=args.interface_only,
                formal_protocol=(not args.interface_only and args.trials == 50 and args.latency_aware
                                 and not rgb_config.enabled and formal_inference_settings(hello)),
                success_version=scenes["success_version"], collection_success_version=collection["success_version"], physics_hz=collection["physics_hz"], control_hz=30, event_hz=240,
                imaging=scenes.get("imaging"), step_budget=scenes.get("step_budget"), background=background,
                renderer="pathtracing", pathtracing_spp=collection["pathtracing_spp"],
                source_sha256=source_hashes(),
                evaluator_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(__file__).parent.glob("*.py"))},
                server_optimization={k:hello.get(k) for k in ("compile_core_active", "compile_mode", "cuda_graphs_enabled", "cache_context_kv", "warmup_iters")})
            results = prepare_results(output, manifest, scenes, args)
            attempted = len(results)
            summarize()
            for trial in range(len(results), args.trials):
                attempted += 1
                spec = scenes["episodes"][trial]
                if hasattr(runtime, "motion_enabled"):
                    runtime.motion_enabled = False
                light_state = lights.apply(spec['imaging'].get('intensity_multiplier', 1.)) if lights else None
                env.reset(seed=spec["seed"])
                settle_action = getattr(runtime, "settle_action", lambda: zero)
                for i in range(45 if args.task.startswith("tennis_") else shared.RESET_WARMUP_STEPS):
                    module.set_views(env)
                    env.step(settle_action())
                    if i % 10 == 0:
                        heartbeat.touch()
                initial = initialize(runtime, spec)
                runtime._eval_opened = initial_grip > 0
                if args.task.startswith("tennis_"):
                    from collections import deque
                    runtime.collection_catch_window = deque(maxlen=runtime.tracker.window_samples)
                if dynamic:
                    from piper_bench.protocols.evaluation_imaging import TrialImaging, EvaluationEvents
                    runtime._eval_event_recorder = EvaluationEvents()
                    lights.verify(spec['imaging'].get('intensity_multiplier', 1.))
                    runtime._eval_rgb = TrialImaging(runtime, dvs, spec['imaging'], light_state, background)
                else:
                    runtime._eval_rgb = TrialRGBOOD(rgb_config, trial)
                dvs.reset(torch.tensor([0], dtype=torch.long, device="cuda:0"))
                dvs.recorder._events.pop(0, None)
                dvs._conveyor_snapshot = shared.snapshot(dvs, env)
                obs = shared.observation(runtime, dvs, cameras, rgb_transform=runtime._eval_rgb.apply_rgb)
                if np.any(obs["obs_voxel"]):
                    raise ValueError("first event frame must be empty after reset")
                write_json(output / f"trial_{trial:03d}_initial_state.json",
                           dict(trial=trial, spec=spec, initial=initial, rgb_ood=runtime._eval_rgb.metadata()))
                cv2.imwrite(str(output / f"trial_{trial:03d}_initial.png"), shared.preview(obs, 0))
                video_path = output / f"trial_{trial:03d}_h264_recording.mp4"
                video = shared.TrialVideo(video_path, cameras,
                                          show_prediction=args.decode_predicted_events and not shared.recording_enabled(args),
                                          scene_label=f"{args.task} {args.condition} {rgb_config.label()}")
                rows, chunks, timings = [], [], []
                started = time.perf_counter()
                try:
                    shared.begin_recording(args, conn, hello, shared.request)
                    video.write(obs, 0, 0)
                    positional = (args, runtime, dvs, conn, hello, checkpoint_step, cameras,
                                  obs, video, rows, chunks, timings, heartbeat, output, trial)
                    if args.latency_aware:
                        outcome = shared.run_latency_loop(*positional, capture=capture_step,
                                                        prompt=module.PROMPT, stable_steps=stable_steps)
                    else:
                        outcome = run_paused_loop(*positional)
                    obs, success, reason, steps, replans, inference, round_trip, latency = outcome
                finally:
                    video.close()
                    if sys.exc_info()[0] is not None:
                        shared.end_recording(args, conn, shared.request)
                prediction_recording = shared.finish_recording(args, conn, hello, shared.request, video, rows, heartbeat)
                event_archive = runtime._eval_event_recorder.save(output / f"trial_{trial:03d}_events") if dynamic else None
                terminal = terminal_state(runtime, args.task, runtime._eval_opened)
                if success and not terminal_valid(args.task, spec["config"], terminal):
                    raise RuntimeError("runtime success disagrees with collector terminal predicate")
                final_video = output / f"trial_{trial:03d}_h264_{'success' if success else 'failure'}.mp4"
                video_path.rename(final_video)
                subprocess.run([video.ffmpeg, "-v", "error", "-i", str(final_video), "-f", "null", "-"], check=True)
                cv2.imwrite(str(output / f"trial_{trial:03d}_terminal.png"), shared.preview(obs, steps))
                result = dict(trial=trial, sim_seed=spec["seed"], steps=steps, frames=len(rows),
                    replans=replans, success=bool(success), reason=reason or ("success" if success else "step_limit"),
                    video=str(final_video), video_decoded=True, spec=spec, terminal=terminal,
                    stability_steps=stable_steps(runtime), trial_wall_seconds=time.perf_counter()-started,
                    inference_timing=shared.timing_stats(inference), round_trip_timing=shared.timing_stats(round_trip),
                    execution_chunks=chunks, execution_horizon_statistics=shared.horizon_statistics(chunks),
                    latency_timing=latency, rgb_ood=runtime._eval_rgb.metadata(), event_archive=event_archive,
                    prediction_recording=prediction_recording)
                if latency and latency.get("controller") == "continuous":
                    from .reporting import plan_statistics
                    result["continuous_plan_statistics"] = plan_statistics([latency])
                if timings:
                    result["step_timing"] = {k:shared.timing_stats([x[k] for x in timings]) for k in timings[0]}
                write_json(output / f"trial_{trial:03d}_steps.json", rows)
                write_json(output / f"trial_{trial:03d}_result.json", result)
                results.append(result)
                heartbeat.touch()
                summarize()
            successes = sum(bool(r["success"]) for r in results)
            write_json(output / "complete.json", dict(complete=True, task=args.task, sim_task=module.TASK,
                condition=args.condition, requested_trials=args.trials, trials=len(results), successes=successes,
                success_rate=None if args.interface_only else successes/args.trials,
                success_rate_denominator="requested_trials", interface_only=args.interface_only,
                formal_protocol=manifest["formal_protocol"], simulation_clock=shared.simulation_clock(args),
                video_decoded=all(r["video_decoded"] for r in results), results=results,
                breakdown=condition_breakdown(results, interface_only=args.interface_only)))
    except BaseException as exc:
        summarize(error=str(exc))
        raise
    finally:
        env.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Help and scene planning are usable on CPU login nodes without Isaac imports.
    cpu_only = "--plan-only" in argv or "--help" in argv or "-h" in argv
    launcher = None
    if not cpu_only:
        from piper_bench.backend.isaac import launcher_class
        launcher = launcher_class()
    parser = build_parser(launcher)
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    if args.plan_only:
        print(json.dumps(settings[-1], indent=2, allow_nan=False))
        return
    from piper_bench.backend.isaac import launch
    app = launch(args)
    exit_code = 0
    try:
        run(args, *settings)
    except BaseException as exc:
        exit_code = 1
        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "failure.json", dict(type=type(exc).__name__, message=str(exc),
                                                         traceback=traceback.format_exc()))
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
