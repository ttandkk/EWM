"""Collect Piper tasks with legacy and eight-dynamic-task sampling protocols."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import traceback

from piper_bench.tasks import load_task
from piper_bench.tasks import register_tasks
from piper_bench.collection.protocol import COLLECTION_TASKS
from piper_bench.collection.protocol import DYNAMIC_VERSION
from piper_bench.collection.protocol import NARROW_VERSION
from piper_bench.collection.protocol import CONTAINER_SPATIAL_VERSIONS
from piper_bench.collection.protocol import VERSION
from piper_bench.collection.protocol import base_config
from piper_bench.collection.protocol import check_attempts
from piper_bench.collection.protocol import episode_spec
from piper_bench.collection.protocol import make_plan
from piper_bench.collection.protocol import source_hashes
from piper_bench.collection.protocol import write_json
from piper_bench.collection.protocol import protocol_camera
from piper_bench.collection.sampling import CARTON_LOWWALL_VERSION
from piper_bench.collection.sampling import LOW_GEOMETRY_VERSION


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=COLLECTION_TASKS, required=True)
    p.add_argument("--collection-protocol", choices=(VERSION, DYNAMIC_VERSION, NARROW_VERSION, *CONTAINER_SPATIAL_VERSIONS, CARTON_LOWWALL_VERSION, LOW_GEOMETRY_VERSION),
                   help="Use the versioned eight-dynamic-task protocol; legacy tasks retain their default")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path,
                   help="Keep collection progress and intermediate audit files outside the dataset")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--split", choices=("train", "validation"), default="train")
    p.add_argument("--acceptance-trials", type=int, default=10)
    p.add_argument("--acceptance-rate", type=float, default=.9)
    p.add_argument("--acceptance-only", action="store_true",
                   help="Run expert preflight and diagnostics without creating a training dataset")
    p.add_argument("--boundary-preflight", action="store_true",
                   help="Push-T only: require all 35 fixed boundary cases before random acceptance")
    p.add_argument("--pathtracing-spp", type=int, default=64)
    p.add_argument("--writer-python", type=Path,
                   default=os.environ.get("PIPER_WRITER_PYTHON"),
                   help="Python from the independent writer environment; defaults to the configured writer profile")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-raw-events", action="store_true",
                   help="Keep unclipped XYTP events and aligned event videos alongside count PNGs")
    p.add_argument("--plan-only", action="store_true", help="Print sampling plan without Isaac or file writes")
    return p


def resolve_writer_python(explicit=None):
    """Resolve the independently configured dataset writer interpreter."""
    value = explicit or os.environ.get("PIPER_WRITER_PYTHON")
    if value is None:
        from piper_bench.deployment.config import load_config
        value = load_config().get("profiles", {}).get("writer", {}).get("python")
    if not value:
        raise ValueError("Configure the writer environment with 'ewm setup --profile writer' "
                         "or pass --writer-python; the simulator and writer use separate environments")
    value = Path(value).expanduser()
    if not value.is_file():
        raise ValueError(f"writer Python not found: {value}")
    # Keep the venv path: resolving bin/python's symlink bypasses pyvenv.cfg.
    return value.absolute()


def rollout(runtime, task, spec, progress, dvs=None, *, save_raw_events=False, frame_observer=None, oracle_factory=None):
    from collections import deque
    import numpy as np
    import torch
    from piper_bench.tasks.conveyor.core import CAMERAS
    from piper_bench.tasks.conveyor.core import CONTROL_HZ
    from piper_bench.sensors.capture import snapshot
    from piper_bench.sensors.event_frames import events_to_frame
    from piper_bench.collection.runtime import initialize
    from piper_bench.collection.runtime import step_result
    from piper_bench.collection.runtime import terminal_state
    from piper_bench.collection.runtime import terminal_valid
    from piper_bench.collection.runtime import frame
    env, name = runtime.env, spec["task"]
    if hasattr(runtime, "motion_enabled"):
        runtime.motion_enabled = False
    env.reset()
    zero = torch.tensor([[0., 0., 0., 0., 0., 0., getattr(runtime, "initial_gripper_action", 1.)]], device=env.device)
    settle_action = getattr(runtime, "settle_action", lambda: zero)
    for _ in range(45 if name.startswith("tennis_") else 60):
        task.set_views(env)
        env.step(settle_action())
    initial = initialize(runtime, spec)
    oracle = (oracle_factory or task.Oracle)(runtime)
    if name.startswith("tennis_"):
        runtime.collection_catch_window = deque(maxlen=runtime.tracker.window_samples)
    buffer, event_mass = [], {cam: 0 for cam in CAMERAS}
    if dvs is not None:
        dvs.reset(torch.tensor([0], dtype=torch.long, device="cuda:0"))
        dvs.recorder._events.pop(0, None)
        previous = snapshot(dvs, env)
    success, reason, opened = False, None, True
    for step in range(round(runtime.cfg.max_seconds * CONTROL_HZ)):
        # Legacy tasks keep their recorded initial hold. Tennis can launch at
        # t=0: let its expert consume that initial observation immediately,
        # matching the preview instead of adding an unobserved 33 ms delay.
        command_stage = oracle.stage
        action = zero if step == 0 and not name.startswith("tennis_") else oracle.action()
        if dvs is not None:
            dvs.warp_and_process(previous, previous, 1, runtime.time, 1/240)
            chunks = dvs.recorder._events.pop(0, {})
            # Integer control ticks avoid subtraction roundoff at the open left boundary.
            start_t, end_t = (step-1)/CONTROL_HZ, step/CONTROL_HZ
            if save_raw_events:
                from piper_bench.sensors.event_archive import checked_events
                raw = {cam: checked_events(chunks.get("DVS/"+cam, []), start_t, end_t)
                       for cam in CAMERAS}
                chunks = {"DVS/"+cam: [values] for cam, values in raw.items()}
            events = {cam: events_to_frame(chunks.get("DVS/"+cam, []), start_t, end_t)
                      for cam in CAMERAS}
            if step == 0 and any(np.any(e) for e in events.values()):
                raise RuntimeError("first event frame must be zero")
            for cam, e in events.items():
                event_mass[cam] += int(e[:, :, :2].sum())
            buffer.append(frame(runtime, name, action, events))
            if save_raw_events:
                buffer[-1]["_raw_events"] = raw
            if frame_observer is not None:
                frame_observer(buffer[-1], previous)
        previous_time = runtime.time
        if name == "turntable_cups":
            runtime.before_step()
        elif name.startswith("tennis_"):
            runtime.before_step(action)
        task.set_views(env)
        env.step(action)
        success, reason, opened = step_result(runtime, name, action)
        if name.startswith("tennis_"):
            from piper_bench.tasks.tennis.collection import success_sample
            runtime.collection_catch_window.append(success_sample(runtime))
        reason = getattr(oracle, "failure", None) or reason
        if reason:
            success = False
        if dvs is not None:
            current = snapshot(dvs, env)
            dvs.warp_and_process(previous, current, 8, previous_time, 1/240)
            previous = current
        diagnose = name in ('push_t', 'drawer_cup', 'turntable_cups', 'moving_carton')
        pose_diagnostic = diagnose and (step % 15 == 0 or command_stage != oracle.stage)
        if step % 60 == 0 or success or reason or pose_diagnostic:
            state = dict(task=name, split=spec["split"], slot=spec["slot"], retry=spec["retry"],
                         step=step+1, time=runtime.time, stage=oracle.stage, speed=spec["speed"],
                         success=success, reason=reason, poses=runtime.state()[0].tolist(),
                         tcp=runtime.state()[2].tolist())
            if diagnose:
                target_q = getattr(oracle, 'target_quaternion', None)
                state.update(command_stage=command_stage,
                             target_tcp=None if oracle.target is None else oracle.target.tolist(),
                             target_quaternion=None if target_q is None else target_q.tolist(),
                             tcp_quaternion=env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().tolist(),
                             robot_joints=env.scene['robot'].data.joint_pos[0].detach().cpu().tolist(),
                             robot_joint_targets=env.scene['robot'].data.joint_pos_target.torch[0].detach().cpu().tolist(),
                             orientation_error_rad=float(oracle.orientation_error) if math.isfinite(oracle.orientation_error) else None,
                             action=action[0].detach().cpu().tolist())
            if name == 'turntable_cups':
                state.update(object_index=oracle.index, held_offset=oracle.held_offset.tolist())
            if name == 'moving_carton':
                state.update(object_index=oracle.index, release_checks=getattr(oracle, 'release_checks', None),
                             stack_diagnostics=runtime.tracker.last_diagnostics)
            if name == "drawer_cup":
                state.update(opening=runtime.opening()[0],
                             cabinet_pose=env.scene["cabinet"].data.root_pose_w[0].detach().cpu().tolist())
            write_json(progress, state)
            print("[collect-progress] " + json.dumps(state), flush=True)
        if success or reason:
            break
    if dvs is not None and save_raw_events:
        # Preserve the final executed action's events, but never put future
        # events into the last pre-action observation/video frame.
        dvs.warp_and_process(previous, previous, 1, runtime.time, 1/240)
        chunks = dvs.recorder._events.pop(0, {})
        buffer[-1]["_raw_event_tail"] = {
            cam: checked_events(chunks.get("DVS/"+cam, []), step/CONTROL_HZ, runtime.time)
            for cam in CAMERAS}
    terminal = terminal_state(runtime, name, opened)
    if success and not terminal_valid(name, spec["config"], terminal):
        raise RuntimeError("runtime success disagrees with terminal predicate")
    if success and dvs is not None and not all(event_mass.values()):
        raise RuntimeError("successful trajectory has an empty event stream")
    record = dict(spec=spec, initial=initial, success=success, reason=reason or ("success" if success else "step_limit"),
                  frames=step+1, terminal=terminal, event_mass=event_mass, oracle_profile=oracle.PROFILE)
    return record, buffer


def collect(args, plan):
    import fcntl
    out = args.output.resolve()
    audit = args.audit_dir.resolve() if args.audit_dir else out.parent / (out.name + "_collection_audit")
    if audit == out or audit.is_relative_to(out):
        raise ValueError("audit directory must be outside the dataset")
    audit.mkdir(parents=True, exist_ok=args.resume)
    with (audit / ".collector.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another collector is using {out}") from exc
        from piper_bench.deployment.process import package_environment
        with package_environment() as environment:
            _collect(args, plan, audit, writer_environment=environment)


def _collect(args, plan, audit, *, writer_environment):
    import gymnasium as gym
    from piper_bench.sensors.capture import configure_renderer
    from piper_bench.sensors.capture import send_writer_message
    from piper_bench.sensors.capture import writer_reply
    from piper_bench.sensors.preview import create_event_camera
    from piper_bench.sensors.preview import event_metadata
    out = args.output.resolve()
    from piper_bench import SOURCE_LAYOUT_VERSION
    from piper_bench.collection.sampling import physics_hz
    # Existing output directories are never overwritten.
    task = load_task(args.task)
    config = dict(version=plan["version"], task=args.task, task_id=task.TASK, plan=plan,
                  task_prompt=task.PROMPT,
                  camera=protocol_camera(args.task, plan), base_config=base_config(args.task, plan).metadata(),
                  oracle_profile=task.Oracle.PROFILE, success_version=getattr(task,"SUCCESS_VERSION","existing_physical_predicates_v1"),
                  source_layout=SOURCE_LAYOUT_VERSION,
                  source_sha256=source_hashes(), pathtracing_spp=args.pathtracing_spp,
                  acceptance_trials=args.acceptance_trials, acceptance_rate=args.acceptance_rate,
                  initial_hold_steps=0 if args.task.startswith("tennis_") else 1,
                  control_hz=30, physics_hz=physics_hz(args.task), event_hz=240,
                  renderer="pathtracing", standard_imaging=True, text_overlays=False)
    if args.task == 'moving_carton' and plan['version'] in (CARTON_LOWWALL_VERSION, LOW_GEOMETRY_VERSION):
        from piper_bench.tasks.moving_carton.lowwall import VERSION as success_version
        config['success_version'] = success_version
        config['oracle_profile'] = 'moving_carton_stack_v11_lowwall_direct_transfer'
    if args.task == 'turntable_cups' and plan['version'] == LOW_GEOMETRY_VERSION:
        from piper_bench.tasks.turntable_cups.scene import BOWL_SUCCESS_VERSION
        config['success_version'] = BOWL_SUCCESS_VERSION
    if args.save_raw_events:
        from piper_bench.sensors.event_archive import CONTRACT
        config["event_archive"] = CONTRACT
    if args.boundary_preflight:
        config["boundary_preflight"] = "push_t_pose_v2_35_of_35"
    config_path = audit / "collection_config.json"
    attempts_path = audit / "attempts.json"
    if args.resume:
        if json.loads(config_path.read_text()) != config:
            raise ValueError("resume requires the same code, task, sampling plan, cameras and renderer")
        attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    else:
        if out.exists():
            raise FileExistsError(f"dataset exists: {out}; use --resume with matching settings")
        attempts = []
        write_json(config_path, config)
        write_json(attempts_path, attempts)
    register_tasks()
    configure_renderer(args.pathtracing_spp)
    env = gym.make(task.TASK, cfg=task.make_env_cfg(base_config(args.task, plan), device="cpu", seed=args.seed)).unwrapped
    writer = None
    try:
        runtime = task.Runtime(env, base_config(args.task, plan))
        preflights = [("acceptance", make_plan(args.task, args.acceptance_trials, args.seed, "acceptance",
                                             version=plan["version"]),
                       args.acceptance_rate)]
        if args.boundary_preflight:
            from piper_bench.tasks.push_t.protocol import boundary_plan
            preflights.insert(0, ("boundary_acceptance", boundary_plan(args.seed), 1.))
        for name, preflight, rate in preflights:
            acceptance_path = audit / (name+".json")
            if args.resume and acceptance_path.exists() and json.loads(acceptance_path.read_text())["passed"]:
                continue
            reports = []
            for slot in range(preflight["episodes"]):
                record, _ = rollout(runtime, task, episode_spec(preflight, slot), audit / "progress.json")
                reports.append(record)
                write_json(acceptance_path, dict(plan=preflight, results=reports, passed=False))
            successes = sum(r["success"] for r in reports)
            passed = successes >= math.ceil(preflight["episodes"]*rate)
            write_json(acceptance_path, dict(plan=preflight, results=reports, successes=successes, passed=passed))
            if not passed:
                raise RuntimeError(f"expert acceptance failed: {successes}/{preflight['episodes']}; inspect {acceptance_path}")
        if plan["version"] in CONTAINER_SPATIAL_VERSIONS:
            from piper_bench.collection.container_diagnostic import run as fixed_center_diagnostic
            fixed_center_diagnostic(runtime, task, preflights[-1][1], audit, rollout, args.resume)
        if args.acceptance_only:
            return
        resume_writer = args.resume and out.exists()
        if resume_writer:
            saved = json.loads((out / "meta/info.json").read_text())["total_episodes"]
            # Recover the single crash window between writer save and audit commit.
            pending = audit / "pending_save.json"
            if pending.exists():
                record = json.loads(pending.read_text())
                if len(attempts) == record["attempt"] and saved == record["saved_episode"]+1:
                    attempts.append(record)
                    write_json(attempts_path, attempts)
                    write_json(out / "meta/piper_task_attempts.json", attempts)
                elif len(attempts) == record["attempt"] and saved == record["saved_episode"]:
                    # No dataset episode committed: retry the same slot. Any
                    # staged event sidecars are atomically replaced on success.
                    pass
                elif not (len(attempts) == record["attempt"]+1 and attempts[-1] == record):
                    raise ValueError("interrupted writer save is ambiguous; keep data and inspect pending_save.json")
                pending.unlink()
            retry = check_attempts(plan, attempts, saved)
            for marker in (".collection_complete", ".export_complete"):
                (out / marker).unlink(missing_ok=True)
        else:
            saved, retry = 0, check_attempts(plan, attempts, 0)
        writer_python = resolve_writer_python(args.writer_python)
        writer = subprocess.Popen([str(writer_python), "-u", "-m", "piper_bench.collection.writer",
                                   "--task", args.task, "--output", str(out),
                                   "--collection-protocol", plan["version"],
                                   *(["--raw-event-archive"] if args.save_raw_events else []),
                                   "--repo-id", f"local/piper_{args.task}_{args.split}",
                                   *(["--resume"] if resume_writer else [])], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, env=writer_environment)
        if writer_reply(writer, "READY") != saved:
            raise ValueError("writer and attempt metadata disagree")
        write_json(out / "meta/piper_collection_config.json", config)
        write_json(out / "meta/piper_camera_calibration.json", protocol_camera(args.task, plan))
        with tempfile.TemporaryDirectory(prefix="piper_tasks_evis_") as temp:
            dvs = create_event_camera(env, Path(temp))
            sensor = event_metadata(dvs)
            # event_metadata is also used by post-step preview videos. Collection
            # includes t=0 and records pre-action observations in separate streams.
            sensor.update(video_clock="pre-action observation k at t=k/30; initial t=0 included",
                          video_layout="separate RGB and event streams per camera")
            write_json(out / "meta/piper_event_sensor.json", sensor)
            while saved < args.episodes and len(attempts) < args.max_attempts:
                record, buffer = rollout(runtime, task, episode_spec(plan, saved, retry), audit / "progress.json", dvs,
                                         save_raw_events=args.save_raw_events)
                record["attempt"] = len(attempts)
                if record["success"]:
                    record["saved_episode"] = saved
                    write_json(audit / "pending_save.json", record)
                    if args.save_raw_events:
                        from piper_bench.sensors.event_archive import write_episode
                        write_episode(out, saved, buffer, record["terminal"]["time"])
                    for item in buffer:
                        item.pop("_raw_events", None)
                        item.pop("_raw_event_tail", None)
                        send_writer_message(writer, dict(command="frame", frame=item))
                    send_writer_message(writer, dict(command="save"))
                    writer.stdin.flush()
                    if writer_reply(writer, "SAVED") != saved+1:
                        raise ValueError("unexpected saved episode index")
                    saved, retry = saved+1, 0
                else:
                    retry += 1
                attempts.append(record)
                write_json(attempts_path, attempts)
                write_json(out / "meta/piper_task_attempts.json", attempts)
                (audit / "pending_save.json").unlink(missing_ok=True)
                print(f"[collection] {args.task} saved={saved}/{args.episodes} attempts={len(attempts)}", flush=True)
                dvs.recorder._events.pop(0, None)
            if saved != args.episodes:
                raise RuntimeError(f"only {saved}/{args.episodes} successes in {len(attempts)} attempts; resume keeps pending speed")
    finally:
        try:
            if writer is not None:
                writer.stdin.close()
                try:
                    code = writer.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    writer.terminate()
                    try:
                        writer.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        writer.kill()
                        writer.wait()
                    raise
                if code:
                    raise RuntimeError(f"LeRobot writer failed: exit {code}")
        finally:
            env.close()
    write_json(out / "meta/piper_acceptance.json", json.loads(acceptance_path.read_text()))
    if args.boundary_preflight:
        write_json(out / "meta/piper_boundary_acceptance.json", json.loads((audit / "boundary_acceptance.json").read_text()))
    write_json(out / ".collection_complete", dict(episodes=saved, attempts=len(attempts), version=plan["version"]))


def main(argv=None):
    p = parser()
    early, _ = p.parse_known_args(argv)
    if early.plan_only:
        print(json.dumps(make_plan(early.task, early.episodes, early.seed, early.split,
                                   version=early.collection_protocol), indent=2))
        return
    from piper_bench.backend.isaac import launcher_class
    AppLauncher = launcher_class()
    AppLauncher.add_app_launcher_args(p)
    p.set_defaults(device="cpu")
    args = p.parse_args(argv)
    if args.boundary_preflight and args.task != "push_t":
        p.error("--boundary-preflight is only supported for Push-T")
    if (args.device != "cpu" or args.episodes < 1 or args.seed < 0
            or args.max_attempts < args.episodes or args.acceptance_trials < 1
            or not 0 < args.acceptance_rate <= 1 or args.pathtracing_spp < 1):
        p.error("require CPU physics, attempts >= episodes, positive acceptance trials/SPP and acceptance rate in (0,1]")
    if not args.acceptance_only:
        try:
            args.writer_python = resolve_writer_python(args.writer_python)
        except ValueError as exc:
            p.error(str(exc))
    import faulthandler
    faulthandler.dump_traceback_later(300)
    try:
        from piper_bench.backend.isaac import launch
        app = launch(args)
    finally:
        faulthandler.cancel_dump_traceback_later()
    exit_code = 0
    try:
        # Isaac's platform-info fork can deadlock if NumPy/OpenBLAS is imported
        # before SimulationApp. Keep sampling and task imports after startup.
        plan = make_plan(args.task, args.episodes, args.seed, args.split, version=args.collection_protocol)
        collect(args, plan)
    except BaseException:
        # Kit fast shutdown exits the process, so print before closing and pass
        # the real status explicitly instead of silently reporting success.
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
