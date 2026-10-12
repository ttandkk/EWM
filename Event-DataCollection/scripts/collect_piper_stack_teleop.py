"""Headless browser teleoperation for aligned Piper RGB/event-frame LeRobot demos.

Launch with Isaac Lab's Python, for example:
  cd /projects/hdd/myproject/CVPR2027/Event-DataCollection
  /projects/hdd/myproject/CVPR2027/isaacsim601/env/bin/python \
    scripts/collect_piper_stack_teleop.py \
    --output /projects/hdd/myproject/CVPR2027/datasets/piper_teleop_test --episodes 1

The collector reuses the production scene, EVIS and writer functions without
running the oracle. Browser preview and commands are served on a local TCP port.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from piper_wrist_camera import (
    CALIBRATION_ID,
    CAM0_EYE_WORLD,
    CAM0_TARGET_WORLD,
    CAM0_FOCAL_LENGTH_CM,
    CAM0_HORIZONTAL_APERTURE_CM,
    WRIST_EYE_LOCAL,
    WRIST_TARGET_LOCAL,
    WRIST_FOCAL_LENGTH_CM,
    WRIST_HORIZONTAL_APERTURE_CM,
)


def _teleop_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--teleop-host", default="127.0.0.1")
    parser.add_argument("--teleop-port", type=int, default=8765)
    parser.add_argument("--preview-fps", type=float, default=10)
    parser.add_argument("--pos-step", type=float, default=0.015,
                        help="Maximum raw relative-IK translation per control step.")
    parser.add_argument("--rot-step", type=float, default=0.025,
                        help="Maximum raw relative-IK rotation per control step.")
    parser.add_argument("--command-timeout", type=float, default=0.35)
    parser.add_argument("--wrist-eye-local", type=float, nargs=3, default=WRIST_EYE_LOCAL,
                        metavar=("X", "Y", "Z"), help="Camera eye offset in the TCP frame, metres.")
    parser.add_argument("--wrist-target-local", type=float, nargs=3, default=WRIST_TARGET_LOCAL,
                        metavar=("X", "Y", "Z"), help="Look-at point in the TCP frame, metres.")
    parser.add_argument("--wrist-focal-length", type=float, default=WRIST_FOCAL_LENGTH_CM,
                        help="Wrist camera focal length in Isaac Lab cm; lower gives a wider view.")
    parser.add_argument("--camera-probe-output", type=Path,
                        help="With --startup-smoke-test, save one static PNG per camera here.")
    parser.add_argument("--startup-smoke-test", action="store_true",
                        help="Initialize the scene, EVIS and two cameras, then exit without opening a dataset.")
    args, collector_args = parser.parse_known_args()
    if "--headless" not in collector_args:
        collector_args.append("--headless")
    if "--enable_cameras" not in collector_args:
        collector_args.append("--enable_cameras")
    # Keep the oracle collector's proven path-tracing/64-SPP defaults.  The
    # realtime override produced stochastic RGB and millions of false events.
    return args, collector_args


teleop_args, forwarded_args = _teleop_args()
original_argv = sys.argv
try:
    sys.argv = [str(Path(__file__).with_name("collect_piper_stack_evis.py")), *forwarded_args]
    import collect_piper_stack_evis as collector
finally:
    sys.argv = original_argv

import numpy as np
import torch
import gymnasium as gym
from PIL import Image

from piper_teleop_server import TeleopServer, TeleopState
from piper_teleop_geometry import wrist_planar_delta


def _jpeg(rgb: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="JPEG", quality=76, subsampling=0)
    return stream.getvalue()


def _set_wrist_view(env) -> None:
    """Set the adjustable TCP-local wrist view; defaults match the collector."""
    ee_frame = env.scene["ee_frame"].data
    tcp_pos = ee_frame.target_pos_w[:, 0, :]
    tcp_quat = ee_frame.target_quat_w[:, 0, :]
    eye_local = torch.tensor([teleop_args.wrist_eye_local], dtype=torch.float32,
                             device=env.device).expand_as(tcp_pos)
    target_local = torch.tensor([teleop_args.wrist_target_local], dtype=torch.float32,
                                device=env.device).expand_as(tcp_pos)
    eye = tcp_pos + collector.quat_apply(tcp_quat, eye_local)
    target = tcp_pos + collector.quat_apply(tcp_quat, target_local)
    env.scene["cam_wrist"].set_world_poses_from_view(eye, target)


def _reset(env, dvs):
    env.sim.reset()
    env.reset()
    collector.set_fixed_view(env)
    _set_wrist_view(env)
    warmup = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=env.device)
    for _ in range(collector.RESET_WARMUP_STEPS):
        collector.set_fixed_view(env)
        _set_wrist_view(env)
        env.step(warmup)
    collector.set_fixed_view(env)
    _set_wrist_view(env)
    dvs.reset(torch.tensor([0], dtype=torch.long, device=env.device))
    dvs.recorder._events.pop(0, None)
    return dvs.snapshot(), 0.0


def _event_frame_at_observation(dvs, previous, time_s: float):
    # Identical to the oracle collector: close (t_(i-1), t_i] before row i.
    dvs.warp_and_process(previous, previous, 1, time_s, 1.0 / collector.EVENT_HZ)
    chunks = dvs.recorder._events.pop(0, {})
    return {
        name: collector.events_to_frame(
            chunks.get(f"DVS/{name}", []), time_s - 1.0 / collector.CONTROL_HZ, time_s
        )
        for name in collector.CAMERA_NAMES
    }


def _rgb_views(env) -> tuple[np.ndarray, np.ndarray]:
    views = []
    for name in collector.CAMERA_NAMES:
        image = env.scene[name].data.output["rgb"][0, ..., :3]
        views.append(image.detach().cpu().numpy().astype(np.uint8))
    return views[0], views[1]


def _wrist_relative_action(env, values: tuple[float, ...]) -> torch.Tensor:
    """W/S and A/D track wrist-image up/right; Q/E remains world Z."""
    ee_frame = env.scene["ee_frame"].data
    tcp_quat = ee_frame.target_quat_w[:, 0, :]
    view_local = torch.tensor(
        [tuple(t - e for e, t in zip(teleop_args.wrist_eye_local, teleop_args.wrist_target_local))],
        dtype=torch.float32, device=env.device,
    )
    forward_world = collector.quat_apply(tcp_quat, view_local)[0].detach().cpu().numpy()
    dx, dy = wrist_planar_delta(forward_world, values[0], values[1])
    return torch.tensor([[dx, dy, *values[2:]]], dtype=torch.float32, device=env.device)


def _write_capture_metadata(output: Path, args) -> None:
    path = output / "meta" / "teleop_capture.json"
    shared_preset = (
        tuple(teleop_args.wrist_eye_local) == WRIST_EYE_LOCAL
        and tuple(teleop_args.wrist_target_local) == WRIST_TARGET_LOCAL
        and teleop_args.wrist_focal_length == WRIST_FOCAL_LENGTH_CM
    )
    expected = {
        "source": "browser_teleoperation",
        "renderer_mode": args.renderer_mode,
        "pathtracing_spp": args.pathtracing_spp if args.renderer_mode == "pathtracing" else None,
        "translation_frame": "wrist_image_xy_world_z_v1",
        "camera_calibration_id": CALIBRATION_ID if shared_preset else "custom",
        "cam0_eye_world_m": list(CAM0_EYE_WORLD),
        "cam0_target_world_m": list(CAM0_TARGET_WORLD),
        "cam0_focal_length_cm": CAM0_FOCAL_LENGTH_CM,
        "cam0_horizontal_aperture_cm": CAM0_HORIZONTAL_APERTURE_CM,
        "wrist_eye_local": list(teleop_args.wrist_eye_local),
        "wrist_target_local": list(teleop_args.wrist_target_local),
        "wrist_focal_length_cm": teleop_args.wrist_focal_length,
        "wrist_horizontal_aperture_cm": WRIST_HORIZONTAL_APERTURE_CM,
    }
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError(f"cannot mix teleop renderer settings in {output}")
        return
    if args.resume:
        raise ValueError(f"cannot resume a non-teleop dataset without {path}")
    path.write_text(json.dumps({
        **expected,
        "control_hz": collector.CONTROL_HZ,
        "event_warp": collector.WARP,
        "event_contract": "aligned_event_frame_pre_action_v1",
        "pos_step": teleop_args.pos_step,
        "rot_step": teleop_args.rot_step,
        "note": "Only 64-SPP path tracing has been validated as stable for event-frame training.",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = collector.args_cli
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--episodes and --max_steps must be positive")
    eye_local = np.asarray(teleop_args.wrist_eye_local, dtype=np.float64)
    target_local = np.asarray(teleop_args.wrist_target_local, dtype=np.float64)
    if (not np.isfinite(eye_local).all() or not np.isfinite(target_local).all()
            or np.linalg.norm(target_local - eye_local) < 0.02):
        raise ValueError("wrist eye/target offsets must be finite and at least 2 cm apart")
    if not 5.0 <= teleop_args.wrist_focal_length <= 25.0:
        raise ValueError("--wrist-focal-length must be in [5, 25] cm")
    if teleop_args.camera_probe_output and not teleop_args.startup_smoke_test:
        raise ValueError("--camera-probe-output requires --startup-smoke-test")
    if not 0 < teleop_args.preview_fps <= collector.CONTROL_HZ:
        raise ValueError("--preview-fps must be in (0, 30]")
    output = args.output.resolve()
    if not teleop_args.startup_smoke_test and not args.writer_python.is_file():
        raise FileNotFoundError(args.writer_python)
    if not teleop_args.startup_smoke_test and not args.resume and output.exists():
        raise FileExistsError(f"{output} exists; choose a fresh path or --resume")

    torch.manual_seed(args.seed)
    collector.configure_clean_event_renderer()
    env_cfg, success_term = collector.configure_env()
    env_cfg.scene.cam_wrist.spawn.focal_length = teleop_args.wrist_focal_length
    env = gym.make(collector.TASK, cfg=env_cfg).unwrapped
    collector.add_cube_patterns(env)
    collector.alias_legacy_hdr_output(env.scene, collector.CAMERA_NAMES)
    if teleop_args.startup_smoke_test:
        try:
            with tempfile.TemporaryDirectory(prefix="piper_teleop_smoke_") as event_dir:
                dvs = collector.DVSCamera.from_scene(
                    env.scene, list(collector.CAMERA_NAMES), out_dir=event_dir,
                    compression="lzf", event_source="hdr",
                    antialiasing=None if args.renderer_mode == "pathtracing" else args.dvs_antialiasing,
                )
                previous, time_s = _reset(env, dvs)
                event_frames = _event_frame_at_observation(dvs, previous, time_s)
                cam0, wrist = _rgb_views(env)
                assert all(image.shape == (256, 256, 3) for image in (cam0, wrist))
                assert all(image.shape == (256, 256, 3) for image in event_frames.values())
                assert all(np.count_nonzero(image) == 0 for image in event_frames.values())
                action = _wrist_relative_action(env, (0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
                assert action.shape == (1, 7) and bool(torch.isfinite(action).all())
                assert abs(float(torch.linalg.vector_norm(action[0, :2])) - 0.015) < 1e-5
                if teleop_args.camera_probe_output:
                    probe_dir = teleop_args.camera_probe_output.resolve()
                    probe_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(cam0).save(probe_dir / "cam0.png")
                    Image.fromarray(wrist).save(probe_dir / "cam_wrist.png")
                    print(f"[teleop] camera probe PNGs: {probe_dir}", flush=True)
                print("[teleop] startup smoke passed: dual RGB, zero event-frames and wrist action", flush=True)
        finally:
            env.close()
        return

    state = TeleopState(pos_step=teleop_args.pos_step, rot_step=teleop_args.rot_step,
                        timeout_s=teleop_args.command_timeout)
    server = TeleopServer(state, host=teleop_args.teleop_host, port=teleop_args.teleop_port,
                          preview_fps=teleop_args.preview_fps)
    writer: subprocess.Popen | None = None
    try:
        server.start()
        print(f"[teleop] browser URL (through a private tunnel): "
              f"http://127.0.0.1:{teleop_args.teleop_port}/?token={server.token}", flush=True)
        writer = subprocess.Popen(
            [str(args.writer_python), "-u",
             str(Path(collector.__file__).with_name("write_piper_stack_lerobot.py")),
             "--output", str(output), "--repo-id", args.repo_id,
             *(["--resume"] if args.resume else [])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        saved = collector.writer_reply(writer, "READY")
        _write_capture_metadata(output, args)
        with tempfile.TemporaryDirectory(prefix="piper_teleop_evis_") as event_dir:
            dvs = collector.DVSCamera.from_scene(
                env.scene, list(collector.CAMERA_NAMES), out_dir=event_dir,
                compression="lzf", event_source="hdr",
                antialiasing=None if args.renderer_mode == "pathtracing" else args.dvs_antialiasing,
            )
            previous, time_s = _reset(env, dvs)
            frames: list[dict] = []
            step = success_steps = attempts = 0
            attempts += 1
            pending_event_frames = None
            last_preview = 0.0
            server.set_message("Connected browser must press Continue to start.")
            while saved < args.episodes and collector.simulation_app.is_running():
                reset = False
                for command in state.take_commands():
                    if command == "save":
                        if success_steps < 10 or not frames:
                            server.set_message("Save refused: success has not held for 10 steps.")
                            continue
                        if len(frames) <= 24:
                            server.set_message("Save refused: need at least 25 frames for EWM K=24.")
                            continue
                        for frame in frames:
                            collector.send_writer_message(writer, {"command": "frame", "frame": frame})
                        collector.send_writer_message(writer, {"command": "save"})
                        writer.stdin.flush()
                        saved = collector.writer_reply(writer, "SAVED")
                        server.set_message(f"Saved successful demonstration {saved}/{args.episodes}.")
                        print(f"[teleop] saved {saved}/{args.episodes}: {len(frames)} frames", flush=True)
                        reset = True
                    elif command in ("reset", "discard"):
                        server.set_message(f"Discarded {len(frames)} unsaved frames; resetting.")
                        reset = True
                if saved >= args.episodes:
                    break
                if step >= args.max_steps:
                    server.set_message(f"Maximum {args.max_steps} steps reached; discarded attempt.")
                    reset = True
                if reset:
                    dvs.recorder._events.pop(0, None)
                    previous, time_s = _reset(env, dvs)
                    frames, step, success_steps = [], 0, 0
                    pending_event_frames = None
                    attempts += 1
                    continue

                _set_wrist_view(env)
                if pending_event_frames is None:
                    pending_event_frames = _event_frame_at_observation(dvs, previous, time_s)
                now = time.monotonic()
                if now - last_preview >= 1 / teleop_args.preview_fps:
                    cam0, wrist = _rgb_views(env)
                    server.publish(_jpeg(cam0), _jpeg(wrist), {
                        "attempt": attempts, "saved": saved, "step": step,
                        "stable_success_steps": success_steps,
                        "renderer": args.renderer_mode,
                    })
                    last_preview = now

                values = state.action()
                if values is None:
                    time.sleep(0.02)
                    continue
                action = _wrist_relative_action(env, values)
                frame = collector.capture_frame(env, action, pending_event_frames)
                tick = time.monotonic()
                _, _, terminated, truncated, _ = env.step(action)
                current = dvs.snapshot()
                if bool(torch.logical_or(terminated, truncated).any()):
                    server.set_message("Environment terminated; discarding and resetting.")
                    previous, time_s = _reset(env, dvs)
                    frames, step, success_steps = [], 0, 0
                    pending_event_frames = None
                    attempts += 1
                    continue
                dvs.warp_and_process(previous, current, collector.WARP,
                                     time_s, 1.0 / collector.EVENT_HZ)
                frames.append(frame)
                step += 1
                success_steps = (success_steps + 1 if
                                 bool(success_term.func(env, **success_term.params)[0]) else 0)
                if success_steps == 10:
                    server.set_message("Success stable for 10 steps; Save is now enabled.")
                previous, time_s = current, time_s + 1.0 / collector.CONTROL_HZ
                pending_event_frames = None
                time.sleep(max(0.0, 1.0 / collector.CONTROL_HZ - (time.monotonic() - tick)))
        print(f"[teleop] complete: {saved} demonstrations after {attempts} attempts -> {output}", flush=True)
    finally:
        server.stop()
        env.close()
        if writer is not None:
            try:
                writer.stdin.close()
            except BrokenPipeError:
                pass
            writer.wait(timeout=300)
            if writer.returncode:
                raise RuntimeError(f"LeRobot writer exited with code {writer.returncode}")


if __name__ == "__main__":
    try:
        main()
    finally:
        collector.simulation_app.close()
