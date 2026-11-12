"""Run one or more closed-loop EWM trials in the training Piper/EVIS scene.

This client runs under the Isaac Sim Python environment. The EWM model lives
in a separate OpenWAM WebSocket server on the same node.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "OpenWAM"))
sys.path.insert(0, str(PROJECT / "Event-DataCollection/scripts"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ready-file", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--resume", action="store_true", help="Resume a contiguous prefix of completed trials")
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=500)
parser.add_argument("--sampling-steps", type=int, default=4)
parser.add_argument("--execution-horizon", "--action-chunk-size", dest="execution_horizon",
                    type=int, default=16, help="Number of predicted actions executed before replanning")
parser.add_argument("--decode-predicted-events", action="store_true",
                    help="Decode future event frames and add a predicted-event row to the MP4")
parser.add_argument("--predicted-event-threshold", type=float, default=1 / 30)
parser.add_argument("--predicted-event-min-neighbors", type=int, default=2)
parser.add_argument("--profile-timing", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--seed", type=int, default=42, help="Base seed for model sampling")
parser.add_argument("--scene-seed", type=int, help="Separate seed for simulator resets (defaults to --seed)")
parser.add_argument("--scene-seed-mode", choices=("per_trial", "stream"), default="per_trial",
                    help="Use an independent seed per trial, or seed once like the data collector")
parser.add_argument("--checkpoint-step", type=int)
parser.add_argument("--renderer-mode", choices=("pathtracing", "realtime"), default="pathtracing")
parser.add_argument("--pathtracing-spp", type=int, default=64)
parser.add_argument("--rgb-ood", action=argparse.BooleanOptionalAction, default=False,
                    help="Enable RGB-only lighting and directional motion-blur OOD together")
parser.add_argument("--rgb-lighting-ood", action=argparse.BooleanOptionalAction, default=None,
                    help="Override lighting OOD separately (RGB postprocessing only)")
parser.add_argument("--rgb-motion-blur-ood", action=argparse.BooleanOptionalAction, default=None,
                    help="Override directional motion blur separately (RGB only)")
parser.add_argument("--rgb-ood-mode",
                    choices=("legacy", "clean", "blur-3", "blur-5",
                             "exposure-0.8", "exposure-0.6", "exposure-0.4"),
                    default="legacy",
                    help="Fixed pure RGB corruption; legacy retains existing random OOD switches")
parser.add_argument("--rgb-ood-severity", choices=("mild", "moderate", "strong"), default="moderate")
parser.add_argument("--rgb-ood-seed", type=int, help="Independent OOD seed; defaults to model --seed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.rgb_ood_mode != "legacy" and (
        args.rgb_ood or args.rgb_lighting_ood is not None or args.rgb_motion_blur_ood is not None):
    parser.error("--rgb-ood-mode cannot combine with legacy RGB OOD component switches")
scene_seed = args.seed if args.scene_seed is None else args.scene_seed
if not 0 <= scene_seed < 2**63:
    parser.error("scene-seed must be in [0, 2**63)")
if min(args.trials, args.max_steps, args.sampling_steps, args.execution_horizon, args.pathtracing_spp) <= 0:
    parser.error("trials, max-steps, sampling-steps and execution-horizon must be positive")
if not 0 < args.predicted_event_threshold <= 1 or not 1 <= args.predicted_event_min_neighbors <= 9:
    parser.error("invalid predicted-event threshold or neighbor filter")
rgb_ood_seed = args.seed if args.rgb_ood_seed is None else args.rgb_ood_seed
if not 0 <= rgb_ood_seed < 2**63:
    parser.error("RGB OOD seed must be in [0,2**63)")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
import cv2
import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import quat_apply

from _isaacsim6_compat import (
    HDR_DATA_TYPE,
    alias_legacy_hdr_output,
    install_legacy_extension_utils,
    migrate_piper_stack_cfg,
    quat_from_wxyz,
)
from piper_wrist_camera import (
    CAM0_EYE_WORLD,
    CAM0_TARGET_WORLD,
    CAM0_FOCAL_LENGTH_CM,
    CAM0_HORIZONTAL_APERTURE_CM,
    WRIST_EYE_LOCAL,
    WRIST_TARGET_LOCAL,
    WRIST_FOCAL_LENGTH_CM,
    WRIST_HORIZONTAL_APERTURE_CM,
)

install_legacy_extension_utils()
import agx_teleop.tasks  # noqa: F401
from dvs_gen.sensors import DVSCamera
from pxr import Sdf, UsdShade
from websockets.sync.client import connect as ws_connect

from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message
from scripts.eval_ewm.ewm_eval_time_audit import EVENT_TIME_CONTRACT
from scripts.eval_ewm.ewm_event_visualization import predicted_event_counts
from scripts.eval_ewm.ewm_eval_recovery import archive_interrupted_trial, load_completed_results
from scripts.eval_ewm.ewm_scene_rng import restore_scene_rng, save_scene_rng, state_path
from scripts.eval_ewm.ewm_eval_summary import write_summary as _write_summary
from scripts.eval_ewm.ewm_eval_summary import horizon_statistics
from scripts.eval_ewm.ewm_rgb_ood import RGBOODConfig, TrialRGBOOD, check_resume_rgb_ood

rgb_ood_config = RGBOODConfig(
    lighting=args.rgb_ood if args.rgb_lighting_ood is None else args.rgb_lighting_ood,
    motion_blur=args.rgb_ood if args.rgb_motion_blur_ood is None else args.rgb_motion_blur_ood,
    severity=args.rgb_ood_severity, seed=rgb_ood_seed, mode=args.rgb_ood_mode,
)

TASK = "Isaac-Stack-Cube-Piper-IK-Rel-v0"
PROMPT = "Stack the blue-and-yellow patterned block on the red-and-white block, then stack the green-and-white block on top."
CONTROL_HZ = 30
WARP = 8
CAMERAS = ("cam0", "cam_wrist")
RESET_WARMUP_STEPS = 60


def camera(name: str) -> CameraCfg:
    fixed = name == "cam0"
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        update_period=1.0 / CONTROL_HZ, height=256, width=256,
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", HDR_DATA_TYPE],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM0_FOCAL_LENGTH_CM if fixed else WRIST_FOCAL_LENGTH_CM,
            focus_distance=400.0,
            horizontal_aperture=CAM0_HORIZONTAL_APERTURE_CM if fixed else WRIST_HORIZONTAL_APERTURE_CM,
            clipping_range=(0.1, 20.0) if fixed else (0.03, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.55, 0.75) if fixed else (0.0, 0.0, 0.0),
            rot=quat_from_wxyz((0.7071, 0.0, 0.7071, 0.0)) if fixed
            else quat_from_wxyz((1.0, 0.0, 0.0, 0.0)),
            convention="world",
        ),
    )


def set_views(env) -> None:
    eye = torch.tensor([CAM0_EYE_WORLD], dtype=torch.float32, device=env.device)
    target = torch.tensor([CAM0_TARGET_WORLD], dtype=torch.float32, device=env.device)
    env.scene["cam0"].set_world_poses_from_view(eye, target)
    from piper_wrist_camera import set_wrist_view
    set_wrist_view(env)


def configure_env():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1, use_fabric=True)
    migrate_piper_stack_cfg(cfg)
    from piper_scene_background import configure_room_background
    configure_room_background(cfg.scene)
    cfg.seed = scene_seed  # Stream mode matches the collector: seed once, then reset without a seed.
    cfg.events.init_piper_arm_pose = None
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 1.0 / 120.0, 4, 4
    cfg.episode_length_s = float(args.max_steps + RESET_WARMUP_STEPS + 2) / CONTROL_HZ
    cfg.scene.cam0, cfg.scene.cam_wrist = camera("cam0"), camera("cam_wrist")
    from piper_wrist_camera import configure_wrist_camera
    configure_wrist_camera(cfg.scene.cam_wrist)
    cfg.scene.light.spawn.color, cfg.scene.light.spawn.intensity = (1.0, 1.0, 1.0), 6000.0
    cfg.scene.table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.16, 0.16, 0.16), roughness=1.0, metallic=0.0)
    cfg.scene.table.spawn.visual_material_path = "visual_material_clean_evis"
    cfg.scene.clean_tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CleanTabletop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0001)),
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.90, 0.0002),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.16, 0.16, 0.16), roughness=1.0, metallic=0.0),
        ),
    )
    success_term = cfg.terminations.success
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    return cfg, success_term


def apply_patterns(env) -> None:
    texture_dir = PROJECT / "Event-DataCollection/assets/piper_cube_patterns"
    stage = sim_utils.get_current_stage()
    for cube, filename in {
        "cube_1": "red_stripes.png",
        "cube_2": "blue_cross.png",
        "cube_3": "green_checker.png",
    }.items():
        material_path = f"/World/Looks/EvisCubePatterns/{cube}"
        surface = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, material_path + "/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, material_path + "/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(texture_dir / filename)))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb")
        material = UsdShade.Material.Define(stage, material_path)
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        cube_root = env.scene[cube].root_physx_view.prim_paths[0]
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(cube_root)).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def event_voxel(chunks, start_t: float, end_t: float) -> tuple[np.ndarray, int]:
    """Match EWMDataset's causal (start,end] bin and OFF/ON count clipping."""
    output = np.zeros((2, 256, 256), dtype=np.float32)
    total = 0
    for x, y, t, p in chunks:
        keep = (t > start_t) & (t <= end_t)
        if not np.any(keep):
            continue
        x, y, p = x[keep], y[keep], p[keep]
        if np.any(x >= 256) or np.any(y >= 256) or not np.all((p == -1) | (p == 1)):
            raise ValueError("EVIS coordinates or polarity violate training codec")
        channel = (p > 0).astype(np.int64)
        flat = (channel * 256 + y.astype(np.int64)) * 256 + x.astype(np.int64)
        output += np.bincount(flat, minlength=2 * 256 * 256).reshape(2, 256, 256)
        total += len(x)
    return np.minimum(output, 15.0)[None], total


def simulation_time(env) -> float:
    """Return Isaac 6 simulation time from its monotonic physics-step counter."""
    return float(env.sim.get_physics_step_count()) * float(env.sim.get_physics_dt())


def capture_step(env, dvs, action: torch.Tensor, *, event_cameras=CAMERAS, profile_timing=False,
                 rgb_ood=None) -> tuple[dict, bool, bool, dict]:
    capture_started = time.perf_counter() if profile_timing else None
    phase_started = capture_started
    profile = {}

    def mark(name):
        nonlocal phase_started
        if phase_started is not None:
            now = time.perf_counter()
            profile[name] = now - phase_started
            phase_started = now

    start_t = simulation_time(env)
    previous = dvs.snapshot()
    mark("snapshot_before")
    set_views(env)
    mark("set_views")
    _, reward, terminated, truncated, info = env.step(action)
    mark("env_step")
    end_t = simulation_time(env)
    if not math.isfinite(end_t) or abs(end_t - start_t - 1 / CONTROL_HZ) > 0.001:
        raise ValueError(f"control/event interval is not 30 Hz: {start_t} -> {end_t}")
    current = dvs.snapshot()
    mark("snapshot_after")
    dvs.warp_and_process(previous, current, WARP, start_t, 1.0 / (CONTROL_HZ * WARP))
    # Match the collector: emit the right endpoint before recording this row.
    dvs.warp_and_process(current, current, 1, end_t, 1.0 / (CONTROL_HZ * WARP))
    mark("dvs_warp_process")
    chunks = dvs.recorder._events.pop(0, {})
    event_frames = [event_voxel(chunks.get(f"DVS/{name}", []), start_t, end_t)
                    for name in event_cameras]
    voxel = np.concatenate([frame for frame, _ in event_frames], axis=0)
    event_counts = {name: count for name, (_, count) in zip(event_cameras, event_frames, strict=True)}
    mark("event_voxelize")
    rgb = np.stack([
        env.scene[name].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        for name in CAMERAS
    ]).transpose(0, 3, 1, 2).copy()
    mark("rgb_readback")
    state = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32, copy=True)
    mark("state_readback")
    if rgb.shape != (2, 3, 256, 256) or rgb.dtype != np.uint8 or state.shape != (8,):
        raise ValueError(f"unexpected simulator RGB/state shapes: {rgb.shape}, {state.shape}")
    # EVIS snapshots/warping/voxelization above always use clean simulator data.
    # Only replace the copied RGB observation, never the camera buffer or events.
    if rgb_ood is not None and rgb_ood.config.enabled:
        rgb = rgb_ood.apply_rgb(rgb)
        mark("rgb_ood")
    if capture_started is not None:
        profile["capture_total"] = time.perf_counter() - capture_started
    observation = {
        "obs_video": rgb, "obs_voxel": voxel, "proprio": state,
        "timing": {
            "obs_video_time_s": end_t, "proprio_time_s": end_t,
            "obs_event_start_s": start_t, "obs_event_end_s": end_t,
        },
        "event_count": sum(event_counts.values()), "event_counts": event_counts,
    }
    if profile_timing:
        observation["profile_seconds"] = profile
    done = bool(torch.logical_or(terminated, truncated).any())
    return observation, done, bool(terminated.any()), {"reward": float(reward[0]), "info": info}


def request(conn, header: dict, arrays=None):
    conn.send(encode_message(header, arrays))
    reply, values = decode_message(conn.recv(timeout=600))
    if not reply.get("ok"):
        raise RuntimeError(f"EWM server: {reply.get('error', 'unknown failure')}")
    return reply, values


def summarize_step_timings(records: list[dict], trial_wall_seconds: float) -> dict:
    """Aggregate per-control-step host wall timings without hiding tail latency."""
    phases = {}
    for name in records[0]:
        values = np.asarray([record[name] for record in records], dtype=np.float64)
        total = float(values.sum())
        phases[name] = {
            "total_seconds": total,
            "mean_seconds": float(values.mean()),
            "p50_seconds": float(np.percentile(values, 50)),
            "p95_seconds": float(np.percentile(values, 95)),
            "max_seconds": float(values.max()),
            "share_of_trial": total / trial_wall_seconds,
        }
    return phases


def write_summary(output: Path, *, max_steps: int, sampling_steps: int,
                  requested_trials: int, attempted_trials: int,
                  results: list[dict], decode_predicted_events: bool = False,
                  error: str | None = None) -> None:
    _write_summary(output, max_steps=max_steps, sampling_steps=sampling_steps,
                   requested_trials=requested_trials, attempted_trials=attempted_trials,
                   results=results, task_name=TASK,
                   decode_predicted_events=decode_predicted_events, error=error,
                   rgb_ood=rgb_ood_config.metadata())

class TrialVideo:
    """Stream RGB, observed events, and optionally predicted events to H.264."""

    def __init__(self, path: Path, event_cameras: tuple[str, ...], *, show_prediction: bool = False,
                 rgb_ood_label: str = ""):
        self.event_cameras = event_cameras
        self.rgb_ood_label = rgb_ood_label
        self.show_prediction = show_prediction
        bundled = sorted((PROJECT / "envs/openwam/lib").glob(
            "python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        ))
        system_ffmpeg = Path("/usr/bin/ffmpeg")
        if bundled:
            ffmpeg = str(bundled[0])
        elif system_ffmpeg.is_file():
            ffmpeg = str(system_ffmpeg)
        else:
            ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for evaluation video output")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-video_size", "512x768" if show_prediction else "512x512",
                "-framerate", str(CONTROL_HZ),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")

    @staticmethod
    def _label(image: np.ndarray, text: str) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
        cv2.putText(image, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def write(self, observation: dict, step: int, success_steps: int,
              *, predicted_counts: np.ndarray | None = None, replan_label: str = "") -> None:
        rgb = observation["obs_video"].transpose(0, 2, 3, 1)
        rgb_row = np.hstack((
            self._label(rgb[0], f"cam0 step={step}"),
            self._label(rgb[1], f"wrist success={success_steps}"),
        ))
        if getattr(self, "rgb_ood_label", ""):
            cv2.rectangle(rgb_row, (0, 232), (512, 256), (0, 0, 0), -1)
            cv2.putText(rgb_row, self.rgb_ood_label, (7, 249),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        event_images = {}
        for name, counts in zip(self.event_cameras, observation["obs_voxel"], strict=True):
            image = np.full((256, 256, 3), 255, dtype=np.uint8)
            image[counts[0] > 0] = (255, 0, 0)  # OFF: blue in BGR.
            image[counts[1] > 0] = (0, 0, 255)  # ON: red in BGR.
            cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
            cv2.putText(image, f"{name} events={observation['event_counts'][name]}",
                        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            event_images[name] = image
        event_row = np.hstack(tuple(
            event_images[name] if name in event_images else np.full((256, 256, 3), 255, dtype=np.uint8)
            for name in CAMERAS
        ))
        rows = [rgb_row, event_row]
        if self.show_prediction:
            if predicted_counts is not None and predicted_counts.shape != (len(self.event_cameras), 2, 256, 256):
                raise ValueError("predicted counts must be [camera,2,256,256]")
            predicted_images = {}
            if predicted_counts is not None:
                for name, counts in zip(self.event_cameras, predicted_counts, strict=True):
                    image = np.full((256, 256, 3), 255, dtype=np.uint8)
                    image[counts[0] > 0] = (255, 0, 0)  # OFF: blue in BGR.
                    image[counts[1] > 0] = (0, 0, 255)  # ON: red in BGR.
                    cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
                    count = int((counts > 0).sum())
                    cv2.putText(image, f"pred {name} active={count}",
                                (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    predicted_images[name] = image
            predicted_row = np.hstack(tuple(
                predicted_images[name] if name in predicted_images else self._blank_prediction(name, step)
                for name in CAMERAS
            ))
            rows.append(predicted_row)
        frame = np.vstack(rows)
        if replan_label:
            cv2.rectangle(frame, (0, frame.shape[0] - 24), (512, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, replan_label, (7, frame.shape[0] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    @staticmethod
    def _blank_prediction(name: str, step: int) -> np.ndarray:
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
        label = f"pred {name} pending" if step == 0 else f"pred {name} unavailable"
        cv2.putText(image, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with exit status {self.process.returncode}")


def main() -> None:
    output = args.output_dir.resolve()
    if (output / "complete.json").exists():
        raise FileExistsError(f"refusing to overwrite completed trial: {output}")
    output.mkdir(parents=True, exist_ok=True)
    heartbeat = output / "progress.heartbeat"
    heartbeat.touch()
    torch.manual_seed(scene_seed)
    np.random.seed(scene_seed)
    settings = carb.settings.get_settings()
    for name, value in {
        "/rtx/dldenoiser/responsiveDenoising": False,
        "/rtx/post/tvNoise/enabled": False,
        "/rtx/post/tvNoise/enableFilmGrain": False,
    }.items():
        settings.set(name, value)
    if args.renderer_mode == "pathtracing":
        import omni.replicator.core as rep
        rep.settings.set_render_pathtraced(samples_per_pixel=args.pathtracing_spp)
        settings.set("/rtx/pathtracing/totalSpp", args.pathtracing_spp)
    cfg, success_term = configure_env()
    env = gym.make(TASK, cfg=cfg).unwrapped
    apply_patterns(env)
    alias_legacy_hdr_output(env.scene, CAMERAS)
    dvs = DVSCamera.from_scene(
        env.scene, list(CAMERAS), out_dir=str(output / "dvs_tmp"),
        compression="lzf", event_source="hdr",
        antialiasing=None if args.renderer_mode == "pathtracing" else "Off",
    )
    ready = json.loads(args.ready_file.read_text())
    with ws_connect(ready["url"], open_timeout=30, max_size=MAX_MESSAGE,
                    compression=None) as conn:
        hello, _ = request(conn, {"op": "hello"})
        if hello.get("event_time_contract") != EVENT_TIME_CONTRACT:
            raise ValueError("server event-time contract differs")
        if hello.get("inference_observation_row_offset") != 0:
            raise ValueError("server is not aligned to same-row EWM observations")
        mode = hello.get("event_camera_mode")
        if (mode, hello.get("event_cameras")) not in (("dual", 2), ("wrist_only", 1), ("global_only", 1)):
            raise ValueError(f"unsupported EWM event camera mode/count: {mode}, {hello.get('event_cameras')}")
        event_cameras = {"dual": CAMERAS, "wrist_only": ("cam_wrist",), "global_only": ("cam0",)}[mode]
        step = args.checkpoint_step or max(hello["checkpoints"])
        if step not in hello["checkpoints"]:
            raise ValueError(f"checkpoint step {step} unavailable")
        if args.execution_horizon > hello["horizon"]:
            raise ValueError("execution-horizon exceeds the trained action horizon")
        manifest = {
            "rgb_ood": rgb_ood_config.metadata(),
            "checkpoint_step": step, "checkpoint_dir": ready.get("checkpoint_dir"),
            "training_observation_row_offset": hello["training_observation_row_offset"],
            "inference_observation_row_offset": 0,
            "event_time_contract": EVENT_TIME_CONTRACT,
            "trials": args.trials, "max_steps": args.max_steps,
            "sampling_steps": args.sampling_steps,
            "decode_predicted_events": args.decode_predicted_events,
            "predicted_event_threshold": args.predicted_event_threshold if args.decode_predicted_events else None,
            "predicted_event_min_neighbors": args.predicted_event_min_neighbors if args.decode_predicted_events else None,
            "action_chunk_size": args.execution_horizon, "execution_horizon": args.execution_horizon,
            "seed": args.seed,
            "scene_seed": scene_seed, "scene_seed_mode": args.scene_seed_mode,
            "trial_seed_rule": (
                "sim_seed = scene_seed + trial_index; model_seed = base_seed + trial_index * max_steps + control_step"
                if args.scene_seed_mode == "per_trial" else
                "scene seeded once at environment creation; subsequent reset(seed=None); model_seed = base_seed + trial_index * max_steps + control_step"
            ),
            "sim_task": TASK, "prompt": PROMPT, "control_hz": CONTROL_HZ,
            "event_warp": WARP, "event_cameras": list(event_cameras),
            "renderer_mode": args.renderer_mode,
            "pathtracing_spp": args.pathtracing_spp if args.renderer_mode == "pathtracing" else None,
            "profile_timing": args.profile_timing,
            "timing_metrics": {
                "inference_seconds": "CUDA-synchronized server sampling latency",
                "round_trip_seconds": "client request/response wall time",
            },
            "server_optimization": {
                "compile_core_requested": hello.get("compile_core_requested"),
                "compile_core_active": hello.get("compile_core_active"),
                "compile_mode": hello.get("compile_mode"),
                "cache_context_kv": hello.get("cache_context_kv"),
                "warmup_iters": hello.get("warmup_iters"),
                "startup_benchmark": hello.get("startup_benchmark"),
            },
        }
        manifest_path = output / "manifest.json"
        if args.resume:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"cannot resume without a manifest: {manifest_path}")
            saved_manifest = json.loads(manifest_path.read_text())
            check_resume_rgb_ood(saved_manifest.get("rgb_ood"), manifest["rgb_ood"])
            immutable = ("checkpoint_step", "trials", "max_steps", "sampling_steps",
                         "decode_predicted_events", "predicted_event_threshold",
                         "predicted_event_min_neighbors", "action_chunk_size", "seed",
                         "scene_seed", "scene_seed_mode",
                         "sim_task", "prompt", "event_time_contract", "event_cameras",
                         "renderer_mode", "pathtracing_spp")
            mismatches = [key for key in immutable if saved_manifest.get(key) != manifest.get(key)]
            if mismatches:
                raise ValueError(f"resume settings differ from saved manifest: {mismatches}")
            results = load_completed_results(output, args.trials)
            archived = archive_interrupted_trial(output, len(results))
            print(f"[ewm-sim] resuming after {len(results)} completed trials; archived={archived}", flush=True)
        else:
            if manifest_path.exists():
                raise FileExistsError(f"existing run requires --resume: {output}")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            results = []
        attempted_trials = len(results)
        write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                      requested_trials=args.trials, decode_predicted_events=args.decode_predicted_events,
                      attempted_trials=attempted_trials, results=results)
        try:
            for trial in range(len(results), args.trials):
                heartbeat.touch()
                rgb_ood = TrialRGBOOD(rgb_ood_config, trial, cameras=CAMERAS)
                if rgb_ood_config.enabled:
                    print(f"[ewm-sim] RGB OOD={json.dumps(rgb_ood.metadata())}", flush=True)
                attempted_trials += 1
                trial_seed = scene_seed + trial if args.scene_seed_mode == "per_trial" else scene_seed
                if args.scene_seed_mode == "stream":
                    if args.resume and state_path(output, trial).is_file():
                        restore_scene_rng(output, trial)
                    elif args.resume and trial > 0:
                        raise FileNotFoundError(f"missing scene RNG checkpoint for trial {trial}: {state_path(output, trial)}")
                    else:
                        save_scene_rng(output, trial)
                print(f"[ewm-sim] trial={trial + 1}/{args.trials} scene_seed={trial_seed} mode={args.scene_seed_mode} reset", flush=True)
                env.sim.reset()
                env.reset(seed=trial_seed if args.scene_seed_mode == "per_trial" else None)
                set_views(env)
                warmup = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=env.device)
                for warmup_index in range(RESET_WARMUP_STEPS):
                    set_views(env)
                    env.step(warmup)
                    if warmup_index % 10 == 9:
                        heartbeat.touch()
                print(f"[ewm-sim] trial={trial + 1}/{args.trials} warmup complete", flush=True)
                set_views(env)
                dvs.reset(torch.tensor([0], dtype=torch.long, device=env.device))
                observation, _, _, _ = capture_step(env, dvs, warmup, event_cameras=event_cameras,
                                                    rgb_ood=rgb_ood)
                initial_cube_pose = np.stack([
                    np.concatenate((env.scene[name].data.root_pos_w[0].detach().cpu().numpy(),
                                    env.scene[name].data.root_quat_w[0].detach().cpu().numpy()))
                    for name in ("cube_1", "cube_2", "cube_3")
                ]).astype(np.float32).reshape(-1).tolist()
                initial_joint_state = observation["proprio"].tolist()
                (output / f"trial_{trial:03d}_initial_state.json").write_text(json.dumps({
                    "trial": trial, "scene_seed": trial_seed,
                    "scene_seed_mode": args.scene_seed_mode,
                    "cube_pose": initial_cube_pose, "joint_state": initial_joint_state,
                    "rgb_ood": rgb_ood.metadata(),
                }, indent=2) + "\n")
                print(f"[ewm-sim] trial={trial + 1}/{args.trials} initial capture complete", flush=True)
                stable_success = 0
                rows = []
                chunks = []
                done = False
                replans = 0
                inference_times = []
                round_trip_times = []
                step_timing_records = []
                steps_json_write_seconds = 0.0
                trial_started = time.perf_counter()
                video_path = output / f"trial_{trial:03d}_h264{'_pred_events' if args.decode_predicted_events else ''}_recording.mp4"
                video_open_started = time.perf_counter() if args.profile_timing else None
                ood_label = rgb_ood_config.label()
                video = TrialVideo(video_path, event_cameras, show_prediction=args.decode_predicted_events,
                                   rgb_ood_label=ood_label)
                video_open_seconds = time.perf_counter() - video_open_started if video_open_started is not None else 0.0
                video_close_seconds = 0.0
                try:
                    initial_video_started = time.perf_counter() if args.profile_timing else None
                    video.write(observation, 0, stable_success)
                    initial_video_write_seconds = time.perf_counter() - initial_video_started if initial_video_started is not None else 0.0
                    control_step = 0
                    while control_step < args.max_steps:
                        execution_horizon = min(args.execution_horizon, args.max_steps - control_step)
                        replan_index = replans
                        request_started = time.perf_counter()
                        reply, values = request(conn, {
                            "op": "predict", "step": step,
                            "seed": args.seed + trial * args.max_steps + control_step,
                            "sampling_steps": args.sampling_steps,
                            "action_chunk_size": execution_horizon,
                            "prompt": PROMPT, "event_time_contract": EVENT_TIME_CONTRACT,
                            "observation_row_offset": 0,
                            "raw_observation": True, "timing": observation["timing"],
                            "decode_future_events": args.decode_predicted_events,
                        }, {
                            "obs_video": observation["obs_video"],
                            "obs_voxel": observation["obs_voxel"],
                            "proprio": observation["proprio"],
                        })
                        round_trip_seconds = time.perf_counter() - request_started
                        inference_seconds = float(reply.get("inference_seconds", math.nan))
                        if not math.isfinite(inference_seconds) or inference_seconds < 0:
                            raise ValueError("server returned invalid inference timing")
                        inference_times.append(inference_seconds)
                        round_trip_times.append(round_trip_seconds)
                        actions = values.get("executed_actions")
                        if (actions is None or actions.shape != (execution_horizon, 7)
                                or actions.dtype != np.float32 or not np.isfinite(actions).all()):
                            raise ValueError("server returned invalid physical action chunk")
                        predicted_counts = None
                        if args.decode_predicted_events:
                            prediction = values.get("future_voxel")
                            expected = (execution_horizon, len(event_cameras), 2, 256, 256)
                            if reply.get("future_decoded") is not True or prediction is None or prediction.shape != expected:
                                raise ValueError(f"server did not return decoded future event frames {expected}")
                            predicted_counts = predicted_event_counts(
                                prediction, count_clip=int(hello["count_clip"]),
                                occupancy_threshold=args.predicted_event_threshold,
                                min_neighbors=args.predicted_event_min_neighbors,
                            )
                        replans += 1
                        for chunk_index, action in enumerate(actions):
                            phase_started = time.perf_counter() if args.profile_timing else None
                            command = torch.from_numpy(action.copy())[None].to(env.device)
                            command_to_device_seconds = time.perf_counter() - phase_started if phase_started is not None else 0.0
                            observation, done, terminated, extra = capture_step(
                                env, dvs, command, event_cameras=event_cameras,
                                profile_timing=args.profile_timing, rgb_ood=rgb_ood)
                            control_step += 1
                            phase_started = time.perf_counter() if args.profile_timing else None
                            success_now = bool(success_term.func(env, **success_term.params)[0])
                            success_check_seconds = time.perf_counter() - phase_started if phase_started is not None else 0.0
                            stable_success = stable_success + 1 if success_now else 0
                            end_reason = (
                                "success" if stable_success >= 10 else "terminated" if done else
                                "step_budget" if control_step >= args.max_steps else
                                "max_horizon" if chunk_index + 1 == execution_horizon else None)
                            phase_started = time.perf_counter() if args.profile_timing else None
                            video.write(
                                observation, control_step, stable_success,
                                predicted_counts=None if predicted_counts is None else predicted_counts[chunk_index],
                                replan_label="",
                            )
                            video_write_seconds = time.perf_counter() - phase_started if phase_started is not None else 0.0
                            if args.profile_timing:
                                step_timing = {
                                    **observation["profile_seconds"],
                                    "command_to_device": command_to_device_seconds,
                                    "success_check": success_check_seconds,
                                    "video_write": video_write_seconds,
                                }
                                step_timing_records.append(step_timing)
                            row = {
                                "trial": trial, "step": control_step,
                                "replan": replan_index, "chunk_index": chunk_index,
                                "action": action.tolist(),
                                "event_count": observation["event_count"],
                                "event_counts": observation["event_counts"],
                                "event_end_s": observation["timing"]["obs_event_end_s"],
                                "success_now": success_now, "stable_success": stable_success,
                                "terminated": terminated, "reward": extra["reward"],
                                "chunk_end_reason": end_reason,
                            }
                            if args.profile_timing:
                                row["timing_seconds"] = step_timing
                            rows.append(row)
                            if control_step % 10 == 0 or done or stable_success >= 10:
                                json_write_started = time.perf_counter() if args.profile_timing else None
                                (output / f"trial_{trial:03d}_steps.json").write_text(json.dumps(rows, indent=2) + "\n")
                                heartbeat.touch()
                                if json_write_started is not None:
                                    steps_json_write_seconds += time.perf_counter() - json_write_started
                                print(f"[ewm-sim] trial={trial + 1}/{args.trials} step={control_step}/{args.max_steps} replan={replans} inference={inference_seconds:.3f}s round_trip={round_trip_seconds:.3f}s events={row['event_count']} stable_success={stable_success}", flush=True)
                            if end_reason is not None:
                                chunks.append({"replan": replan_index,
                                               "executed_horizon": chunk_index + 1,
                                               "end_reason": end_reason})
                                break
                        if done or stable_success >= 10:
                            break
                finally:
                    video_close_started = time.perf_counter() if args.profile_timing else None
                    video.close()
                    if video_close_started is not None:
                        video_close_seconds = time.perf_counter() - video_close_started
                trial_wall_seconds = time.perf_counter() - trial_started
                trial_success = stable_success >= 10
                final_video_path = output / f"trial_{trial:03d}_h264{'_pred_events' if args.decode_predicted_events else ''}_{'success' if trial_success else 'failure'}.mp4"
                video_path.rename(final_video_path)
                inference_array = np.asarray(inference_times, dtype=np.float64)
                round_trip_array = np.asarray(round_trip_times, dtype=np.float64)
                inference_total = float(inference_array.sum())
                result = {
                    "trial": trial, "sim_seed": trial_seed, "scene_seed_mode": args.scene_seed_mode,
                    "initial_cube_pose": initial_cube_pose, "initial_joint_state": initial_joint_state,
                    "steps": len(rows),
                    "replans": replans,
                    "execution_chunks": chunks,
                    "execution_horizon_statistics": horizon_statistics(chunks),
                    "success": trial_success,
                    "terminated": done,
                    "video": str(final_video_path),
                    "trial_wall_seconds": trial_wall_seconds,
                    "inference_timing": {
                        "calls": len(inference_times),
                        "total_seconds": inference_total,
                        "mean_seconds": float(inference_array.mean()),
                        "p50_seconds": float(np.percentile(inference_array, 50)),
                        "p95_seconds": float(np.percentile(inference_array, 95)),
                        "milliseconds_per_executed_action": 1000.0 * inference_total / len(rows),
                        "executed_actions_per_inference_second": len(rows) / inference_total,
                    },
                    "round_trip_timing": {
                        "total_seconds": float(round_trip_array.sum()),
                        "mean_seconds": float(round_trip_array.mean()),
                        "p50_seconds": float(np.percentile(round_trip_array, 50)),
                        "p95_seconds": float(np.percentile(round_trip_array, 95)),
                    },
                    "profile_timing": args.profile_timing,
                    "rgb_ood": rgb_ood.metadata(),
                }
                if args.profile_timing:
                    step_timing_summary = summarize_step_timings(step_timing_records, trial_wall_seconds)
                    accounted_seconds = (
                        step_timing_summary["capture_total"]["total_seconds"]
                        + step_timing_summary["command_to_device"]["total_seconds"]
                        + step_timing_summary["success_check"]["total_seconds"]
                        + step_timing_summary["video_write"]["total_seconds"]
                        + float(round_trip_array.sum()) + steps_json_write_seconds
                        + video_open_seconds + initial_video_write_seconds + video_close_seconds
                    )
                    result["step_timing"] = step_timing_summary
                    result["other_timing"] = {
                        "video_open_seconds": video_open_seconds,
                        "initial_video_write_seconds": initial_video_write_seconds,
                        "video_close_seconds": video_close_seconds,
                        "steps_json_write_seconds": steps_json_write_seconds,
                        "unattributed_seconds": max(0.0, trial_wall_seconds - accounted_seconds),
                    }
                (output / f"trial_{trial:03d}_steps.json").write_text(json.dumps(rows, indent=2) + "\n")
                if args.scene_seed_mode == "stream":
                    save_scene_rng(output, trial + 1)
                (output / f"trial_{trial:03d}_result.json").write_text(json.dumps(result, indent=2) + "\n")
                results.append(result)
                heartbeat.touch()
                write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                              requested_trials=args.trials, decode_predicted_events=args.decode_predicted_events,
                              attempted_trials=attempted_trials, results=results)
                print(f"[ewm-sim] result={json.dumps(result)}", flush=True)
            write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                          requested_trials=args.trials, decode_predicted_events=args.decode_predicted_events,
                          attempted_trials=attempted_trials, results=results)
            (output / "complete.json").write_text(json.dumps({
                "complete": True, "results": results,
                "rgb_ood": rgb_ood_config.metadata(),
                "execution_horizon_statistics": horizon_statistics(
                    [chunk for result in results for chunk in result.get("execution_chunks", [])]),
                "success_rate": sum(r["success"] for r in results) / len(results),
            }, indent=2) + "\n")
            request(conn, {"op": "shutdown"})
        except BaseException as exc:
            for recording in output.glob("trial_*_h264*_recording.mp4"):
                recording.rename(recording.with_name(recording.name.replace("_recording.mp4", "_error.mp4")))
            write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                          requested_trials=args.trials, decode_predicted_events=args.decode_predicted_events,
                          attempted_trials=attempted_trials, results=results, error=str(exc))
            failure = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(failure["traceback"], file=sys.stderr, flush=True)
            raise
        finally:
            env.close()
            simulation_app.close()


if __name__ == "__main__":
    main()
