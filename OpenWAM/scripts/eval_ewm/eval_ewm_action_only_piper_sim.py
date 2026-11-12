"""Run independent action-only EWM trials in the same training Piper scene.

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
parser.add_argument("--profile-timing", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--seed", type=int, default=42, help="Base seed for model sampling")
parser.add_argument("--scene-seed", type=int, help="Separate seed for simulator resets (defaults to --seed)")
parser.add_argument("--scene-seed-mode", choices=("per_trial", "stream"), default="per_trial",
                    help="Use an independent seed per trial, or seed once like the data collector")
parser.add_argument("--checkpoint-step", type=int)
parser.add_argument("--scene-rng-reference-dir", type=Path,
                    help="Replay trusted scene RNG snapshots from a previous evaluation for paired trials")
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
from pxr import Sdf, UsdShade
from websockets.sync.client import connect as ws_connect

from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message
from scripts.eval_ewm.ewm_action_only_protocol import OBSERVATION_CONTRACT
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


def simulation_time(env) -> float:
    """Return Isaac 6 simulation time from its monotonic physics-step counter."""
    return float(env.sim.get_physics_step_count()) * float(env.sim.get_physics_dt())


def capture_step(env, action: torch.Tensor, *, profile_timing=False, rgb_ood=None) -> tuple[dict, bool, bool, dict]:
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
    set_views(env)
    mark("set_views")
    _, reward, terminated, truncated, info = env.step(action)
    mark("env_step")
    end_t = simulation_time(env)
    if not math.isfinite(end_t) or abs(end_t - start_t - 1 / CONTROL_HZ) > 0.001:
        raise ValueError(f"control interval is not 30 Hz: {start_t} -> {end_t}")
    rgb = np.stack([
        env.scene[name].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        for name in CAMERAS
    ]).transpose(0, 3, 1, 2).copy()
    mark("rgb_readback")
    state = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32, copy=True)
    mark("state_readback")
    if rgb.shape != (2, 3, 256, 256) or rgb.dtype != np.uint8 or state.shape != (8,):
        raise ValueError(f"unexpected simulator RGB/state shapes: {rgb.shape}, {state.shape}")
    # Apply OOD only to the copied RGB, never to renderer or physical state.
    if rgb_ood is not None and rgb_ood.config.enabled:
        rgb = rgb_ood.apply_rgb(rgb)
        mark("rgb_ood")
    if capture_started is not None:
        profile["capture_total"] = time.perf_counter() - capture_started
    observation = {
        "obs_video": rgb, "proprio": state,
        "timing": {
            "obs_video_time_s": end_t, "proprio_time_s": end_t,
        },
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
    with (output / "summary.txt").open("a", encoding="utf-8") as handle:
        handle.write("training_mode: action_only\nevent_input: false\n")

class TrialVideo:
    """Stream the two conditioning RGB views to H.264; no event renderer."""

    def __init__(self, path: Path, *, rgb_ood_label: str = ""):
        self.rgb_ood_label = rgb_ood_label
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
                "-video_size", "512x256",
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

    def write(self, observation: dict, step: int, success_steps: int) -> None:
        rgb = observation["obs_video"].transpose(0, 2, 3, 1)
        rgb_row = np.hstack((
            self._label(rgb[0], f"cam0 step={step}"),
            self._label(rgb[1], f"wrist success={success_steps}"),
        ))
        if getattr(self, "rgb_ood_label", ""):
            cv2.rectangle(rgb_row, (0, 232), (512, 256), (0, 0, 0), -1)
            cv2.putText(rgb_row, self.rgb_ood_label, (7, 249),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.process.stdin.write(np.ascontiguousarray(rgb_row).tobytes())

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
    if args.scene_rng_reference_dir:
        if args.scene_seed_mode != "stream":
            raise ValueError("scene RNG replay requires --scene-seed-mode stream")
        reference = args.scene_rng_reference_dir.resolve()
        if reference == output:
            raise ValueError("scene RNG reference must differ from the output directory")
        reference_manifest = json.loads((reference / "manifest.json").read_text())
        if reference_manifest.get("scene_seed_mode") != "stream" or reference_manifest.get("scene_seed") != scene_seed:
            raise ValueError("reference scene seed/mode differs")
        for trial in range(args.trials):
            if not state_path(reference, trial).is_file():
                raise FileNotFoundError(f"missing trusted reference RNG: {state_path(reference, trial)}")
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
    # DVSCamera used to set realtime AA to Off. Preserve that setting without EVIS.
    if args.renderer_mode == "realtime":
        import omni.replicator.core as rep
        rep.settings.set_render_rtx_realtime(antialiasing="Off")
    ready = json.loads(args.ready_file.read_text())
    with ws_connect(ready["url"], open_timeout=30, max_size=MAX_MESSAGE,
                    compression=None) as conn:
        hello, _ = request(conn, {"op": "hello"})
        if hello.get("training_mode") != "action_only" or hello.get("observation_contract") != OBSERVATION_CONTRACT:
            raise ValueError("this client requires the independent action-only server")
        if hello.get("inference_observation_row_offset") != 0:
            raise ValueError("server is not aligned to same-row RGB/state observations")
        step = args.checkpoint_step or max(hello["checkpoints"])
        if step not in hello["checkpoints"]:
            raise ValueError(f"checkpoint step {step} unavailable")
        if args.execution_horizon > hello["horizon"]:
            raise ValueError("execution-horizon exceeds the trained action horizon")
        manifest = {
            "rgb_ood": rgb_ood_config.metadata(),
            "training_mode": "action_only",
            "observation_contract": OBSERVATION_CONTRACT,
            "input_keys": ["obs_video", "proprio", "prompt"],
            "event_input": False, "event_prediction": False,
            "scene_rng_reference_dir": (str(args.scene_rng_reference_dir.resolve())
                                        if args.scene_rng_reference_dir else None),
            "config_sha256": hello["config_sha256"],
            "normalization_sha256": hello["normalization_sha256"],
            "checkpoint_sha256": hello["checkpoint_sha256"][str(step)],
            "checkpoint_step": step, "checkpoint_dir": ready.get("checkpoint_dir"),
            "training_observation_row_offset": hello["training_observation_row_offset"],
            "inference_observation_row_offset": 0,
            "trials": args.trials, "max_steps": args.max_steps,
            "sampling_steps": args.sampling_steps,
            "decode_predicted_events": False,
            "action_chunk_size": args.execution_horizon, "execution_horizon": args.execution_horizon,
            "seed": args.seed,
            "scene_seed": scene_seed, "scene_seed_mode": args.scene_seed_mode,
            "trial_seed_rule": ("restore reference scene RNG before each trial" if args.scene_rng_reference_dir else (
                "sim_seed = scene_seed + trial_index; model_seed = base_seed + trial_index * max_steps + control_step"
                if args.scene_seed_mode == "per_trial" else
                "scene seeded once at environment creation; subsequent reset(seed=None); model_seed = base_seed + trial_index * max_steps + control_step"
            )),
            "sim_task": TASK, "prompt": PROMPT, "control_hz": CONTROL_HZ,
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
                         "action_chunk_size", "seed",
                         "scene_seed", "scene_seed_mode",
                         "sim_task", "prompt", "observation_contract", "training_mode",
                         "scene_rng_reference_dir", "checkpoint_sha256", "config_sha256",
                         "normalization_sha256",
                         "renderer_mode", "pathtracing_spp")
            mismatches = [key for key in immutable if saved_manifest.get(key) != manifest.get(key)]
            if mismatches:
                raise ValueError(f"resume settings differ from saved manifest: {mismatches}")
            results = load_completed_results(output, args.trials)
            archived = archive_interrupted_trial(output, len(results))
            print(f"[ewm-action-sim] resuming after {len(results)} completed trials; archived={archived}", flush=True)
        else:
            if manifest_path.exists():
                raise FileExistsError(f"existing run requires --resume: {output}")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            results = []
        attempted_trials = len(results)
        write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                      requested_trials=args.trials, decode_predicted_events=False,
                      attempted_trials=attempted_trials, results=results)
        try:
            for trial in range(len(results), args.trials):
                heartbeat.touch()
                rgb_ood = TrialRGBOOD(rgb_ood_config, trial, cameras=CAMERAS)
                if rgb_ood_config.enabled:
                    print(f"[ewm-action-sim] RGB OOD={json.dumps(rgb_ood.metadata())}", flush=True)
                attempted_trials += 1
                trial_seed = scene_seed + trial if args.scene_seed_mode == "per_trial" else scene_seed
                if args.scene_rng_reference_dir:
                    restore_scene_rng(args.scene_rng_reference_dir, trial)
                    save_scene_rng(output, trial)
                elif args.scene_seed_mode == "stream":
                    if args.resume and state_path(output, trial).is_file():
                        restore_scene_rng(output, trial)
                    elif args.resume and trial > 0:
                        raise FileNotFoundError(f"missing scene RNG checkpoint for trial {trial}: {state_path(output, trial)}")
                    else:
                        save_scene_rng(output, trial)
                print(f"[ewm-action-sim] trial={trial + 1}/{args.trials} scene_seed={trial_seed} mode={args.scene_seed_mode} reset", flush=True)
                env.sim.reset()
                env.reset(seed=trial_seed if args.scene_seed_mode == "per_trial" else None)
                set_views(env)
                warmup = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=env.device)
                for warmup_index in range(RESET_WARMUP_STEPS):
                    set_views(env)
                    env.step(warmup)
                    if warmup_index % 10 == 9:
                        heartbeat.touch()
                print(f"[ewm-action-sim] trial={trial + 1}/{args.trials} warmup complete", flush=True)
                set_views(env)
                observation, _, _, _ = capture_step(env, warmup, rgb_ood=rgb_ood)
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
                print(f"[ewm-action-sim] trial={trial + 1}/{args.trials} initial capture complete", flush=True)
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
                video_path = output / f"trial_{trial:03d}_h264_recording.mp4"
                video_open_started = time.perf_counter() if args.profile_timing else None
                ood_label = rgb_ood_config.label()
                video = TrialVideo(video_path, rgb_ood_label=ood_label)
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
                            "prompt": PROMPT, "observation_contract": OBSERVATION_CONTRACT,
                            "observation_row_offset": 0,
                            "raw_observation": True, "timing": observation["timing"],
                        }, {
                            "obs_video": observation["obs_video"],
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
                        replans += 1
                        for chunk_index, action in enumerate(actions):
                            phase_started = time.perf_counter() if args.profile_timing else None
                            command = torch.from_numpy(action.copy())[None].to(env.device)
                            command_to_device_seconds = time.perf_counter() - phase_started if phase_started is not None else 0.0
                            observation, done, terminated, extra = capture_step(
                                env, command, profile_timing=args.profile_timing, rgb_ood=rgb_ood)
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
                            video.write(observation, control_step, stable_success)
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
                                "observation_time_s": observation["timing"]["obs_video_time_s"],
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
                                print(f"[ewm-action-sim] trial={trial + 1}/{args.trials} step={control_step}/{args.max_steps} replan={replans} inference={inference_seconds:.3f}s round_trip={round_trip_seconds:.3f}s stable_success={stable_success}", flush=True)
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
                final_video_path = output / f"trial_{trial:03d}_h264_{'success' if trial_success else 'failure'}.mp4"
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
                              requested_trials=args.trials, decode_predicted_events=False,
                              attempted_trials=attempted_trials, results=results)
                print(f"[ewm-action-sim] result={json.dumps(result)}", flush=True)
            write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                          requested_trials=args.trials, decode_predicted_events=False,
                          attempted_trials=attempted_trials, results=results)
            (output / "complete.json").write_text(json.dumps({
                "complete": True, "results": results, "training_mode": "action_only",
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
                          requested_trials=args.trials, decode_predicted_events=False,
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


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
