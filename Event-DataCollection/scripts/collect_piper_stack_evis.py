"""Collect Piper stack demonstrations directly into aligned LeRobot v3 rows."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect Piper Stack RGB + EVIS demonstrations.")
parser.add_argument("--episodes", type=int, default=50, help="Number of successful demonstrations.")
parser.add_argument("--output", type=Path, required=True, help="New LeRobot v3 dataset directory.")
parser.add_argument("--resume", action="store_true", help="Append to an existing aligned event-frame LeRobot dataset.")
parser.add_argument("--repo-id", default="local/piper_stack_evis_event_frames")
parser.add_argument("--writer-python", type=Path, default=Path(__file__).resolve().parents[2] / "envs/lerobot/bin/python")
parser.add_argument("--max_steps", type=int, default=600, help="Maximum control steps per attempt.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--dvs_antialiasing",
    choices=("Off", "FXAA", "DLSS", "TAA", "DLAA"),
    default="Off",
    help="Global RTX AA mode. Off is the production default for motion-vector event warping.",
)
parser.add_argument(
    "--renderer_mode",
    choices=("realtime", "pathtracing"),
    default="pathtracing",
    help="Path tracing is the stable production default; realtime is available for speed diagnostics.",
)
parser.add_argument("--pathtracing_spp", type=int, default=64, help="Per-frame SPP for --renderer_mode pathtracing.")
parser.add_argument(
    "--static_rgb_steps",
    type=int,
    default=0,
    help="If positive, save this many 30 Hz RGB frames after a no-motion warm-up and exit (renderer stability probe).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import gymnasium as gym
import numpy as np
import torch
import carb
from PIL import Image

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from isaaclab.assets import AssetBaseCfg
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

install_legacy_extension_utils()
import agx_teleop.tasks  # noqa: F401 -- registers Piper tasks
from dvs_gen.sensors import DVSCamera
from piper_event_frames import EVENT_FRAME_KEY, events_to_frame


TASK = "Isaac-Stack-Cube-Piper-IK-Rel-v0"
CONTROL_HZ = 30
WARP = 8
EVENT_HZ = CONTROL_HZ * WARP
# Let the renderer, contacts, and wrist-mounted camera settle before setting
# the DVS reference frame. This removes the reset-time event burst.
RESET_WARMUP_STEPS = 60
CAMERA_NAMES = ("cam0", "cam_wrist")
CAMERA_RESOLUTION = (256, 256)  # (height, width), shared by both views
# The front global view and wrist view use the same shared calibration as
# browser teleoperation.  The world poses are reapplied after each reset.
# Keep these values synchronized with browser teleoperation via the shared
# piper_wrist_camera module; never change a dataset's view while resuming it.


def camera_calibration_metadata() -> dict:
    return {
        "calibration_id": CALIBRATION_ID,
        "cam0_eye_world_m": list(CAM0_EYE_WORLD),
        "cam0_target_world_m": list(CAM0_TARGET_WORLD),
        "cam0_focal_length_cm": CAM0_FOCAL_LENGTH_CM,
        "cam0_horizontal_aperture_cm": CAM0_HORIZONTAL_APERTURE_CM,
        "wrist_eye_local_m": list(WRIST_EYE_LOCAL),
        "wrist_target_local_m": list(WRIST_TARGET_LOCAL),
        "wrist_focal_length_cm": WRIST_FOCAL_LENGTH_CM,
        "wrist_horizontal_aperture_cm": WRIST_HORIZONTAL_APERTURE_CM,
        "resolution_hw": list(CAMERA_RESOLUTION),
    }


class StackOracle:
    """Deterministic Cartesian controller equivalent to record_ik_stack.py.

    It emits the task's 7-D relative Differential-IK command:
    ``[dx, dy, dz, drx, dry, drz, gripper]``.
    """

    OPEN, CLOSE = 1.0, -1.0

    def __init__(self, env: gym.Env):
        self.env = env
        self.robot = env.scene["robot"]
        self.ee_frame = env.scene["ee_frame"]
        self.cubes = [env.scene[f"cube_{i}"] for i in range(1, 4)]
        self.path: torch.Tensor | None = None
        self.grippers: list[float] = []
        self.step_idx = 0
        self.settle_steps = 0

    @staticmethod
    def _offset(pos: torch.Tensor, dz: float) -> torch.Tensor:
        out = pos.clone()
        out[:, 2:3] += dz
        return out

    def plan(self) -> None:
        """Use the proven smooth trajectory from record_ik_stack.py."""
        cube_height = self.cubes[0].data.root_pos_w[:, 2:3].clone() * 2.0
        base = self.cubes[0].data.root_pos_w.clone()
        waypoints: list[torch.Tensor] = []
        gripper_values: list[float] = []
        durations: list[int] = []
        for cube in self.cubes[1:]:
            source = cube.data.root_pos_w.clone()
            place = self._offset(base, cube_height.item())
            stages = [
                (self._offset(source, 0.18), self.OPEN, 30),
                (self._offset(source, 0.00), self.OPEN, 30),
                (self._offset(source, 0.00), self.CLOSE, 30),
                (self._offset(source, 0.25), self.CLOSE, 30),
                (self._offset(place, 0.18), self.CLOSE, 30),
                (self._offset(place, 0.005), self.CLOSE, 30),
                (self._offset(place, 0.005), self.OPEN, 20),
                (self._offset(place, 0.20), self.OPEN, 30),
            ]
            for target, grip, duration in stages:
                waypoints.append(target.squeeze(0))
                gripper_values.append(grip)
                durations.append(duration)
            # Match the source collector: an extra second at the retreat pose.
            durations[-1] += 30
            base = place
        self.path = self._smooth_path(waypoints, durations)
        self.grippers = []
        for grip, duration in zip(gripper_values, durations):
            self.grippers.extend([grip] * duration)
        self.grippers = self.grippers[: self.path.shape[0]]
        self.step_idx = 0
        self.settle_steps = 0
        print(f"[StackOracle] trajectory planned: {len(self.grippers)} steps", flush=True)

    @staticmethod
    def _smooth_path(waypoints: list[torch.Tensor], durations: list[int]) -> torch.Tensor:
        """Cubic Hermite/PCHIP-style interpolation used by the source demo."""
        wp = torch.stack(waypoints)
        tangents = torch.zeros_like(wp)
        tangents[0] = wp[1] - wp[0]
        tangents[-1] = wp[-1] - wp[-2]
        for index in range(1, len(wp) - 1):
            tangents[index] = (wp[index + 1] - wp[index - 1]) / 2.0
        segments: list[torch.Tensor] = []
        for index in range(len(wp) - 1):
            p0, p1 = wp[index], wp[index + 1]
            m0, m1 = tangents[index], tangents[index + 1]
            steps = durations[index]
            if torch.norm(p1 - p0) < 1e-4:
                segment = p0.unsqueeze(0).repeat(steps, 1)
            else:
                s = torch.linspace(0, 1, steps, device=p0.device).unsqueeze(-1)
                segment = (
                    (2 * s**3 - 3 * s**2 + 1) * p0
                    + (s**3 - 2 * s**2 + s) * m0
                    + (-2 * s**3 + 3 * s**2) * p1
                    + (s**3 - s**2) * m1
                )
            segments.append(segment)
        return torch.cat(segments, dim=0).unsqueeze(1)

    def action(self) -> torch.Tensor:
        if self.path is None:
            raise RuntimeError("Call plan() before action().")
        if self.step_idx >= len(self.grippers):
            self.settle_steps += 1
            return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.OPEN]], device=self.env.device)
        current = self.ee_frame.data.target_pos_w[:, 0, :]
        # ``path`` is [T, 1, 3]; index (rather than slice) preserves the
        # action term's expected [1, 3] Cartesian command shape.
        delta_pos = self.path[self.step_idx] - current
        delta_rot = torch.zeros((1, 3), device=self.env.device)
        grip = torch.tensor([[self.grippers[self.step_idx]]], device=self.env.device)
        self.step_idx += 1
        return torch.cat((delta_pos, delta_rot, grip), dim=-1)

    def finished(self) -> bool:
        return self.step_idx >= len(self.grippers) and self.settle_steps >= 60


def external_camera() -> CameraCfg:
    """Fixed table camera; its RGB and events share this render sensor."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam0",
        update_period=1.0 / CONTROL_HZ,
        height=CAMERA_RESOLUTION[0],
        width=CAMERA_RESOLUTION[1],
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", HDR_DATA_TYPE],
        # The wider front view keeps the robot, cubes, and near table in frame.
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM0_FOCAL_LENGTH_CM, focus_distance=400.0,
            horizontal_aperture=CAM0_HORIZONTAL_APERTURE_CM, clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.55, 0.75), rot=quat_from_wxyz((0.7071, 0.0, 0.7071, 0.0)), convention="world"
        ),
    )


def wrist_camera() -> CameraCfg:
    """RGB/Event camera whose world pose is updated rigidly from the TCP."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_wrist",
        update_period=1.0 / CONTROL_HZ,
        height=CAMERA_RESOLUTION[0],
        width=CAMERA_RESOLUTION[1],
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", HDR_DATA_TYPE],
        spawn=sim_utils.PinholeCameraCfg(
            # Wider than the table camera so both fingertips and the object
            # workspace survive small pose/calibration changes.
            focal_length=WRIST_FOCAL_LENGTH_CM, focus_distance=400.0,
            horizontal_aperture=WRIST_HORIZONTAL_APERTURE_CM, clipping_range=(0.03, 20.0)
        ),
        # The actual pose is set by ``set_wrist_view``.  This neutral initial
        # pose only makes construction of the camera prim well-defined.
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=quat_from_wxyz((1.0, 0.0, 0.0, 0.0)), convention="world"
        ),
    )


def set_fixed_view(env: gym.Env) -> None:
    """Aim the one shared RGB/Event camera at the Piper table workspace."""
    eye = torch.tensor([CAM0_EYE_WORLD], dtype=torch.float32, device=env.device)
    target = torch.tensor([CAM0_TARGET_WORLD], dtype=torch.float32, device=env.device)
    env.scene["cam0"].set_world_poses_from_view(eye, target)


def set_wrist_view(env: gym.Env) -> None:
    """Place a side/front wrist camera and look into the grasp workspace."""
    ee_frame = env.scene["ee_frame"].data
    tcp_pos = ee_frame.target_pos_w[:, 0, :]
    tcp_quat = ee_frame.target_quat_w[:, 0, :]
    eye_local = torch.tensor(WRIST_EYE_LOCAL, dtype=torch.float32, device=env.device).expand_as(tcp_pos)
    target_local = torch.tensor(WRIST_TARGET_LOCAL, dtype=torch.float32, device=env.device).expand_as(tcp_pos)
    eye = tcp_pos + quat_apply(tcp_quat, eye_local)
    target = tcp_pos + quat_apply(tcp_quat, target_local)
    env.scene["cam_wrist"].set_world_poses_from_view(eye, target)


def configure_env():
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, use_fabric=True)
    migrate_piper_stack_cfg(cfg)
    # Isaac Lab seeds Python's random module (used by cube pose sampling),
    # NumPy and Torch once at environment creation. Later reset() calls pass
    # no seed, so each episode advances the RNG rather than repeating a layout.
    cfg.seed = args_cli.seed
    # Isaac Lab 3 freezes articulation defaults after the asset is primed.
    # Piper's asset already carries this exact home pose, so the legacy reset
    # term is redundant and would attempt an illegal default-data mutation.
    cfg.events.init_piper_arm_pose = None
    cfg.sim.dt = 1.0 / 120.0
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.episode_length_s = float(args_cli.max_steps) / CONTROL_HZ
    cfg.scene.cam0 = external_camera()
    cfg.scene.cam_wrist = wrist_camera()
    # The stock Piper task uses a grey DomeLight at intensity 3000.  Use a
    # neutral, brighter illumination for camera data without changing physics.
    cfg.scene.light.spawn.color = (1.0, 1.0, 1.0)
    cfg.scene.light.spawn.intensity = 6000.0
    # The stock SeattleLabTable material contains sharp specular/texture
    # highlights that look like white dots and create large false HDR events.
    # Override visuals only: table collision, dimensions and cube physics stay
    # untouched.  A rough dielectric is intentionally stable for clean EWM data.
    cfg.scene.table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.16, 0.16, 0.16), roughness=1.0, metallic=0.0
    )
    cfg.scene.table.spawn.visual_material_path = "visual_material_clean_evis"
    # SeattleLabTable is instanceable and retains a few high-contrast surface
    # details despite a material override.  This is a visual-only skin placed
    # 0.2 mm above its physical top: cubes still collide with the original USD
    # table, while RGB/HDR sees a uniform non-specular manipulation surface.
    cfg.scene.clean_tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CleanTabletop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0001)),
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.90, 0.0002),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.16, 0.16, 0.16), roughness=1.0, metallic=0.0
            ),
        ),
    )
    # The built-in success term remains callable but is not allowed to reset
    # before the raw event chunk has been flushed by this collector.
    success = cfg.terminations.success
    cfg.terminations.success = None
    cfg.terminations.time_out = None
    return cfg, success


def configure_clean_event_renderer() -> None:
    """Remove renderer-introduced brightness flicker before EVIS sees HDR.

    EVIS deliberately keeps temporal AA off: temporal reconstruction creates
    false frame-to-frame log-intensity changes.  The DL denoiser's *responsive*
    mode can independently leave white speckles on simple materials, so disable
    only that mode while retaining the denoiser itself.  Film-grain/TV-noise is
    also explicitly disabled to keep this synthetic dataset clean.
    """
    settings = carb.settings.get_settings()
    desired = {
        "/rtx/dldenoiser/responsiveDenoising": False,
        "/rtx/post/tvNoise/enabled": False,
        "/rtx/post/tvNoise/enableFilmGrain": False,
    }
    for key, value in desired.items():
        settings.set(key, value)
    if args_cli.renderer_mode == "pathtracing":
        import omni.replicator.core as rep

        rep.settings.set_render_pathtraced(samples_per_pixel=args_cli.pathtracing_spp)
        # Do not carry samples from a preceding camera pose into the next one.
        settings.set("/rtx/pathtracing/totalSpp", args_cli.pathtracing_spp)
        print(f"[render] PathTracing enabled: spp={args_cli.pathtracing_spp}", flush=True)
    observed = ", ".join(f"{key}={settings.get(key)!r}" for key in desired)
    print(f"[render] clean-event renderer: {observed}", flush=True)


def capture_frame(env: gym.Env, action: torch.Tensor, event_frames: dict) -> dict:
    """Snapshot the observation at the start of action's control interval."""
    robot = env.scene["robot"].data
    ee = env.scene["ee_frame"].data
    cubes = [env.scene[f"cube_{i}"].data for i in range(1, 4)]
    frame = {
        "action": action[0].detach().cpu().numpy().astype(np.float32),
        "observation.state": robot.joint_pos[0].detach().cpu().numpy().astype(np.float32),
        "observation.joint_vel": robot.joint_vel[0].detach().cpu().numpy().astype(np.float32),
        "observation.ee_pose": np.concatenate((
            ee.target_pos_w[0, 0].detach().cpu().numpy(),
            ee.target_quat_w[0, 0].detach().cpu().numpy(),
        )).astype(np.float32),
        "observation.cube_pose": np.stack([
            np.concatenate((cube.root_pos_w[0].cpu().numpy(), cube.root_quat_w[0].cpu().numpy()))
            for cube in cubes
        ]).astype(np.float32).reshape(-1),
    }
    for name in CAMERA_NAMES:
        output = env.scene[name].data.output
        frame[f"observation.images.{name}"] = output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        frame[f"{EVENT_FRAME_KEY}.{name}"] = event_frames[name]
    return frame


def send_writer_message(writer: subprocess.Popen, message: dict) -> None:
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    writer.stdin.write(len(payload).to_bytes(8, "little"))
    writer.stdin.write(payload)


def writer_reply(writer: subprocess.Popen, expected: str) -> int:
    line = writer.stdout.readline().decode("utf-8", errors="replace").strip()
    if not line.startswith(expected + " "):
        raise RuntimeError(f"LeRobot writer expected {expected}, got {line!r}; exit={writer.poll()}")
    return int(line.split()[1])


def reset_episode(env: gym.Env, dvs: DVSCamera, oracle: StackOracle):
    env.sim.reset()
    env.reset()
    set_fixed_view(env)
    set_wrist_view(env)
    # Do not record the period in which the renderer and physics settle after
    # reset. In particular, a wrist-camera pose update otherwise produces a
    # one-frame 70k--100k-event brightness-reference transient.
    warmup_action = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, StackOracle.OPEN]], device=env.device
    )
    for _ in range(RESET_WARMUP_STEPS):
        set_fixed_view(env)
        set_wrist_view(env)
        env.step(warmup_action)
    set_fixed_view(env)
    set_wrist_view(env)
    dvs.reset(torch.tensor([0], dtype=torch.long, device=env.device))
    dvs.recorder._events.pop(0, None)
    oracle.plan()
    action = oracle.action()
    return [], dvs.snapshot(), 0.0, action


def write_static_rgb_probe(env: gym.Env, output: Path, steps: int) -> None:
    """Capture a no-command sequence to separate renderer flicker from motion."""
    env.sim.reset()
    env.reset()
    set_fixed_view(env)
    set_wrist_view(env)
    action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, StackOracle.OPEN]], device=env.device)
    # Let gravity/contact and the renderer settle before evaluating static pixels.
    for _ in range(60):
        set_wrist_view(env)
        env.step(action)
    frames: dict[str, list[np.ndarray]] = {name: [] for name in CAMERA_NAMES}
    for _ in range(steps):
        set_wrist_view(env)
        env.step(action)
        for name in CAMERA_NAMES:
            frames[name].append(env.scene[name].data.output["rgb"][0, ..., :3].detach().cpu().numpy().copy())
    path = output / "static_rgb_probe.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["fps"] = CONTROL_HZ
        h5.attrs["warmup_steps"] = 60
        h5.attrs["camera_calibration_id"] = CALIBRATION_ID
        for name, sequence in frames.items():
            h5.create_dataset(name, data=np.asarray(sequence, dtype=np.uint8), compression="lzf")
    cam0_last = frames["cam0"][-1][..., :3].astype(np.uint8)
    wrist_last = frames["cam_wrist"][-1][..., :3].astype(np.uint8)
    Image.fromarray(cam0_last).save(output / "cam0.png")
    Image.fromarray(wrist_last).save(output / "cam_wrist.png")
    Image.fromarray(np.concatenate((cam0_last, wrist_last), axis=1)).save(output / "preview.png")
    wrist = np.asarray(frames["cam_wrist"], dtype=np.int16)
    # Mean inter-frame absolute difference is an auditable scalar for the
    # stationary-render test; it does not replace visual inspection.
    mad = float(np.abs(np.diff(wrist, axis=0)).mean()) if steps > 1 else 0.0
    print(f"[render] static RGB probe: frames={steps} wrist_mean_abs_frame_diff={mad:.6f} -> {path}", flush=True)


def add_cube_patterns(env: gym.Env) -> None:
    """Bind deterministic UV textures directly to the physical cube meshes.

    This avoids fragile thin visual-only cuboids. The root material binding
    changes every visible cube mesh but leaves collision and rigid-body schemas.
    """
    from pxr import Sdf, UsdShade

    texture_dir = Path(__file__).resolve().parent.parent / "assets" / "piper_cube_patterns"
    patterns = {
        "cube_1": "red_stripes.png",
        "cube_2": "blue_cross.png",
        "cube_3": "green_checker.png",
    }
    stage = sim_utils.get_current_stage()
    for cube_name, filename in patterns.items():
        texture_path = texture_dir / filename
        if not texture_path.is_file():
            raise FileNotFoundError(f"Missing cube texture: {texture_path}")
        material_path = f"/World/Looks/EvisCubePatterns/{cube_name}"
        material = UsdShade.Material.Define(stage, material_path)
        surface = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        st_reader = UsdShade.Shader.Define(stage, f"{material_path}/stReader")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        cube_root = env.scene[cube_name].root_physx_view.prim_paths[0]
        binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(cube_root))
        binding.Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    print("[scene] bound UV cube textures: red-stripes, blue-cross, green-checker", flush=True)


def main() -> None:
    if args_cli.episodes <= 0 or args_cli.max_steps <= 0:
        raise ValueError("--episodes and --max_steps must be positive")
    output = args_cli.output.resolve()
    camera_meta_path = output / "meta" / "piper_camera_calibration.json"
    if args_cli.resume:
        if (not camera_meta_path.is_file() or
                json.loads(camera_meta_path.read_text(encoding="utf-8")) != camera_calibration_metadata()):
            raise ValueError(f"cannot resume an old or mismatched camera-calibration dataset: {output}")
    if args_cli.static_rgb_steps > 0:
        output.mkdir(parents=True, exist_ok=True)
    if not args_cli.writer_python.is_file() and args_cli.static_rgb_steps == 0:
        raise FileNotFoundError(f"LeRobot Python environment is missing: {args_cli.writer_python}")
    torch.manual_seed(args_cli.seed)
    configure_clean_event_renderer()
    env_cfg, success_term = configure_env()
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    add_cube_patterns(env)
    alias_legacy_hdr_output(env.scene, CAMERA_NAMES)
    if args_cli.static_rgb_steps > 0:
        write_static_rgb_probe(env, output, args_cli.static_rgb_steps)
        (output / "camera_calibration.json").write_text(
            json.dumps(camera_calibration_metadata(), indent=2) + "\n", encoding="utf-8"
        )
        print("[collect] static RGB probe complete", flush=True)
        env.close()
        simulation_app.close()
        return
    writer = subprocess.Popen(
        [str(args_cli.writer_python), "-u", str(Path(__file__).with_name("write_piper_stack_lerobot.py")),
         "--output", str(output), "--repo-id", args_cli.repo_id, *(["--resume"] if args_cli.resume else [])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    try:
        saved = writer_reply(writer, "READY")
        if not args_cli.resume:
            camera_meta_path.write_text(
                json.dumps(camera_calibration_metadata(), indent=2) + "\n", encoding="utf-8"
            )
            randomization = env_cfg.events.randomize_cube_positions.params
            (output / "meta" / "piper_collection_config.json").write_text(
                json.dumps({
                    "requested_successful_episodes": args_cli.episodes,
                    "seed": args_cli.seed,
                    "seed_scope": "Isaac Lab seeds Python/NumPy/Torch once; reset() advances RNG",
                    "cube_pose_range": randomization["pose_range"],
                    "cube_min_separation_m": randomization["min_separation"],
                    "joint_gaussian_std": env_cfg.events.randomize_piper_joint_state.params["std"],
                    "max_steps_per_attempt": args_cli.max_steps,
                    "renderer_mode": args_cli.renderer_mode,
                    "pathtracing_spp": args_cli.pathtracing_spp if args_cli.renderer_mode == "pathtracing" else None,
                    "cameras": list(CAMERA_NAMES),
                    "camera_calibration_id": CALIBRATION_ID,
                }, indent=2) + "\n", encoding="utf-8"
            )
        if saved and not args_cli.resume:
            raise RuntimeError("writer unexpectedly reported existing episodes")
        if saved:
            print(f"[collect] resuming from {saved} complete episodes", flush=True)
        oracle = StackOracle(env)
        attempts = 0
        success_steps = 0
        with tempfile.TemporaryDirectory(prefix="piper_evis_") as event_dir:
            dvs = DVSCamera.from_scene(
                env.scene, list(CAMERA_NAMES), out_dir=event_dir,
                compression="lzf", event_source="hdr",
                antialiasing=None if args_cli.renderer_mode == "pathtracing" else args_cli.dvs_antialiasing,
            )
            buffer, prev, t_prev, action = reset_episode(env, dvs, oracle)
            while saved < args_cli.episodes and simulation_app.is_running():
                attempts += 1
                for step_index in range(args_cli.max_steps):
                    set_wrist_view(env)
                    # EVIS emits the right endpoint of an interval only when
                    # that keyframe becomes the *next* gap's first frame.
                    # Feed it before taking row i, then pop (t_(i-1), t_i].
                    dvs.warp_and_process(prev, prev, 1, t_prev, 1.0 / EVENT_HZ)
                    chunks = dvs.recorder._events.pop(0, {})
                    event_frames = {
                        name: events_to_frame(chunks.get(f"DVS/{name}", []),
                                              t_prev - 1.0 / CONTROL_HZ, t_prev)
                        for name in CAMERA_NAMES
                    }
                    frame = capture_frame(env, action, event_frames)
                    _obs, _reward, terminated, truncated, _info = env.step(action)
                    current = dvs.snapshot()
                    if bool(torch.logical_or(terminated, truncated).any()):
                        print(f"[collect] attempt {attempts} stopped at step {step_index + 1}", flush=True)
                        break
                    dvs.warp_and_process(prev, current, WARP, t_prev, 1.0 / EVENT_HZ)
                    buffer.append(frame)
                    success_steps = success_steps + 1 if bool(success_term.func(env, **success_term.params)[0]) else 0
                    if success_steps >= 10:
                        for saved_frame in buffer:
                            send_writer_message(writer, {"command": "frame", "frame": saved_frame})
                        send_writer_message(writer, {"command": "save"})
                        writer.stdin.flush()
                        saved = writer_reply(writer, "SAVED")
                        print(f"[collect] saved {saved}/{args_cli.episodes} ({len(buffer)} frames)", flush=True)
                        break
                    prev, t_prev = current, t_prev + 1.0 / CONTROL_HZ
                    action = oracle.action()
                dvs.recorder._events.pop(0, None)
                if saved >= args_cli.episodes:
                    break
                buffer, prev, t_prev, action = reset_episode(env, dvs, oracle)
                success_steps = 0
    finally:
        env.close()
        try:
            writer.stdin.close()
        except BrokenPipeError:
            pass
        writer.wait(timeout=300)
        if writer.returncode:
            raise RuntimeError(f"LeRobot writer exited with code {writer.returncode}")
    print(f"[collect] complete: {saved} successful trajectories after {attempts} attempts -> {output}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
