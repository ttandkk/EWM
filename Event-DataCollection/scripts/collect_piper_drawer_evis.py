"""Headless Piper drawer-opening demonstrations with shared RGB/EVIS cameras.

Each saved episode contains successful 30 Hz RGB/depth/action data and raw HDR events from the fixed and wrist views.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect Piper Stack RGB + EVIS demonstrations.")
parser.add_argument("--episodes", type=int, default=50, help="Number of successful demonstrations.")
parser.add_argument("--output", type=Path, required=True, help="Output directory for episode HDF5 files.")
parser.add_argument("--resume", action="store_true", help="Continue from contiguous, validated episode files already in --output.")
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
    default="realtime",
    help="Realtime is the production default; pathtracing is for quality diagnostics only.",
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

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.sensors import CameraCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_apply, quat_from_matrix, quat_inv, quat_mul, quat_slerp

import robot_lab.tasks  # noqa: F401 -- registers PiPER drawer tasks
from dvs_gen.sensors import DVSCamera


TASK = "Isaac-Open-Drawer-Piper-IK-Rel-v0"
CONTROL_HZ = 30
WARP = 8
EVENT_HZ = CONTROL_HZ * WARP
# Let the renderer, contacts, and wrist-mounted camera settle before setting
# the DVS reference frame. This removes the reset-time event burst.
RESET_WARMUP_STEPS = 60
CAMERA_NAMES = ("cam0", "cam_wrist")
CAMERA_RESOLUTION = (256, 256)  # (height, width), shared by both views
# A fixed front/upper table view.  The pose is applied using the sensor's
# eye/look-at API (rather than a hand-derived quaternion) after each reset.
CAMERA_EYE = (0.00, -1.25, 0.80)
CAMERA_TARGET = (0.60, 0.0, 0.45)
# Wrist-relative side/front mounting points.  The optical axis is aimed *past*
# the TCP into the manipulation workspace.  This deliberately excludes most
# of the forearm: the useful composition is fingertips in the lower edge and
# the candidate object/contact region near the image centre.
# Empirically calibrated midpoint between the rear view (which exposed too
# much forearm) and the front view (which hid the gripper entirely).
# Piper's two fingers open along the gripper frame's Y axis.  Looking mainly
# from -Y makes one finger occlude the other, so use a -X-dominant oblique
# viewpoint and only a small Y component to expose the jaw gap/contact.
WRIST_EYE_LOCAL = (-0.15, -0.04, -0.07)
WRIST_TARGET_LOCAL = (0.0, 0.0, 0.11)


class DrawerOracle:
    """Contact-aware, velocity-limited drawer-opening controller."""

    OPEN, CLOSE = 1.0, -1.0
    PREGRASP_SPEED, GRASP_SPEED, HOLD_SPEED, PULL_SPEED = 0.035, 0.010, 0.004, 0.0030
    PREGRASP_ROT, GRASP_ROT, HOLD_ROT, PULL_ROT = 0.35, 0.12, 0.08, 0.10

    def __init__(self, env: gym.Env):
        self.env = env
        self.ee_frame = env.scene["ee_frame"]
        self.stage = "pre_grasp"
        self.stage_step = 0
        self.pre_grasp: torch.Tensor | None = None
        self.grasp: torch.Tensor | None = None
        self.pull_dir: torch.Tensor | None = None
        self.desired_quat: torch.Tensor | None = None

    def _set_stage(self, name: str) -> None:
        self.stage, self.stage_step = name, 0
        print(f"[DrawerOracle] stage={name}", flush=True)


    @staticmethod
    def _limit(vector: torch.Tensor, limit: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-6)
        return vector * torch.clamp(limit / norm, max=1.0)

    def plan(self) -> None:
        handle_frame = self.env.scene["cabinet_frame"].data
        handle = handle_frame.target_pos_w[:, 0, :].clone()
        handle_rot = matrix_from_quat(handle_frame.target_quat_w[:, 0, :].clone())
        desired_rot = torch.stack((-handle_rot[..., 1], handle_rot[..., 2], -handle_rot[..., 0]), dim=-1)
        self.desired_quat = quat_from_matrix(desired_rot)
        current = self.ee_frame.data.target_pos_w[:, 0, :].clone()
        self.pull_dir = current - handle
        self.pull_dir[:, 2] = 0.0
        self.pull_dir = self.pull_dir / torch.clamp(torch.linalg.vector_norm(self.pull_dir, dim=-1, keepdim=True), min=1e-6)
        self.pre_grasp, self.grasp = handle + 0.12 * self.pull_dir, handle
        self._set_stage("pre_grasp")
        print(f"[DrawerOracle] contact-aware trajectory planned: handle={handle[0].tolist()}, pull_dir={self.pull_dir[0].tolist()}", flush=True)

    def action(self) -> torch.Tensor:
        if self.pre_grasp is None or self.grasp is None or self.pull_dir is None or self.desired_quat is None:
            raise RuntimeError("Call plan() before action().")
        current = self.ee_frame.data.target_pos_w[:, 0, :]
        if self.stage == "pre_grasp":
            target, speed, rot_speed, grip = self.pre_grasp, self.PREGRASP_SPEED, self.PREGRASP_ROT, self.OPEN
            if self.stage_step >= 8 and torch.linalg.vector_norm(current - target, dim=-1).max().item() < 0.014:
                self._set_stage("grasp")
                return self.action()
        elif self.stage == "grasp":
            target, speed, rot_speed, grip = self.grasp, self.GRASP_SPEED, self.GRASP_ROT, self.OPEN
            if self.stage_step >= 8 and torch.linalg.vector_norm(current - target, dim=-1).max().item() < 0.007:
                self._set_stage("close")
                return self.action()
        elif self.stage == "close":
            target, speed, rot_speed, grip = self.grasp, self.HOLD_SPEED, self.HOLD_ROT, self.CLOSE
            if self.stage_step >= 18:
                self._set_stage("pull")
                return self.action()
        else:
            target = self.grasp + min(self.stage_step * self.PULL_SPEED, 0.42) * self.pull_dir
            speed, rot_speed, grip = self.PULL_SPEED, self.PULL_ROT, self.CLOSE
        delta_pos = self._limit(target - current, speed)
        current_quat = self.ee_frame.data.target_quat_w[:, 0, :]
        delta_rot = axis_angle_from_quat(quat_mul(self.desired_quat, quat_inv(current_quat)))
        delta_rot = self._limit(delta_rot, rot_speed)
        self.stage_step += 1
        return torch.cat((delta_pos, delta_rot, torch.tensor([[grip]], device=self.env.device)), dim=-1)


def drawer_is_open(env: gym.Env, threshold: float = 0.30) -> bool:
    cabinet = env.scene["cabinet"]
    joint_id = cabinet.find_joints("drawer_top_joint")[0][0]
    return bool(cabinet.data.joint_pos[0, joint_id] > threshold)


def external_camera() -> CameraCfg:
    """Fixed table camera; its RGB and events share this render sensor."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam0",
        update_period=1.0 / CONTROL_HZ,
        height=CAMERA_RESOLUTION[0],
        width=CAMERA_RESOLUTION[1],
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", "HdrColor"],
        # A 256×256 sensor would otherwise reveal ~33% more vertical FOV than
        # the previous 640×480 view.  32 mm gives this square output the former
        # 4:3 image's vertical framing (a centered square crop, not a wider view).
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=28.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -0.55, 0.75), rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"),
    )


def wrist_camera() -> CameraCfg:
    """RGB/Event camera whose world pose is updated rigidly from the TCP."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_wrist",
        update_period=1.0 / CONTROL_HZ,
        height=CAMERA_RESOLUTION[0],
        width=CAMERA_RESOLUTION[1],
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", "HdrColor"],
        spawn=sim_utils.PinholeCameraCfg(
            # Wider than the table camera so both fingertips and the object
            # workspace survive small pose/calibration changes.
            focal_length=12.5, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.03, 20.0)
        ),
        # The actual pose is set by ``set_wrist_view``.  This neutral initial
        # pose only makes construction of the camera prim well-defined.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def set_fixed_view(env: gym.Env) -> None:
    """Aim the one shared RGB/Event camera at the Piper table workspace."""
    eye = torch.tensor([CAMERA_EYE], dtype=torch.float32, device=env.device)
    target = torch.tensor([CAMERA_TARGET], dtype=torch.float32, device=env.device)
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
    cfg.sim.dt = 1.0 / 120.0
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.episode_length_s = float(args_cli.max_steps) / CONTROL_HZ
    cfg.scene.cam0 = external_camera()
    cfg.scene.cam_wrist = wrist_camera()
    # These task defaults are useful when interactively debugging poses, but
    # the RGB/event cameras must never record the colored frame markers.
    cfg.scene.ee_frame.debug_vis = False
    cfg.scene.cabinet_frame.debug_vis = False
    # Avoid clipping the white cabinet while retaining a clear grasp scene.
    cfg.scene.light.spawn.color = (1.0, 1.0, 1.0)
    cfg.scene.light.spawn.intensity = 2500.0
    # robot_lab cabinet has no success termination; drawer displacement is
    # evaluated explicitly by the drawer collector.
    cfg.terminations.time_out = None
    return cfg, None


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


def new_buffer() -> dict[str, list | dict[str, list]]:
    return {
        **{key: [] for key in ("t", "action", "joint_pos", "joint_vel", "ee_pose", "cabinet_joint_pos")},
        "rgb": {name: [] for name in CAMERA_NAMES},
        "depth": {name: [] for name in CAMERA_NAMES},
    }


def append_frame(buf: dict, env: gym.Env, action: torch.Tensor, timestamp: float) -> None:
    robot = env.scene["robot"].data
    ee = env.scene["ee_frame"].data
    buf["t"].append(float(timestamp))
    buf["action"].append(action[0].detach().cpu().numpy().astype(np.float32))
    buf["joint_pos"].append(robot.joint_pos[0].detach().cpu().numpy().astype(np.float32))
    buf["joint_vel"].append(robot.joint_vel[0].detach().cpu().numpy().astype(np.float32))
    buf["ee_pose"].append(
        np.concatenate((ee.target_pos_w[0, 0].detach().cpu().numpy(), ee.target_quat_w[0, 0].detach().cpu().numpy())).astype(np.float32)
    )
    buf["cabinet_joint_pos"].append(env.scene["cabinet"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32))
    for name in CAMERA_NAMES:
        output = env.scene[name].data.output
        buf["rgb"][name].append(output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
        buf["depth"][name].append(output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy().astype(np.float16))


def write_episode(path: Path, buffer: dict, event_path: Path, episode_id: int) -> bool:
    """Atomically export an episode only when every shared camera has events."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(event_path, "r") as raw:
            missing = [name for name in CAMERA_NAMES if f"DVS/{name}" not in raw]
            if missing:
                print(f"[collect] retrying episode {episode_id}: missing EVIS streams {missing}", flush=True)
                return False
            with h5py.File(temporary, "w") as out:
                meta = out.create_group("meta")
                meta.attrs.update(
                    task=TASK,
                    success=True,
                    episode_id=episode_id,
                    control_hz=CONTROL_HZ,
                    event_hz=EVENT_HZ,
                    warp=WARP,
                    event_source="hdr",
                    camera_names=np.asarray(CAMERA_NAMES, dtype=h5py.string_dtype()),
                    camera_resolution=np.array([CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0]], dtype=np.int32),
                    action_semantics="relative_ee_delta_xyz_rotvec_plus_binary_gripper",
                )
                obs = out.create_group("observation")
                obs.create_dataset("t", data=np.asarray(buffer["t"], np.float64))
                for key in ("joint_pos", "joint_vel", "ee_pose", "cabinet_joint_pos"):
                    obs.create_dataset(key, data=np.asarray(buffer[key], np.float32), compression="lzf")
                rgb = obs.create_group("rgb")
                depth = obs.create_group("depth")
                for name in CAMERA_NAMES:
                    rgb.create_dataset(name, data=np.asarray(buffer["rgb"][name], np.uint8), compression="lzf")
                    depth.create_dataset(name, data=np.asarray(buffer["depth"][name], np.float16), compression="lzf")
                out.create_dataset("action", data=np.asarray(buffer["action"], np.float32), compression="lzf")
                events = out.create_group("events")
                for name in CAMERA_NAMES:
                    raw.copy(f"DVS/{name}", events, name=name)
        temporary.replace(path)
        return True
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def completed_episode_count(output: Path) -> int:
    """Return the contiguous prefix of complete, dual-camera HDF5 episodes."""
    completed = 0
    for episode_id in range(args_cli.episodes):
        path = output / f"episode_{episode_id:05d}.h5"
        if not path.is_file():
            break
        try:
            with h5py.File(path, "r") as h5:
                if not all(f"events/{name}/t" in h5 for name in CAMERA_NAMES):
                    raise KeyError("missing camera event stream")
        except (OSError, KeyError) as exc:
            print(f"[collect] resume ignores incomplete {path.name}: {exc}", flush=True)
            break
        completed += 1
    return completed


def reset_episode(env: gym.Env, dvs: DVSCamera, oracle: StackOracle):
    env.sim.reset()
    env.reset()
    set_fixed_view(env)
    set_wrist_view(env)
    # Do not record the period in which the renderer and physics settle after
    # reset. In particular, a wrist-camera pose update otherwise produces a
    # one-frame 70k--100k-event brightness-reference transient.
    warmup_action = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DrawerOracle.OPEN]], device=env.device
    )
    for _ in range(RESET_WARMUP_STEPS):
        set_fixed_view(env)
        set_wrist_view(env)
        env.step(warmup_action)
    set_fixed_view(env)
    set_wrist_view(env)
    dvs.reset(torch.tensor([0], dtype=torch.long, device=env.device))
    oracle.plan()
    action = oracle.action()
    return new_buffer(), dvs.snapshot(), float(env.sim.current_time), action


def write_static_rgb_probe(env: gym.Env, output: Path, steps: int) -> None:
    """Capture a no-command sequence to separate renderer flicker from motion."""
    env.sim.reset()
    env.reset()
    set_fixed_view(env)
    set_wrist_view(env)
    action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DrawerOracle.OPEN]], device=env.device)
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
        for name, sequence in frames.items():
            h5.create_dataset(name, data=np.asarray(sequence, dtype=np.uint8), compression="lzf")
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
    output = args_cli.output.resolve()
    event_dir = output / ".dvs_tmp"
    output.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(exist_ok=True)
    torch.manual_seed(args_cli.seed)
    configure_clean_event_renderer()
    env_cfg, success_term = configure_env()
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    dvs = DVSCamera.from_scene(
        env.scene,
        list(CAMERA_NAMES),
        out_dir=str(event_dir),
        compression="lzf",
        event_source="hdr",
        # Explicit rather than relying on EVIS's current default: no temporal
        # accumulation is allowed in inputs that are motion-vector warped.
        # `set_render_rtx_realtime` must not override the diagnostic path tracer.
        antialiasing=None if args_cli.renderer_mode == "pathtracing" else args_cli.dvs_antialiasing,
    )
    if args_cli.static_rgb_steps > 0:
        write_static_rgb_probe(env, output, args_cli.static_rgb_steps)
        print("[collect] static RGB probe complete", flush=True)
        env.close()
        simulation_app.close()
        return
    oracle = DrawerOracle(env)
    saved = completed_episode_count(output) if args_cli.resume else 0
    attempts = 0
    if saved:
        print(f"[collect] resuming from {saved} complete episodes", flush=True)
    buffer, prev, t_prev, action = reset_episode(env, dvs, oracle)
    success_steps = 0
    try:
        while saved < args_cli.episodes and simulation_app.is_running():
            attempts += 1
            for _ in range(args_cli.max_steps):
                # The wrist view is expressed in the current TCP frame, so it
                # moves with the gripper while keeping the grasp point centered.
                set_wrist_view(env)
                _, _, terminated, truncated, _ = env.step(action)
                current = dvs.snapshot()
                if bool(torch.logical_or(terminated, truncated).any()):
                    print(f"[collect] attempt {attempts} stopped at step {_ + 1}", flush=True)
                    break
                dvs.warp_and_process(prev, current, WARP, t_prev, 1.0 / EVENT_HZ)
                append_frame(buffer, env, action, t_prev)
                if drawer_is_open(env):
                    success_steps += 1
                else:
                    success_steps = 0
                if success_steps >= 10:
                    dvs.flush(0, saved)
                    raw_path = event_dir / f"env0_ep{saved}.h5"
                    if raw_path.exists() and buffer["t"]:
                        if write_episode(output / f"episode_{saved:05d}.h5", buffer, raw_path, saved):
                            saved += 1
                            print(f"[collect] saved {saved}/{args_cli.episodes}", flush=True)
                        raw_path.unlink(missing_ok=True)
                    break
                prev, t_prev = current, float(env.sim.current_time)
                action = oracle.action()
            dvs.recorder._events.pop(0, None)
            if saved >= args_cli.episodes:
                break
            buffer, prev, t_prev, action = reset_episode(env, dvs, oracle)
            success_steps = 0
    finally:
        env.close()
        shutil.rmtree(event_dir, ignore_errors=True)
    print(f"[collect] complete: {saved} successful trajectories after {attempts} attempts -> {output}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
