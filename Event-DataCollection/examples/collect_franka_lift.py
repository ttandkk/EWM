"""Headless Franka lift demonstrations with stereo RGB/depth/DVS observations.

The controller is Isaac Lab's GPU Warp pick-and-lift state machine.  One HDF5
file is written per *successful* episode.  The file contains synchronized 30 Hz
control frames and the two asynchronous DVS streams generated at 240 Hz.

Run through Isaac Lab, for example::

    ./isaaclab.sh -p /path/to/collect_franka_lift.py --headless --enable_cameras \
        --episodes 50 --output /path/to/franka_lift_50
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect successful headless Franka lift demonstrations with EVIS.")
parser.add_argument("--episodes", type=int, default=50, help="Number of successful trajectories to save.")
parser.add_argument("--output", type=Path, required=True, help="Output directory (one HDF5 file per trajectory).")
parser.add_argument("--max_steps", type=int, default=180, help="Maximum 30 Hz control steps attempted per episode.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401 -- registers the Gym task
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from dvs_gen.sensors import DVSCamera

TASK = "Isaac-Lift-Cube-Franka-IK-Abs-v0"
CONTROL_HZ = 30
WARP = 8
EVENT_HZ = CONTROL_HZ * WARP


class PickAndLiftOracle:
    """Single-environment Torch form of Isaac Lab's five-stage lift oracle.

    Keeping it here avoids importing the official executable state-machine script,
    whose module-level code creates another Isaac Sim application.
    """

    REST, ABOVE, APPROACH, GRASP, LIFT = range(5)

    def __init__(self, dt: float, device: torch.device | str):
        self.dt, self.device = float(dt), device
        self.state = self.REST
        self.wait = 0.0
        self.last_pose = torch.tensor([[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]], device=device)

    def reset(self) -> None:
        self.state, self.wait = self.REST, 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, goal_pose: torch.Tensor) -> torch.Tensor:
        obj = object_pose[0, :3]
        ee = ee_pose[0, :3]
        target = self.last_pose.clone()
        gripper = 1.0
        if self.state == self.REST:
            target[0] = ee_pose[0]
            if self.wait >= 0.2:
                self.state, self.wait = self.ABOVE, 0.0
        elif self.state == self.ABOVE:
            target[0, :3] = obj + torch.tensor([0.0, 0.0, 0.10], device=self.device)
            target[0, 3:] = ee_pose[0, 3:]
            if torch.linalg.vector_norm(ee - target[0, :3]) < 0.01 and self.wait >= 0.5:
                self.state, self.wait = self.APPROACH, 0.0
        elif self.state == self.APPROACH:
            target[0, :3], target[0, 3:] = obj, ee_pose[0, 3:]
            if torch.linalg.vector_norm(ee - obj) < 0.01 and self.wait >= 0.6:
                self.state, self.wait = self.GRASP, 0.0
        elif self.state == self.GRASP:
            target[0, :3], target[0, 3:] = obj, ee_pose[0, 3:]
            gripper = -1.0
            if self.wait >= 0.3:
                self.state, self.wait = self.LIFT, 0.0
        else:
            target[0], gripper = goal_pose[0], -1.0
        self.last_pose = target
        self.wait += self.dt
        return torch.cat((target, torch.tensor([[gripper]], device=self.device)), dim=-1)


def stereo_camera(name: str, pos: tuple[float, float, float]) -> CameraCfg:
    """A fixed table-view camera.  ``world`` convention looks along +X."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        update_period=1.0 / CONTROL_HZ,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "motion_vectors", "HdrColor"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"),
    )


def configure_env() -> LiftEnvCfg:
    cfg: LiftEnvCfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1, use_fabric=True)
    # Preserve 120 Hz physics, but expose a camera/control keyframe every 1/30 s.
    cfg.sim.dt = 1.0 / 120.0
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.episode_length_s = float(args_cli.max_steps) / CONTROL_HZ
    # A small y-baseline gives a calibrated-style stereo pair around the table view.
    cfg.scene.cam0 = stereo_camera("cam0", (0.0, -0.30, 1.00))
    cfg.scene.cam1 = stereo_camera("cam1", (0.0, -0.24, 1.00))
    return cfg


def new_buffer() -> dict[str, list[np.ndarray] | list[float]]:
    return {"t": [], "action": [], "joint_pos": [], "joint_vel": [], "ee_pose": [], "object_pose": [],
            "rgb_cam0": [], "rgb_cam1": [], "depth_cam0": [], "depth_cam1": []}


def append_frame(buf: dict, env, action: torch.Tensor, t: float) -> None:
    scene = env.scene
    robot = scene["robot"].data
    obj: RigidObjectData = scene["object"].data
    ee = scene["ee_frame"].data
    buf["t"].append(float(t))
    buf["action"].append(action[0].detach().cpu().numpy().astype(np.float32))
    buf["joint_pos"].append(robot.joint_pos[0].detach().cpu().numpy().astype(np.float32))
    buf["joint_vel"].append(robot.joint_vel[0].detach().cpu().numpy().astype(np.float32))
    buf["ee_pose"].append(np.concatenate((ee.target_pos_w[0, 0].detach().cpu().numpy(), ee.target_quat_w[0, 0].detach().cpu().numpy())).astype(np.float32))
    buf["object_pose"].append(np.concatenate((obj.root_pos_w[0].detach().cpu().numpy(), obj.root_quat_w[0].detach().cpu().numpy())).astype(np.float32))
    for cam in ("cam0", "cam1"):
        out = scene[cam].data.output
        # RGB has an alpha channel in some Isaac Sim builds; retain RGB only.
        buf[f"rgb_{cam}"].append(out["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
        buf[f"depth_{cam}"].append(out["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy().astype(np.float16))


def write_episode(path: Path, buffer: dict, dvs_path: Path, episode_id: int) -> None:
    """Combine control-rate observations and EVIS's asynchronous event stream."""
    with h5py.File(path, "w") as out, h5py.File(dvs_path, "r") as events:
        meta = out.create_group("meta")
        meta.attrs.update(task=TASK, success=True, control_hz=CONTROL_HZ, event_hz=EVENT_HZ,
                          warp=WARP, episode_id=episode_id, camera_resolution=np.array([640, 480], np.int32))
        obs = out.create_group("observation")
        obs.create_dataset("t", data=np.asarray(buffer["t"], np.float64))
        for key in ("joint_pos", "joint_vel", "ee_pose", "object_pose"):
            obs.create_dataset(key, data=np.asarray(buffer[key], np.float32), compression="lzf")
        rgb = obs.create_group("rgb")
        depth = obs.create_group("depth")
        for cam in ("cam0", "cam1"):
            rgb.create_dataset(cam, data=np.asarray(buffer[f"rgb_{cam}"], np.uint8), compression="lzf")
            depth.create_dataset(cam, data=np.asarray(buffer[f"depth_{cam}"], np.float16), compression="lzf")
        out.create_dataset("action", data=np.asarray(buffer["action"], np.float32), compression="lzf")
        dvs = out.create_group("events")
        for cam in ("cam0", "cam1"):
            events.copy(f"DVS/{cam}", dvs, name=cam)


def reset_episode(env, dvs: DVSCamera, sm: PickAndLiftOracle) -> tuple[dict, dict, float, torch.Tensor]:
    env.reset()
    ids = torch.tensor([0], device=env.device, dtype=torch.long)
    sm.reset()
    dvs.reset(ids)
    actions = torch.zeros(env.action_space.shape, device=env.device)
    actions[:, 3] = 1.0
    env.step(actions)  # render a clean post-reset reference frame for the DVS model
    return new_buffer(), dvs.snapshot(), float(env.sim.current_time), actions


def main() -> None:
    output = args_cli.output.resolve()
    temp_events = output / ".dvs_tmp"
    output.mkdir(parents=True, exist_ok=True)
    temp_events.mkdir(exist_ok=True)
    torch.manual_seed(args_cli.seed)

    env = gym.make(TASK, cfg=configure_env()).unwrapped
    dvs = DVSCamera.from_scene(
        env.scene, ["cam0", "cam1"], out_dir=str(temp_events), compression="lzf",
        margin=(0, 0, 0, 0), event_source="hdr",
    )
    sm = PickAndLiftOracle(1.0 / CONTROL_HZ, env.device)
    desired_orientation = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=env.device)
    saved = attempts = 0
    buffer, prev, t_prev, actions = reset_episode(env, dvs, sm)

    try:
        while saved < args_cli.episodes and simulation_app.is_running():
            attempts += 1
            for _ in range(args_cli.max_steps):
                _, _, terminated, truncated, _ = env.step(actions)
                cur = dvs.snapshot()
                # Do not warp across an automatic reset; it would create false events.
                if bool(torch.logical_or(terminated, truncated).any()):
                    break
                dvs.warp_and_process(prev, cur, WARP, t_prev, 1.0 / EVENT_HZ)
                append_frame(buffer, env, actions, t_prev)
                ee = env.scene["ee_frame"].data
                obj: RigidObjectData = env.scene["object"].data
                ee_pose = torch.cat((ee.target_pos_w[..., 0, :] - env.scene.env_origins, ee.target_quat_w[..., 0, :]), dim=-1)
                obj_pose = torch.cat((obj.root_pos_w - env.scene.env_origins, desired_orientation), dim=-1)
                goal = env.command_manager.get_command("object_pose")
                actions = sm.compute(ee_pose, obj_pose, torch.cat((goal[..., :3], desired_orientation), dim=-1))
                prev, t_prev = cur, float(env.sim.current_time)
                if bool(lift_mdp.object_reached_goal(env, threshold=0.025).any()):
                    dvs.flush(0, saved)
                    dvs_path = temp_events / f"env0_ep{saved}.h5"
                    if dvs_path.exists() and buffer["t"]:
                        write_episode(output / f"episode_{saved:05d}.h5", buffer, dvs_path, saved)
                        dvs_path.unlink()
                        saved += 1
                        print(f"[collect] saved {saved}/{args_cli.episodes}: {output / f'episode_{saved - 1:05d}.h5'}", flush=True)
                    break
            # Drop failed / unfinished event chunks and restart the oracle cleanly.
            dvs.recorder._events.pop(0, None)
            buffer, prev, t_prev, actions = reset_episode(env, dvs, sm)
    finally:
        env.close()
        shutil.rmtree(temp_events, ignore_errors=True)
    print(f"[collect] complete: {saved} successful trajectories after {attempts} attempts -> {output}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
