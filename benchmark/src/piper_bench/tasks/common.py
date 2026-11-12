"""Shared table, rigid wrist camera, physical state and smooth Cartesian control."""
import math
import numpy as np
from piper_bench.tasks.conveyor.core import ConveyorConfig
from piper_bench.tasks.conveyor.core import ConveyorRuntime
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import CAMERAS
from piper_bench.tasks.conveyor.core import make_env_cfg as conveyor_env_cfg
from piper_bench.tasks.rotating_tube.scene import reset_scene
from piper_bench.tasks.rotating_tube.scene import tabletop_camera_metadata as camera_metadata
from piper_bench.tasks.rotating_tube.scene import set_tabletop_views as set_views
from piper_bench.tasks.rotating_tube.oracle import TubeOracle
from piper_bench.tasks.rotating_tube.oracle import yaw_pitch_quaternion
from piper_bench.tasks.rotating_tube.oracle import interpolate_quaternion
from piper_bench.robots.piper.motion import line_motion
from piper_bench.robots.piper.motion import quintic


def tabletop_orientation(target):
    """Keep contact nearly vertical; tilt gradually aloft to respect wrist limits."""
    pitch = math.pi-.75*quintic((target[2]-.04)/.20)
    return yaw_pitch_quaternion(math.atan2(target[1], target[0]), pitch)


def approach_pose(start, start_quat, end, u):
    target, orientation = TubeOracle.approach_pose(start, start_quat, end, u)
    # Tube's initial radial path avoids the zero-home wrist singularity. Adapt
    # its terminal pitch for the higher clearances required by these tasks.
    from piper_bench.tasks.conveyor.core import rotation_xyzw
    initial_pitch = math.acos(np.clip(rotation_xyzw(start_quat)[2, 2], -1., 1.))
    final_pitch = math.pi-.75*quintic((end[2]-.04)/.20)
    orientation = yaw_pitch_quaternion(math.atan2(target[1], target[0]), (1-u)*initial_pitch+u*final_pitch)
    return target, orientation


def upright_grasp(tool_quat, object_quat, reference_quat):
    """Level a measured grasp while preserving the reference wrist heading."""
    from scipy.spatial.transform import Rotation
    from piper_bench.tasks.conveyor.core import rotation_xyzw
    tool = Rotation.from_quat(tool_quat)
    up = rotation_xyzw(object_quat)[:, 2]
    axis = np.cross(up, [0., 0., 1.]); sine = np.linalg.norm(axis)
    if sine > 1e-8:
        correction = Rotation.from_rotvec(axis*(math.atan2(sine, up[2])/sine))
    else:
        correction = Rotation.from_rotvec([0. if up[2] >= 0 else math.pi, 0., 0.])
    level = correction*tool
    heading = lambda matrix: math.atan2(matrix[1, 0], matrix[0, 0])
    yaw = heading(rotation_xyzw(reference_quat))-heading(level.as_matrix())
    return (Rotation.from_rotvec([0., 0., yaw])*level).as_quat()


def table_env_cfg(config, device, seed):
    cfg = conveyor_env_cfg(ConveyorConfig(max_seconds=config.max_seconds), device=device, seed=seed)
    cfg.events.reset_all.func = reset_scene
    for name in ('belt', 'belt_roller_0', 'belt_roller_1', 'basket', 'cube_2'):
        setattr(cfg.scene, name, None)
    for name in CAMERAS:
        getattr(cfg.scene, name).spawn.clipping_range = (.03, 100.)
    cfg.scene.cam_wrist.spawn.focal_length = camera_metadata()['wrist_focal_length_cm']
    return cfg


class StaticRuntime:
    """Object pose writes are restricted to episode initialization."""
    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.objects = [env.scene['cube_1']]
        self.reset(42)

    @property
    def time(self):
        return self.control_step / CONTROL_HZ

    state = ConveyorRuntime.state

    def reset(self, seed):
        self.seed, self.control_step, self.stable_steps = seed, 0, 0
        self.supplied = False

    def supply(self):
        import torch
        if self.supplied:
            return []
        rng = np.random.default_rng(self.seed)
        pose = self.initial_pose(rng)
        obj = self.objects[0]
        obj.write_root_pose_to_sim(torch.tensor(pose[None], dtype=torch.float32, device=self.env.device))
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.env.device))
        self.supplied = True
        return [0]

    def after_step(self, gripper_open=True):
        self.control_step += 1
        metrics = self.metrics(gripper_open)
        self.stable_steps = self.stable_steps + 1 if metrics['valid'] else 0
        success = self.stable_steps >= round(self.cfg.stable_seconds * CONTROL_HZ)
        pose = self.state()[0][0]
        reason = 'object_fell' if pose[2] < -.04 else None
        if self.time >= self.cfg.max_seconds and not success:
            reason = 'timeout'
        return bool(success), reason


class CartesianOracle:
    """Bounded feedback around C2 references, using the existing Piper IK action."""
    def __init__(self, runtime):
        self.runtime, self.env, self.cfg = runtime, runtime.env, runtime.cfg
        self.stage, self.started, self.target = 'approach', None, None
        self.orientation_error = math.inf
        self.failure = None

    def begin(self, stage, tcp):
        self.stage, self.started = stage, self.runtime.time
        self.start = tcp.copy()
        self.start_q = self.env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().numpy().copy()

    def move(self, end, seconds):
        return line_motion(self.start, end, self.runtime.time-self.started, seconds)

    def reached(self, end, seconds, tolerance=.006):
        tcp = self.runtime.state()[2]
        return (self.runtime.time-self.started >= seconds and np.linalg.norm(tcp-end) < tolerance
                and self.orientation_error < .06)

    def command(self, target, orientation, grip, translation_limit=.012, *, angular_limit=None):
        import torch
        from isaaclab.utils.math import compute_pose_error
        from isaaclab.utils.math import quat_apply_inverse
        frame = self.env.scene['ee_frame'].data
        current_p, current_q = frame.target_pos_w[:, 0], frame.target_quat_w[:, 0]
        delta = np.asarray(target)-current_p[0].detach().cpu().numpy()
        norm = np.linalg.norm(delta)
        if norm > translation_limit:
            delta *= translation_limit/norm
        desired_q = torch.tensor(np.asarray(orientation)[None], dtype=torch.float32, device=self.env.device)
        _, error = compute_pose_error(current_p, current_q, current_p, desired_q, rot_error_type='axis_angle')
        angular = error[0].detach().cpu().numpy()
        self.orientation_error = float(np.linalg.norm(angular))
        limit = (.20 if self.stage == 'approach' else .08) if angular_limit is None else angular_limit
        if self.orientation_error > limit:
            angular *= limit/self.orientation_error
        base_q = self.env.scene['robot'].data.root_quat_w.torch
        def local(value):
            return quat_apply_inverse(base_q, torch.tensor(value[None], dtype=torch.float32, device=self.env.device))[0]
        self.target = np.asarray(target).copy()
        return torch.cat((2*local(delta), 2*local(angular), torch.tensor([grip], device=self.env.device))).unsqueeze(0)
