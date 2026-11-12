import math
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.common import StaticRuntime
from piper_bench.tasks.common import table_env_cfg
from piper_bench.tasks.common import camera_metadata as base_camera_metadata
from piper_bench.tasks.common import set_views as base_set_views
from piper_bench.tasks.common import upright_grasp as upright_bowl_grasp
from piper_bench.tasks.rotating_tube.scene import reset_scene
from piper_bench.tasks.drawer_cup.config import DrawerCupConfig

TASK = 'Isaac-Drawer-Cup-Piper-IK-Rel-v0'
PROMPT = 'Open the drawer, pick up the bowl by one side of its rim, place it upright inside, and close the drawer.'


def camera_metadata():
    value = base_camera_metadata()
    value.update(calibration_id='piper_drawer_bowl_left_aligned_v3',
                 cam0_eye_world_m=[.98, .65, .60], cam0_target_world_m=[.30, -.01, .045])
    return value


def set_views(env):
    import torch
    # Keep the rigid wrist calibration; only the global view changes for this task.
    base_set_views(env)
    cal = camera_metadata()
    tensor = lambda x: torch.tensor([x], dtype=torch.float32, device=env.device)
    env.scene['cam0'].set_world_poses_from_view(tensor(cal['cam0_eye_world_m']), tensor(cal['cam0_target_world_m']))


def drawer_rotation(cfg):
    c, s = math.cos(cfg.cabinet_yaw), math.sin(cfg.cabinet_yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def drawer_to_world(point, opening, cfg):
    return (drawer_rotation(cfg)@(np.asarray(point)+[opening, 0., 0.])
            +[cfg.cabinet_x, cfg.cabinet_y, cfg.drawer_z])


def handle_position(opening, cfg):
    return drawer_to_world([.305*cfg.scale_x+cfg.handle_extension, 0., .009*cfg.scale_z], opening, cfg)


def placement_position(opening, cfg):
    # Offset within the tray to leave room for the single-rim grip and housing.
    return drawer_to_world([.125*cfg.scale_x, -.04,
                            -.040356*cfg.scale_z+cfg.cup_height/2+.001], opening, cfg)


def bowl_rim_grasp(pose, cfg, *, heading=None):
    """Pinch a lateral wall with radial tool yaw and pads parallel to its taper."""
    from scipy.spatial.transform import Rotation
    from piper_bench.tasks.common import yaw_pitch_quaternion
    radial = math.atan2(pose[1], pose[0]) if heading is None else heading
    # This side is nearer the hand after opening the drawer and keeps the
    # fingers away from the drawer's front panel during placement.
    normal = np.array([math.sin(radial), -math.cos(radial), 0.])
    # Wall-aligned pads retain contact during the lift. Keep the full open
    # finger mesh above the table, including its tilted pad corners.
    # Selecting the lateral rim avoids a quarter-turn about the wrist axis.
    depth = .010
    taper = (cfg.cup_radius-cfg.bowl_bottom_radius)/cfg.cup_height
    radius = cfg.cup_radius-depth*taper-cfg.bowl_wall/2
    point = pose[:3]+radius*normal+np.array([0., 0., cfg.cup_height/2-depth])
    base = Rotation.from_quat(yaw_pitch_quaternion(radial, math.pi))
    wall_aligned = base*Rotation.from_rotvec([-math.atan(taper), 0., 0.])
    # Tilt along the wall tangent for wrist-joint clearance. Rotating about
    # local Y leaves the closing direction normal to the sloped wall.
    return point, (wall_aligned*Rotation.from_rotvec([0., -.25, 0.])).as_quat()


def cup_in_drawer(pose, opening, cfg):
    # Keep the legacy function name; the contained object is now a shallow bowl.
    local = drawer_rotation(cfg).T@(pose[:3]-[cfg.cabinet_x, cfg.cabinet_y, cfg.drawer_z])-[opening, 0., 0.]
    upright = rotation_xyzw(pose[3:])[2, 2] > .98
    return bool(abs(local[0])+cfg.cup_radius < .245*cfg.scale_x and
                abs(local[1])+cfg.cup_radius < .338*cfg.scale_y and upright and
                abs(pose[2]-cfg.cup_height/2-cfg.floor_z) < .004)


def bowl_ready_to_release(pose, velocity, opening, cfg):
    """Drop an aligned bowl from up to 35 mm; success still requires settling."""
    target = placement_position(opening, cfg)
    gap = pose[2]-cfg.cup_height/2-cfg.floor_z
    local = drawer_rotation(cfg).T@(pose[:3]-[cfg.cabinet_x, cfg.cabinet_y, cfg.drawer_z])-[opening, 0., 0.]
    # Keep the entire rim over the tray even when accepting a loose XY target.
    over_tray = (abs(local[0])+cfg.cup_radius < .245*cfg.scale_x and
                 abs(local[1])+cfg.cup_radius < .338*cfg.scale_y)
    return bool(opening >= cfg.open_threshold and over_tray
                and np.linalg.norm(pose[:2]-target[:2]) < .020 and -.001 <= gap <= .035
                and rotation_xyzw(pose[3:])[2, 2] > .98 and np.linalg.norm(velocity[:3]) < .03)


class SequenceTracker:
    def __init__(self):
        self.opened = self.placed = False

    def update(self, pose, vel, tcp, opening, drawer_speed, gripper_open, cfg):
        self.opened |= opening >= cfg.open_threshold
        inside = cup_in_drawer(pose, opening, cfg)
        settled = np.linalg.norm(vel[:3]) < .02 and np.linalg.norm(vel[3:]) < .2
        if self.opened and opening >= cfg.open_threshold and inside and settled and gripper_open:
            self.placed = True
        closed = opening < cfg.closed_threshold
        valid = (self.opened and self.placed and inside and settled and closed and abs(drawer_speed) < .01
                 and gripper_open and np.linalg.norm(tcp-pose[:3]) > .12)
        return dict(valid=bool(valid), opened=bool(self.opened), placed=bool(self.placed),
                    cup_inside=inside, closed=bool(closed), opening_m=opening, drawer_speed_m_s=drawer_speed)


def reset_drawer_scene(env, env_ids):
    reset_scene(env, env_ids)
    cabinet = env.scene['cabinet']
    pos = cabinet.data.default_joint_pos.torch[env_ids].clone()
    vel = cabinet.data.default_joint_vel.torch[env_ids].clone()
    cabinet.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
    cabinet.write_joint_velocity_to_sim_index(velocity=vel, env_ids=env_ids)
    cabinet.set_joint_velocity_target_index(target=vel, env_ids=env_ids)


def make_env_cfg(config=DrawerCupConfig(), *, device='cpu', seed=42):
    import isaaclab.sim as sim
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from piper_bench.tasks.turntable_cups.config import CupConfig
    from piper_bench.tasks.turntable_cups.geometry import spawn_cup
    from piper_bench.tasks.drawer_cup.geometry import spawn_drawer
    cfg = table_env_cfg(config, device, seed)
    cfg.events.reset_all.func = reset_drawer_scene
    cfg.actions.gripper_action.open_command_expr = {
        'gripper_joint1': config.gripper_open_half_width,
        'gripper_joint2': -config.gripper_open_half_width}
    cfg.gripper_open_val = [config.gripper_open_half_width, -config.gripper_open_half_width]
    cup = cfg.scene.cube_1
    cup.spawn.func = spawn_cup
    cup.spawn.cup_config = CupConfig(object_size=config.cup_height, cup_height=config.cup_height,
        cup_bottom_radius=config.bowl_bottom_radius, cup_top_radius=config.cup_radius,
        cup_wall=config.bowl_wall, cup_floor=config.bowl_floor).metadata()
    cup.spawn.mass_props.mass = .035
    cup.spawn.visual_material.diffuse_color = (.28, .38, .22)
    cup.spawn.rigid_props.solver_position_iteration_count = 32
    cup.spawn.rigid_props.solver_velocity_iteration_count = 8
    spawn = sim.CuboidCfg(func=spawn_drawer, size=(1., 1., 1.))
    spawn.drawer_config = config.metadata()
    cfg.scene.cabinet = ArticulationCfg(prim_path='{ENV_REGEX_NS}/DrawerCabinet', spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(pos=(config.cabinet_x, config.cabinet_y, config.drawer_z),
            rot=(0., 0., math.sin(config.cabinet_yaw/2), math.cos(config.cabinet_yaw/2)),
            joint_pos={'drawer_joint': 0.}),
        actuators={'slide': ImplicitActuatorCfg(joint_names_expr=['drawer_joint'],
            stiffness=0., damping=2., effort_limit_sim=15., velocity_limit_sim=.12)})
    return cfg


class DrawerCupRuntime(StaticRuntime):
    def reset(self, seed):
        super().reset(seed)
        self.tracker = SequenceTracker()

    def initial_pose(self, rng):
        c = self.cfg
        xy = np.array([c.source_x, c.source_y])+rng.uniform(-c.position_jitter, c.position_jitter, 2)
        yaw = rng.uniform(-c.yaw_jitter, c.yaw_jitter)
        return np.r_[xy, c.cup_height/2+.001, 0., 0., math.sin(yaw/2), math.cos(yaw/2)]

    def opening(self):
        cabinet = self.env.scene['cabinet']
        index = cabinet.find_joints('drawer_joint')[0][0]
        return float(cabinet.data.joint_pos[0, index]), float(cabinet.data.joint_vel[0, index])

    def handle(self):
        return handle_position(self.opening()[0], self.cfg)

    def placement(self):
        return placement_position(self.opening()[0], self.cfg)

    def metrics(self, gripper_open=True):
        poses, vel, tcp = self.state()
        return self.tracker.update(poses[0], vel[0], tcp, *self.opening(), gripper_open, self.cfg)

    def metadata(self):
        return dict(task=TASK, prompt=PROMPT, config=self.cfg.metadata(), seed=self.seed,
                    object='shallow hollow bowl, single lateral rim pinch with wall-aligned fingers',
                    drawer='compact axis-aligned Sektion tray on the left; opens towards workspace centre',
                    drawer_motion='passive prismatic joint; physical pull and push only',
                    no_grasp_attachments=True, clock='fixed simulation time')
