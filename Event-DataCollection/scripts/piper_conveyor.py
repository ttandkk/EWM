"""Shared Piper conveyor task; simulator imports require a running AppLauncher."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import product
import numpy as np

TASK = "Isaac-Conveyor-Pick-Piper-IK-Rel-v0"
PROMPT = "Pick up both objects arriving on the conveyor belt and place them in the basket."
CONTROL_HZ = 30
CAMERAS = ("cam0", "cam_wrist")
# Allow contact-solver/float rounding at the containment boundary (0.1 mm).
BASKET_GEOMETRY_TOLERANCE_M = 1e-4
BASKET_SUCCESS_CRITERION = "clipped_basket_v2"


@dataclass(frozen=True)
class ConveyorConfig:
    speed: float = 0.06
    arm_damping_scale: float = 0.5
    spawn_interval: float = 0.0
    object_spacing: float = 0.14
    object_spacing_jitter: float = 0.0
    max_seconds: float = 35.0
    belt_x: float = 0.33
    belt_length: float = 0.64
    belt_width: float = 0.12
    belt_top: float = 0.03
    spawn_y: float = -0.26
    pick_y: float = -0.10
    basket_x: float = 0.18
    basket_y: float = 0.18
    basket_length: float = 0.12
    basket_width: float = 0.10
    basket_height: float = 0.05
    wall_thickness: float = 0.002
    basket_taper: float = 0.006
    object_size: float = 0.04
    position_jitter: float = 0.01
    yaw_jitter: float = float(np.deg2rad(10))
    stable_seconds: float = 1.0
    linear_threshold: float = 0.02
    angular_threshold: float = 0.2
    retreat_distance: float = 0.10

    def __post_init__(self):
        if not (0 < self.speed <= 0.2 and self.spawn_interval >= 0 and self.max_seconds > 0):
            raise ValueError("invalid speed, spawn interval or duration")
        if not (0 < self.arm_damping_scale <= 1):
            raise ValueError("invalid arm damping scale")
        if self.object_spacing_jitter < 0:
            raise ValueError("spacing jitter must be nonnegative")
        if self.spawn_interval == 0 and (self.object_spacing-self.object_spacing_jitter <= self.object_size*np.sqrt(2)+.01
                or self.spawn_y+self.object_spacing+self.object_spacing_jitter+self.object_size*np.sqrt(2)/2 > self.belt_length/2):
            raise ValueError("simultaneous objects must start separated and on the belt")
        if min(self.basket_length, self.basket_width) <= self.object_size:
            raise ValueError("basket must fit the objects")
        if self.position_jitter < 0 or self.yaw_jitter < 0:
            raise ValueError("randomization ranges must be nonnegative")

    def metadata(self):
        return asdict(self)

    def supply_positions(self, seed):
        """Both objects start together, separated along the belt to clear the fingers."""
        rng = np.random.default_rng(seed)
        x = self.belt_x+rng.uniform(-self.position_jitter,self.position_jitter,2)
        y = np.full(2,self.spawn_y)
        if self.spawn_interval == 0:
            y[0] += self.object_spacing
        yaw = rng.uniform(-self.yaw_jitter,self.yaw_jitter,2)
        if self.spawn_interval == 0 and self.object_spacing_jitter:
            y[0] += rng.uniform(-self.object_spacing_jitter,self.object_spacing_jitter)
        return x,y,yaw


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def object_in_basket(pose, cfg: ConveyorConfig):
    """Contain the cube portion below the rim; allow protrusion and stacking."""
    signs = np.array(list(product((-0.5, 0.5), repeat=3)))
    corners = signs * cfg.object_size @ rotation_xyzw(pose[3:7]).T + np.asarray(pose[:3])
    tolerance = BASKET_GEOMETRY_TOLERANCE_M
    rim = cfg.wall_thickness + cfg.basket_height
    if corners[:, 2].min() < cfg.wall_thickness-0.002-tolerance:
        return False
    # A cube entirely above the opening, or merely resting on the rim, is outside.
    if corners[:, 2].min() >= rim-tolerance:
        return False
    if (abs(pose[0]-cfg.basket_x) > cfg.basket_length/2+cfg.basket_taper+tolerance
            or abs(pose[1]-cfg.basket_y) > cfg.basket_width/2+cfg.basket_taper+tolerance):
        return False
    vertices = list(corners[corners[:, 2] <= rim])
    # Include intersections of all twelve cube edges with the rim plane. Checking
    # just low corners would miss a cube passing through a wall near its top.
    for i in range(8):
        for j in range(i+1, 8):
            if np.count_nonzero(signs[i] != signs[j]) != 1:
                continue
            if (corners[i, 2]-rim)*(corners[j, 2]-rim) < 0:
                a = (rim-corners[i, 2])/(corners[j, 2]-corners[i, 2])
                vertices.append(corners[i]+a*(corners[j]-corners[i]))
    vertices = np.asarray(vertices)
    expansion = np.clip((vertices[:, 2]-cfg.wall_thickness)/cfg.basket_height, 0, 1)*cfg.basket_taper
    return bool(np.all(np.abs(vertices[:, 0]-cfg.basket_x) <= cfg.basket_length/2+expansion+tolerance)
                and np.all(np.abs(vertices[:, 1]-cfg.basket_y) <= cfg.basket_width/2+expansion+tolerance))


def success_now(poses, velocities, tcp, active, cfg: ConveyorConfig):
    return bool(all(active) and all(object_in_basket(p, cfg) for p in poses)
                and np.all(np.linalg.norm(np.asarray(velocities)[:, :3], axis=1) < cfg.linear_threshold)
                and np.all(np.linalg.norm(np.asarray(velocities)[:, 3:], axis=1) < cfg.angular_threshold)
                and np.all(np.linalg.norm(np.asarray(poses)[:, :3]-tcp, axis=1) >= cfg.retreat_distance))


class SuccessTracker:
    def __init__(self, cfg):
        self.cfg, self.steps = cfg, 0

    def update(self, poses, velocities, tcp, active):
        self.steps = self.steps+1 if success_now(poses, velocities, tcp, active, self.cfg) else 0
        return self.steps >= round(self.cfg.stable_seconds*CONTROL_HZ)


def wrist_mount_rotation(eye=(-.12, 0., -.07), target=(0., 0., .17)):
    """Fixed OpenGL camera frame in TCP coordinates; never recompute world up."""
    backward = np.asarray(eye, dtype=np.float64)-np.asarray(target, dtype=np.float64)
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0., 0., 1.]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return np.stack((right, up, backward), axis=1)


def camera_metadata():
    from piper_wrist_camera import (WRIST_EYE_LOCAL, WRIST_TARGET_LOCAL,
                                  WRIST_FOCAL_LENGTH_CM, WRIST_HORIZONTAL_APERTURE_CM)
    return dict(calibration_id="piper_conveyor_front_wrist_v3_rigid", resolution_hw=[256, 256],
                cam0_eye_world_m=[0.88, 0.0, 0.46], cam0_target_world_m=[0.14, 0.0, 0.08],
                cam0_focal_length_cm=24.0, cam0_horizontal_aperture_cm=20.955,
                wrist_eye_local_m=[-0.12, 0.0, -0.07], wrist_target_local_m=[0.0, 0.0, 0.17],
                wrist_focal_length_cm=WRIST_FOCAL_LENGTH_CM,
                wrist_horizontal_aperture_cm=WRIST_HORIZONTAL_APERTURE_CM,
                wrist_mount_mode="rigid_tcp",
                wrist_mount_rotation_local_opengl=wrist_mount_rotation().tolist())


def reset_conveyor_scene(env, env_ids):
    """Reset dynamic entities; the kinematic belt keeps its authored surface speed."""
    robot = env.scene["robot"]
    positions = robot.data.default_joint_pos.torch[env_ids].clone()
    velocities = robot.data.default_joint_vel.torch[env_ids].clone()
    robot.write_joint_position_to_sim_index(position=positions, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=velocities, env_ids=env_ids)
    robot.set_joint_position_target_index(target=positions, env_ids=env_ids)
    robot.set_joint_velocity_target_index(target=velocities, env_ids=env_ids)
    for name in ("cube_1", "cube_2"):
        obj = env.scene[name]
        pose = obj.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim_index(root_velocity=obj.data.default_root_vel.torch[env_ids].clone(),
                                            env_ids=env_ids)


def make_env_cfg(config=ConveyorConfig(), *, device="cpu", seed=42):
    import gymnasium as gym
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, EventTermCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from _isaacsim6_compat import (HDR_DATA_TYPE, install_legacy_extension_utils,
                                migrate_piper_stack_cfg, quat_from_wxyz)
    install_legacy_extension_utils()
    import agx_teleop.tasks  # noqa: F401
    from isaaclab.envs import mdp
    from pxr import PhysxSchema, Gf

    @configclass
    class Policy(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel)

    @configclass
    class Observations:
        policy = Policy()

    @configclass
    class Terminations:
        pass

    cfg = parse_env_cfg("Isaac-Stack-Cube-Piper-IK-Rel-v0", device=device, num_envs=1, use_fabric=True)
    migrate_piper_stack_cfg(cfg)
    from piper_conveyor_ik import ConveyorIKAction
    cfg.actions.arm_action.class_type = ConveyorIKAction
    # Relative IK updates at 30 Hz; the stack task's heavily damped drives lag
    # fast belt interception. Isolate a faster tracking profile to this task.
    cfg.scene.robot = cfg.scene.robot.copy()
    for name in ('piper_shoulder','piper_forearm'):
        cfg.scene.robot.actuators[name].damping *= config.arm_damping_scale
    @configclass
    class Events:
        reset_all = EventTermCfg(func=reset_conveyor_scene, mode="reset")

    # The user-approved initial image uses Piper's straight, zero-angle pose.
    # Explicit defaults make this pose reproducible across episode resets.
    cfg.scene.robot.init_state.joint_pos = {f"joint{i}": 0.0 for i in range(1, 7)}
    cfg.scene.robot.init_state.joint_pos.update(gripper_joint1=0.05, gripper_joint2=-0.05)
    cfg.seed, cfg.events = seed, Events()
    cfg.observations, cfg.terminations = Observations(), Terminations()
    cfg.sim.dt, cfg.decimation, cfg.sim.render_interval = 1/120, 4, 4
    cfg.sim.physx.enable_scene_query_support = True
    cfg.episode_length_s = config.max_seconds+3.0
    cfg.scene.cube_3 = None
    # This task uses one environment; inter-environment filtering is unnecessary.
    cfg.scene.filter_collisions = False
    cfg.scene.replicate_physics = False
    cfg.scene.light.spawn.color, cfg.scene.light.spawn.intensity = (1.0, 1.0, 1.0), 3000.0

    def material(color):
        return sim.PreviewSurfaceCfg(diffuse_color=color, roughness=1.0, metallic=0.0)

    cfg.scene.table.spawn.visual_material = material((0.42, 0.42, 0.40))
    cfg.scene.table.spawn.visual_material_path = "visual_material_clean_evis"
    cfg.scene.clean_tabletop = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CleanTabletop",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0001)),
        spawn=sim.CuboidCfg(size=(1.2, 0.9, 0.0002), visual_material=material((0.42, 0.42, 0.40))))

    def spawn_belt(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
        prim = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
        surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
        surface.CreateSurfaceVelocityEnabledAttr().Set(True)
        surface.CreateSurfaceVelocityLocalSpaceAttr().Set(False)
        surface.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(0.0, config.speed, 0.0))
        return prim

    cfg.scene.belt = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Belt",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(config.belt_x, 0.0, config.belt_top/2)),
        spawn=sim.CuboidCfg(func=spawn_belt, size=(config.belt_width, config.belt_length, config.belt_top),
            rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=material((0.10, 0.13, 0.17))))
    for index, sign in enumerate((-1, 1)):
        cfg.scene.__dict__[f"belt_roller_{index}"] = AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/BeltRoller_{index}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(config.belt_x, sign*config.belt_length/2, config.belt_top/2)),
            spawn=sim.CylinderCfg(radius=config.belt_top/2+0.001, height=config.belt_width+0.014,
                                 axis="X", visual_material=material((0.035, 0.035, 0.035))))
    for index, color in enumerate(((0.8, 0.10, 0.08), (0.08, 0.35, 0.85)), 1):
        setattr(cfg.scene, f"cube_{index}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/ConveyorObject_{index}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, index*0.1, 0.05)),
            spawn=sim.CuboidCfg(size=(config.object_size,)*3,
                rigid_props=sim.RigidBodyPropertiesCfg(solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4, max_depenetration_velocity=1.0),
                mass_props=sim.MassPropertiesCfg(mass=0.08),
                collision_props=sim.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                physics_material=sim.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
                visual_material=material(color))))
    from piper_conveyor_basket import spawn_basket
    basket_spawn = sim.CuboidCfg(func=spawn_basket, size=(1.0, 1.0, 1.0))
    basket_spawn.basket_config = config.metadata()
    cfg.scene.basket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(config.basket_x, config.basket_y, 0.0)),
        spawn=basket_spawn)
    cal = camera_metadata()
    for name in CAMERAS:
        prefix = "cam0" if name == "cam0" else "wrist"
        setattr(cfg.scene, name, CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}", update_period=1/CONTROL_HZ, height=256, width=256,
            data_types=["rgb", "distance_to_image_plane", "motion_vectors", HDR_DATA_TYPE],
            spawn=sim.PinholeCameraCfg(focal_length=cal[f"{prefix}_focal_length_cm"],
                horizontal_aperture=cal[f"{prefix}_horizontal_aperture_cm"], clipping_range=(0.03, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0),
                rot=quat_from_wxyz((1.0, 0.0, 0.0, 0.0)), convention="world")))
    if TASK not in gym.registry:
        gym.register(TASK, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                     kwargs={"env_cfg_entry_point": "piper_conveyor:make_env_cfg"})
    return cfg


def set_views(env):
    import torch
    from isaaclab.utils.math import quat_apply,quat_from_matrix,quat_mul
    cal = camera_metadata()
    def tensor(value):
        return torch.tensor([value], dtype=torch.float32, device=env.device)
    env.scene["cam0"].set_world_poses_from_view(tensor(cal["cam0_eye_world_m"]), tensor(cal["cam0_target_world_m"]))
    frame = env.scene["ee_frame"].data
    position, rotation = frame.target_pos_w[:, 0], frame.target_quat_w[:, 0]
    # A wrist camera has a rigid mount. World-up look-at becomes singular when
    # pointing down, adding spurious 180-degree image rolls to smooth arm motion.
    local_rotation = torch.tensor(cal["wrist_mount_rotation_local_opengl"],
                                  dtype=torch.float32, device=env.device)[None]
    orientation = quat_mul(rotation, quat_from_matrix(local_rotation))
    env.scene["cam_wrist"].set_world_poses(
        positions=position+quat_apply(rotation, tensor(cal["wrist_eye_local_m"])),
        orientations=orientation, convention="opengl")


class ConveyorRuntime:
    """Fixed-clock object supply and state shared by collection and evaluation."""
    def __init__(self, env, cfg=ConveyorConfig()):
        self.env, self.cfg = env, cfg
        self.objects = [env.scene[f"cube_{i}"] for i in (1, 2)]
        self.reset(42)

    def reset(self, seed):
        self.seed = int(seed)
        self.spawn_x,self.spawn_y,self.spawn_yaw = self.cfg.supply_positions(seed)
        self.active = [False, False]
        self.control_step, self.tracker = 0, SuccessTracker(self.cfg)

    @property
    def time(self):
        return self.control_step/CONTROL_HZ

    def supply(self):
        import torch
        spawned = []
        for i, obj in enumerate(self.objects):
            if not self.active[i] and self.time+1e-8 >= i*self.cfg.spawn_interval:
                yaw = self.spawn_yaw[i]
                pose = [[self.spawn_x[i], self.spawn_y[i],
                         self.cfg.belt_top+self.cfg.object_size/2+0.001,
                         0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)]]
                obj.write_root_pose_to_sim(torch.tensor(pose, dtype=torch.float32, device=self.env.device))
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.env.device))
                self.active[i] = True
                spawned.append(i)
        return spawned

    def state(self):
        def cpu(t):
            return t[0].detach().cpu().numpy().copy()
        poses = np.stack([np.concatenate((cpu(o.data.root_pos_w), cpu(o.data.root_quat_w))) for o in self.objects])
        velocities = np.stack([cpu(o.data.root_vel_w) for o in self.objects])
        tcp = cpu(self.env.scene["ee_frame"].data.target_pos_w)[0]
        return poses, velocities, tcp

    def after_step(self):
        self.control_step += 1
        poses, velocities, tcp = self.state()
        success = self.tracker.update(poses, velocities, tcp, self.active)
        reason = None
        for i, pose in enumerate(poses):
            if self.active[i] and not object_in_basket(pose, self.cfg):
                if pose[2] < -0.04:
                    reason = f"object_{i+1}_fell"
                elif pose[1] > self.cfg.belt_length/2+self.cfg.object_size and pose[2] < self.cfg.belt_top+0.08:
                    reason = f"object_{i+1}_missed"
        if self.time >= self.cfg.max_seconds and not success:
            reason = "timeout"
        return success, reason

    def metadata(self):
        return dict(scene_seed=self.seed, spawn_x=self.spawn_x.tolist(),
                    spawn_y=self.spawn_y.tolist(), spawn_yaw=self.spawn_yaw.tolist(), spawn_times=[0.0, self.cfg.spawn_interval],
                    config=self.cfg.metadata(), clock="fixed_simulation_time_inference_paused")
