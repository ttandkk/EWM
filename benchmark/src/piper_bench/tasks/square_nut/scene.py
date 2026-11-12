"""Fixed square peg and a freely grasped nut, with geometric seating checks."""
import math
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.common import StaticRuntime
from piper_bench.tasks.common import table_env_cfg
from piper_bench.tasks.square_nut.config import SquareNutConfig

TASK = 'Isaac-Square-Nut-Assembly-Piper-IK-Rel-v0'
PROMPT = 'Pick up the light-blue square nut by its handle and fit it onto the fixed orange square peg.'


def assembly_metrics(pose, velocity, tcp, gripper_open, cfg):
    rotation = rotation_xyzw(pose[3:])
    axis = rotation[:, 2]
    tilt = math.acos(float(np.clip(axis[2], -1., 1.)))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    # A square fits at four equivalent headings; the side handle need not face
    # one particular direction for the physical assembly to be successful.
    yaw_error = abs(math.atan2(math.sin(4*(yaw-cfg.peg_yaw)), math.cos(4*(yaw-cfg.peg_yaw)))) / 4
    corners = cfg.peg_width/2 * np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    cs, sn = math.cos(cfg.peg_yaw), math.sin(cfg.peg_yaw)
    corners = corners @ np.array([[cs, sn], [-sn, cs]]) + [cfg.peg_x, cfg.peg_y]
    extent = math.inf
    if axis[2] > .1:
        # Intersect all four vertical peg edges with BOTH nut faces. Checking
        # the centre alone would falsely accept a rotated or tilted wall perch.
        local = []
        for face in (-cfg.thickness/2, cfg.thickness/2):
            dz = (face - (corners-pose[:2]) @ axis[:2]) / axis[2]
            points = np.column_stack((corners-pose[:2], dz)) @ rotation
            local.append(points[:, :2])
        extent = float(np.max(np.abs(local)))
    bottom = pose[2] - cfg.thickness/2*axis[2]
    floor_error = float(bottom-cfg.base_top)
    bottom_spread = cfg.outer_width/2 * (abs(rotation[2, 0])+abs(rotation[2, 1]))
    hole_clear = extent <= cfg.hole_width/2 + .00005
    seated = bool(hole_clear and tilt < math.radians(4) and yaw_error < math.radians(6)
                  and abs(floor_error) < cfg.support_tolerance
                  and bottom_spread < cfg.support_tolerance
                  and pose[2]+cfg.thickness/2 < cfg.base_top+cfg.peg_height-.005)
    settled = np.linalg.norm(velocity[:3]) < .015 and np.linalg.norm(velocity[3:]) < .15
    valid = seated and settled and gripper_open and np.linalg.norm(tcp-pose[:3]) > .10
    return dict(valid=bool(valid), seated=seated, hole_clear=bool(hole_clear),
                position_error_m=float(np.linalg.norm(pose[:2]-[cfg.peg_x, cfg.peg_y])),
                yaw_error_deg=math.degrees(yaw_error), tilt_deg=math.degrees(tilt),
                floor_error_m=floor_error, bottom_spread_m=float(bottom_spread),
                bore_extent_m=extent if math.isfinite(extent) else None,
                insertion_depth_m=float(cfg.base_top+cfg.peg_height-bottom), settled=bool(settled))


def make_env_cfg(config=SquareNutConfig(), *, device='cpu', seed=42):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    from piper_bench.tasks.square_nut.geometry import spawn_nut
    cfg = table_env_cfg(config, device, seed)
    nut = cfg.scene.cube_1
    nut.spawn.func = spawn_nut
    nut.spawn.nut_config = config.metadata()
    nut.spawn.mass_props.mass = .055
    nut.spawn.rigid_props.solver_position_iteration_count = 32
    nut.spawn.rigid_props.solver_velocity_iteration_count = 8
    nut.spawn.physics_material = sim.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1., restitution=0.)
    nut.spawn.visual_material.diffuse_color = (.32, .55, .68)
    nut.spawn.visual_material.roughness = .55
    material = sim.PreviewSurfaceCfg(diffuse_color=(.65, .40, .22), roughness=.65)
    collision = sim.CollisionPropertiesCfg(contact_offset=.0003, rest_offset=0.)
    physical = sim.RigidBodyMaterialCfg(static_friction=.6, dynamic_friction=.5, restitution=0.)
    orientation = (0., 0., math.sin(config.peg_yaw/2), math.cos(config.peg_yaw/2))
    for name, size, z in (
        ('assembly_base', (config.base_width, config.base_width, config.base_height), config.table_z+config.base_height/2),
        ('square_peg', (config.peg_width, config.peg_width, config.peg_height), config.base_top+config.peg_height/2),
    ):
        setattr(cfg.scene, name, AssetBaseCfg(prim_path='{ENV_REGEX_NS}/'+name,
            spawn=sim.CuboidCfg(size=size, collision_props=collision, physics_material=physical, visual_material=material),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(config.peg_x, config.peg_y, z), rot=orientation)))
    return cfg


class SquareNutRuntime(StaticRuntime):
    def initial_pose(self, rng):
        c = self.cfg
        xy = np.array([c.source_x, c.source_y]) + rng.uniform(-c.position_jitter, c.position_jitter, 2)
        yaw = c.source_yaw + rng.uniform(-c.yaw_jitter, c.yaw_jitter)
        return np.r_[xy, c.table_z+c.thickness/2+.002, 0., 0., math.sin(yaw/2), math.cos(yaw/2)]

    def metrics(self, gripper_open=True):
        poses, velocity, tcp = self.state()
        return assembly_metrics(poses[0], velocity[0], tcp, gripper_open, self.cfg)

    def metadata(self):
        return dict(task=TASK, prompt=PROMPT, config=self.cfg.metadata(), seed=self.seed,
                    no_grasp_attachments=True, object_motion='passive contact only',
                    clock='fixed simulation time', initial_pose=self.state()[0][0].tolist())
