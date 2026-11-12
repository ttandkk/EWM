import math
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.common import StaticRuntime
from piper_bench.tasks.common import table_env_cfg
from piper_bench.tasks.push_t.config import PushTConfig
from piper_bench.tasks.push_t.geometry import coverage

TASK = 'Isaac-Push-T-Piper-IK-Rel-v0'
PROMPT = 'Push the T-shaped block onto the matching green target without lifting it.'


def pose_metrics(pose, vel, tcp, cfg):
    rot = rotation_xyzw(pose[3:])
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    angle = abs(math.atan2(math.sin(yaw-cfg.target_yaw), math.cos(yaw-cfg.target_yaw)))
    position = float(np.linalg.norm(pose[:2]-[cfg.target_x, cfg.target_y]))
    overlap = coverage(cfg, pose[:2], yaw)
    valid = (position < cfg.position_tolerance and angle < cfg.yaw_tolerance and overlap >= cfg.min_coverage
             and abs(pose[2]-(cfg.height/2-.003)) < .004 and rot[2, 2] > .995
             and np.linalg.norm(vel[:3]) < .015 and np.linalg.norm(vel[3:]) < .15
             and np.linalg.norm(tcp-pose[:3]) > .10)
    return dict(valid=bool(valid), position_error_m=position, yaw_error_deg=math.degrees(angle),
                target_coverage=float(overlap), yaw_rad=yaw, upright=bool(rot[2, 2] > .995))


def make_env_cfg(config=PushTConfig(), *, device='cpu', seed=42):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    from piper_bench.tasks.push_t.geometry import spawn_t
    cfg = table_env_cfg(config, device, seed)
    block = cfg.scene.cube_1
    block.spawn.func = spawn_t
    block.spawn.t_config = config.metadata()
    block.spawn.mass_props.mass = .07
    block.spawn.physics_material = sim.RigidBodyMaterialCfg(static_friction=.45, dynamic_friction=.35)
    block.spawn.visual_material.diffuse_color = (.48, .24, .16)
    goal = sim.CuboidCfg(func=spawn_t, size=(1., 1., 1.),
                        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.16, .34, .23), roughness=1.))
    goal.t_config, goal.goal_only = config.metadata(), True
    cfg.scene.goal = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/TGoal', spawn=goal,
        init_state=AssetBaseCfg.InitialStateCfg(pos=(config.target_x, config.target_y, .0004),
            rot=(0., 0., math.sin(config.target_yaw/2), math.cos(config.target_yaw/2))))
    return cfg


class PushTRuntime(StaticRuntime):
    def initial_pose(self, rng):
        c = self.cfg
        xy = np.array([c.source_x, c.source_y])+rng.uniform(-c.position_jitter, c.position_jitter, 2)
        yaw = rng.uniform(-c.yaw_jitter, c.yaw_jitter)
        return np.r_[xy, c.height/2+.001, 0., 0., math.sin(yaw/2), math.cos(yaw/2)]

    def metrics(self, gripper_open=True):
        poses, vel, tcp = self.state()
        return pose_metrics(poses[0], vel[0], tcp, self.cfg)

    def metadata(self):
        return dict(task=TASK, prompt=PROMPT, config=self.cfg.metadata(), seed=self.seed,
                    no_grasp_attachments=True, object_motion='passive contact only', clock='fixed simulation time')
