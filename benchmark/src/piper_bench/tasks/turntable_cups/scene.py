"""Rotating disk, green cup into basket, then purple cup nested inside green."""
import math
import numpy as np
from piper_bench.tasks.turntable_cups.turntable import TurntableConfig
from piper_bench.tasks.turntable_cups.turntable import TurntableRuntime
from piper_bench.tasks.turntable_cups.turntable import make_env_cfg as base_env_cfg
from piper_bench.tasks.turntable_cups.turntable import DISK_SEGMENTS
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.conveyor.core import camera_metadata as original_camera_metadata
from piper_bench.tasks.turntable_cups.geometry import spawn_cup
from piper_bench.tasks.turntable_cups.geometry import spawn_brown_basket
from piper_bench.tasks.turntable_cups.geometry import cup_rings
from piper_bench.tasks.turntable_cups.geometry import WALL_SEGMENTS

TASK='Isaac-Turntable-Cup-Nesting-Piper-IK-Rel-v0'
PROMPT='Place the green bowl in the basket, then nest the purple bowl inside the green bowl.'
BOWL_SUCCESS_VERSION='bowl_nested_upright_stable_1s_v2_depth_tol1mm'

def camera_metadata():
    value=original_camera_metadata()
    value.update(calibration_id='piper_turntable_cups_conveyor_front_v3')
    return value


from piper_bench.tasks.turntable_cups.config import CupConfig
from piper_bench.tasks.turntable_cups.config import BowlConfig


def cup_in_basket(pose,cfg):
    # Conservative containment of the complete footprint; protruding rims are allowed.
    points=cup_rings(cfg,32).reshape(-1,3)@rotation_xyzw(pose[3:7]).T+pose[:3]
    low=points[:,2].min()
    return bool(low>=cfg.wall_thickness-.003 and low<cfg.wall_thickness+cfg.basket_height-.005
        and np.max(np.abs(points[:,0]-cfg.basket_x))<cfg.basket_length/2
        and np.max(np.abs(points[:,1]-cfg.basket_y))<cfg.basket_width/2)


def nesting_metrics(poses,cfg):
    rotations=[rotation_xyzw(p[3:7]) for p in poses]
    relative=rotations[0].T@(poses[1,:3]-poses[0,:3])
    tilt=[math.degrees(math.acos(np.clip(r[2,2],-1,1))) for r in rotations]
    gap=float(relative[2]); xy=float(np.linalg.norm(relative[:2]))
    floor=float(poses[0,2]-cfg.cup_height/2-cfg.wall_thickness)
    # Require half-depth nesting, with 1 mm seating tolerance for shallow bowls.
    # Reject rim perches, side-by-side cups, inversion and interpenetration.
    min_gap=cfg.cup_wall/cfg.taper-.003
    max_gap=cfg.cup_height*.5
    if cfg.cup_height < 2*cfg.cup_top_radius:
        # Shallow bowls need a floor/wall clearance bound; the legacy 3 mm
        # tolerance is comparable to their whole seating gap.
        min_gap=max(cfg.cup_floor, cfg.cup_wall/cfg.taper)-.0005
        max_gap+=.001
    valid=all(cup_in_basket(p,cfg) for p in poses) and max(tilt)<8 and abs(floor)<.003
    valid=valid and xy<.005 and min_gap<gap<max_gap
    return dict(valid=bool(valid),relative_xy_m=xy,base_height_gap_m=gap,
                insertion_depth_m=cfg.cup_height-gap,tilt_deg=tilt,lower_floor_error_m=floor)


class NestingTracker:
    def __init__(self,cfg):self.cfg,self.steps=cfg,0
    def update(self,poses,velocities,tcp,active):
        valid=all(active) and nesting_metrics(poses,self.cfg)['valid']
        valid=valid and np.all(np.linalg.norm(velocities[:,:3],axis=1)<self.cfg.linear_threshold)
        valid=valid and np.all(np.linalg.norm(velocities[:,3:],axis=1)<self.cfg.angular_threshold)
        valid=valid and np.all(np.linalg.norm(poses[:,:3]-tcp,axis=1)>=self.cfg.retreat_distance)
        self.steps=self.steps+1 if valid else 0
        return self.steps>=round(self.cfg.stable_seconds*CONTROL_HZ)


def set_views(env):
    """Use the conveyor front pose and retain the rigid wrist mount."""
    import torch
    from piper_bench.tasks.conveyor.core import set_views as base_views
    base_views(env)
    cal=camera_metadata()
    env.scene['cam0'].set_world_poses_from_view(
        torch.tensor([cal['cam0_eye_world_m']],dtype=torch.float32,device=env.device),
        torch.tensor([cal['cam0_target_world_m']],dtype=torch.float32,device=env.device))


def make_env_cfg(config=BowlConfig(),*,device='cpu',seed=42):
    import gymnasium as gym
    import isaaclab.sim as sim
    from piper_bench.sensors.lighting import spawn_light_without_wrist_shadows
    cfg=base_env_cfg(config,device=device,seed=seed)
    cfg.scene.light.spawn.func=spawn_light_without_wrist_shadows
    cfg.scene.cam0.spawn.focal_length=camera_metadata()['cam0_focal_length_cm']
    cfg.scene.belt.spawn.visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.36,.32,.25),roughness=.62,metallic=0.)
    cfg.scene.basket.spawn.func=spawn_brown_basket
    for i,color in enumerate(((.28,.38,.22),(.38,.26,.46)),1):
        obj=getattr(cfg.scene,f'cube_{i}')
        obj.spawn.func=spawn_cup
        obj.spawn.cup_config=config.metadata()
        obj.spawn.visual_material=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.38,metallic=0.)
        obj.spawn.mass_props.mass=.035
        obj.spawn.rigid_props.solver_position_iteration_count=24
        obj.spawn.rigid_props.solver_velocity_iteration_count=8
    from piper_bench.scenes.task_assets import configure_task_assets
    configure_task_assets(cfg.scene, 'turntable_cups', seed)
    if TASK not in gym.registry:
        gym.register(TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
                     kwargs={'env_cfg_entry_point':'piper_bench.tasks.turntable_cups.scene:make_env_cfg'})
    return cfg


class CupRuntime(TurntableRuntime):
    def __init__(self,env,cfg):
        super().__init__(env,cfg)
        from isaaclab.sim.utils.stage import get_current_stage
        from piper_bench.tasks.turntable_cups.geometry import add_stone_surface
        add_stone_surface(get_current_stage(),str(env.scene.env_prim_paths[0])+'/TurntableMarks',cfg.disk_radius)
    def reset(self,seed):
        super().reset(seed)
        self.tracker=NestingTracker(self.cfg)
        self.off_counts=[0,0]
        from piper_bench.scenes.task_assets import reset_task_assets
        self.tabletop_appearance=reset_task_assets(self.env, 'turntable_cups', seed)
    def after_step(self):
        self.control_step+=1
        poses,velocities,tcp=self.state()
        success=self.tracker.update(poses,velocities,tcp,self.active)
        reason=None
        for i,p in enumerate(poses):
            outside=np.linalg.norm(p[:2]-self.cfg.center)>self.cfg.disk_radius+self.cfg.cup_top_radius
            low=p[2]<self.cfg.belt_top+self.cfg.cup_height/2-.01
            failed=self.active[i] and outside and low and np.linalg.norm(p[:3]-tcp)>.10 and not cup_in_basket(p,self.cfg)
            self.off_counts[i]=self.off_counts[i]+1 if failed else 0
            if p[2]<-.04 or self.off_counts[i]>=6:reason=f'cup_{i+1}_fell_or_off_disk'
        if self.time>=self.cfg.max_seconds and not success:reason='timeout'
        return success,reason
    def metadata(self):
        data=super().metadata()
        data.update(task=TASK,prompt=PROMPT,object_shape='open_tapered_cup',
            wall_convex_parts=WALL_SEGMENTS,cup_order=['green','purple'],
            failure_protocol='cup_aware_low_away_six_ticks',success_protocol='nested_upright_stable_1s',
            appearance=self.tabletop_appearance)
        if self.cfg.cup_height < 2*self.cfg.cup_top_radius:
            data.update(object_shape='open_tapered_bowl',
                        success_protocol=BOWL_SUCCESS_VERSION)
        return data
