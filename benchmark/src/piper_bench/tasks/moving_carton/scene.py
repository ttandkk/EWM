"""Static-source pick/place into a physically moving basket, isolated from conveyor tasks."""
from dataclasses import dataclass
import math
import numpy as np
from piper_bench.tasks.conveyor.core import ConveyorConfig
from piper_bench.tasks.conveyor.core import ConveyorRuntime
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import object_in_basket
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.conveyor.core import make_env_cfg as conveyor_env_cfg
from piper_bench.tasks.conveyor.core import camera_metadata as conveyor_camera_metadata

TASK = 'Isaac-Moving-Carton-Stack-Piper-IK-Rel-v0'
PROMPT = 'Place the red gift box in the moving cardboard box, then stack the small taped cardboard parcel on top.'

# Evaluation-only revision; keep historical collection defaults reproducible.
STACK_SUCCESS_V2 = 'moving_carton_stack_v2_xy12mm_angular0p3_stable1s'
STACK_SUCCESS_V2_OVERRIDES = dict(stack_xy_tolerance=.012, angular_threshold=.3)
STACK_SUCCESS_V3 = 'moving_carton_stack_v3_xy15mm_overhang2mm_angular0p3_stable1s'
STACK_SUCCESS_V3_OVERRIDES = dict(stack_xy_tolerance=.015, angular_threshold=.3,
                                  orange_overhang_tolerance=.002)


@dataclass(frozen=True)
class MovingBasketConfig(ConveyorConfig):
    # Preserve this task's default independently of conveyor grasp randomization.
    yaw_jitter: float = math.radians(10)
    object_size: float = .03
    belt_width: float = .12
    belt_length: float = .32
    basket_x: float = .33
    basket_y: float = 0.
    basket_z: float = .03
    basket_length: float = .065
    basket_width: float = .065
    basket_height: float = .035
    basket_taper: float = 0.
    amplitude: float = .06
    peak_speed: float = .05
    source_x: float = .18
    source_y: float = .19
    source_spacing: float = .12
    position_jitter: float = .004
    stack_xy_tolerance: float = .006
    stack_z_tolerance: float = .004
    # None preserves historical collection and v2 evaluation contracts.
    orange_overhang_tolerance: float | None = None
    stack_success_mode: str = 'legacy'
    controlled_placement: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.stack_success_mode not in ('legacy', 'lowwall_v4'):
            raise ValueError('unknown stack success mode')
        if self.stack_success_mode == 'lowwall_v4' and not 0 < self.basket_height <= self.object_size-.005+1e-9:
            raise ValueError('low-wall stack requires the rim at least 5 mm below the blue top')
        if self.orange_overhang_tolerance is not None and (
                not math.isfinite(self.orange_overhang_tolerance) or self.orange_overhang_tolerance < 0):
            raise ValueError('orange overhang tolerance must be finite and nonnegative')
        values = (self.object_size, self.basket_x, self.basket_y, self.basket_z,
                  self.amplitude, self.peak_speed, self.source_x, self.source_y,
                  self.source_spacing, self.belt_length, self.belt_width)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('moving-basket geometry and motion must be finite')
        if not (0 < self.amplitude <= .10 and 0 < self.peak_speed <= .15 and self.basket_z >= 0):
            raise ValueError('invalid motion amplitude, peak speed or basket height')
        if self.object_size <= 0:
            raise ValueError('invalid source geometry')
        margin = self.object_size/math.sqrt(2)+self.position_jitter+.005
        if self.source_x+margin+.02 >= self.basket_x-self.belt_width/2:
            raise ValueError('tabletop source must clear the conveyor edge')
        if self.source_x-margin < .10 or abs(self.source_y)+self.source_spacing/2+margin > .44:
            raise ValueError('tabletop source must clear the base and table edge')
        if self.source_spacing-2*self.position_jitter <= self.object_size*math.sqrt(2)+.01:
            raise ValueError('source cubes must remain separated')
        if not math.isclose(self.basket_z,self.belt_top,abs_tol=1e-8):
            raise ValueError('basket bottom must sit on the belt surface')
        outer_x = self.basket_length/2+self.basket_taper+self.wall_thickness
        outer_y = self.basket_width/2+self.basket_taper+self.wall_thickness
        if outer_x+.005 >= self.belt_width/2 or self.amplitude+outer_y+.015 >= self.belt_length/2:
            raise ValueError('basket travel must fit fully on the short conveyor')
        if self.basket_x-self.belt_width/2 <= .10:
            raise ValueError('short conveyor too close to the robot base')

    @property
    def omega(self):
        return self.peak_speed/self.amplitude

    @property
    def period(self):
        return 2*math.pi/self.omega

    def conveyor_config(self):
        return ConveyorConfig(**{k:getattr(self,k) for k in ConveyorConfig.__dataclass_fields__})

    def metadata(self):
        value = super().metadata()
        if self.stack_success_mode == 'legacy' and not self.controlled_placement:
            # Old plans compare complete serialized episode specs on replay.
            value.pop('stack_success_mode')
            value.pop('controlled_placement')
        return value

    def supply_positions(self, seed):
        rng = np.random.default_rng(seed)
        x = self.source_x+rng.uniform(-self.position_jitter,self.position_jitter,2)
        y = self.source_y+np.array([-.5,.5])*self.source_spacing+rng.uniform(-self.position_jitter,self.position_jitter,2)
        return x, y, rng.uniform(-self.yaw_jitter,self.yaw_jitter,2)


@dataclass(frozen=True)
class HalfWallConfig(MovingBasketConfig):
    # Half of the 25 mm walls in the current low-wall-v4 dataset.
    basket_height: float = .0125
    stack_success_mode: str | None = None
    controlled_placement: bool | None = None

    def __post_init__(self):
        # Historical 35 mm configs omitted these fields. Loading an explicit
        # old height must not silently enable the current low-wall predicate.
        if self.stack_success_mode is None:
            object.__setattr__(self, 'stack_success_mode',
                               'lowwall_v4' if self.basket_height == .0125 else 'legacy')
        if self.controlled_placement is None:
            object.__setattr__(self, 'controlled_placement', self.stack_success_mode == 'lowwall_v4')
        super().__post_init__()


def basket_motion(cfg, time, phase):
    angle = phase+cfg.omega*time
    position = np.array([cfg.basket_x, cfg.basket_y+cfg.amplitude*math.sin(angle), cfg.basket_z])
    velocity = np.array([0., cfg.peak_speed*math.cos(angle), 0.])
    return position, velocity


def basket_local_poses(poses, basket_pose, cfg):
    """Transform translation-only basket coordinates to the existing containment convention."""
    result = np.array(poses, dtype=float, copy=True)
    result[..., :3] -= np.asarray(basket_pose[:3])
    result[..., 0] += cfg.basket_x
    result[..., 1] += cfg.basket_y
    return result


def orange_overhang(orange_pose, basket_pose, cfg):
    """Maximum full-cube projection beyond the inner opening, in basket coordinates.

    Unlike object_in_basket, include the part above the rim. The absolute
    rotation matrix gives the exact corner extents for an oriented cube.
    """
    basket_rotation = rotation_xyzw(basket_pose[3:7])
    center = basket_rotation.T @ (np.asarray(orange_pose[:3])-np.asarray(basket_pose[:3]))
    rotation = basket_rotation.T @ rotation_xyzw(orange_pose[3:7])
    half_extents = np.abs(rotation).sum(axis=1)*cfg.object_size/2
    opening = np.array([cfg.basket_length, cfg.basket_width])/2+cfg.basket_taper
    return np.maximum(np.abs(center[:2])+half_extents[:2]-opening, 0.)


def moving_success_diagnostics(poses, velocities, tcp, active, basket_pose, basket_velocity, cfg):
    """Explain the exact predicate using simulator truth, never policy inputs."""
    if cfg.stack_success_mode == 'lowwall_v4':
        from piper_bench.tasks.moving_carton.lowwall import support_metrics
        return support_metrics(poses, basket_pose, tcp, active, cfg)
    local = basket_local_poses(poses, basket_pose, cfg)
    relative_velocity = np.array(velocities, dtype=float, copy=True)
    relative_velocity[:, :3] -= np.asarray(basket_velocity[:3])
    geometry = stack_geometry(poses, basket_pose, cfg)
    linear = np.linalg.norm(relative_velocity[:, :3], axis=1)
    angular = np.linalg.norm(relative_velocity[:, 3:], axis=1)
    retreat = np.linalg.norm(np.asarray(poses)[:, :3]-np.asarray(tcp), axis=1)
    contained = [bool(object_in_basket(p, cfg)) for p in local]
    overhang = orange_overhang(poses[1], basket_pose, cfg)
    checks = dict(active=bool(all(active)), containment=all(contained),
                  xy=geometry['xy_error_m'] < cfg.stack_xy_tolerance,
                  height=abs(geometry['height_gap_m']-cfg.object_size) < cfg.stack_z_tolerance,
                  lower_seated=geometry['lower_floor_error_m'] < cfg.stack_z_tolerance,
                  flat=geometry['flat'], linear=bool(np.all(linear < cfg.linear_threshold)),
                  angular=bool(np.all(angular < cfg.angular_threshold)),
                  retreat=bool(np.all(retreat >= cfg.retreat_distance)))
    if cfg.orange_overhang_tolerance is not None:
        checks['orange_boundary'] = bool(np.all(overhang <= cfg.orange_overhang_tolerance+1e-9))
    return dict(valid=all(checks.values()), checks=checks,
                failed_conditions=[key for key, passed in checks.items() if not passed],
                **{k:v for k,v in geometry.items() if k != 'valid'},
                contained=contained, relative_linear_speed_m_s=linear.tolist(),
                angular_speed_rad_s=angular.tolist(), tcp_distance_m=retreat.tolist(),
                orange_overhang_m=float(np.max(overhang)), orange_overhang_by_axis_m=overhang.tolist(),
                thresholds=dict(xy_m=cfg.stack_xy_tolerance, height_error_m=cfg.stack_z_tolerance,
                                orange_overhang_m=cfg.orange_overhang_tolerance,
                                tilt_deg=10., linear_m_s=cfg.linear_threshold,
                                angular_rad_s=cfg.angular_threshold, retreat_m=cfg.retreat_distance,
                                stable_seconds=cfg.stable_seconds))


def moving_success_now(poses, velocities, tcp, active, basket_pose, basket_velocity, cfg):
    return moving_success_diagnostics(poses, velocities, tcp, active, basket_pose, basket_velocity, cfg)['valid']


def stack_geometry(poses,basket_pose,cfg):
    """Blue cube 0 supports orange cube 1; side-by-side or unsupported stacks fail."""
    poses=np.asarray(poses)
    xy_error=float(np.linalg.norm(poses[1,:2]-poses[0,:2]))
    height_gap=float(poses[1,2]-poses[0,2])
    floor_error=float(abs(poses[0,2]-(basket_pose[2]+cfg.wall_thickness+cfg.object_size/2)))
    vertical = [float(np.max(np.abs(rotation_xyzw(p[3:7])[2]))) for p in poses]
    flat = all(v >= math.cos(math.radians(10.)) for v in vertical)
    tilt = [math.degrees(math.acos(np.clip(v, 0., 1.))) for v in vertical]
    valid=(xy_error<cfg.stack_xy_tolerance and abs(height_gap-cfg.object_size)<cfg.stack_z_tolerance
           and floor_error<cfg.stack_z_tolerance and flat)
    return dict(valid=bool(valid),xy_error_m=xy_error,height_gap_m=height_gap,
                lower_floor_error_m=floor_error,flat=bool(flat),tilt_degrees=tilt,order='object_1_bottom_object_2_top')


class MovingSuccessTracker:
    def __init__(self, cfg):
        from collections import deque
        self.cfg, self.steps = cfg, 0
        self.last_diagnostics = None
        self.pose_window = deque(maxlen=round(cfg.stable_seconds*CONTROL_HZ)+1)

    def update(self, poses, velocities, tcp, active, basket_pose, basket_velocity):
        self.last_diagnostics = moving_success_diagnostics(
            poses, velocities, tcp, active, basket_pose, basket_velocity, self.cfg)
        if self.cfg.stack_success_mode == 'lowwall_v4':
            from piper_bench.tasks.moving_carton.lowwall import relative_corners
            from piper_bench.tasks.moving_carton.lowwall import stable_window
            if self.last_diagnostics['valid']:
                self.pose_window.append(relative_corners(poses, basket_pose, self.cfg))
            else:
                self.pose_window.clear()
            stable = stable_window(self.pose_window, self.cfg)
            self.steps = len(self.pose_window) if stable else 0
            self.last_diagnostics.update(stable_steps=self.steps,
                observed_valid_samples=len(self.pose_window), pose_window_stable=stable,
                required_stable_steps=self.pose_window.maxlen)
            return stable
        self.steps = self.steps+1 if self.last_diagnostics['valid'] else 0
        self.last_diagnostics.update(stable_steps=self.steps,
                                     required_stable_steps=round(self.cfg.stable_seconds*CONTROL_HZ))
        return self.steps >= round(self.cfg.stable_seconds*CONTROL_HZ)


def camera_metadata():
    value = conveyor_camera_metadata()
    value.update(calibration_id='piper_moving_carton_conveyor_front_v3')
    return value


def set_views(env):
    """Use the conveyor front pose and retain the rigid wrist mount."""
    import torch
    from piper_bench.tasks.conveyor.core import set_views as base_views
    base_views(env)
    cal=camera_metadata()
    env.scene['cam0'].set_world_poses_from_view(
        torch.tensor([cal['cam0_eye_world_m']],dtype=torch.float32,device=env.device),
        torch.tensor([cal['cam0_target_world_m']],dtype=torch.float32,device=env.device))


def make_env_cfg(config=HalfWallConfig(), *, device='cpu', seed=42):
    import gymnasium as gym
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.assets import RigidObjectCfg
    from pxr import UsdPhysics
    from pxr import PhysxSchema
    from pxr import Gf
    from piper_bench.tasks.moving_carton.geometry import spawn_carton
    from piper_bench.tasks.moving_carton.geometry import add_conveyor_frames
    from piper_bench.tasks.moving_carton.control import MovingBasketIKAction
    from piper_bench.sensors.lighting import spawn_light_without_wrist_shadows

    cfg = conveyor_env_cfg(config.conveyor_config(), device=device, seed=seed)
    cfg.scene.light.spawn.func = spawn_light_without_wrist_shadows
    cfg.actions.arm_action.class_type = MovingBasketIKAction
    cfg.scene.cube_1.spawn.visual_material = sim.PreviewSurfaceCfg(diffuse_color=(.12,.48,.72),roughness=.75)
    cfg.scene.cube_2.spawn.visual_material = sim.PreviewSurfaceCfg(diffuse_color=(.90,.34,.13),roughness=.75)
    def spawn_conveyor(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
        prim = sim.spawn_cuboid(prim_path,spawn_cfg,translation,orientation,**kwargs)
        surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
        surface.CreateSurfaceVelocityEnabledAttr(True)
        surface.CreateSurfaceVelocityLocalSpaceAttr(False)
        surface.CreateSurfaceVelocityAttr(Gf.Vec3f(0.,0.,0.))
        return prim

    # Reuse the original conveyor geometry, shortened to 32 cm, along the same Y axis.
    cfg.scene.belt = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Belt',
        init_state=RigidObjectCfg.InitialStateCfg(pos=(config.basket_x,config.basket_y,config.belt_top/2)),
        spawn=sim.CuboidCfg(func=spawn_conveyor,size=(config.belt_width,config.belt_length,config.belt_top),
            rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.9,dynamic_friction=.8),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.055,.060,.067),roughness=.85,metallic=0.)))
    for index,sign in enumerate((-1,1)):
        setattr(cfg.scene,f'belt_roller_{index}',AssetBaseCfg(
            prim_path=f'{{ENV_REGEX_NS}}/BeltRoller_{index}',
            init_state=AssetBaseCfg.InitialStateCfg(pos=(config.basket_x,
                config.basket_y+sign*config.belt_length/2,config.belt_top/2)),
            spawn=sim.CylinderCfg(radius=config.belt_top/2+.001,height=config.belt_width+.014,
                axis='X',visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.035,.035,.035)))))

    add_conveyor_frames(cfg.scene,config)

    def spawn_moving_basket(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
        prim = spawn_carton(prim_path,spawn_cfg,translation,orientation,**kwargs)
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateRigidBodyEnabledAttr(True)
        body.CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.)
        material_path = str(prim.GetPath())+'/contact_material'
        material = sim.RigidBodyMaterialCfg(static_friction=.9,dynamic_friction=.8)
        material.func(material_path,material)
        for child in prim.GetChildren():
            if child.HasAPI(UsdPhysics.CollisionAPI):
                sim.bind_physics_material(str(child.GetPath()),material_path)
        return prim

    spawn = sim.CuboidCfg(func=spawn_moving_basket,size=(1.,1.,1.))
    spawn.basket_config = config.conveyor_config().metadata()
    cfg.scene.basket = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Basket',spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(config.basket_x,config.basket_y,config.basket_z)))
    from piper_bench.scenes.task_assets import configure_task_assets
    configure_task_assets(cfg.scene, 'moving_carton', seed)
    if TASK not in gym.registry:
        gym.register(TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
                     kwargs={'env_cfg_entry_point':'piper_bench.tasks.moving_carton.scene:make_env_cfg'})
    return cfg


class MovingBasketRuntime(ConveyorRuntime):
    def __init__(self, env, cfg):
        import omni.physics.tensors as tensors
        import torch
        self.basket = env.scene['basket']
        # A kinematic-only Torch view avoids mixing dynamic cubes into target writes.
        self.physics_view = tensors.create_simulation_view('torch', stage_id=env.scene.stage_id)
        self.physics_view.set_subspace_roots('/')
        self.basket_view = self.physics_view.create_rigid_body_view(str(env.scene.env_prim_paths[0])+'/Basket')
        if self.basket_view.count != 1:
            raise RuntimeError('moving basket probe requires exactly one kinematic basket')
        self.target_pose = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],dtype=torch.float32,device=env.device)
        self.target_ids = torch.tensor([0],dtype=torch.int32,device=env.device)
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import PhysxSchema
        belt_prim = get_current_stage().GetPrimAtPath(str(env.scene.env_prim_paths[0])+'/Belt')
        self.surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI(belt_prim).GetSurfaceVelocityAttr()
        self.motion_enabled = False
        super().__init__(env,cfg)
        env._moving_basket_runtime = self

    def reset(self, seed):
        import torch
        super().reset(seed)
        self.phase = float(np.random.default_rng(seed+171).uniform(-math.pi,math.pi))
        self.physics_tick = 0
        self.tracker = MovingSuccessTracker(self.cfg)
        pos,_ = self.basket_target(0.)
        self.target_pose[0,:3] = torch.as_tensor(pos,dtype=torch.float32,device=self.env.device)
        self.basket.write_root_pose_to_sim(self.target_pose)
        self.basket_view.set_kinematic_targets(self.target_pose,self.target_ids)
        from pxr import Gf
        self.surface_velocity.Set(Gf.Vec3f(*self.basket_target(0.)[1]))
        self.motion_enabled = True
        from piper_bench.scenes.task_assets import reset_task_assets
        self.tabletop_appearance = reset_task_assets(self.env, 'moving_carton', seed)

    def supply(self):
        """Reset-only spawning directly on the original tabletop (z=0), beside the belt."""
        import torch
        spawned = []
        for i,obj in enumerate(self.objects):
            if self.active[i]:
                continue
            yaw = self.spawn_yaw[i]
            pose = [[self.spawn_x[i],self.spawn_y[i],self.cfg.object_size/2+.001,
                     0.,0.,np.sin(yaw/2),np.cos(yaw/2)]]
            obj.write_root_pose_to_sim(torch.tensor(pose,dtype=torch.float32,device=self.env.device))
            obj.write_root_velocity_to_sim(torch.zeros((1,6),device=self.env.device))
            self.active[i] = True
            spawned.append(i)
        return spawned

    def basket_target(self, time):
        return basket_motion(self.cfg,time,self.phase)

    def advance_basket(self):
        import torch
        if not self.motion_enabled:
            return
        self.physics_tick += 1
        pos,velocity = self.basket_target(self.physics_tick*self.env.physics_dt)
        from pxr import Gf
        self.surface_velocity.Set(Gf.Vec3f(*velocity))
        self.target_pose[0,:3] = torch.as_tensor(pos,dtype=torch.float32,device=self.env.device)
        self.basket_view.set_kinematic_targets(self.target_pose,self.target_ids)

    def basket_state(self):
        return (self.basket.data.root_pose_w[0].detach().cpu().numpy().copy(),
                self.basket.data.root_vel_w[0].detach().cpu().numpy().copy())

    def objects_in_basket(self, poses):
        basket_pose,_ = self.basket_state()
        return [object_in_basket(p,self.cfg) for p in basket_local_poses(poses,basket_pose,self.cfg)]

    def preload_contact_probe(self):
        """Reset-only placement inside the basket, solely for contact transport validation."""
        import torch
        base,_ = self.basket_target(0.)
        for i,obj in enumerate(self.objects):
            xyz = base+np.array([0.,0.,self.cfg.wall_thickness+(i+.5)*self.cfg.object_size+.001*(i+1)])
            pose = np.r_[xyz,0.,0.,0.,1.][None]
            obj.write_root_pose_to_sim(torch.as_tensor(pose,dtype=torch.float32,device=self.env.device))
            obj.write_root_velocity_to_sim(torch.zeros((1,6),device=self.env.device))
            self.active[i] = True

    def after_step(self):
        self.control_step += 1
        if self.physics_tick != self.control_step*self.env.cfg.decimation:
            raise RuntimeError('basket must advance at every physics substep')
        poses,velocities,tcp = self.state()
        basket_pose,basket_velocity = self.basket_state()
        success = self.tracker.update(poses,velocities,tcp,self.active,basket_pose,basket_velocity)
        reason = next((f'object_{i+1}_fell' for i,p in enumerate(poses) if self.active[i] and p[2]<-.04),None)
        if self.time>=self.cfg.max_seconds and not success:
            reason = 'timeout'
        return success,reason

    def metadata(self):
        data = super().metadata()
        data.update(task=TASK,prompt=PROMPT,basket_phase_rad=self.phase,basket_period_s=self.cfg.period,
                    basket_motion='prescribed_kinematic_motion_on_short_conveyor_at_physics_hz',
                    conveyor_motion='Y_reciprocation_surface_speed_matches_basket',
                    object_source='original_tabletop_z0_beside_conveyor',
                    basket_drive='prescribed_trajectory_not_friction_driven',
                    success_criterion=(STACK_SUCCESS_V3 if all(getattr(self.cfg, k) == v
                        for k, v in STACK_SUCCESS_V3_OVERRIDES.items()) else
                        STACK_SUCCESS_V2 if all(getattr(self.cfg, k) == v
                        for k, v in STACK_SUCCESS_V2_OVERRIDES.items()) else
                        'moving_carton_stack_blue_bottom_orange_top_stable_1s_tcp_retreat'),
                    container_style='open_cardboard_box',cube_colors=['red_and_gold','kraft_brown'],
                    cube_assets=['gift_box','cardboard_box'],appearance=self.tabletop_appearance,
                    object_pose_override='spawn_and_reset_only')
        if self.cfg.stack_success_mode == 'lowwall_v4':
            from piper_bench.tasks.moving_carton.lowwall import VERSION
            data['success_criterion'] = VERSION
        return data
