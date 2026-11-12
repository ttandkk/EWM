"""Piper places a tabletop test tube into a continuously rotating orange rack."""
import math
import numpy as np
from piper_bench.tasks.conveyor.core import ConveyorConfig
from piper_bench.tasks.conveyor.core import ConveyorRuntime
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import CAMERAS
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.conveyor.core import make_env_cfg as conveyor_env_cfg
from piper_bench.tasks.conveyor.core import camera_metadata as original_camera

TASK='Isaac-Rotating-Tube-Insertion-Piper-IK-Rel-v0'
SUCCESS_VERSION='piper_rotating_tube_bore_seated_v2'
PROMPT='Pick up the test tube and insert it upright into the rotating orange rack.'
INITIAL_ARM_JOINTS=(0.,0.,0.,0.,0.,0.)


from piper_bench.tasks.rotating_tube.config import TubeConfig


def hole_motion(cfg,time,phase=0.,index=0):
    angle=phase+cfg.omega*time+index*2*math.pi/cfg.holes
    radial=cfg.hole_orbit*np.array([math.cos(angle),math.sin(angle)])
    pos=np.r_[np.array([cfg.rack_x,cfg.rack_y])+radial,cfg.plate_top]
    vel=np.array([-cfg.omega*radial[1],cfg.omega*radial[0],0.])
    return pos,vel


def insertion_metrics(pose,velocity,tcp,cfg,hole,gripper_open):
    axis=rotation_xyzw(pose[3:])[:,2]
    tilt=math.degrees(math.acos(np.clip(axis[2],-1,1)))
    bottom=pose[:3]-axis*(cfg.tube_length/2-cfg.tube_radius)
    bottom[2]-=cfg.tube_radius  # lowest point of the round bottom, including tilt
    floor_error=bottom[2]-cfg.base_thickness
    # Axis intersection with the top guide plate; tube center alone can accept a tilted rim perch.
    if axis[2]>.1:
        planes=np.array([cfg.plate_top-cfg.plate_thickness,cfg.plate_top])
        centers=pose[:2]+(planes[:,None]-pose[2])*axis[None,:2]/axis[2]
        offsets=np.linalg.norm(centers-hole[0][None,:2],axis=1)
        xy=float(offsets[-1])
        # Bound the tilted cylinder's elliptical cross-section at both faces.
        # Legal contact with the hole wall must not be rejected as misalignment.
        bore_extent=float(offsets.max()+cfg.tube_radius/axis[2])
    else:xy,bore_extent=np.inf,np.inf
    relative_speed=float(np.linalg.norm(velocity[:3]-hole[1]))
    # A gravity-released tube may lean against the bore wall. Accept its
    # measured cross-section at both plate faces instead of an angle cutoff.
    valid=(bore_extent<=cfg.hole_radius+.00005 and abs(floor_error)<.0005
           and cfg.plate_top-bottom[2]>.035 and relative_speed<.02 and gripper_open
           and np.linalg.norm(tcp-pose[:3])>.10)
    return dict(valid=bool(valid),axis_xy_error_m=xy if math.isfinite(xy) else None,tilt_deg=tilt,
                bottom_z_m=float(bottom[2]),floor_error_m=float(floor_error),
                insertion_depth_m=float(cfg.plate_top-bottom[2]),relative_speed_m_s=relative_speed,
                bore_outer_radius_m=bore_extent if math.isfinite(bore_extent) else None)


class InsertionTracker:
    def __init__(self,cfg):self.cfg,self.steps=cfg,0
    def update(self,pose,vel,tcp,hole,gripper_open):
        metrics=insertion_metrics(pose,vel,tcp,self.cfg,hole,gripper_open)
        self.steps=self.steps+1 if metrics['valid'] else 0
        return self.steps>=round(self.cfg.stable_seconds*CONTROL_HZ)


def tabletop_camera_metadata():
    """Preserve the established view used by static and tennis tasks."""
    value=original_camera()
    value.update(calibration_id='piper_rotating_tube_front_v2_wrist_v1',
                 cam0_eye_world_m=[1.12,0.,.62],cam0_target_world_m=[.34,0.,.03])
    return value


def set_tabletop_views(env):
    import torch
    cal=tabletop_camera_metadata()
    def tensor(value):
        return torch.tensor([value],dtype=torch.float32,device=env.device)
    env.scene['cam0'].set_world_poses_from_view(
        tensor(cal['cam0_eye_world_m']),tensor(cal['cam0_target_world_m']))
    from piper_bench.robots.piper.cameras import set_wrist_view
    set_wrist_view(env)


def camera_metadata():
    value=original_camera()
    value['calibration_id']='piper_rotating_tube_conveyor_front_v3'
    return value


def set_views(env):
    from piper_bench.tasks.conveyor.core import set_views as conveyor_views
    conveyor_views(env)


def reset_scene(env,env_ids):
    robot=env.scene['robot']
    pos=robot.data.default_joint_pos.torch[env_ids].clone();vel=robot.data.default_joint_vel.torch[env_ids].clone()
    robot.write_joint_position_to_sim_index(position=pos,env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=vel,env_ids=env_ids)
    robot.set_joint_position_target_index(target=pos,env_ids=env_ids)
    robot.set_joint_velocity_target_index(target=vel,env_ids=env_ids)
    obj=env.scene['cube_1'];pose=obj.data.default_root_pose.torch[env_ids].clone()
    pose[:,:3]+=env.scene.env_origins[env_ids]
    obj.write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
    obj.write_root_velocity_to_sim_index(root_velocity=obj.data.default_root_vel.torch[env_ids].clone(),env_ids=env_ids)


def make_env_cfg(config=TubeConfig(),*,device='cpu',seed=42):
    import gymnasium as gym
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    from piper_bench.tasks.rotating_tube.geometry import spawn_rack
    from piper_bench.tasks.rotating_tube.geometry import spawn_tube
    from piper_bench.tasks.rotating_tube.control import RotatingRackIKAction
    from piper_bench.sensors.lighting import spawn_light_without_wrist_shadows
    cfg=conveyor_env_cfg(ConveyorConfig(max_seconds=config.max_seconds),device=device,seed=seed)
    cfg.scene.light.spawn.func=spawn_light_without_wrist_shadows
    cfg.actions.arm_action.class_type=RotatingRackIKAction
    # Pinch the narrower tube body with a small preload.
    # Excess squeezing leaves a lateral/angular impulse when the fingers open.
    half_gap=config.tube_radius-.00375
    cfg.actions.gripper_action.close_command_expr={
        'gripper_joint1':half_gap,'gripper_joint2':-half_gap}
    # Inherit the original conveyor reset: six zero arm joints, fingers open.
    cfg.events.reset_all.func=reset_scene
    for name in ('belt','belt_roller_0','belt_roller_1','basket','cube_2'):
        setattr(cfg.scene,name,None)
    for name in CAMERAS:getattr(cfg.scene,name).spawn.clipping_range=(.03,100.)
    cfg.scene.cam_wrist.spawn.focal_length=camera_metadata()['wrist_focal_length_cm']
    tube=cfg.scene.cube_1
    tube.spawn.func=spawn_tube;tube.spawn.tube_config=config.metadata()
    tube.spawn.mass_props.mass=.025
    tube.spawn.rigid_props.solver_position_iteration_count=32
    tube.spawn.rigid_props.solver_velocity_iteration_count=8
    tube.spawn.collision_props.contact_offset=.0003
    tube.spawn.physics_material=sim.RigidBodyMaterialCfg(static_friction=1.2,dynamic_friction=1.0,restitution=0.)
    spawn=sim.CuboidCfg(func=spawn_rack,size=(1.,1.,1.),
        rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),
        collision_props=sim.CollisionPropertiesCfg(contact_offset=.0003,rest_offset=0.),
        mass_props=sim.MassPropertiesCfg(mass=1.),
        physics_material=sim.RigidBodyMaterialCfg(static_friction=.9,dynamic_friction=.8,restitution=0.),
        visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.48,.30,.18),roughness=.38))
    spawn.tube_config=config.metadata()
    cfg.scene.rack=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/RotatingRack',spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(config.rack_x,config.rack_y,0.)))
    from piper_bench.scenes.task_assets import configure_task_assets
    configure_task_assets(cfg.scene, 'rotating_tube', seed)
    if TASK not in gym.registry:
        gym.register(TASK,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
                     kwargs={'env_cfg_entry_point':'piper_bench.tasks.rotating_tube.scene:make_env_cfg'})
    return cfg


class TubeRuntime:
    def __init__(self,env,cfg):
        import torch
        import omni.physics.tensors as tensors
        self.env,self.cfg=env,cfg;self.objects=[env.scene['cube_1']]
        self.rack=env.scene['rack'];self.motion_enabled=False
        self.physics_view=tensors.create_simulation_view('torch',stage_id=env.scene.stage_id)
        self.physics_view.set_subspace_roots('/')
        self.rack_view=self.physics_view.create_rigid_body_view(str(env.scene.env_prim_paths[0])+'/RotatingRack')
        assert self.rack_view.count==1
        self.target_pose=torch.tensor([[cfg.rack_x,cfg.rack_y,0.,0.,0.,0.,1.]],dtype=torch.float32,device=env.device)
        self.target_ids=torch.tensor([0],dtype=torch.int32,device=env.device)
        env._rotating_tube_runtime=self
        self.reset(42)

    @property
    def time(self):return self.control_step/CONTROL_HZ
    state=ConveyorRuntime.state

    def reset(self,seed):
        self.seed=seed;self.control_step=0;self.physics_tick=0
        self.phase=float(np.random.default_rng(seed).uniform(-math.pi,math.pi))
        self.target_index=0;self.tracker=InsertionTracker(self.cfg);self.active=[False]
        self.target_pose[0,5]=math.sin(self.phase/2);self.target_pose[0,6]=math.cos(self.phase/2)
        self.rack.write_root_pose_to_sim(self.target_pose)
        self.rack_view.set_kinematic_targets(self.target_pose,self.target_ids)
        self.motion_enabled=True
        from piper_bench.scenes.task_assets import reset_task_assets
        self.tabletop_appearance=reset_task_assets(self.env, 'rotating_tube', seed)

    def supply(self):
        import torch
        if self.active[0]:return []
        c=self.cfg;rng=np.random.default_rng(self.seed+91)
        x,y=c.tube_x+rng.uniform(-.003,.003),c.tube_y+rng.uniform(-.003,.003)
        self.tube_yaw=math.atan2(y,x)
        from piper_bench.tasks.rotating_tube.oracle import yaw_pitch_quaternion
        q=yaw_pitch_quaternion(self.tube_yaw,math.pi/2)
        pose=np.r_[x,y,c.cap_radius+.001,q][None]
        self.objects[0].write_root_pose_to_sim(torch.as_tensor(pose,dtype=torch.float32,device=self.env.device))
        self.objects[0].write_root_velocity_to_sim(torch.zeros((1,6),device=self.env.device))
        self.active[0]=True
        return [0]

    def advance_rack(self):
        if not self.motion_enabled:return
        self.physics_tick+=1
        angle=self.phase+self.cfg.omega*self.physics_tick*self.env.physics_dt
        self.target_pose[0,5]=math.sin(angle/2);self.target_pose[0,6]=math.cos(angle/2)
        self.rack_view.set_kinematic_targets(self.target_pose,self.target_ids)

    def hole_target(self,lead=0.):return hole_motion(self.cfg,self.time+lead,self.phase,self.target_index)

    def preload_contact_probe(self):
        import torch
        hole,_=self.hole_target();hole[2]=self.cfg.plate_top+.002+self.cfg.tube_length/2
        pose=np.r_[hole,0.,0.,0.,1.][None]
        self.objects[0].write_root_pose_to_sim(torch.as_tensor(pose,dtype=torch.float32,device=self.env.device))
        velocity=np.r_[self.hole_target()[1],0.,0.,self.cfg.omega][None]
        self.objects[0].write_root_velocity_to_sim(torch.as_tensor(velocity,dtype=torch.float32,device=self.env.device))

    def after_step(self,gripper_open=True):
        self.control_step+=1
        assert self.physics_tick==self.control_step*self.env.cfg.decimation
        poses,vel,tcp=self.state()
        holes=[hole_motion(self.cfg,self.time,self.phase,i) for i in range(self.cfg.holes)]
        self.accepted_index=int(np.argmin([np.linalg.norm(poses[0,:2]-h[0][:2]) for h in holes]))
        success=self.tracker.update(poses[0],vel[0],tcp,holes[self.accepted_index],gripper_open)
        reason='tube_fell' if poses[0,2]<-.04 else None
        if self.time>=self.cfg.max_seconds and not success:reason='timeout'
        return success,reason

    def metadata(self):
        return dict(task=TASK,prompt=PROMPT,config=self.cfg.metadata(),seed=self.seed,
                    rack_phase_rad=self.phase,target_index=self.target_index,
                    motion='continuous kinematic rotation at 120 Hz; tube fully dynamic',
                    tube_spawn='horizontal on original tabletop; no support fixture',
                    no_grasp_attachments=True,clock='fixed simulation time',
                    appearance=self.tabletop_appearance)
