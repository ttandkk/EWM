"""Shared real-physics tennis scene; no constraints attach ball or box to Piper."""
import numpy as np
from piper_bench.tasks.conveyor.core import ConveyorRuntime
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.common import table_env_cfg
from piper_bench.tasks.tennis.motion import launch_state
from piper_bench.tasks.tennis.motion import launch_time
from piper_bench.tasks.tennis.metrics import CatchTracker
from piper_bench.tasks.tennis.metrics import gripper_metrics
from piper_bench.tasks.tennis.metrics import container_metrics
from piper_bench.scenes.ball_props import MODE_SKINS
from piper_bench.scenes.ball_props import appearance_id

MODE_TASKS = {'roll': 'tennis_rolling', 'fall': 'tennis_falling',
              'lob': 'tennis_parabolic', 'container': 'tennis_container'}

TASK_INFO = {
    'roll': ('Isaac-Tennis-Rolling-Intercept-Piper-IK-Rel-v0', 'Intercept the rolling tennis ball, grasp it and lift it off the table.'),
    'fall': ('Isaac-Tennis-Free-Fall-Catch-Piper-IK-Rel-v0', 'Catch the vertically falling soccer ball with the gripper and hold it aloft.'),
    'lob': ('Isaac-Tennis-Parabolic-Catch-Piper-IK-Rel-v0', 'Catch the thrown basketball in flight with the gripper and hold it aloft.'),
    'container': ('Isaac-Tennis-Container-Catch-Piper-IK-Rel-v0', 'Hold the blue container and catch the thrown baseball without letting it bounce out.'),
}


def reset_tennis_scene(env, env_ids):
    from piper_bench.tasks.rotating_tube.scene import reset_scene
    reset_scene(env, env_ids)
    # Reset the preheld body before the closed-gripper settling steps, including
    # repeated trials where the preceding episode may have dropped the box.
    if 'receiver' in env.scene.rigid_objects:
        receiver = env.scene['receiver']
        pose = receiver.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        receiver.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        receiver.write_root_velocity_to_sim_index(
            root_velocity=receiver.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids)


def make_tennis_env_cfg(config, *, device='cpu', seed=42):
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    from piper_bench.tasks.tennis.control import TennisIKAction
    cfg = table_env_cfg(config, device, seed)
    cfg.actions.arm_action.class_type = TennisIKAction
    if config.mode == 'lob':
        # Fixed observing TCP near (0.28, 0, 0.50), independent of every launch.
        # Pitch 1.0 keeps the incoming arc in the unchanged rigid wrist camera.
        ready = (0., .816200196, -.721389450, 0., -.578340599, 0.)
        cfg.scene.robot.init_state.joint_pos = {f'joint{i+1}': v for i, v in enumerate(ready)}
        cfg.scene.robot.init_state.joint_pos.update(gripper_joint1=.04, gripper_joint2=-.04)
        cfg.actions.gripper_action.open_command_expr = {'gripper_joint1': .04, 'gripper_joint2': -.04}
    if config.mode == 'roll':
        # Fixed forward-looking neutral pose: TCP (.30, 0, .12), pitch 2.0.
        # The rigid wrist sees the incoming lane; motion still waits for launch.
        ready = (0., 1.120653033, -.161312161, 0., -.442870725, 0.)
        cfg.scene.robot.init_state.joint_pos = {f'joint{i+1}': v for i, v in enumerate(ready)}
        cfg.scene.robot.init_state.joint_pos.update(gripper_joint1=.05, gripper_joint2=-.05)
    if config.mode == 'fall':
        # Fixed retracted observing pose: TCP (.26, 0, .48), pitch .8.
        # The rigid wrist sees the release area before the arm moves down to catch.
        ready = (0., .865840307, -.515552765, 0., -1.033817395, 0.)
        cfg.scene.robot.init_state.joint_pos = {f'joint{i+1}': v for i, v in enumerate(ready)}
        cfg.scene.robot.init_state.joint_pos.update(gripper_joint1=.05, gripper_joint2=-.05)
    if config.mode == 'container':
        cfg.events.reset_all.func = reset_tennis_scene
        # Resolve the handle pinch at 240 Hz while preserving 30 Hz actions.
        cfg.sim.dt = 1./240.
        cfg.decimation = 8
        cfg.sim.render_interval = 8
        # The smaller opening needs faster lateral tracking after ball release.
        # Keep the existing stiffness, effort and joint-speed limits.
        for name in ('piper_shoulder', 'piper_forearm'):
            cfg.scene.robot.actuators[name].damping *= .5
        # Clamp the horizontal pan handle firmly within the existing 22 N effort limit.
        cfg.scene.robot.actuators['gripper'].stiffness = 1600.
        cfg.scene.robot.actuators['gripper'].damping = 20.
        # Fixed retracted horizontal grip on the horizontal pan handle. The free container is supported
        # exclusively by finger contacts; no attachment or in-flight pose writes.
        ready = (0., .329087525, -.433331600, 0., .191510679, 0.)
        cfg.scene.robot.init_state.joint_pos = {f'joint{i+1}': v for i, v in enumerate(ready)}
        cfg.scene.robot.init_state.joint_pos.update(gripper_joint1=config.handle_width/2, gripper_joint2=-config.handle_width/2)
    from piper_bench.tasks.tennis.camera import camera_metadata_for_config
    cfg.scene.cam0.spawn.focal_length = camera_metadata_for_config(config)['cam0_focal_length_cm']
    if config.mode in ('lob', 'container'):
        from piper_bench.tasks.tennis.wrist import configure_wrist_camera
        configure_wrist_camera(cfg.scene.cam_wrist, config.mode)
    cfg.sim.gravity = (0., 0., -config.gravity)
    cfg.sim.physx.enable_ccd = True
    from piper_bench.tasks.tennis.geometry import spawn_ball
    cfg.scene.cube_1 = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/TennisBall',
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3. if config.mode == 'container' else -.7, 0., -.5)),
        spawn=sim.SphereCfg(func=spawn_ball, radius=config.radius,
            rigid_props=sim.RigidBodyPropertiesCfg(linear_damping=0., angular_damping=0.,
                solver_position_iteration_count=32, solver_velocity_iteration_count=8,
                max_depenetration_velocity=1.),
            mass_props=sim.MassPropertiesCfg(mass=config.mass),
            collision_props=sim.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
            physics_material=sim.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.6, restitution=config.restitution),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.65, .86, .045), roughness=.95)))
    if config.mode == 'container':
        from piper_bench.tasks.tennis.geometry import spawn_container
        spawn = sim.CuboidCfg(func=spawn_container, size=(1., 1., 1.),
            rigid_props=sim.RigidBodyPropertiesCfg(solver_position_iteration_count=32,
                solver_velocity_iteration_count=8), mass_props=sim.MassPropertiesCfg(mass=.12))
        spawn.tennis_config = config.metadata()
        cfg.scene.receiver = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/TennisReceiver', spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(config.container_x, config.container_y, config.container_hold_z),
                rot=(0., 0., np.sin(config.initial_container_yaw/2), np.cos(config.initial_container_yaw/2))))
    cfg.scene.cube_1.spawn.ball_skin = MODE_SKINS[config.mode]
    from piper_bench.scenes.task_assets import configure_task_assets
    configure_task_assets(cfg.scene, MODE_TASKS[config.mode], seed)
    return cfg


class TennisRuntime:
    """A scheduled one-shot launch, followed exclusively by passive ball physics."""
    # Shared preview must not spend flight time settling after supply().
    settle_steps = 0
    preview_steps = 480
    state = ConveyorRuntime.state

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.objects = [env.scene['cube_1']]
        self.initial_gripper_action = -1. if cfg.mode == 'container' else 1.
        self.reset(42)
        env._tennis_runtime = self

    @property
    def time(self):
        return self.control_step/CONTROL_HZ

    def write_initial(self, obj, position, velocity):
        import torch
        obj.write_root_pose_to_sim(torch.tensor([list(position)+[0., 0., 0., 1.]], dtype=torch.float32, device=self.env.device))
        obj.write_root_velocity_to_sim(torch.tensor(np.asarray(velocity)[None], dtype=torch.float32, device=self.env.device))

    def reset(self, seed):
        self.seed, self.control_step = seed, 0
        self.launched, self.table_hit = False, False
        self.active = [False]
        self.tracker = CatchTracker(self.cfg)
        self.initial_position, self.initial_velocity = launch_state(self.cfg, seed)
        self.launch_time_s = launch_time(self.cfg, seed)
        # Keep the parked ball outside the expanded container camera until launch.
        park_x = -3. if self.cfg.mode == 'container' else -.7
        self.write_initial(self.objects[0], [park_x, 0., -.5], np.zeros(6))
        self.previous_tcp = self.state()[2]
        self.tcp_velocity = np.zeros(3)
        self.gripper_closed = self.cfg.mode == 'container'
        from piper_bench.scenes.task_assets import reset_task_assets
        self.tabletop_appearance = reset_task_assets(self.env, MODE_TASKS[self.cfg.mode], seed)

    def settle_action(self):
        if self.cfg.mode != 'container':
            import torch
            return torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=self.env.device)
        from piper_bench.tasks.common import CartesianOracle
        from piper_bench.tasks.common import yaw_pitch_quaternion
        # A zero relative action follows gravity-induced deflection on each
        # settling step. Hold an absolute pose instead, preserving the front-handle grip.
        from piper_bench.tasks.conveyor.core import rotation_xyzw
        yaw = self.cfg.initial_container_yaw
        rotation = rotation_xyzw(yaw_pitch_quaternion(yaw, 0.))
        tool_q = np.asarray(self.cfg.initial_tool_quaternion)
        target = np.array([self.cfg.container_x, self.cfg.container_y, self.cfg.container_hold_z])
        target += rotation@np.asarray(self.cfg.grip_offset)+.0192*rotation_xyzw(tool_q)[:, 2]
        return CartesianOracle(self).command(target, tool_q, -1., translation_limit=.02)

    def supply(self):
        # Called before every control step. The launch schedule is independent
        # of oracle decisions and identical for preview and future policies.
        if self.launched or self.time+1e-8 < self.launch_time_s:
            return []
        self.write_initial(self.objects[0], self.initial_position, self.initial_velocity)
        self.launched = True
        self.active = [True]
        return [0]

    def before_step(self, action):
        self.supply()
        if self.cfg.mode == 'container':
            action[0, -1] = -1.
        self.gripper_closed = bool(float(action[0, -1]) < 0)

    def receiver_state(self):
        obj = self.env.scene['receiver']
        return (obj.data.root_pose_w[0].detach().cpu().numpy().copy(),
                obj.data.root_vel_w[0].detach().cpu().numpy().copy())

    def metrics(self, gripper_open=True):
        poses, velocities, tcp = self.state()
        joints = self.env.scene['robot'].data.joint_pos[0].detach().cpu().numpy()
        aperture = float(abs(joints[-2]-joints[-1]))
        q = self.env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().numpy()
        if self.cfg.mode == 'container':
            receiver, speed = self.receiver_state()
            result = container_metrics(poses[0], velocities[0], receiver, speed, tcp, aperture, self.gripper_closed, self.cfg)
            result.update(container_pose=receiver.tolist(), container_velocity=speed.tolist())
        else:
            result = gripper_metrics(poses[0], velocities[0], tcp, q, self.tcp_velocity, aperture, self.gripper_closed, self.cfg)
        if self.launched:
            from piper_bench.tasks.tennis.camera import camera_metadata_for_config
            from piper_bench.tasks.tennis.camera import wrist_ball_projection
            result['wrist_projection'] = wrist_ball_projection(
                poses[0, :3], tcp, q, self.cfg.radius,
                calibration=camera_metadata_for_config(self.cfg))
        result.update(stable_position_diameter_m=self.tracker.position_diameter_m,
                      stable_window_span_s=max(0, self.tracker.steps-1)/CONTROL_HZ)
        result.update(launched=self.launched, table_hit=self.table_hit, stable_steps=self.tracker.steps)
        result['valid'] = bool(result['valid'] and self.launched and not self.table_hit)
        return result

    def observe_ball(self):
        # Monitor each physics substep: a tennis ball can touch and bounce off the table
        # between two 30 Hz observations. The small clearance is conservative;
        # successful airborne catches must occur much higher than this plane.
        if self.launched and self.cfg.mode != 'roll':
            z = float(self.objects[0].data.root_pos_w[0, 2])
            if self.cfg.mode == 'container':
                # Contact-height proxy with the collider's 0.5 mm margin,
                # rather than the other tasks' conservative 12 mm exclusion.
                from piper_bench.tasks.conveyor.core import rotation_xyzw
                pose, _ = self.receiver_state()
                rotation = rotation_xyzw(pose[3:])
                half = self.cfg.container_inner/2+self.cfg.wall
                bottom = pose[2]-half*(abs(rotation[2, 0])+abs(rotation[2, 1]))
                plane = self.cfg.landing_table_z+.0005
                self.table_hit |= bool(z <= self.cfg.radius+plane or bottom <= plane)
            else:
                self.table_hit |= z <= self.cfg.radius+.012

    def after_step(self, gripper_open=True):
        self.control_step += 1
        poses, _, tcp = self.state()
        self.tcp_velocity = (tcp-self.previous_tcp)*CONTROL_HZ
        self.previous_tcp = tcp.copy()
        self.observe_ball()
        metrics = self.metrics(gripper_open)
        success = self.tracker.update(metrics, self.launched, self.table_hit)
        reason = None
        if self.table_hit:
            reason = ('ball_or_container_hit_table' if self.cfg.mode == 'container'
                      else 'ball_hit_table_before_stable_catch')
        elif self.launched and (poses[0, 2] < -.04 or abs(poses[0, 1]) > (1.5 if self.cfg.mode == 'container' else .46) or poses[0, 0] > (1.6 if self.cfg.mode in ('lob', 'container') else 1.1)):
            reason = 'ball_left_workspace'
        if self.time >= self.cfg.max_seconds and not success:
            reason = reason or 'timeout'
        return bool(success), reason

    def metadata(self):
        task, prompt = TASK_INFO[self.cfg.mode]
        value = dict(task=task, prompt=prompt, config=self.cfg.metadata(), seed=self.seed,
                    ball_appearance=appearance_id(MODE_SKINS[self.cfg.mode]),
                    ball_skin=MODE_SKINS[self.cfg.mode], appearance=self.tabletop_appearance,
                    oracle_observation=('post-launch measured position and velocity' if self.cfg.mode in ('lob', 'roll', 'container') else 'predictive state expert'),
                    launch_position_m=self.initial_position.tolist(), launch_velocity=self.initial_velocity.tolist(),
                    launch_time_s=self.launch_time_s, launch_clock='fixed simulation time at control boundary',
                    launch_time_range_s=[self.cfg.launch_delay_min, self.cfg.launch_delay_max],
                    launch_time_sampling='uniform inclusive 30 Hz ticks; independent schedule RNG',
                    object_pose_override='reset parking and one scheduled launch only',
                    no_grasp_attachments=True, ball_motion='passive PhysX gravity and contacts after launch',
                    success='lifted stable gripper hold' if self.cfg.mode != 'container' else 'ball retained in airborne container for 0.5 s without table/ground contact')

        if self.cfg.mode == 'lob':
            vx, vy = self.initial_velocity[:2]
            value.update(success='airborne stable gripper hold without table contact',
                         success_protocol='lob_finger_region_v3',
                         grasp_region='fixed fingertip frame; independent of expert grasp_depth',
                         lift_required_for_success=False,
                         stable_position_diameter_m=self.cfg.stable_position_diameter_m,
                         stable_window_seconds=self.cfg.stable_seconds,
                         stable_window_samples=self.tracker.window_samples,
                         velocity_used_for_success=False,
                         launch_horizontal_speed_m_s=float(np.hypot(vx, vy)),
                         launch_yaw_deg=float(np.degrees(np.arctan2(vy, -vx))))
        if self.cfg.mode == 'roll':
            vx, vy = self.initial_velocity[:2]
            value.update(success=f'lifted gripper hold for {self.cfg.stable_seconds:g} s',
                         success_protocol='rolling_lifted_hold_v3',
                         grasp_region='finger-aligned envelope with palm clearance',
                         stable_position_diameter_m=self.cfg.stable_position_diameter_m,
                         stable_window_seconds=self.cfg.stable_seconds,
                         stable_window_samples=self.tracker.window_samples,
                         velocity_used_for_success=False, position_stability_used_for_success=False,
                         lift_required_for_success=True,
                         launch_horizontal_speed_m_s=float(np.hypot(vx, vy)),
                         launch_yaw_deg=float(np.degrees(np.arctan2(vy, -vx))),
                         launch_sampling='independent uniform draws conditioned on fixed receiving workspace',
                         receiving_plane_x_m=self.cfg.intercept_x,
                         receiving_y_range_m=[self.cfg.intercept_y_min, self.cfg.intercept_y_max],
                         prelaunch_motion=False, reaction_delay_s=self.cfg.reaction_delay,
                         launch_direction_reference='negative world X toward robot base',
                         initial_pose='fixed forward-looking neutral pose independent of launch',
                         waiting_point_independent_of_seed=True)
        if self.cfg.mode == 'fall':
            value.update(launch_downward_speed_m_s=float(-self.initial_velocity[2]),
                         success_protocol='falling_relative_position_v1',
                         initial_pose='fixed retracted observing pose; move to catch after observation',
                         prelaunch_motion=False, reaction_delay_s=self.cfg.reaction_delay,
                         stable_position_diameter_m=self.cfg.stable_position_diameter_m,
                         stable_window_seconds=self.cfg.stable_seconds,
                         stable_window_samples=self.tracker.window_samples,
                         velocity_used_for_success=False, lift_required_for_success=True,
                         downward_speed_range_m_s=[self.cfg.downward_speed_min, self.cfg.downward_speed_max],
                         launch_direction_reference='negative world Z; no horizontal velocity')
        if self.cfg.mode == 'container':
            vx, vy = self.initial_velocity[:2]
            value.update(initial_pose='fixed physically preheld container',
                         success_protocol='container_retention_v3',
                         stable_position_diameter_m=self.cfg.stable_position_diameter_m,
                         stable_window_samples=self.tracker.window_samples,
                         stable_window_seconds=self.cfg.stable_seconds,
                         container_instantaneous_angular_velocity_used_for_success=False,
                         container_uprightness_used_for_success=False,
                         velocity_used_for_success=False,
                         position_stability_used_for_success=False,
                         table_contact_scope='ball and container; physics-substep geometric contact-height check',
                         gripper_closed_entire_episode=True, prelaunch_motion=False,
                         grasp_type='top-bottom pinch of horizontal pan handle crossbar',
                         initial_tool_quaternion_xyzw=list(self.cfg.initial_tool_quaternion),
                         physics_hz=240, control_hz=30,
                         initial_container_yaw_rad=self.cfg.initial_container_yaw,
                         grip_offset_m=list(self.cfg.grip_offset),
                         receiving_radius_max_m=self.cfg.receiving_radius_max,
                         reaction_delay_s=self.cfg.reaction_delay,
                         launch_horizontal_speed_m_s=float(np.hypot(vx, vy)),
                         launch_yaw_deg=float(np.degrees(np.arctan2(vy, -vx))),
                         launch_sampling='paired frontal launches sharing release time/position and flight duration; distinct direction/speed within fixed tabletop receiving region; no live robot target',
                         launch_clock='scheduled simulation time at control boundary',
                         launch_time_range_s=[self.cfg.launch_delay_min, self.cfg.launch_delay_max],
                         launch_time_sampling='uniform inclusive 30 Hz ticks, independent RNG stream per pair',
                         launch_pair_id=int(self.seed)//2,
                         launch_pair_member=int(self.seed)%2,
                         paired_seed=int(self.seed)^1,
                         launch_x_range_m=[self.cfg.source_x-self.cfg.release_x_jitter, self.cfg.source_x+self.cfg.release_x_jitter],
                         launch_z_range_m=[self.cfg.source_z-self.cfg.release_z_jitter, self.cfg.source_z+self.cfg.release_z_jitter],
                         flight_time_range_s=[self.cfg.flight_time_min, self.cfg.flight_time_max],
                         minimum_receiver_displacement_m=self.cfg.minimum_receiver_displacement,
                         landing_region_x_m=[self.cfg.landing_x_min, self.cfg.landing_x_max],
                         landing_region_y_m=[-self.cfg.landing_y_limit, self.cfg.landing_y_limit],
                         landing_edge_clearance_m=self.cfg.landing_edge_margin,
                         receiving_radius_min_m=self.cfg.receiving_radius_min,
                         receiving_x_range_m=[self.cfg.receiving_x_min, self.cfg.receiving_x_max],
                         receiving_y_range_m=[-self.cfg.receiving_y_limit, self.cfg.receiving_y_limit])
            if not self.cfg.paired_launch:
                value.update(launch_sampling='independent frontal launch within narrow receiving region',
                             launch_pair_id=None, launch_pair_member=None, paired_seed=None,
                             launch_time_sampling='fixed control tick')
            from piper_bench.tasks.tennis.motion import container_table_landing
            from piper_bench.tasks.tennis.motion import interception
            value['predicted_table_landing_m'] = container_table_landing(
                self.initial_position, self.initial_velocity, self.cfg).tolist()
            point, seconds = interception(self.initial_position, self.initial_velocity, self.cfg)
            value.update(predicted_rim_entry_m=point.tolist(), predicted_flight_time_s=seconds)
        return value
