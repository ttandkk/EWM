"""Predictive gripper timing and contact-only container pickup/catching expert."""
import math
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.common import CartesianOracle
from piper_bench.tasks.common import approach_pose
from piper_bench.tasks.common import interpolate_quaternion
from piper_bench.tasks.common import quintic
from piper_bench.tasks.common import yaw_pitch_quaternion
from piper_bench.tasks.tennis.motion import interception
from piper_bench.tasks.tennis.motion import predict


class TennisOracle(CartesianOracle):
    PROFILE = 'tennis_predictive_contact_v1'

    def __init__(self, runtime):
        super().__init__(runtime)
        self.catch_point, self.arrival = None, None
        # All interception estimates come from post-launch observations.
        self.first_observed = None
        self.capture_target = None
        self.catch_q, self.capture_q = None, None
        self.rolling_heading = 0.
        self.falling_close_time = None

    def catch_orientation(self):
        # For rolling, align the open jaw entrance with measured planar travel.
        # For flight, horizontal fingers leave the vertical path open.
        return yaw_pitch_quaternion(self.rolling_heading, math.pi) if self.cfg.mode == 'roll' else yaw_pitch_quaternion(0., math.pi/2)

    def update_prediction(self):
        if self.runtime.launched:
            poses, velocity, _ = self.runtime.state()
            result = interception(poses[0, :3], velocity[0], self.cfg)
            if result is not None:
                point, seconds = result
                self.catch_point, self.arrival = point, self.runtime.time+seconds
                if self.cfg.mode == 'fall':
                    # Align impact with a 30 Hz closure command at or before the
                    # nominal crossing. Rounding later would lower the receiving
                    # plane and can drive the lowered waiting pose into a wrist limit.
                    if self.falling_close_time is None:
                        self.falling_close_time = math.floor((self.arrival-self.cfg.close_lead)*30+1e-8)/30
                    self.arrival = self.falling_close_time+self.cfg.close_lead
                    self.catch_point, _ = predict(poses[0, :3], velocity[0],
                                                  max(0., self.arrival-self.runtime.time), self.cfg)
                if self.cfg.mode == 'roll':
                    self.rolling_heading = math.atan2(-velocity[0, 1], -velocity[0, 0])
                if self.cfg.mode == 'lob':
                    # Tool +Z faces the incoming ball; update attitude from the
                    # measured flight, without exposing launcher state.
                    _, arrival_velocity = predict(poses[0, :3], velocity[0], seconds, self.cfg)
                    incoming = -arrival_velocity
                    yaw = math.atan2(incoming[1], incoming[0])
                    pitch = math.atan2(np.linalg.norm(incoming[:2]), incoming[2])
                    self.catch_q = yaw_pitch_quaternion(yaw, pitch)

    def action(self):
        if self.cfg.mode == 'roll':
            return self.reactive_rolling_action()
        if self.cfg.mode == 'lob':
            return self.reactive_lob_action()
        return self.container_action() if self.cfg.mode == 'container' else self.gripper_action()

    def gripper_action(self):
        _, _, tcp = self.runtime.state()
        if self.started is None:
            self.begin('wait' if self.cfg.mode == 'fall' else 'approach', tcp)
        if self.cfg.mode == 'fall' and not self.runtime.launched:
            return self.command(self.start.copy(), self.start_q.copy(), 1., translation_limit=.02)
        elapsed, stage = self.runtime.time-self.started, self.stage
        if stage in ('approach', 'wait'):
            self.update_prediction()
        if self.cfg.mode == 'fall' and stage == 'wait':
            if self.first_observed is None:
                self.first_observed = self.runtime.time
            if self.runtime.time-self.first_observed+1e-8 < self.cfg.reaction_delay:
                # Accumulate visible motion at the fixed observing pose before
                # translating underneath the vertical flight.
                return self.command(self.start.copy(), self.start_q.copy(), 1., translation_limit=.02)
        q = self.catch_orientation()
        center = self.catch_point
        end = center+(.0042+getattr(self.cfg, 'grasp_depth', 0.))*rotation_xyzw(q)[:, 2]
        grip = 1.
        if stage == 'approach':
            # Use the established radial home escape before reorienting the
            # wrist to the catch frame; position and attitude settle before launch.
            target, radial_q = approach_pose(self.start, self.start_q, end, quintic(elapsed/2.5))
            q = interpolate_quaternion(radial_q, q, quintic((elapsed-1.2)/1.3))
            if elapsed >= 2.5:
                self.begin('wait', tcp)
        elif stage == 'wait':
            target = end
            if self.runtime.time+1e-8 >= self.arrival-self.cfg.close_lead:
                self.capture_target = end.copy()
                self.begin('close', tcp)
                grip = -1.
        elif stage == 'close':
            target, grip = self.capture_target, -1.
            if self.runtime.time >= self.arrival+.12:
                self.begin('lift', tcp)
        elif stage == 'lift':
            grip = -1.
            target = self.move(self.capture_target+np.array([0., 0., self.cfg.lift_height]), .7)
            if elapsed >= .7:
                self.begin('hold', tcp)
        else:
            target, grip = self.capture_target+np.array([0., 0., self.cfg.lift_height]), -1.
        if self.cfg.mode == 'fall':
            action = self.command(target, q, grip,
                                  translation_limit=self.cfg.translation_limit,
                                  angular_limit=self.cfg.angular_limit)
            if self.stage == 'wait':
                # Close the position lag during the short observed approach.
                # Joint speed/effort limits and passive ball physics are unchanged.
                action[:, :3] *= self.cfg.approach_gain
            return action
        return self.command(target, q, grip, translation_limit=.02)

    def reactive_rolling_action(self):
        """Keep the reset pose until a launched ball has actually been observed."""
        _, _, tcp = self.runtime.state()
        if self.started is None:
            self.begin('observe', tcp)
        elapsed = self.runtime.time-self.started
        if self.stage == 'observe':
            target, q = self.start.copy(), self.start_q.copy()
            if self.runtime.launched:
                if self.first_observed is None:
                    self.first_observed = self.runtime.time
                if self.runtime.time-self.first_observed+1e-8 >= self.cfg.reaction_delay:
                    self.update_prediction()
                    if self.catch_point is not None:
                        self.begin('approach', tcp)
            return self.command(target, q, 1., translation_limit=self.cfg.translation_limit)
        if self.stage == 'approach':
            self.update_prediction()
            q = self.catch_orientation()
            end = self.catch_point+.0042*rotation_xyzw(q)[:, 2]
            duration = max(.3, min(self.cfg.intercept_seconds, self.arrival-self.started-.12))
            u = quintic(elapsed/duration)
            target, radial_q = approach_pose(self.start, self.start_q, end, u)
            q = interpolate_quaternion(radial_q, q, u)
            grip = 1.
            if self.runtime.time >= self.arrival-self.cfg.close_lead:
                self.capture_target, self.capture_q = end.copy(), self.catch_orientation()
                self.begin('close', tcp)
                target, q, grip = end, self.capture_q, -1.
        elif self.stage == 'close':
            target, q, grip = self.capture_target, self.capture_q, -1.
            if self.runtime.time >= self.arrival+.12:
                self.begin('lift', tcp)
        elif self.stage == 'lift':
            target = self.move(self.capture_target+[0., 0., self.cfg.lift_height], .7)
            q, grip = self.capture_q, -1.
            if elapsed >= .7:
                self.begin('hold', tcp)
        else:
            target, q, grip = self.capture_target+[0., 0., self.cfg.lift_height], self.capture_q, -1.
        return self.command(target, q, grip, translation_limit=self.cfg.translation_limit)

    def reactive_lob_action(self):
        """React to observed flight only; the launcher never queries this expert."""
        _, _, tcp = self.runtime.state()
        q = yaw_pitch_quaternion(0., 1.0)
        if self.started is None:
            self.begin('observe', tcp)
        elapsed = self.runtime.time-self.started
        if self.stage == 'observe':
            target = self.start.copy()
            if self.runtime.launched:
                if self.first_observed is None:
                    self.first_observed = self.runtime.time
                if self.runtime.time-self.first_observed +1e-8 >= self.cfg.reaction_delay:
                    self.update_prediction()
                    if self.catch_point is not None:
                        self.begin('intercept', tcp)
            return self.command(target, q, 1., translation_limit=self.cfg.translation_limit)
        if self.stage == 'intercept':
            self.update_prediction()
            q = self.catch_q
            end = self.catch_point+(.0042+self.cfg.grasp_depth)*rotation_xyzw(q)[:, 2]
            duration = max(.10, min(self.cfg.intercept_seconds, self.arrival-self.started-.12))
            target = self.start+(end-self.start)*quintic(elapsed/duration)
            grip = 1.
            if self.runtime.time >= self.arrival-self.cfg.close_lead:
                self.capture_target = end.copy()
                self.capture_q = q.copy()
                self.begin('close', tcp)
                target, grip = end, -1.
        elif self.stage == 'close':
            target, grip = self.capture_target, -1.
            if self.runtime.time >= self.arrival+.10:
                self.begin('lift', tcp)
        elif self.stage == 'lift':
            target, grip = self.move(self.capture_target+[0., 0., self.cfg.lift_height], .7), -1.
            if elapsed >= .7:
                self.begin('hold', tcp)
        else:
            target, grip = self.capture_target+[0., 0., self.cfg.lift_height], -1.
        if self.stage in ('close', 'lift', 'hold'):
            q = self.capture_q
        return self.command(target, q, grip, translation_limit=self.cfg.translation_limit)

    def container_action(self):
        """Start with the front handle clamped; react only to measured free flight."""
        c = self.cfg
        _, _, tcp = self.runtime.state()
        pose, _ = self.runtime.receiver_state()
        q = np.asarray(c.initial_tool_quaternion)
        if self.started is None:
            self.begin('observe', tcp)
        if self.stage == 'observe':
            target = self.start.copy()
            if self.runtime.launched:
                if self.first_observed is None:
                    self.first_observed = self.runtime.time
                if self.runtime.time-self.first_observed+1e-8 >= c.reaction_delay:
                    self.begin('intercept', tcp)
            return self.command(target, q, -1., translation_limit=c.translation_limit)
        if self.stage == 'intercept':
            self.update_prediction()
            if self.catch_point is not None:
                # Center the path between rim entry and floor impact, allowing
                # the incoming ball's horizontal travel inside the open box.
                from piper_bench.tasks.tennis.motion import descending_time
                poses, velocities, _ = self.runtime.state()
                floor_time = descending_time(poses[0, :3], velocities[0],
                                             c.container_hold_z+c.wall+c.radius, c.gravity)
                floor_point = self.catch_point if floor_time is None else predict(
                    poses[0, :3], velocities[0], floor_time, c)[0]
                center = .5*(self.catch_point[:2]+floor_point[:2])
                # Yaw with the receiving location while keeping the box level.
                # Fixed world heading is unreachable at wide lateral extremes.
                self.hold_yaw = math.atan2(center[1], center[0])
                self.hold_center = np.r_[center, c.container_hold_z]
                if self.runtime.time >= self.arrival-.025:
                    self.begin('hold', tcp)
            else:
                return self.command(self.start.copy(), q, -1., translation_limit=c.translation_limit)
        # Level the freely held box after impact-induced handle slip. Only the
        # robot target changes; the container stays a passive contact body.
        from scipy.spatial.transform import Rotation
        yaw = self.hold_yaw
        desired = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                            [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
        correction = desired@rotation_xyzw(pose[3:]).T
        tool_q = self.env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().numpy()
        q = Rotation.from_matrix(correction@rotation_xyzw(tool_q)).as_quat()
        target = self.hold_center+correction@(tcp-pose[:3])
        return self.command(target, q, -1., translation_limit=c.translation_limit)
