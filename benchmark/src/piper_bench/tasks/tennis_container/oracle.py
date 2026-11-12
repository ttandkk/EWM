"""Delay interception until lateral flight is visible, then move continuously."""
import math
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.robots.piper.motion import quintic
from piper_bench.tasks.tennis.oracle import TennisOracle
from piper_bench.tasks.tennis.motion import descending_time
from piper_bench.tasks.tennis.motion import predict
from piper_bench.tasks.tennis.wrist import wrist_metadata


class Oracle(TennisOracle):
    PROFILE = 'tennis_container_visible_smooth_intercept_v15'
    OBSERVATION_SECONDS = .30
    LATERAL_SIGNAL_PX = 1.5
    CENTRAL_BAND_M = .035
    ARRIVAL_MARGIN_SECONDS = .055
    MIN_INTERCEPT_SECONDS = .20
    TRANSLATION_LIMIT = .024

    def lateral_signal_pixels(self, ball, tcp, tool_q):
        """Projected displacement from the central flight plane in the wrist view.

        This is an observability gate on measured state, not an image detector.
        Launcher seeds, scheduled velocity and future state are never consulted.
        """
        cal = wrist_metadata('container')
        tool = rotation_xyzw(tool_q)
        eye = tcp + tool @ np.asarray(cal['wrist_eye_local_m'])
        camera = tool @ np.asarray(cal['wrist_mount_rotation_local_opengl'])
        centerline = np.array(ball, copy=True)
        centerline[1] = self.cfg.container_y
        points = (np.stack([ball, centerline])-eye) @ camera
        depths = -points[:, 2]
        if np.any(depths <= 0):
            return 0.
        focal = cal['wrist_intrinsic_matrix'][0][0]
        return float(abs(focal*(points[0, 0]/depths[0]-points[1, 0]/depths[1])))

    def container_action(self):
        from scipy.spatial.transform import Rotation
        c = self.cfg
        poses, velocities, tcp = self.runtime.state()
        box, _ = self.runtime.receiver_state()
        tool_q = self.env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().numpy()
        if self.started is None:
            self.begin('observe', tcp)
        if self.stage == 'observe':
            if self.runtime.launched:
                if self.first_observed is None:
                    self.first_observed = self.runtime.time
                if self.runtime.time-self.first_observed+1e-8 >= self.OBSERVATION_SECONDS:
                    self.update_prediction()
                    floor_t = descending_time(poses[0, :3], velocities[0],
                                              c.container_hold_z+c.wall+c.radius, c.gravity)
                    if self.catch_point is not None and floor_t is not None:
                        floor = predict(poses[0, :3], velocities[0], floor_t, c)[0]
                        center = .5*(self.catch_point[:2]+floor[:2])
                        visible = self.lateral_signal_pixels(poses[0, :3], tcp, tool_q)
                        central = abs(center[1]-c.container_y) <= self.CENTRAL_BAND_M
                        if central or visible >= self.LATERAL_SIGNAL_PX:
                            self.hold_center = np.r_[center, c.container_hold_z]
                            self.hold_yaw = math.atan2(center[1], center[0])
                            self.intercept_start_center = box[:3].copy()
                            self.intercept_start_yaw = math.atan2(rotation_xyzw(box[3:])[1, 0],
                                                                  rotation_xyzw(box[3:])[0, 0])
                            # A late observation must never collapse the eased
                            # reference into a one-tick jump, even if that
                            # leaves too little time to catch an OOD launch.
                            self.intercept_duration = max(self.MIN_INTERCEPT_SECONDS,
                                self.arrival-self.runtime.time-self.ARRIVAL_MARGIN_SECONDS)
                            self.begin('intercept', tcp)
            return self.command(self.start.copy(), np.asarray(c.initial_tool_quaternion), -1.,
                                translation_limit=self.TRANSLATION_LIMIT)
        if self.stage == 'intercept':
            elapsed = self.runtime.time-self.started
            u = quintic(elapsed/self.intercept_duration)
            center = (1-u)*self.intercept_start_center+u*self.hold_center
            yaw = (1-u)*self.intercept_start_yaw+u*self.hold_yaw
            if elapsed >= self.intercept_duration:
                self.begin('hold', tcp)
        else:
            center, yaw = self.hold_center, self.hold_yaw
        desired = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                            [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
        correction = desired @ rotation_xyzw(box[3:]).T
        q = Rotation.from_matrix(correction @ rotation_xyzw(tool_q)).as_quat()
        target = center+correction @ (tcp-box[:3])
        return self.command(target, q, -1., translation_limit=self.TRANSLATION_LIMIT)
