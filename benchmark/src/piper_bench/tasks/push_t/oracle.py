"""Reposition above the table, then push with closed fingers; no object state writes."""
import math
import numpy as np
from piper_bench.tasks.common import CartesianOracle
from piper_bench.tasks.common import approach_pose
from piper_bench.tasks.common import tabletop_orientation
from piper_bench.tasks.common import quintic
from piper_bench.tasks.common import yaw_pitch_quaternion
from piper_bench.tasks.common import interpolate_quaternion
from piper_bench.tasks.push_t.geometry import rectangles


def contact_on_ray(cfg, direction, yaw):
    """First real T boundary encountered when approaching its centre.

    A support-plane projection can lie in the empty corner of the concave T,
    leaving short diagonal corrections without any contact.
    """
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    ray = -rot.T @ direction
    farthest = 0.
    for center, size in rectangles(cfg):
        low, high = np.asarray(center)-np.asarray(size)/2, np.asarray(center)+np.asarray(size)/2
        entry, leave = 0., math.inf
        for axis in range(2):
            if abs(ray[axis]) < 1e-9:
                if not low[axis] <= 0 <= high[axis]:
                    leave = -math.inf
            else:
                a, b = sorted((low[axis]/ray[axis], high[axis]/ray[axis]))
                entry, leave = max(entry, a), min(leave, b)
        if leave >= entry:
            farthest = max(farthest, leave)
    return rot @ (ray*farthest)


def lift_orientation(start_quat, start_z, target, contact_yaw):
    """Retain the contact heading near the T; relax the wrist after clearance."""
    end = np.r_[target[:2], .14]
    u = np.clip((target[2]-start_z)/max(.14-start_z, 1e-6), 0., 1.)
    pitch = math.pi-.75*quintic((.14-.04)/.20)
    contact = interpolate_quaternion(start_quat, yaw_pitch_quaternion(contact_yaw, pitch), u)
    # At 55 mm TCP height the closed fingertips are above the 25 mm T block.
    # Keeping a +/-45 degree contact yaw all the way up can hit joint 5's limit.
    relax = quintic((target[2]-.055)/(.14-.055))
    return interpolate_quaternion(contact, tabletop_orientation(end), relax)


class PushTOracle(CartesianOracle):
    PROFILE = 'push_t_pose_v2_contact_reach_v9'
    APPROACH_PADDING = .019
    contact_pitch = math.pi
    PUSH_SPEED = .06
    MAX_TRANSLATION = .30

    def plan(self, pose):
        c = self.cfg
        self.push_origin = pose[:2].copy()
        yaw = self.runtime.metrics()['yaw_rad']
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        error = math.atan2(math.sin(c.target_yaw-yaw), math.cos(c.target_yaw-yaw))
        self.aligning = abs(error) > math.radians(3)
        if self.aligning:
            # Use the left bar end for either yaw sign. Switching the face
            # keeps the corrective force directed inward at the outer X boundary.
            side = -1. if error > 0 else 1.
            self.direction = rot@np.array([0., side])
            edge_y = c.length/2 if side < 0 else c.length/2-c.bar
            self.contact = pose[:2]+rot@np.array([-c.width*.38, edge_y])
            self.distance = float(np.clip(abs(error)*.045, .008, .040))
        else:
            delta = rot.T@(np.array([c.target_x, c.target_y])-pose[:2])
            axis = int(np.argmax(np.abs(delta)))
            direction = np.zeros(2); direction[axis] = math.copysign(1., delta[axis])
            self.distance = min(abs(float(delta[axis])), self.MAX_TRANSLATION)
            self.direction = rot@direction
            # Correct one body axis at a time. A diagonal approach can put the
            # finger on the broad bar before it reaches the intended stem.
            contact = contact_on_ray(c, self.direction, yaw)
            # Keep lateral contact on the stem centreline. The old offset
            # toward the bar produced large yaw drift with the finite fingers
            # in physical replay, despite being closer to the geometric COM.
            self.contact = pose[:2]+contact
        self.behind = np.r_[self.contact-self.direction*.020, .012]
        self.endpoint = self.behind+np.r_[self.direction*(self.distance+self.APPROACH_PADDING), 0.]
        # Keep the fingertip face parallel to the T instead of to the robot's
        # radial heading. An oblique face torques the stem during translation,
        # forcing repeated lift/reposition/angle-correction cycles.
        self.push_yaw = yaw
        # A vertical wrist at the far edge saturates joint 5. Use a small
        # fixed tilt over the entire stroke, including the descent endpoint.
        radius = max(np.linalg.norm(self.behind[:2]), np.linalg.norm(self.endpoint[:2]))
        self.contact_pitch = math.pi-float(np.clip((radius-.42)*1.8, 0., .10))

    def action(self):
        pose, _, tcp = self.runtime.state()
        pose = pose[0]
        if self.started is None:
            self.plan(pose)
            self.begin('approach', tcp)
        elapsed = self.runtime.time-self.started
        stage = self.stage
        # A transition can reset self.start or plan a new heading below. The
        # command being completed still belongs to the old segment.
        command_yaw = self.push_yaw
        if stage in ('descend', 'lift', 'reposition'):
            segment_z, segment_q = float(self.start[2]), self.start_q.copy()
        high = self.behind.copy(); high[2] = .12
        q = tabletop_orientation(high)
        if stage == 'approach':
            target, q = approach_pose(self.start, self.start_q, high, quintic(elapsed/2.5))
            if self.reached(high, 2.5):
                self.begin('descend', tcp)
        elif stage == 'reposition':
            target = self.move(high, .9)
            if self.reached(high, .9):
                self.begin('descend', tcp)
        elif stage == 'descend':
            target = self.move(self.behind, .8)
            if self.reached(self.behind, .8, .004):
                self.begin('push', tcp)
            elif elapsed > 2. and np.linalg.norm(tcp-self.behind) > .12:
                self.failure = 'ik_tracking_diverged'
        elif stage == 'push':
            # Keep small position/angle corrections gentle: at 30 Hz the
            # faster main stroke would otherwise overshoot by a few mm.
            speed = .025 if self.aligning or self.distance < .04 else self.PUSH_SPEED
            duration = (self.distance+self.APPROACH_PADDING)/speed
            target = self.move(self.endpoint, duration)
            metric = self.runtime.metrics()
            aligned = metric['yaw_error_deg'] < 2.5
            arrived = metric['position_error_m'] < .005 and metric['target_coverage'] >= self.cfg.min_coverage
            # The closed fingers' contact extent varies with heading. Stop on
            # measured block travel instead of consuming the full TCP approach
            # padding, which can push the block beyond the goal by ~1 cm.
            translated = float((pose[:2]-self.push_origin) @ self.direction) >= self.distance-.001
            if elapsed >= duration or (elapsed > .5 and ((self.aligning and aligned) or (not self.aligning and (arrived or translated)))):
                self.begin('lift', tcp)
        elif stage == 'lift':
            end = self.start.copy(); end[2] = .14
            target = self.move(end, .65)
            if self.reached(end, .65):
                metric = self.runtime.metrics()
                if metric['position_error_m'] < .006 and metric['yaw_error_deg'] < 4 and metric['target_coverage'] >= self.cfg.min_coverage:
                    self.begin('done', tcp)
                else:
                    self.plan(pose)
                    self.begin('reposition', tcp)
        else:
            target = self.start.copy()
        if stage != 'approach':
            pitch = math.pi-.75*quintic((target[2]-.04)/.20)
            # Keep the fingers' yaw fixed during contact. Following the TCP's
            # changing radial heading rotates the T during an otherwise
            # straight push and creates repeated final corrections.
            q = yaw_pitch_quaternion(command_yaw, self.contact_pitch if stage == 'push' else pitch)
            if stage == 'lift':
                q = lift_orientation(segment_q, segment_z, target, command_yaw)
            elif stage == 'descend':
                end_z, end_pitch = .012, self.contact_pitch
                # The generic height/pitch curve exceeds joint 5's 70-degree
                # limit midway down the outer approach. Linear pose progress
                # keeps that entire segment reachable, not just its endpoints.
                # On descent, do not rotate ahead of the measured height.
                z = max(float(target[2]), float(tcp[2])) if stage == 'descend' else float(target[2])
                u = (z-segment_z)/(end_z-segment_z) if abs(end_z-segment_z) > 1e-6 else 1.
                q = interpolate_quaternion(segment_q, yaw_pitch_quaternion(command_yaw, end_pitch), np.clip(u, 0., 1.))
            elif stage == 'reposition':
                q = interpolate_quaternion(segment_q, tabletop_orientation(target), quintic(elapsed/.9))
        self.target_quaternion = np.asarray(q).copy()
        # Keep the original translation/rotation balance. Halving only the
        # translation limit lets pitch advance ahead of the arm position,
        # saturating joint 5 during approach/descent (208921 physical replay).
        return self.command(target, q, -1., translation_limit=.012)
