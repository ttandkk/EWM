"""Smooth feedback-gated open/place/close sequence; no drawer or cup state writes."""
import math
import numpy as np
from piper_bench.tasks.common import CartesianOracle
from piper_bench.tasks.common import approach_pose
from piper_bench.tasks.common import tabletop_orientation
from piper_bench.tasks.common import quintic
from piper_bench.tasks.common import yaw_pitch_quaternion
from piper_bench.tasks.common import interpolate_quaternion
from piper_bench.tasks.drawer_cup.scene import bowl_rim_grasp
from piper_bench.tasks.drawer_cup.scene import upright_bowl_grasp
from piper_bench.tasks.drawer_cup.scene import bowl_ready_to_release


def handle_hook_target(handle, cfg):
    # Preserve the inner finger's 10 mm offset behind the handle even when
    # the configured opening changes. Only the TCP/outer finger move closer.
    return (handle+(cfg.gripper_open_half_width-.010)*np.asarray(cfg.pull_direction)
            -np.array([0., 0., .008]))


def bowl_approach_pose(start, start_q, rim, rim_q, u):
    """Unhook, arc towards the bowl and descend without intermediate stops.

    Match the wrist's tilt to the elevated control points. The final rim
    orientation is reachable near the table, but not at the old 75 mm stop.
    """
    a, b = np.r_[start[:2], .20], np.r_[rim[:2], .17]
    target = (1-u)**3*start+3*(1-u)**2*u*a+3*(1-u)*u*u*b+u**3*rim
    quats = [start_q, tabletop_orientation(a), tabletop_orientation(b), rim_q]
    while len(quats) > 1:
        quats = [interpolate_quaternion(quats[i], quats[i+1], u) for i in range(len(quats)-1)]
    return target, quats[0]


class DrawerCupOracle(CartesianOracle):
    PROFILE = 'drawer_bowl_continuous_rim_drop_v18'

    def __init__(self, runtime):
        super().__init__(runtime)
        self.held_offset = np.array([0., 0., -.015])
        self.grasp_lifted = False

    def action(self):
        poses, vel, tcp = self.runtime.state()
        pose = poses[0]; c = self.cfg
        if self.started is None:
            self.begin('approach', tcp)
        stage = self.stage; elapsed = self.runtime.time-self.started
        handle = self.runtime.handle()
        opening, _ = self.runtime.opening()
        pull = np.asarray(c.pull_direction)
        # Open jaws run along the pull direction: the inner fingertip enters
        # behind the handle, then pulls by contact without a pinch/release cycle.
        handle_q = yaw_pitch_quaternion(c.cabinet_yaw-math.pi/2, math.pi)
        hook = handle_hook_target(handle, c)
        grip = -1. if stage in ('cup_close', 'cup_lift', 'transfer', 'place_align', 'place', 'close_approach', 'push') else 1.
        q = tabletop_orientation(tcp)
        # Keep the near-side single-rim grasp, turning it away from the drawer
        # front. Together with the narrower opening this leaves >8 mm in the
        # 60 sampled protocol poses checked against the actual finger meshes.
        rim_heading = math.atan2(pose[1], pose[0])+math.radians(18)
        rim, rim_q = bowl_rim_grasp(pose, c, heading=rim_heading)
        if stage in ('transfer', 'place_align', 'place'):
            self.held_offset = .8*self.held_offset+.2*(pose[:3]-tcp)
        if stage == 'approach':
            end = hook+np.array([0., 0., .08])
            target, q = approach_pose(self.start, self.start_q, end, quintic(elapsed/2.))
            if self.reached(end, 2.):
                self.begin('handle_descend', tcp)
        elif stage == 'handle_descend':
            target = self.move(hook, 1.1)
            q = interpolate_quaternion(self.start_q, handle_q, quintic(elapsed/1.1))
            if self.reached(hook, 1.1, .003):
                self.pull_origin = tcp.copy()
                self.begin('pull', tcp)
        elif stage == 'pull':
            end = self.pull_origin+(c.travel+.020)*pull
            target = self.move(end, 2.4)
            # The sequence predicate's open threshold is not the placement
            # clearance: pull nearly to the stop so the bowl clears the lid.
            if elapsed >= 2.4 and opening >= c.travel-.003:
                self.rim_q = rim_q.copy()
                self.begin('cup_approach', tcp)
            elif elapsed > 6.:
                self.failure = 'drawer_not_opened'
        elif stage == 'cup_approach':
            target, q = bowl_approach_pose(self.start, self.start_q, rim, self.rim_q, quintic(elapsed/3.2))
            if self.reached(rim, 3.2, .004):
                self.begin('cup_close', tcp)
            elif elapsed > 6.:
                self.failure = 'bowl_approach_blocked'
        elif stage == 'cup_close':
            target = self.start.copy()
            if elapsed >= .4:
                self.begin('cup_lift', tcp)
        elif stage == 'cup_lift':
            end = self.start.copy(); end[2] = .23
            target = self.move(end, 1.5)
            # Unload the rim and tilt the wrist continuously as height grows;
            # holding the full rim pose up to 75 mm exceeds joint 5's limit.
            u = np.clip((target[2]-self.start[2])/(end[2]-self.start[2]), 0., 1.)
            q = interpolate_quaternion(self.start_q, tabletop_orientation(end), u)
            self.grasp_lifted |= pose[2] > .050
            if self.reached(end, 1.5) and self.grasp_lifted:
                self.held_offset = pose[:3]-tcp
                self.begin('transfer', tcp)
            elif elapsed > 4.:
                self.failure = 'bowl_not_lifted_by_rim'
        elif stage == 'transfer':
            end = self.runtime.placement()-self.held_offset; end[2] = .24
            target = self.move(end, 1.6)
            if self.reached(end, 1.6):
                # The bowl is axisymmetric: yaw it with the arm's new heading
                # while preserving the tilt relative to its upright grasp.
                _, reference_q = bowl_rim_grasp(np.r_[self.runtime.placement(), 0., 0., 0., 1.], c,
                                               heading=c.cabinet_yaw-5*math.pi/6)
                tool_q = self.env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().numpy()
                self.place_q = upright_bowl_grasp(tool_q, pose[3:], reference_q)
                self.begin('place_align', tcp)
        elif stage == 'place_align':
            # Restore the grasp tilt while clear of the drawer sides.
            # Only world yaw differs, so the bowl returns upright.
            end = self.runtime.placement()-self.held_offset; end[2] = .11
            target = self.move(end, 1.4)
            q = interpolate_quaternion(self.start_q, self.place_q, quintic(elapsed/1.4))
            if self.reached(end, 1.4):
                self.begin('place', tcp)
        elif stage == 'place':
            end = self.runtime.placement()-self.held_offset
            target = self.move(end, 1.5)
            # Release once the bowl is aligned and close enough to fall into
            # the tray; do not keep forcing the fingers down to the floor.
            if elapsed >= 1.5 and bowl_ready_to_release(pose, vel[0], opening, c):
                self.begin('cup_release', tcp)
        elif stage == 'cup_release':
            target = self.start.copy()
            if elapsed >= .5 and self.runtime.metrics(True)['placed']:
                self.begin('cup_withdraw', tcp)
        elif stage == 'cup_withdraw':
            end = self.start.copy(); end[2] = .12
            target = self.move(end, 1.)
            if self.reached(end, 1.):
                self.begin('close_approach', tcp)
        elif stage == 'close_approach':
            end = handle+.025*pull-np.array([0., 0., .008])
            u = quintic(elapsed/1.6)
            # Clear the front panel on one rounded approach, then push with
            # closed fingertips. No handle grasp or quarter-turn is needed.
            via = np.r_[end[:2], self.start[2]+.05]
            target = (1-u)**2*self.start+2*u*(1-u)*via+u*u*end
            q = interpolate_quaternion(self.start_q, handle_q, u)
            if self.reached(end, 1.6):
                self.push_start = tcp.copy()
                self.push_distance = opening+.018
                self.begin('push', tcp)
        elif stage == 'push':
            end = self.push_start-self.push_distance*pull
            target = self.move(end, 2.6)
            if elapsed >= 1. and opening < c.closed_threshold:
                self.begin('retreat', tcp)
            elif elapsed > 6.:
                self.failure = 'drawer_not_closed'
        elif stage == 'retreat':
            end = self.start+.035*pull+np.array([0., 0., .15])
            target = self.move(end, 1.)
            if self.reached(end, 1.):
                self.begin('done', tcp)
        else:
            target = self.start.copy()
        if stage in ('pull', 'push'):
            q = handle_q
        elif stage == 'cup_close':
            q = self.rim_q
        elif stage in ('place', 'cup_release', 'cup_withdraw'):
            q = self.place_q
        elif stage not in ('approach', 'handle_descend', 'cup_approach', 'cup_lift', 'place_align', 'close_approach'):
            q = tabletop_orientation(target)
        if stage in ('handle_descend', 'close_approach') and elapsed > 4.:
            self.failure = 'handle_approach_blocked'
        self.target_quaternion = np.asarray(q).copy()
        return self.command(target, q, grip)
