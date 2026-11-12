"""Short contact-only pick, yaw alignment, insertion and release sequence."""
import math
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.common import CartesianOracle
from piper_bench.tasks.common import approach_pose
from piper_bench.tasks.common import quintic
from piper_bench.tasks.common import yaw_pitch_quaternion
from piper_bench.tasks.common import interpolate_quaternion
from piper_bench.tasks.common import tabletop_orientation
from piper_bench.tasks.square_nut.scene import assembly_metrics


def handle_grasp(pose, cfg):
    rotation = rotation_xyzw(pose[3:])
    target = pose[:3] + rotation @ np.array([cfg.handle_x, 0., 0.])
    # Existing Piper TCP is 4.2 mm below the fingertip centres in a top grasp.
    target[2] -= .0042
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return target, yaw_pitch_quaternion(yaw, math.pi)


def aligned_grasp(pose, tcp, tool_quat, cfg):
    """Preserve the measured grasp transform while aligning the nut to the peg."""
    from scipy.spatial.transform import Rotation
    correction = Rotation.from_euler('z', cfg.peg_yaw) * Rotation.from_quat(pose[3:]).inv()
    return correction.apply(tcp-pose[:3]), (correction * Rotation.from_quat(tool_quat)).as_quat()


class SquareNutOracle(CartesianOracle):
    PROFILE = 'square_nut_handle_pick_insert_v4'

    def __init__(self, runtime):
        super().__init__(runtime)
        self.seated_steps = 0

    def action(self):
        poses, velocity, tcp = self.runtime.state()
        pose, c = poses[0], self.cfg
        if self.started is None:
            self.begin('approach', tcp)
        stage, elapsed = self.stage, self.runtime.time-self.started
        grasp, grasp_q = handle_grasp(pose, c)
        grip = -1. if stage in ('close', 'lift', 'transfer', 'insert') else 1.
        q = self.start_q
        if stage == 'approach':
            end = grasp.copy(); end[2] = .065
            target, q = approach_pose(self.start, self.start_q, end, quintic(elapsed/2.))
            if self.reached(end, 2.):
                self.grasp_q = grasp_q
                self.begin('descend', tcp)
        elif stage == 'descend':
            target = self.move(grasp, .9)
            q = interpolate_quaternion(self.start_q, self.grasp_q, quintic(elapsed/.9))
            if self.reached(grasp, .9, .003):
                self.begin('close', tcp)
        elif stage == 'close':
            target, q = self.start.copy(), self.grasp_q
            if elapsed >= .4:
                self.begin('lift', tcp)
        elif stage == 'lift':
            end = self.start.copy(); end[2] = .065
            target, q = self.move(end, .9), self.grasp_q
            if self.reached(end, .9):
                if pose[2] < .045:
                    self.failure = 'nut_not_lifted'
                else:
                    tool = self.env.scene['ee_frame'].data.target_quat_w[0, 0].detach().cpu().numpy()
                    self.offset, self.insert_q = aligned_grasp(pose, tcp, tool, c)
                    self.begin('transfer', tcp)
        elif stage == 'transfer':
            end = np.array([c.peg_x, c.peg_y, c.base_top+c.peg_height+c.thickness/2+.015])+self.offset
            u = quintic(elapsed/1.8)
            # Keep radius and height within the URDF-checked vertical-grasp
            # workspace, and avoid changing radial direction abruptly.
            start_yaw = math.atan2(self.start[1], self.start[0])
            end_yaw = math.atan2(end[1], end[0])
            yaw = start_yaw+u*math.atan2(math.sin(end_yaw-start_yaw), math.cos(end_yaw-start_yaw))
            radius = (1-u)*np.linalg.norm(self.start[:2])+u*np.linalg.norm(end[:2])
            z = (1-u)*self.start[2]+u*end[2]+.008*u*(1-u)
            target = np.array([radius*math.cos(yaw), radius*math.sin(yaw), z])
            q = interpolate_quaternion(self.start_q, self.insert_q, u)
            if self.reached(end, 1.8, .003):
                self.begin('insert', tcp)
        elif stage == 'insert':
            end = np.array([c.peg_x, c.peg_y, c.base_top+c.thickness/2+.0003])+self.offset
            target, q = self.move(end, 1.5), self.insert_q
            metrics = assembly_metrics(pose, velocity[0], tcp, False, c)
            # Closed fingers induce solver velocity impulses even after the
            # supported pose stops moving. Release after sustained geometric
            # seating; final success still requires free, low-velocity rest.
            self.seated_steps = self.seated_steps+1 if metrics['seated'] else 0
            if elapsed >= 1.5 and self.seated_steps >= 4:
                self.begin('release', tcp)
        elif stage == 'release':
            target, q = self.start.copy(), self.insert_q
            if elapsed >= .4:
                self.begin('withdraw', tcp)
        elif stage == 'withdraw':
            end = self.start+np.array([0., 0., .12])
            target = self.move(end, 1.)
            q = interpolate_quaternion(self.start_q, tabletop_orientation(end), quintic(elapsed/1.))
            if self.reached(end, 1.):
                self.begin('done', tcp)
        else:
            target = self.start.copy()
        if elapsed > 6. and stage != 'done':
            self.failure = f'{stage}_blocked'
        return self.command(target, q, grip)
