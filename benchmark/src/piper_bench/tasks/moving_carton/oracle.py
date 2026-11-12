"""Privileged static-source expert that tracks the basket during placement/release."""
from __future__ import annotations
import numpy as np
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.robots.piper.motion import line_motion
from piper_bench.robots.piper.motion import bezier_motion
from piper_bench.robots.piper.motion import quintic


def cube_face_grasp(pose, reference_yaw, pitch=2.704, *, alternate_faces=False):
    """Align the local-Y closing axis with a cube face normal (xyzw).

    Choose the nearest heading, or the nearest heading on the other face pair
    when alternate_faces is set. Piper's TCP is
    at link6 Z=140 mm; the distal finger contact region is centred near 128 mm.
    Align that region, rather than the TCP/finger tips, with the cube centre.
    """
    cube_rotation = rotation_xyzw(pose[3:7])
    cube_yaw = np.arctan2(cube_rotation[1, 0], cube_rotation[0, 0])
    headings = cube_yaw + np.arange(4)*np.pi/2
    differences = np.arctan2(np.sin(headings-reference_yaw), np.cos(headings-reference_yaw))
    nearest = int(np.argmin(np.abs(differences)))
    if alternate_faces:
        # Adjacent quarter-turn headings close on the other two side faces.
        candidates = np.array([(nearest-1) % 4, (nearest+1) % 4])
        nearest = candidates[np.argmin(np.abs(differences[candidates]))]
    yaw = reference_yaw + differences[nearest]
    sy, cy = np.sin(yaw/2), np.cos(yaw/2)
    sp, cp = np.sin(pitch/2), np.cos(pitch/2)
    quaternion = np.array([-sy*sp, cy*sp, sy*cp, cy*cp])
    target = np.asarray(pose[:3], dtype=float) + rotation_xyzw(quaternion) @ np.array([0., 0., .012])
    return target, quaternion


def smooth_face_yaw(start_yaw, grasp_quaternion, elapsed, duration):
    """Turn along the shortest yaw arc, with zero endpoint speed/acceleration."""
    rotation = rotation_xyzw(grasp_quaternion)
    end_yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    delta = np.arctan2(np.sin(end_yaw-start_yaw), np.cos(end_yaw-start_yaw))
    return float(start_yaw + quintic(elapsed/duration)*delta)


def cube_face_error(pose, tool_quaternion):
    """Angle between the measured closing axis and a cube side's normal."""
    normals = rotation_xyzw(pose[3:7])[:, :2]
    closing_axis = rotation_xyzw(tool_quaternion)[:, 1]
    return float(np.arccos(np.clip(np.max(np.abs(normals.T @ closing_axis)), 0., 1.)))


def placement_checks(poses, velocities, basket_state, index, cfg):
    """Measured object clearance and tracking, never just TCP target arrival."""
    basket, basket_velocity = basket_state
    half = cfg.object_size/2
    bottom = poses[index,2]-np.abs(rotation_xyzw(poses[index,3:7])[2]).sum()*half
    support_z = basket[2]+cfg.wall_thickness
    support_xy = basket[:2]
    if index == 1:
        support_z = poses[0,2]+np.abs(rotation_xyzw(poses[0,3:7])[2]).sum()*half
        support_xy = poses[0,:2]
    gap = float(bottom-support_z)
    xy = float(np.linalg.norm(poses[index,:2]-support_xy))
    speed = float(np.linalg.norm(velocities[index,:3]-basket_velocity[:3]))
    return dict(ready=bool(-.0015<=gap<=.0025 and xy<=.006),
                surface_clearance_m=gap, xy_error_m=xy, relative_speed_m_s=speed)


def placement_window_stable(relative_positions):
    """Four intervals of measured relative motion reject drift, not solver jitter."""
    if len(relative_positions) != 5:
        return False
    values=np.asarray(relative_positions)
    span=float(np.linalg.norm(np.ptp(values,axis=0)))
    speed=float(np.linalg.norm(values[-1]-values[0])/(4/CONTROL_HZ))
    return bool(span<=.002 and speed<=.015)


class MovingBasketOracle:
    PROFILE='moving_carton_stack_v11_direct_transfer'

    def __init__(self,runtime):
        from collections import deque
        self.runtime,self.env,self.cfg=runtime,runtime.env,runtime.cfg
        if self.cfg.controlled_placement:
            self.PROFILE='moving_carton_stack_v11_lowwall_direct_transfer'
        self.index=0;self._stage='approach';self.started=None
        self.start=None;self.start_velocity=np.zeros(3)
        self.target=None;self.reference_velocity=np.zeros(3)
        self.orientation_error=np.inf
        self.grasp_lifted=[False,False];self.retries=[0,0]
        self.held_offset=np.zeros(3)
        self.release_orientation=None
        self.basket_anchor=None
        self.basket_anchor_velocity=np.zeros(3)
        self.grasp_quaternion=None
        self.grasp_target=None
        self.transfer_yaw=None
        self.failure=None
        self.commanded_yaw=None
        self.start_yaw=0.
        self.release_positions=deque(maxlen=5)
        self.release_checks=None

    @property
    def stage(self):return self._stage

    def begin(self,stage,tcp):
        self._stage=stage;self.started=self.runtime.time
        # Preserve the outgoing reference, rather than resetting the turn to
        # the lagging measured wrist pose at each stage boundary.
        if self.commanded_yaw is not None:self.start_yaw=self.commanded_yaw
        if stage in ('wait', 'recover', 'return', 'align') and self.index < 2:
            pose = self.runtime.state()[0][self.index]
            self.grasp_target, self.grasp_quaternion = cube_face_grasp(
                pose, np.arctan2(pose[1], pose[0]), alternate_faces=self.index == 1)
        if stage in ('release','withdraw'):
            basket_pose,basket_velocity=self.runtime.basket_state()
            self.basket_anchor=basket_pose[:3].copy()
            self.basket_anchor_velocity=basket_velocity[:3].copy()
        if stage in ('place','release'):
            self.release_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        if stage == 'transfer':
            # Aim at the placement site, not the intermediate TCP position.
            # The latter first turns toward the source, then reverses in transit.
            poses=self.runtime.state()[0]
            destination=self.destination(poses,tcp,compensate=False)-self.held_offset
            self.transfer_yaw=float(np.arctan2(destination[1],destination[0]))
        if stage == 'place':
            self.release_positions.clear()
        self.start=tcp.copy() if self.target is None else self.target.copy()
        self.start_velocity=self.reference_velocity.copy()
        speed=np.linalg.norm(self.start_velocity)
        if speed>.30:self.start_velocity*=.30/speed

    def destination(self,poses,tcp,compensate=True):
        # Follow the actual lower cube for stacking, including its settled offset in the carton.
        basket,_=self.runtime.basket_target(self.runtime.time+.10)
        place=basket+np.array([0.,0.,self.cfg.wall_thickness+self.cfg.basket_height+.025])
        if self.index==1:
            current_basket=self.runtime.basket_state()[0]
            place[:2]=poses[0,:2]+basket[:2]-current_basket[:2]
            place[2]=poses[0,2]+self.cfg.object_size+.004
        if self.cfg.controlled_placement:
            half_height=np.abs(rotation_xyzw(poses[self.index,3:7])[2]).sum()*self.cfg.object_size/2
            support_z=self.runtime.basket_state()[0][2]+self.cfg.wall_thickness
            if self.index==1:
                support_z=poses[0,2]+np.abs(rotation_xyzw(poses[0,3:7])[2]).sum()*self.cfg.object_size/2
            place[2]=support_z+half_height+.001
        if compensate:
            offset=np.clip(poses[self.index,:3]-tcp,-.03,.03)
            self.held_offset=.8*self.held_offset+.2*offset
            place-=self.held_offset
        return place

    def action(self):
        import torch
        poses,velocities,tcp=self.runtime.state()
        if self.started is None:self.begin('approach',tcp)
        elapsed=self.runtime.time-self.started
        hover=np.array([self.cfg.belt_x,self.cfg.pick_y,.15])
        if self.index<2:
            hover[:2]=poses[self.index,:2]
        source=poses[min(self.index,1),:3].copy()
        if self.grasp_quaternion is not None and self.index < 2:
            source += rotation_xyzw(self.grasp_quaternion) @ np.array([0., 0., .012])
            hover[:2]=source[:2]
        grip=-1. if self.stage in ('close','carry','transfer','place') else 1.
        if self.stage=='approach':
            entry=np.array([.30,0.,.18])
            target=line_motion(self.start,entry,elapsed,.8,self.start_velocity)
            if elapsed>=.8 and np.linalg.norm(tcp-entry)<.025 and self.orientation_error<.25:
                self.begin('wait',tcp)
        elif self.stage=='wait':
            target=line_motion(self.start,hover,elapsed,.9,self.start_velocity)
            if (self.runtime.active[self.index] and elapsed>=.9
                and np.linalg.norm(tcp-hover)<.03):
                self.begin('align',tcp)
        elif self.stage=='recover':
            high=source+np.array([0.,0.,.13])
            target=line_motion(self.start,high,elapsed,.6,self.start_velocity)
            if elapsed>=.6 and (np.linalg.norm(tcp-high)<.025 or elapsed>=1.2):self.begin('align',tcp)
        elif self.stage=='align':
            high=source.copy();high[2]=.15
            target=line_motion(self.start,high,elapsed,.2,self.start_velocity)
            if elapsed>=.2 and np.linalg.norm(tcp-high)<.003 and self.orientation_error<.04:
                self.begin('descend',tcp)
            elif elapsed>4.:
                self.failure='hover_alignment_timeout'
        elif self.stage=='descend':
            target=line_motion(self.start,source,elapsed,.8,self.start_velocity)
            actual_q=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy()
            face_error=cube_face_error(poses[self.index],actual_q)
            if elapsed>=.8 and np.linalg.norm(tcp-source)<.003 and self.orientation_error<.04 and face_error<.04:
                self.grasp_target=source.copy()
                self.begin('close',tcp)
            elif elapsed>3.:
                self.failure='grasp_alignment_timeout'
        elif self.stage=='close':
            target=self.grasp_target.copy()
            if elapsed>=.30:
                self.held_offset=np.clip(poses[self.index,:3]-tcp,-.03,.03)
                self.begin('carry',tcp)
        elif self.stage=='carry':
            self.grasp_lifted[self.index]|=bool(poses[self.index,2]>self.cfg.belt_top+.09)
            # Clear the 3 cm source cubes without a redundant high lift.
            # Continue toward the carton at 19 cm, above its rim and contents.
            high=self.start.copy();high[2]=.17
            target=line_motion(self.start,high,elapsed,.55,self.start_velocity)
            if elapsed>=.55 and not self.grasp_lifted[self.index] and tcp[2]>.13 and self.retries[self.index]<2:
                self.retries[self.index]+=1;self.begin('recover',tcp)
            elif elapsed>=.55 and np.linalg.norm(tcp-high)<.02:self.begin('transfer',tcp)
        elif self.stage=='transfer':
            # Crossing the source at basket height can sweep the other object away.
            # Complete the lift before translating; descend only over the basket.
            above=self.destination(poses,tcp);above[2]=.19
            target=line_motion(self.start,above,elapsed,.70,self.start_velocity)
            if elapsed>=.70 and np.linalg.norm(tcp-above)<.02:self.begin('place',tcp)
        elif self.stage=='place':
            place=self.destination(poses,tcp)
            duration=1.1 if self.cfg.controlled_placement else .70
            target=line_motion(self.start,place,elapsed,duration,self.start_velocity)
            self.grasp_lifted[self.index]|=bool(poses[self.index,2]>self.cfg.belt_top+.09)
            # One descent, then release. Do not chase tool yaw or the last
            # millimetres of cube alignment while the carton keeps moving.
            # The unchanged task predicate still checks the settled stack.
            if self.cfg.controlled_placement:
                self.release_checks=placement_checks(poses, velocities, self.runtime.basket_state(),
                                                     self.index, self.cfg)
                ready=elapsed>=duration and self.release_checks['ready']
                if ready:
                    self.release_positions.append((poses[self.index,:3]-self.runtime.basket_state()[0][:3]).copy())
                else:
                    self.release_positions.clear()
                stable=placement_window_stable(self.release_positions)
                self.release_checks.update(ready=bool(ready and stable), relative_pose_stable=stable,
                                           observed_valid_samples=len(self.release_positions))
                if self.release_checks['ready']:
                    self.begin('release',tcp)
                elif elapsed>4.:
                    self.failure='controlled_placement_timeout'
            elif elapsed>=.70 and np.linalg.norm(tcp-place)<.012:
                self.begin('release',tcp)
        elif self.stage=='release':
            target=self.start.copy()+self.runtime.basket_state()[0][:3]-self.basket_anchor
            if elapsed>=(.35 if self.cfg.controlled_placement else .20):self.begin('withdraw',tcp)
        elif self.stage=='withdraw':
            high=self.start.copy();high[2]=.19
            target=line_motion(self.start,high,elapsed,.45,self.start_velocity-self.basket_anchor_velocity)
            target[:2]+=self.runtime.basket_state()[0][:2]-self.basket_anchor[:2]
            if elapsed>=.45 and tcp[2]>.17:
                self.index+=1
                self.begin('return' if self.index<2 else 'done',tcp)
        elif self.stage=='done':
            # The hand is already >10 cm above the settled stack. Hold here
            # for the success window instead of another lateral return/turn.
            target=self.start.copy()
        else:
            end=hover if self.index<2 else np.array([self.cfg.basket_x,-.08,.19])
            first=self.start.copy();first[2]=max(.19,first[2])
            second=end.copy();second[2]=.19
            duration=1.2 if self.index<2 else .70
            target=bezier_motion(self.start,first,second,end,elapsed,duration,self.start_velocity)
            if self.index<2 and elapsed>=duration and np.linalg.norm(tcp-hover)<.030:
                self.begin('align',tcp)
        if self.target is not None:self.reference_velocity=(target-self.target)*CONTROL_HZ
        self.target=target.copy()
        delta=target-tcp;distance=np.linalg.norm(delta)
        if distance>.03:delta*=.03/distance
        from isaaclab.utils.math import compute_pose_error
        from isaaclab.utils.math import quat_apply_inverse
        frame=self.env.scene['ee_frame'].data
        current_q,current_p=frame.target_quat_w[:,0],frame.target_pos_w[:,0]
        yaw=np.arctan2(target[1],target[0]);sy,cy=np.sin(yaw/2),np.cos(yaw/2)
        sp,cp=np.sin(2.704/2),np.cos(2.704/2)
        if self.grasp_quaternion is not None and (self.stage in ('wait','recover','align','descend','close','carry')
                                                 or (self.stage=='return' and self.index<2)):
            grasp_rotation=rotation_xyzw(self.grasp_quaternion)
            grasp_yaw=np.arctan2(-grasp_rotation[0,1],grasp_rotation[1,1])
            # High lifts need the reachable radial heading. On approach,
            # turn during translation, finishing at the face-aligned hover;
            # do not wait until arrival to command the remaining wrist turn.
            radial_weight=0. if self.stage in ('align','descend','close','carry') else quintic((tcp[2]-.12)/.05)
            yaw=grasp_yaw+radial_weight*np.arctan2(np.sin(yaw-grasp_yaw),np.cos(yaw-grasp_yaw))
            if self.stage in ('wait','return'):
                duration=.9 if self.stage=='wait' else 1.2
                yaw=smooth_face_yaw(self.start_yaw,self.grasp_quaternion,
                                    self.runtime.time-self.started,duration)
            sy,cy=np.sin(yaw/2),np.cos(yaw/2)
        if self.stage == 'transfer':
            # Continue from the held grasp heading along one smooth arc.
            delta_yaw=np.arctan2(np.sin(self.transfer_yaw-self.start_yaw),
                                 np.cos(self.transfer_yaw-self.start_yaw))
            yaw=self.start_yaw+quintic((self.runtime.time-self.started)/.70)*delta_yaw
            sy,cy=np.sin(yaw/2),np.cos(yaw/2)
        self.commanded_yaw=yaw
        desired_q=torch.tensor([[-sy*sp,cy*sp,sy*cp,cy*cp]],dtype=torch.float32,device=self.env.device)
        if self.release_orientation is not None and (self.stage in ('place','release','withdraw','done') or (self.stage=='return' and tcp[2]<.13)):
            desired_q=torch.as_tensor(self.release_orientation[None],dtype=torch.float32,device=self.env.device)
            release_rotation=rotation_xyzw(self.release_orientation)
            self.commanded_yaw=np.arctan2(-release_rotation[0,1],release_rotation[1,1])
        _,error=compute_pose_error(current_p,current_q,current_p,desired_q,rot_error_type='axis_angle')
        angular=error[0].detach().cpu().numpy();norm=np.linalg.norm(angular);self.orientation_error=norm
        if norm>.06:angular*=.06/norm
        base_q=self.env.scene['robot'].data.root_quat_w.torch
        translation=quat_apply_inverse(base_q,torch.as_tensor(delta[None],dtype=torch.float32,device=self.env.device))[0]
        rotation=quat_apply_inverse(base_q,torch.as_tensor(angular[None],dtype=torch.float32,device=self.env.device))[0]
        return torch.cat((2*translation,2*rotation,torch.tensor([grip],device=self.env.device))).unsqueeze(0)
