"""Conveyor oracle adapted to circular interception; placement and IK are retained."""
from __future__ import annotations
import numpy as np
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.common import upright_grasp
from piper_bench.tasks.common import interpolate_quaternion
from piper_bench.tasks.common import yaw_pitch_quaternion
from piper_bench.tasks.turntable_cups.turntable import predict_position
from piper_bench.robots.piper.motion import line_motion
from piper_bench.robots.piper.motion import quintic


def cup_ready_to_release(poses, index, cfg):
    upright=rotation_xyzw(poses[index,3:])[2,2]>np.cos(np.deg2rad(2.))
    if index==0:
        return bool(upright)
    return bool(upright and np.linalg.norm(poses[1,:2]-poses[0,:2])<.0015
        and abs(poses[1,2]-poses[0,2]-cfg.cup_height*.45)<.003)


def cup_grasp_z_offset(cfg):
    """Place the distal finger pads below the shallow bowl rim."""
    return cfg.cup_height*(-.15 if cfg.cup_height < 2*cfg.cup_top_radius else .25)


def observation_orientation(target):
    """Leave wrist margin while looking down from the raised empty-hand pose."""
    pitch=np.pi-.25*quintic((target[2]-.10)/.05)
    return yaw_pitch_quaternion(np.arctan2(target[1],target[0]),pitch)


def empty_return_pose(start, start_quat, end, elapsed, start_velocity):
    """Turn toward the next grasp during the empty transit, with continuous endpoints."""
    duration=.85
    target=line_motion(start,end,elapsed,duration,start_velocity)
    orientation=observation_orientation(end)
    return target,interpolate_quaternion(start_quat,orientation,quintic(elapsed/duration))


class CupOracle:
    PROFILE='turntable_bowl_deep_grasp_observation_v9'
    CARRY_Z=.18
    SECOND_HOVER_Z=.15
    OBSERVE_SECONDS=.20

    def __init__(self,runtime):
        self.runtime,self.env,self.cfg=runtime,runtime.env,runtime.cfg
        self.index=0;self._stage='wait';self.started=None
        self.start=None;self.start_velocity=np.zeros(3)
        self.target=None;self.reference_velocity=np.zeros(3)
        self.orientation_error=np.inf
        self.grasp_lifted=[False,False];self.retries=[0,0]
        self.held_offset=np.zeros(3)
        self.release_orientation=None
        self.place_orientation=None
        self.grasp_orientation=None
        self.return_start_orientation=None

    @property
    def stage(self):return self._stage

    def begin(self,stage,tcp):
        self._stage=stage;self.started=self.runtime.time
        if stage=='descend':
            self.grasp_ready_steps=0
            self.descend_start_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        if stage=='release':
            self.release_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        elif stage=='return':
            self.return_start_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        elif stage=='carry':
            self.grasp_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        elif stage=='transfer':
            self.place_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
            self.align_start_orientation=self.place_orientation.copy()
        self.start=tcp.copy() if self.target is None else self.target.copy()
        self.start_velocity=self.reference_velocity.copy()
        speed=np.linalg.norm(self.start_velocity)
        if speed>.30:self.start_velocity*=.30/speed

    def destination(self,poses,tcp,compensate=True):
        clearance=.002 if self.cfg.cup_height < 2*self.cfg.cup_top_radius else .008
        place=np.array([self.cfg.basket_x,self.cfg.basket_y,
                        self.cfg.wall_thickness+self.cfg.cup_height/2+clearance])
        if self.index==1:
            place=poses[0,:3].copy()
            # Insert past the lower rim before opening. The former 10 mm
            # insertion frequently left a tilted cup perched on that rim.
            place[2]+=self.cfg.cup_height*.45
        if compensate:
            offset=np.clip(poses[self.index,:3]-tcp,-.05,.05)
            self.held_offset=.8*self.held_offset+.2*offset
            place-=self.held_offset
        return place

    def action(self):
        import torch
        poses,velocities,tcp=self.runtime.state()
        if self.started is None:self.begin('wait',tcp)
        elapsed=self.runtime.time-self.started
        hover=np.array([self.cfg.belt_x,self.cfg.pick_y,self.SECOND_HOVER_Z if self.index==1 else .15])
        if self.index<2:
            hover[:2]=predict_position(poses[self.index],self.cfg,.25)[:2]
        source=poses[min(self.index,1),:3].copy()
        # Follow an angular lead rather than extrapolating along the old belt axis.
        if self.index<2 and self.stage in ('descend','close','recover'):
            source=predict_position(source,self.cfg,.10)
        source[2]+=cup_grasp_z_offset(self.cfg)
        stage=self.stage
        # Finish levelling above the lower rim before the final insertion.
        align_height=.12 if self.index==0 else .113
        grip=-1. if stage in ('close','carry','observe_held','transfer','place_align','place') else 1.
        if self.stage=='wait':
            target=line_motion(self.start,hover,elapsed,.6,self.start_velocity)
            if (self.runtime.active[self.index] and elapsed>=.6
                and np.linalg.norm(tcp-hover)<.03):
                self.begin('descend',tcp)
        elif self.stage=='recover':
            high=source+np.array([0.,0.,.13])
            target=line_motion(self.start,high,elapsed,.6,self.start_velocity)
            if elapsed>=.6 and (np.linalg.norm(tcp-high)<.025 or elapsed>=1.2):self.begin('descend',tcp)
        elif self.stage=='descend':
            target=line_motion(self.start,source,elapsed,.6,self.start_velocity)
            if self.cfg.cup_height < 2*self.cfg.cup_top_radius:
                ready=(elapsed>=.6 and np.linalg.norm(tcp[:2]-source[:2])<.006
                       and abs(tcp[2]-source[2])<.003)
                self.grasp_ready_steps=self.grasp_ready_steps+1 if ready else 0
                if self.grasp_ready_steps>=3:self.begin('close',tcp)
            elif elapsed>=.6 and (np.linalg.norm(tcp-source)<.012 or elapsed>=1.25):self.begin('close',tcp)
        elif self.stage=='close':
            target=source
            if elapsed>=.40:
                self.held_offset=np.clip(poses[self.index,:3]-tcp,-.05,.05)
                self.begin('carry',tcp)
        elif self.stage=='carry':
            self.grasp_lifted[self.index]|=bool(poses[self.index,2]>self.cfg.belt_top+.09)
            high=self.start.copy();high[2]=self.CARRY_Z
            target=line_motion(self.start,high,elapsed,.8,self.start_velocity)
            if elapsed>=.8 and not self.grasp_lifted[self.index] and tcp[2]>.13 and self.retries[self.index]<2:
                self.retries[self.index]+=1;self.begin('recover',tcp)
            elif (elapsed>=.8 and np.linalg.norm(tcp-high)<.02
                  and abs(tcp[2]-high[2])<.006):self.begin('observe_held',tcp)
        elif self.stage=='observe_held':
            target=self.start.copy()
            if elapsed>=self.OBSERVE_SECONDS:self.begin('transfer',tcp)
        elif self.stage=='observe_second':
            target=line_motion(self.start,hover,elapsed,self.OBSERVE_SECONDS,self.start_velocity)
            if (elapsed>=self.OBSERVE_SECONDS and np.linalg.norm(tcp-hover)<.008
                and self.orientation_error<.04):self.begin('descend',tcp)
        elif self.stage=='transfer':
            # Crossing the disk at basket height can sweep the other object away.
            # Complete the lift before translating; keep the crossing above both rims.
            above=self.destination(poses,tcp);above[2]=align_height
            target=line_motion(self.start,above,elapsed,1.0,self.start_velocity)
            if elapsed>=1.0 and np.linalg.norm(tcp-above)<.02:self.begin('place_align',tcp)
        elif self.stage=='place_align':
            above=self.destination(poses,tcp);above[2]=align_height
            target=line_motion(self.start,above,elapsed,.20,self.start_velocity)
            upright=rotation_xyzw(poses[self.index,3:])[2,2]>np.cos(np.deg2rad(2.))
            if elapsed>=.20 and np.linalg.norm(tcp-above)<.008 and upright and self.orientation_error<.04:
                self.begin('place',tcp)
        elif self.stage=='place':
            place=self.destination(poses,tcp)
            target=line_motion(self.start,place,elapsed,1.1,self.start_velocity)
            self.grasp_lifted[self.index]|=bool(poses[self.index,2]>self.cfg.belt_top+.09)
            # Release only after the physical hand reaches the basket; motor lag varies by seed.
            if (elapsed>=1.1 and np.linalg.norm(tcp-place)<.006 and self.orientation_error<.04
                and cup_ready_to_release(poses,self.index,self.cfg)):
                self.begin('release',tcp)
        elif self.stage=='release':
            target=self.start.copy()
            if elapsed>=.40:self.begin('withdraw',tcp)
        elif self.stage=='withdraw':
            # Clear the basket and both rims without driving the wrist into its
            # pitch limit; a 23 cm vertical withdrawal pulled the hand sideways.
            high=self.start.copy();high[2]=.11
            target=line_motion(self.start,high,elapsed,.6,self.start_velocity)
            if elapsed>=.6 and np.linalg.norm(tcp-high)<.008 and self.orientation_error<.04:
                self.index+=1;self.begin('return',tcp)
        else:
            end=hover if self.index<2 else np.array([self.cfg.belt_x,0.,.12])
            target,return_orientation=empty_return_pose(self.start,self.return_start_orientation,end,
                                                       elapsed,self.start_velocity)
            if (self.index<2 and elapsed>=.85 and np.linalg.norm(tcp-hover)<.015
                and self.orientation_error<.04):
                self.begin('observe_second',tcp)
        if self.target is not None:self.reference_velocity=(target-self.target)*CONTROL_HZ
        self.target=target.copy()
        delta=target-tcp;distance=np.linalg.norm(delta)
        if distance>.03:delta*=.03/distance
        from isaaclab.utils.math import compute_pose_error
        from isaaclab.utils.math import quat_apply_inverse
        frame=self.env.scene['ee_frame'].data
        current_q,current_p=frame.target_quat_w[:,0],frame.target_pos_w[:,0]
        yaw=np.arctan2(target[1],target[0]);sy,cy=np.sin(yaw/2),np.cos(yaw/2)
        sp,cp=np.sin(np.pi/2),np.cos(np.pi/2)
        desired_q=torch.tensor([[-sy*sp,cy*sp,sy*cp,cy*cp]],dtype=torch.float32,device=self.env.device)
        if stage=='observe_second':
            desired_q=torch.as_tensor(observation_orientation(target)[None],dtype=torch.float32,device=self.env.device)
        if stage=='descend':
            down=desired_q[0].detach().cpu().numpy()
            orientation=interpolate_quaternion(self.descend_start_orientation,down,quintic(elapsed/.6))
            desired_q=torch.as_tensor(orientation[None],dtype=torch.float32,device=self.env.device)
        if stage in ('carry','observe_held'):
            desired_q=torch.as_tensor(self.grasp_orientation[None],dtype=torch.float32,device=self.env.device)
        if stage in ('transfer','place_align','place'):
            # A tapered cup can tilt within the grasp even when the tool is
            # vertical. Level the measured object, then retain that grip while
            # descending. Low-pass the correction to avoid abrupt wrist motion.
            reference=self.grasp_orientation
            level=upright_grasp(current_q[0].detach().cpu().numpy(),poses[self.index,3:],reference)
            if stage=='transfer':
                # Level during the move, retaining the grasp heading. The
                # lower transfer path stays within the wrist's pitch range.
                level=interpolate_quaternion(self.align_start_orientation,level,quintic(elapsed/1.0))
            self.place_orientation=interpolate_quaternion(self.place_orientation,level,.2)
            desired_q=torch.as_tensor(self.place_orientation[None],dtype=torch.float32,device=self.env.device)
        if stage=='return':
            desired_q=torch.as_tensor(return_orientation[None],dtype=torch.float32,device=self.env.device)
        # Use the stage that produced this command, including its last step.
        if self.release_orientation is not None and stage in ('release','withdraw'):
            desired_q=torch.as_tensor(self.release_orientation[None],dtype=torch.float32,device=self.env.device)
        self.target_quaternion=desired_q[0].detach().cpu().numpy().copy()
        _,error=compute_pose_error(current_p,current_q,current_p,desired_q,rot_error_type='axis_angle')
        angular=error[0].detach().cpu().numpy();norm=np.linalg.norm(angular);self.orientation_error=norm
        if norm>.06:angular*=.06/norm
        base_q=self.env.scene['robot'].data.root_quat_w.torch
        translation=quat_apply_inverse(base_q,torch.as_tensor(delta[None],dtype=torch.float32,device=self.env.device))[0]
        rotation=quat_apply_inverse(base_q,torch.as_tensor(angular[None],dtype=torch.float32,device=self.env.device))[0]
        return torch.cat((2*translation,2*rotation,torch.tensor([grip],device=self.env.device))).unsqueeze(0)
