"""Feedback oracle with continuous motion; objects move solely through physics."""
from __future__ import annotations
import numpy as np
from piper_conveyor import CONTROL_HZ
from piper_conveyor_motion import line_motion,bezier_motion,quintic


class ConveyorOracle:
    PROFILE='simultaneous_fast_tracking_v31_random_placement'

    def __init__(self,runtime):
        self.runtime,self.env,self.cfg=runtime,runtime.env,runtime.cfg
        self.index=0;self._stage='wait';self.started=None
        self.start=None;self.start_velocity=np.zeros(3)
        self.target=None;self.reference_velocity=np.zeros(3)
        self.orientation_error=np.inf
        self.grasp_lifted=[False,False];self.retries=[0,0]
        self.held_offset=np.zeros(2)
        self.release_orientation=None

    @property
    def stage(self):return self._stage

    def begin(self,stage,tcp):
        self._stage=stage;self.started=self.runtime.time
        if stage=='release':
            self.release_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        self.start=tcp.copy() if self.target is None else self.target.copy()
        self.start_velocity=self.reference_velocity.copy()
        speed=np.linalg.norm(self.start_velocity)
        if speed>.30:self.start_velocity*=.30/speed

    def destination(self,poses,tcp,compensate=True):
        place=np.array([self.cfg.basket_x+(-.022 if self.index==0 else .030),
                        self.cfg.basket_y+(0. if self.index==0 else .020),(.035 if self.index==0 else 0.065)])
        if compensate:
            offset=np.clip(poses[self.index,:2]-tcp[:2],-.03,.03)
            self.held_offset=.8*self.held_offset+.2*offset
            place[:2]-=self.held_offset
        return place

    def action(self):
        import torch
        poses,velocities,tcp=self.runtime.state()
        if self.started is None:self.begin('wait',tcp)
        elapsed=self.runtime.time-self.started
        hover=np.array([self.cfg.belt_x,self.cfg.pick_y,.15])
        if self.index<2:
            hover[:2]=poses[self.index,:2]
            hover[1]=max(self.cfg.pick_y,hover[1]+self.cfg.speed*.45)
        source=poses[min(self.index,1),:3].copy()
        # A short lead compensates for motor lag while the fingers track the belt.
        if self.index<2 and self.stage in ('descend','close','recover'):
            source[1]+=self.cfg.speed*.12
        grip=-1. if self.stage in ('close','carry','place') else 1.
        if self.stage=='wait':
            target=line_motion(self.start,hover,elapsed,.6,self.start_velocity)
            if (self.runtime.active[self.index] and poses[self.index,1]>=self.cfg.pick_y-.04
                and np.linalg.norm(tcp-hover)<.03):
                self.begin('descend',tcp)
        elif self.stage=='recover':
            high=source+np.array([0.,0.,.13])
            target=line_motion(self.start,high,elapsed,.6,self.start_velocity)
            if elapsed>=.6 and (np.linalg.norm(tcp-high)<.025 or elapsed>=1.2):self.begin('descend',tcp)
        elif self.stage=='descend':
            target=line_motion(self.start,source,elapsed,.6,self.start_velocity)
            if elapsed>=.6 and (np.linalg.norm(tcp-source)<.012 or elapsed>=1.25):self.begin('close',tcp)
        elif self.stage=='close':
            target=source
            if elapsed>=.40:
                self.held_offset=np.clip(poses[self.index,:2]-tcp[:2],-.03,.03)
                self.begin('carry',tcp)
        elif self.stage=='carry':
            self.grasp_lifted[self.index]|=bool(poses[self.index,2]>self.cfg.belt_top+.09)
            place=self.destination(poses,tcp)
            first=self.start.copy();first[2]=.23
            second=place.copy();second[2]=.23
            target=bezier_motion(self.start,first,second,place,elapsed,1.25,self.start_velocity)
            if elapsed>=.8 and not self.grasp_lifted[self.index] and tcp[2]>.13 and self.retries[self.index]<2:
                self.retries[self.index]+=1;self.begin('recover',tcp)
            elif elapsed>=1.25:self.begin('place',tcp)
        elif self.stage=='place':
            target=self.destination(poses,tcp)
            self.grasp_lifted[self.index]|=bool(poses[self.index,2]>self.cfg.belt_top+.09)
            # Release only after the physical hand reaches the basket; motor lag varies by seed.
            if elapsed>=.15 and np.linalg.norm(tcp-target)<(.012 if self.index==0 else .006) and (self.index==0 or self.orientation_error<.02):self.begin('release',tcp)
        elif self.stage=='release':
            target=self.start.copy()
            if elapsed>=.30:
                self.index+=1;self.begin('return',tcp)
        else:
            end=hover if self.index<2 else np.array([self.cfg.basket_x,0.,.22])
            first=self.start.copy();first[2]=max(.22,first[2])
            second=end.copy();second[2]=.22
            target=bezier_motion(self.start,first,second,end,elapsed,.70,self.start_velocity)
            if self.index<2 and elapsed>=.25 and np.linalg.norm(tcp-hover)<.030:
                self.begin('descend',tcp)
        if self.target is not None:self.reference_velocity=(target-self.target)*CONTROL_HZ
        self.target=target.copy()
        delta=target-tcp;distance=np.linalg.norm(delta)
        if distance>.03:delta*=.03/distance
        from isaaclab.utils.math import compute_pose_error,quat_apply_inverse
        frame=self.env.scene['ee_frame'].data
        current_q,current_p=frame.target_quat_w[:,0],frame.target_pos_w[:,0]
        yaw=np.arctan2(target[1],target[0]);sy,cy=np.sin(yaw/2),np.cos(yaw/2)
        sp,cp=np.sin(2.704/2),np.cos(2.704/2)
        desired_q=torch.tensor([[-sy*sp,cy*sp,sy*cp,cy*cp]],dtype=torch.float32,device=self.env.device)
        if self.release_orientation is not None and (self.stage=='release' or (self.stage=='return' and tcp[2]<.13)):
            desired_q=torch.as_tensor(self.release_orientation[None],dtype=torch.float32,device=self.env.device)
        _,error=compute_pose_error(current_p,current_q,current_p,desired_q,rot_error_type='axis_angle')
        angular=error[0].detach().cpu().numpy();norm=np.linalg.norm(angular);self.orientation_error=norm
        if norm>.06:angular*=.06/norm
        base_q=self.env.scene['robot'].data.root_quat_w.torch
        translation=quat_apply_inverse(base_q,torch.as_tensor(delta[None],dtype=torch.float32,device=self.env.device))[0]
        rotation=quat_apply_inverse(base_q,torch.as_tensor(angular[None],dtype=torch.float32,device=self.env.device))[0]
        return torch.cat((2*translation,2*rotation,torch.tensor([grip],device=self.env.device))).unsqueeze(0)
