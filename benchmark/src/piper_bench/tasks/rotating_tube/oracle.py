"""Physical grasp, direct transfer, and release halfway into a moving bore."""
import math
import numpy as np
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.robots.piper.motion import line_motion
from piper_bench.robots.piper.motion import quintic


def yaw_pitch_quaternion(yaw,pitch):
    sy,cy=math.sin(yaw/2),math.cos(yaw/2)
    sp,cp=math.sin(pitch/2),math.cos(pitch/2)
    return np.array([-sy*sp,cy*sp,sy*cp,cy*cp])


def interpolate_quaternion(start,end,u):
    """Shortest continuous rotation; handle equivalent quaternion signs."""
    a=np.asarray(start,dtype=float);a=a/np.linalg.norm(a)
    b=np.asarray(end,dtype=float);b=b/np.linalg.norm(b)
    dot=float(a@b)
    if dot<0:b=-b;dot=-dot
    if dot>.9995:
        q=(1-u)*a+u*b
        return q/np.linalg.norm(q)
    theta=math.acos(np.clip(dot,-1,1))
    return (math.sin((1-u)*theta)*a+math.sin(u*theta)*b)/math.sin(theta)


class TubeOracle:
    PROFILE='physical_rotating_tube_deep_body_grasp_v7'
    PICK_Z_OFFSET=-.0075
    HOVER_Z=.015
    LIFT_Z=.080
    STAGING=(.34,.10,.16)
    PICK_PITCH=math.pi
    INSERT_PITCH=math.radians(105)
    INSERT_SECONDS=.9
    APPROACH_SECONDS=2.0
    UPRIGHT_SECONDS=1.70

    @staticmethod
    def approach_pose(start,start_quat,end,u,end_pitch=math.pi):
        # Match tool yaw to the radial position while leaving the original
        # straight wrist. Cartesian SLERP cuts across incompatible IK branches.
        start_yaw=math.atan2(start[1],start[0]);end_yaw=math.atan2(end[1],end[0])
        yaw=start_yaw+u*math.atan2(math.sin(end_yaw-start_yaw),math.cos(end_yaw-start_yaw))
        radius=(1-u)*np.linalg.norm(start[:2])+u*np.linalg.norm(end[:2])
        target=np.array([radius*math.cos(yaw),radius*math.sin(yaw),(1-u)*start[2]+u*end[2]])
        start_pitch=math.acos(np.clip(rotation_xyzw(start_quat)[2,2],-1,1))
        return target,yaw_pitch_quaternion(yaw,(1-u)*start_pitch+u*end_pitch)

    @staticmethod
    def release_depth(cfg):
        return min(.5*cfg.tube_length,cfg.plate_top-cfg.base_thickness-.005)

    @staticmethod
    def insertion_target(hole,bottom,axis,cfg):
        # Centre the tilted tube through the middle of the guide plate.
        # Its rounded bottom and grasp offset determine the required TCP height.
        z=bottom+cfg.tube_radius+(cfg.tube_length/2-cfg.tube_radius+cfg.grasp_offset)*axis[2]
        mid=cfg.plate_top-cfg.plate_thickness/2
        xy=hole[:2]+(z-mid)*axis[:2]/max(axis[2],.1)
        return np.r_[xy,z]

    @classmethod
    def half_inserted(cls,pose,cfg,hole):
        # Check actual geometry, not elapsed time or the commanded TCP pose.
        from piper_bench.tasks.rotating_tube.scene import insertion_metrics
        metrics=insertion_metrics(pose,np.zeros(6),np.zeros(3),cfg,
                                  (hole,np.zeros(3)),False)
        extent=metrics['bore_outer_radius_m']
        return bool(metrics['insertion_depth_m']>=cls.release_depth(cfg)-.002
                    and extent is not None and extent<=cfg.hole_radius+.00005)

    def __init__(self,runtime):
        self.runtime,self.env,self.cfg=runtime,runtime.env,runtime.cfg
        self.stage='approach';self.started=None;self.start=None
        self.target=None;self.reference_velocity=np.zeros(3);self.start_velocity=np.zeros(3)
        self.held_offset=np.zeros(3);self.grasp_lifted=False;self.retries=0
        self.orientation_error=np.inf
        self.release_orientation=None;self.release_offset=np.zeros(2)
        self.insert_bottom_start=None

    def begin(self,stage,tcp):
        self.stage=stage;self.started=self.runtime.time
        if stage=='transfer':
            # Use the positive-Y approach arc beside the source tube. The
            # tilted grasp permits descent here without following a long arc.
            arrival=self.runtime.time+1.4
            angles=self.runtime.phase+self.cfg.omega*arrival+np.arange(self.cfg.holes)*2*math.pi/self.cfg.holes
            errors=np.arctan2(np.sin(angles-1.7),np.cos(angles-1.7))
            self.runtime.target_index=int(np.argmin(np.abs(errors)))
        if stage=='insert':
            pose=self.runtime.state()[0][0]
            axis=rotation_xyzw(pose[3:])[:,2]
            self.insert_bottom_start=pose[2]-axis[2]*(self.cfg.tube_length/2-self.cfg.tube_radius)-self.cfg.tube_radius
        if stage=='release':
            previous=self.target if self.target is not None else tcp
            self.release_offset=previous[:2]-self.runtime.hole_target(.10)[0][:2]
            self.release_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        if stage=='approach':
            self.approach_orientation=self.env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().numpy().copy()
        if stage=='descend':self.grasp_ready_steps=0
        self.start=tcp.copy() if self.target is None else self.target.copy()
        self.start_velocity=self.reference_velocity.copy()
        speed=np.linalg.norm(self.start_velocity)
        if speed>.25:self.start_velocity*=.25/speed

    def action(self):
        import torch
        from isaaclab.utils.math import compute_pose_error
        from isaaclab.utils.math import quat_apply_inverse
        poses,vel,tcp=self.runtime.state();pose=poses[0];c=self.cfg
        if self.started is None:self.begin('approach',tcp)
        stage=self.stage;elapsed=self.runtime.time-self.started
        axis=rotation_xyzw(pose[3:])[:,2]
        grasp=pose[:3]+c.grasp_offset*axis
        # Pinch the white body 15 mm below the cap grasp. The TCP offset
        # places the fingertips around the tube while clearing the tabletop.
        source=grasp+np.array([0.,0.,self.PICK_Z_OFFSET])
        pitch=self.PICK_PITCH;yaw=self.runtime.tube_yaw
        grip=-1. if stage in ('close','lift','upright','transfer','align','insert') else 1.
        high=source.copy();high[2]=self.HOVER_Z
        if stage=='approach':
            target,_=self.approach_pose(self.start,self.approach_orientation,high,quintic(elapsed/self.APPROACH_SECONDS),self.PICK_PITCH)
            if elapsed>=self.APPROACH_SECONDS and np.linalg.norm(tcp-high)<.006 and self.orientation_error<.025:
                self.begin('descend',tcp)
        elif stage=='descend':
            target=line_motion(self.start,source,elapsed,.65,self.start_velocity)
            ready=(elapsed>=.65 and np.linalg.norm(tcp[:2]-source[:2])<.002
                   and abs(tcp[2]-source[2])<.0015 and self.orientation_error<.025)
            self.grasp_ready_steps=self.grasp_ready_steps+1 if ready else 0
            if self.grasp_ready_steps>=3:
                self.begin('close',tcp)
        elif stage=='close':
            target=self.start.copy()
            if elapsed>=.35:self.begin('lift',tcp)
        elif stage=='lift':
            high=self.start.copy();high[2]=self.LIFT_Z
            target=line_motion(self.start,high,elapsed,.65,self.start_velocity)
            self.grasp_lifted|=bool(pose[2]>.045)
            if elapsed>=.65 and tcp[2]>self.LIFT_Z-.006:
                if self.grasp_lifted:
                    self.begin('upright',tcp)
                elif self.retries<2:
                    self.retries+=1;self.begin('approach',tcp)
        elif stage=='upright':
            staging=np.array(self.STAGING);u=quintic(elapsed/self.UPRIGHT_SECONDS)
            target=line_motion(self.start,staging,elapsed,self.UPRIGHT_SECONDS,self.start_velocity)
            pitch=(1-u)*self.PICK_PITCH+u*self.INSERT_PITCH
            yaw=math.atan2(target[1],target[0])
            if elapsed>=self.UPRIGHT_SECONDS and axis[2]>.94 and np.linalg.norm(tcp-staging)<.01 and self.orientation_error<.035:
                self.held_offset=grasp-tcp;self.begin('transfer',tcp)
        else:
            pitch=self.INSERT_PITCH
            hole,_=self.runtime.hole_target(.10)
            yaw=math.atan2(hole[1],hole[0])
            if stage in ('transfer','align','insert'):
                self.held_offset=.8*self.held_offset+.2*(grasp-tcp)
            bottom=c.plate_top+.020
            desired=self.insertion_target(hole,bottom,axis,c)-self.held_offset
            if stage=='transfer':
                desired[2]=.16
                target=line_motion(self.start,desired,elapsed,.85,self.start_velocity)
                if elapsed>=.85 and np.linalg.norm(tcp-desired)<.01:self.begin('align',tcp)
            elif stage=='align':
                target=line_motion(self.start,desired,elapsed,.55,self.start_velocity)
                actual_hole,_=self.runtime.hole_target()
                from piper_bench.tasks.rotating_tube.scene import insertion_metrics
                m=insertion_metrics(pose,vel[0],tcp,c,(actual_hole,np.zeros(3)),False)
                aligned=m['bore_outer_radius_m'] is not None and m['bore_outer_radius_m']<c.hole_radius-.0001
                if elapsed>=.55 and aligned and np.linalg.norm(tcp-desired)<.004:
                    self.begin('insert',tcp)
            elif stage=='insert':
                u=quintic(elapsed/self.INSERT_SECONDS)
                bottom=(1-u)*self.insert_bottom_start+u*(c.plate_top-self.release_depth(c))
                target=self.insertion_target(hole,bottom,axis,c)-self.held_offset
                actual_hole,_=self.runtime.hole_target()
                if self.half_inserted(pose,c,actual_hole):
                    self.begin('release',tcp)
            elif stage=='release':
                # Open only after measured insertion reaches half the tube.
                # Once the fingers open, lift clear of the moving rack instead
                # of following its near edge at the wrist extension limit.
                height=self.start[2]+.025*quintic((elapsed-1/CONTROL_HZ)/.20)
                target=np.r_[hole[:2]+self.release_offset,height]
                if elapsed>=.40:self.begin('withdraw',tcp)
            elif stage=='withdraw':
                end=np.array(self.STAGING);end[2]=.20
                target=line_motion(self.start,end,elapsed,.8,self.start_velocity)
                if elapsed>=.8 and np.linalg.norm(tcp-end)<.01:self.begin('done',tcp)
            else:target=self.start.copy()
        if self.target is not None:self.reference_velocity=(target-self.target)*CONTROL_HZ
        self.target=target.copy()
        delta=target-tcp;distance=np.linalg.norm(delta)
        if distance>.025:delta*=.025/distance
        frame=self.env.scene['ee_frame'].data
        current_q,current_p=frame.target_quat_w[:,0],frame.target_pos_w[:,0]
        orientation=yaw_pitch_quaternion(yaw,pitch)
        if stage=='approach':
            _,orientation=self.approach_pose(self.start,self.approach_orientation,high,quintic(elapsed/self.APPROACH_SECONDS),self.PICK_PITCH)
        if stage in ('release','withdraw','done') and self.release_orientation is not None:
            orientation=self.release_orientation
        desired_q=torch.tensor(orientation[None],dtype=torch.float32,device=self.env.device)
        _,error=compute_pose_error(current_p,current_q,current_p,desired_q,rot_error_type='axis_angle')
        angular=error[0].detach().cpu().numpy();norm=np.linalg.norm(angular);self.orientation_error=norm
        # The previous 0.06 rad target-error cap saturated the damped wrist
        # drive at about 0.5 rad/s: translation finished while rotation lagged.
        # Allow the smooth reorientation reference to be tracked; retain the
        # small correction bound during grasp, alignment and insertion.
        angular_limit=.20 if stage in ('approach','upright') else .06
        if norm>angular_limit:angular*=angular_limit/norm
        base_q=self.env.scene['robot'].data.root_quat_w.torch
        translation=quat_apply_inverse(base_q,torch.as_tensor(delta[None],dtype=torch.float32,device=self.env.device))[0]
        rotation=quat_apply_inverse(base_q,torch.as_tensor(angular[None],dtype=torch.float32,device=self.env.device))[0]
        return torch.cat((2*translation,2*rotation,torch.tensor([grip],device=self.env.device))).unsqueeze(0)
