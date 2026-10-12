"""Coordinate-indexed event memory: dense fine slots, sparse coarse/fine reads.

Persistent traces use FP32. Only actual event activity writes them. The two
fixed time constants require no learned recurrent gates. No future/noised
features enter this module. Physical frame IDs and resets belong to callers.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class EventMemoryState:
    traces: Tensor       # [B,V,2,G,G,D], FP32
    support: Tensor      # [B,V,2,G,G], FP32
    activity: Tensor     # [B,V,G,G], FP32 occupancy history
    age: Tensor          # [B,V,G,G], FP32 seconds
    seen: Tensor         # [B,V,G,G], bool

    def detach(self):
        return EventMemoryState(*(getattr(self,k).detach() for k in self.__dataclass_fields__))


@dataclass(frozen=True)
class EventMemoryRead:
    tokens: Tensor       # [B,V*(16+3R),D]
    valid: Tensor        # [B,V*(16+3R)]
    coordinates: Tensor  # [B,V*(16+3R),3], time/y/x in EWM patch-grid units
    fine_cells: Tensor   # [B,V,R], stable 4x4 coarse-cell IDs


class CompressedEventMemory(nn.Module):
    def __init__(self, dim=128, cameras=1, refined_cells=4, grid_size=8,
                 tau_fast=.08, tau_slow=.40, max_age=1.60):
        super().__init__()
        if dim <= 0 or cameras not in (1,2) or grid_size != 8:
            raise ValueError('memory requires positive dim, 1/2 cameras and an 8x8 fine grid')
        if not isinstance(refined_cells,int) or isinstance(refined_cells,bool) or not 0<=refined_cells<=16:
            raise ValueError('refined_cells must be an integer in [0,16]')
        if not all(math.isfinite(x) and x>0 for x in (tau_fast,tau_slow,max_age)) or tau_fast>=tau_slow:
            raise ValueError('invalid memory time constants/maximum age')
        self.dim,self.cameras,self.refined_cells,self.grid_size=dim,cameras,refined_cells,grid_size
        self.taus=(tau_fast,tau_slow)
        self.max_age=max_age
        # 256px -> 64px pooled -> 8x8 features; both OFF/ON channels retained.
        self.encoder=nn.Conv2d(2,dim,kernel_size=8,stride=8)
        self.readout=nn.Sequential(nn.Linear(3*dim+3,dim),nn.SiLU(),nn.LayerNorm(dim))
        self.camera=nn.Parameter(torch.randn(cameras,dim)*.02)
        self.scale=nn.Parameter(torch.zeros(2,dim)) # coarse/fine
        self.register_buffer('memory_spec',torch.tensor([1,grid_size,dim,refined_cells,cameras],dtype=torch.long))
        self.register_buffer('time_spec',torch.tensor([round(tau_fast*1e6),round(tau_slow*1e6),round(max_age*1e6)],dtype=torch.long))

    @property
    def tokens_per_camera(self):
        return 16+3*self.refined_cells

    def _dt(self,dt,b,device):
        dt=torch.as_tensor(dt,device=device,dtype=torch.float32)
        return dt.expand(b) if dt.ndim==0 else dt.reshape(b)

    def frame_activity(self,voxel,valid_region=None):
        if voxel.ndim!=6 or voxel.shape[2:4]!=(self.cameras,2) or voxel.shape[-2:]!=(256,256):
            raise ValueError('memory frames must be [B,T,V,2,256,256]')
        b,t,v,_,h,w=voxel.shape
        valid=torch.ones((b,1,h,w),device=voxel.device,dtype=torch.float32) if valid_region is None else valid_region.float()
        if valid.shape!=(b,1,h,w):
            raise ValueError('valid_region must be [B,1,256,256]')
        occupancy=(voxel>0).float().mean(3)*valid[:,None]
        denominator=F.avg_pool2d(valid,32,32)[:,None]
        activity=F.avg_pool2d(occupancy.reshape(b*t,v,h,w),32,32).reshape(b,t,v,8,8)
        return torch.where(denominator>0,activity/denominator.clamp_min(1e-8),0.)

    def encode_frames(self,voxel,valid_region=None):
        if voxel.ndim!=6 or voxel.shape[2:4]!=(self.cameras,2) or voxel.shape[-2:]!=(256,256):
            raise ValueError('memory frames must be [B,T,V,2,256,256]')
        b,t,v,_,h,w=voxel.shape
        valid=torch.ones((b,1,h,w),device=voxel.device,dtype=torch.float32) if valid_region is None else valid_region.float()
        if valid.shape!=(b,1,h,w):
            raise ValueError('valid_region must be [B,1,256,256]')
        raw=voxel.float()*valid[:,None,None]
        activity=self.frame_activity(voxel,valid_region)
        pooled=F.avg_pool2d(raw.reshape(b*t*v,2,h,w),4,4)
        features=self.encoder(pooled.to(self.encoder.weight.dtype)).permute(0,2,3,1).reshape(b,t,v,8,8,self.dim)
        return features,activity

    def update(self,features,activity,dt,state=None,frame_valid=None,reset=None):
        if features.ndim!=5 or features.shape[1:]!=(self.cameras,8,8,self.dim) or activity.shape!=features.shape[:-1]:
            raise ValueError('invalid frame feature/activity axes')
        b=features.shape[0]
        dt=self._dt(dt,b,features.device)
        active=activity>0
        if frame_valid is not None:
            active=active & frame_valid.to(device=features.device,dtype=torch.bool)[:,None,None,None]
        if state is None:
            traces=torch.zeros((b,self.cameras,2,8,8,self.dim),device=features.device,dtype=torch.float32)
            support=torch.zeros_like(traces[...,0])
            density=torch.zeros_like(activity,dtype=torch.float32)
            age=torch.zeros_like(density)
            seen=torch.zeros_like(active)
        else:
            traces,support,density,age,seen=(getattr(state,k) for k in state.__dataclass_fields__)
            if traces.shape!=(b,self.cameras,2,8,8,self.dim) or traces.device!=features.device:
                raise ValueError('state axes/device mismatch; reset after stream changes')
        if reset is not None:
            keep=(~reset.to(device=features.device,dtype=torch.bool))[:,None,None,None]
            traces=traces*keep[:, :, None, :, :, None]
            support=support*keep[:,:,None]
            density=density*keep
            age=age*keep
            seen=seen & keep
        decay=torch.exp(-dt[:,None]/features.new_tensor(self.taus,dtype=torch.float32)[None])[:,None,:,None,None]
        writes=active[:,:,None].float()
        traces=decay[...,None]*traces+(1-decay)[...,None]*writes[...,None]*features.float()[:,:,None]
        support=decay*support+(1-decay)*writes
        slow=decay[:,:,1]
        density=slow*density+(1-slow)*torch.where(active,activity.float(),0.)
        age=torch.where(active,0.,age+dt[:,None,None,None])
        return EventMemoryState(traces,support,density,age,seen|active)

    def decay_to_query(self,state,dt):
        zero=state.traces.new_zeros((state.traces.shape[0],self.cameras,8,8,self.dim))
        return self.update(zero,zero[...,0],dt,state)

    def read(self,state,current_activity,fps,valid_region=None):
        b,v=state.age.shape[:2]
        fps=torch.as_tensor(fps,device=state.age.device,dtype=torch.float32).expand(b)
        allowed=state.seen & (state.age<=self.max_age) & (state.support[:,:,1]>1e-6)
        spatial=torch.ones_like(allowed) if valid_region is None else (F.avg_pool2d(valid_region.float(),32,32)>0).expand(b,v,8,8)
        allowed=allowed & spatial
        # Persistent traces are normalized by support; confidence and age are
        # exposed separately, so empty-frame decay is not hidden by normalization.
        def coarse(x):
            return x.reshape(*x.shape[:-2],4,2,4,2).mean((-3,-1))
        scores=coarse(current_activity.float()+state.activity)
        cell_valid=coarse(spatial.float())>0
        scores=torch.where(cell_valid,scores,torch.full_like(scores,-1)).flatten(2)
        ids=torch.arange(16,device=scores.device)
        tie=torch.argsort((ids//4+ids%4)%2,stable=True)
        order=tie[torch.argsort(scores.index_select(-1,tie),descending=True,stable=True)]
        fine=order[...,:self.refined_cells].sort(-1).values
        coarse_ids=order[...,self.refined_cells:].sort(-1).values
        offsets=fine.new_tensor([0,1,8,9])
        fine_ids=((fine//4)*16+(fine%4)*2)[...,None]+offsets
        fine_ids=fine_ids.flatten(-2)
        normalized=state.traces/state.support[...,None].clamp_min(1e-6)
        fine_features=torch.cat((normalized[:,:,0],normalized[:,:,1],normalized[:,:,0]-normalized[:,:,1]),-1)
        conf=state.support.permute(0,1,3,4,2)
        fine_metadata=torch.cat((torch.log1p(state.age)[...,None],conf),-1)
        fine_raw=torch.cat((fine_features,fine_metadata),-1).flatten(2,3)
        fine_read=fine_raw.gather(2,fine_ids[...,None].expand(b,v,fine_ids.shape[-1],fine_raw.shape[-1]))
        fine_valid=allowed.flatten(2).gather(2,fine_ids)
        fine_age=state.age.flatten(2).gather(2,fine_ids)
        # Aggregate only supported, non-expired fine cells into each coarse token.
        weights=state.support*allowed[:,:,None]
        mass=weights.reshape(b,v,2,4,2,4,2).sum((4,6))
        sums=(normalized*weights[...,None]).reshape(b,v,2,4,2,4,2,self.dim).sum((4,6))
        aggregate=sums/mass[...,None].clamp_min(1e-6)
        coarse_features=torch.cat((aggregate[:,:,0],aggregate[:,:,1],aggregate[:,:,0]-aggregate[:,:,1]),-1)
        minimum_age=torch.where(allowed,state.age,torch.full_like(state.age,float('inf'))).reshape(b,v,4,2,4,2).amin((3,5))
        coarse_valid=allowed.reshape(b,v,4,2,4,2).any((3,5))
        minimum_age=torch.where(coarse_valid,minimum_age,0.)
        coarse_metadata=torch.cat((torch.log1p(minimum_age)[...,None],(mass/4).permute(0,1,3,4,2)),-1)
        coarse_raw=torch.cat((coarse_features,coarse_metadata),-1).flatten(2,3)
        coarse_read=coarse_raw.gather(2,coarse_ids[...,None].expand(b,v,coarse_ids.shape[-1],coarse_raw.shape[-1]))
        coarse_valid=coarse_valid.flatten(2).gather(2,coarse_ids)
        coarse_age=minimum_age.flatten(2).gather(2,coarse_ids)
        values=torch.cat((coarse_read,fine_read),2).to(self.readout[0].weight.dtype)
        token=self.readout(values)+self.camera[None,:,None]
        ncoarse=coarse_ids.shape[-1]
        scales=torch.cat((torch.zeros_like(coarse_ids),torch.ones_like(fine_ids)),2)
        token=token+self.scale[scales]
        valid=torch.cat((coarse_valid,fine_valid),2)
        token=torch.where(valid[...,None],token,0.)
        ages=torch.cat((coarse_age,fine_age),2)
        cyx=torch.stack((2*(coarse_ids//4)+.5,2*(coarse_ids%4)+.5),-1)
        fyx=torch.stack((fine_ids//8,fine_ids%8),-1).float()
        yx=torch.cat((cyx,fyx),2)
        temporal=-ages*fps[:,None,None]
        coord=torch.cat((temporal[...,None],yx),-1)
        return EventMemoryRead(token.flatten(1,2),valid.flatten(1,2),coord.flatten(1,2),fine)
