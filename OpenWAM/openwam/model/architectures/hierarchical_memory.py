"""Short-term RGB semantic memory enriched by ordered event motion features."""
import torch
from torch import nn
from torch.nn import functional as F
from openwam.model.architectures.event_memory import EventMemoryRead

RGB_OFFSETS = (-9, -6, -3, 0)
EVENT_OFFSETS = tuple(range(-9, 1))


def event_difference_inputs(events, valid):
    """Normalized OFF/ON -> floating probe packing -> current + nine deltas.

    Area averaging at 128px is equivalent to the probe's 2x2 sum divided
    by [60,120,60] for Piper count_clip=15; no byte rounding is introduced.
    Missing endpoints never create artificial changes against padding.
    """
    b,t,v,c,h,w = events.shape
    if t != 10 or c != 2 or valid.shape != (b,10):
        raise ValueError('event memory requires ten OFF/ON bins and [B,10] validity')
    x = F.adaptive_avg_pool2d(events.reshape(b*t*v,2,h,w).float(), (128,128))
    off,on = x.unbind(1)
    x = torch.stack((off,(off+on)/2,on),1).reshape(b,t,v,3,128,128)
    pair_valid = valid[:,1:] & valid[:,:-1]
    delta = (x[:,1:] - x[:,:-1]) * pair_valid[:,:,None,None,None,None]
    anchor = x[:,-1:] * valid[:,-1:,None,None,None,None]
    return torch.cat((anchor,delta),1), torch.cat((valid[:,-1:],pair_valid),1)


class RGBEventMemory(nn.Module):
    """Two modality levels over one window; no recurrent EMA or future writes."""
    def __init__(self, rgb_dim, dim=256, cameras=2, heads=4):
        super().__init__()
        if cameras not in (1,2) or dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError('invalid memory width/cameras/heads')
        self.dim,self.cameras = dim,cameras
        # Exact convolution widths/kernels from the 10-bin pixel-difference probe.
        self.event_encoder = nn.Sequential(
            nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,3,2,1),
            nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d((4,4)))
        self.event_time = nn.Sequential(nn.Linear(3,64),nn.SiLU(),nn.Linear(64,64))
        self.event_temporal = nn.Sequential(nn.Linear(10*64,dim),nn.SiLU(),nn.LayerNorm(dim))
        self.rgb_projection = nn.Linear(rgb_dim,dim)
        self.rgb_time = nn.Sequential(nn.Linear(1,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.rgb_temporal = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.motion_attention = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.motion_projection = nn.Linear(dim,dim)
        self.gate = nn.Linear(2*dim,dim)
        self.output_norm = nn.LayerNorm(dim)
        self.camera = nn.Parameter(torch.randn(cameras,dim)*.02)
        self.rgb_position = nn.Parameter(torch.randn(64,dim)*.02)
        self.event_position = nn.Parameter(torch.randn(16,dim)*.02)
        self.register_buffer('memory_spec',torch.tensor([1,10,4,8,4,dim,cameras,heads,rgb_dim]))

    def forward(self, rgb, events, *, rgb_valid, event_valid, fps, valid_region=None):
        # rgb [B,4,V,D,8,8], events [B,10,V,2,H,W]
        b,t,v,d,h,w = rgb.shape
        if (t,v,h,w)!=(4,self.cameras,8,8) or events.shape[:3]!=(b,10,v):
            raise ValueError('memory needs [B,4,V,D,8,8] RGB and [B,10,V,2,H,W] events')
        if rgb_valid.shape!=(b,4) or event_valid.shape!=(b,10):
            raise ValueError('memory validity must be [B,4] and [B,10]')
        if not bool(rgb_valid[:,-1].all() & event_valid[:,-1].all()):
            raise ValueError('current RGB/event observations must be valid')
        fps = torch.as_tensor(fps,device=rgb.device,dtype=torch.float32).expand(b)
        if not bool(torch.isfinite(fps).all() & (fps>0).all()):
            raise ValueError('fps must be finite and positive')
        r = self.rgb_projection(rgb.permute(0,2,4,5,1,3).to(self.rgb_projection.weight.dtype))
        rt = rgb.new_tensor(RGB_OFFSETS,dtype=torch.float32)[None]/fps[:,None]
        r = r + self.rgb_time(rt[:,:,None].to(r.dtype))[:,None,None,None]
        r = r + self.rgb_position.reshape(1,1,8,8,1,-1).to(r)
        r = r.reshape(b*v*64,4,self.dim)
        rvalid = rgb_valid[:,None,None].expand(b,v,64,4).reshape(-1,4)
        semantic = self.rgb_temporal(r[:,-1:],r,r,key_padding_mask=~rvalid,need_weights=False)[0]
        semantic = (semantic[:,0] + r[:,-1]).reshape(b*v,64,self.dim)

        x,xvalid = event_difference_inputs(events,event_valid)
        motion = self.event_encoder(x.reshape(b*10*v,3,128,128).to(self.event_encoder[0].weight.dtype))
        motion = motion.reshape(b,10,v,64,16).permute(0,2,4,1,3)
        # Anchor time=0; deltas use interval midpoints, plus support duration/type.
        et = torch.tensor([0.,-8.5,-7.5,-6.5,-5.5,-4.5,-3.5,-2.5,-1.5,-.5],device=rgb.device)[None]/fps[:,None]
        duration = (1/fps)[:,None].expand(-1,10)
        kind = torch.tensor([0.]+[1.]*9,device=rgb.device)[None].expand(b,-1)
        meta = torch.stack((et,duration,kind),-1)
        motion = motion + self.event_time(meta.to(motion.dtype))[:,None,None]
        motion = motion * xvalid[:,None,None,:,None]
        motion = self.event_temporal(motion.flatten(-2)).reshape(b*v,16,self.dim)
        motion = motion + self.event_position.to(motion)[None]
        spatial8 = torch.ones((b,64),device=rgb.device,dtype=torch.bool)
        spatial4 = torch.ones((b,16),device=rgb.device,dtype=torch.bool)
        if valid_region is not None:
            if valid_region.shape!=(b,1,*events.shape[-2:]):
                raise ValueError('valid_region must match event image geometry')
            spatial8 = F.adaptive_avg_pool2d(valid_region.float(),(8,8)).flatten(1)>0
            spatial4 = F.adaptive_avg_pool2d(valid_region.float(),(4,4)).flatten(1)>0
        evmask = spatial4[:,None].expand(b,v,16).reshape(b*v,16)
        # All-padding images must not produce NaNs in attention.
        safe = evmask.clone(); safe[:,0] |= ~safe.any(-1)
        enriched = self.motion_attention(semantic,motion,motion,key_padding_mask=~safe,need_weights=False)[0]
        enriched = enriched * evmask.any(-1)[:,None,None]
        gate = torch.sigmoid(self.gate(torch.cat((semantic,enriched),-1)))
        fused = self.output_norm(semantic + gate*self.motion_projection(enriched))
        fused = fused.reshape(b,v,64,self.dim) + self.camera.to(fused)[None,:,None]
        valid = spatial8[:,None].expand(b,v,64).reshape(b,v*64)
        fused = torch.where(valid[...,None],fused.flatten(1,2),0.)
        yy,xx = torch.meshgrid(torch.arange(8,device=rgb.device),torch.arange(8,device=rgb.device),indexing='ij')
        # Semantic slots are anchored at current time; history timing lives inside.
        coords = torch.stack((torch.zeros_like(yy),yy,xx),-1).float().reshape(1,1,64,3).expand(b,v,-1,-1)
        return EventMemoryRead(fused,valid,coords.reshape(b,v*64,3),torch.empty(b,v,0,device=rgb.device,dtype=torch.long))
