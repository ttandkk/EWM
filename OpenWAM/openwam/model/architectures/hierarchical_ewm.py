"""Dense EWM with fused RGB-semantic/event-motion short-term memory."""
import time
import torch
from torch import nn
from torch.nn import functional as F
from openwam.model.action_backbone.components import RMSNorm
from openwam.model.architectures.ewm import EWMModel
from openwam.model.architectures.ewm_memory import MemoryEWMModel
from .hierarchical_memory import RGBEventMemory


class HierarchicalMemoryEWMModel(EWMModel):
    checkpoint_format = 'ewm_rgb_event_memory_latent_v1'
    _MEMORY_PREFIXES = ('rgb_event_memory.','memory_projectors.','memory_key_norms.')
    project_memory_kv = MemoryEWMModel.project_memory_kv

    def __init__(self, *, memory_enabled=True, memory_dim=256, memory_heads=4,
                 memory_dino_batch_size=16, **kwargs):
        super().__init__(**kwargs)
        if self.training_mode not in ('joint','event_only') or self.context_encoder is None:
            raise ValueError('RGB/event memory requires joint/event_only with a frozen DINO context encoder')
        if self.event_codec_impl != 'leanvae16_joint' or self.latent_patch_size != 4:
            raise ValueError('first RGB/event memory integration requires LeanVAE joint patch4')
        if memory_dino_batch_size<=0:
            raise ValueError('memory_dino_batch_size must be positive')
        self.memory_enabled = bool(memory_enabled)
        self.memory_dino_batch_size = int(memory_dino_batch_size)
        self.rgb_event_memory = RGBEventMemory(self.context_encoder.dino.properties.z_dim,
            memory_dim,self.num_event_cameras,memory_heads)
        width = self.event_input.out_channels
        self.memory_projectors = nn.ModuleList([nn.Sequential(nn.LayerNorm(memory_dim),nn.Linear(memory_dim,2*width)) for _ in self.blocks])
        self.memory_key_norms = nn.ModuleList([RMSNorm(width) for _ in self.blocks])
        if not self.memory_enabled:
            for module in (self.rgb_event_memory,self.memory_projectors,self.memory_key_norms):
                module.requires_grad_(False)

    def _with_current_features(self,batch):
        if not self.memory_enabled or 'current_dino_features' in batch:
            return batch
        result = dict(batch)
        with torch.no_grad():
            result['current_dino_features'] = self.context_encoder.encode_rgb_features(batch['obs_video'])
        return result

    def _context(self,batch,context):
        if context is not None:
            return context
        if 'current_dino_features' in batch:
            return self.context_encoder.encode(batch['obs_video'],batch['prompt'],rgb_features=batch['current_dino_features'])
        return super()._context(batch,context)

    def forward(self,batch,**kwargs):
        profiling = bool(kwargs.get('profile_timing',False))
        device = batch['obs_voxel'].device
        if profiling and device.type == 'cuda': torch.cuda.synchronize(device)
        start = time.perf_counter() if profiling else None
        prepared = self._with_current_features(batch)
        if profiling and device.type == 'cuda': torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-start if profiling else None
        result = super().forward(prepared,**kwargs)
        if profiling: result['_profile']['memory_current_dino'] = elapsed
        return result

    compute_loss = forward

    @torch.no_grad()
    def sample_policy(self,batch,**kwargs):
        return super().sample_policy(self._with_current_features(batch),**kwargs)

    def _rgb_memory_features(self,batch):
        current = batch.get('current_dino_features')
        if current is None:
            with torch.no_grad():
                current = self.context_encoder.encode_rgb_features(batch['obs_video'])
        b,v,d,h,w = current.shape
        if v != 2:
            raise ValueError('current DINO features must contain both RGB cameras')
        current = F.adaptive_avg_pool2d(current.reshape(b*v,d,h,w),(8,8)).reshape(b,v,d,8,8).to(torch.bfloat16)
        if 'memory_rgb_features' in batch:
            history = batch['memory_rgb_features']
            if history.shape!=(b,3,2,d,8,8):
                raise ValueError('cached RGB history must be [B,3,2,DINO_D,8,8]')
            history = history.to(current)
        else:
            if 'memory_rgb' not in batch or batch['memory_rgb'].shape[:4]!=(b,3,2,3):
                raise ValueError('memory needs three historical RGB frames or cached DINO features')
            raw = batch['memory_rgb']; raw = raw.reshape(b*3,2,3,*raw.shape[-2:])
            with torch.no_grad():
                encoded=[]
                for chunk in raw.split(self.memory_dino_batch_size):
                    f = self.context_encoder.encode_rgb_features(chunk)
                    encoded.append(F.adaptive_avg_pool2d(f.flatten(0,1),(8,8)).reshape(len(chunk),2,d,8,8))
                history = torch.cat(encoded).reshape(b,3,2,d,8,8)
        features = torch.cat((history.to(current),current[:,None]),1)
        if self.event_camera_mode=='wrist_only': features=features[:,:,1:2]
        elif self.event_camera_mode=='global_only': features=features[:,:,:1]
        return features

    def prepare_memory_kv(self,batch):
        if not self.memory_enabled:
            return None
        events = batch['memory_event']
        if not torch.equal(events[:,-1].detach(),batch['obs_voxel'].detach()):
            raise ValueError('memory event window must end at the current observation')
        if 'memory_indices' in batch:
            idx,anchor= batch['memory_indices'],batch['memory_anchor']
            valid=batch['memory_event_valid']
            expected = anchor[:,None]+torch.arange(-9,1,device=idx.device)[None]
            if bool(((idx!=expected)&valid).any()):
                raise ValueError('memory indices must be causal consecutive bins ending at anchor')
        read = self.rgb_event_memory(self._rgb_memory_features(batch),events,
            rgb_valid=batch['memory_rgb_valid'],event_valid=batch['memory_event_valid'],
            fps=batch.get('fps',30.),valid_region=batch.get('valid_region_mask'))
        return self.project_memory_kv(read)

    def token_counts(self,horizon=24):
        if horizon<=0 or horizon%self.event_temporal_compression:
            raise ValueError('horizon must be a positive codec-stride multiple')
        e=self.num_event_cameras*(32//self.latent_patch_size)**2*(1+horizon//self.event_temporal_compression)
        a=0 if self.training_mode=='event_only' else horizon+1
        memory=64*self.num_event_cameras if self.memory_enabled else 0
        return dict(query=e+a,key_value=e+a+memory,memory=memory)

    def initialize_from_dense_policy(self,state):
        expected={k for k in self.policy_state_dict() if not k.startswith(self._MEMORY_PREFIXES)}
        if set(state)!=expected:
            raise RuntimeError('dense initialization has missing or unexpected policy weights')
        return self.load_state_dict(state,strict=False)

    def load_policy_state_dict(self,state):
        key='rgb_event_memory.memory_spec'
        if key not in state or not torch.equal(state[key].cpu(),self.rgb_event_memory.memory_spec.cpu()):
            raise RuntimeError('incompatible RGB/event memory checkpoint contract')
        return super().load_policy_state_dict(state)
