"""Dense-future EWM with compressed, read-only event-memory joint K/V."""
from dataclasses import dataclass
import torch
from torch import nn
from openwam.model.action_backbone.components import RMSNorm
from .ewm import EWMModel, multimodal_rope, multimodal_rope_frequencies
from .event_memory import CompressedEventMemory, EventMemoryState, EventMemoryRead

@dataclass(frozen=True)
class EventMemoryStream:
    state: EventMemoryState
    frame_ids: torch.Tensor

    def detach(self):
        return EventMemoryStream(self.state.detach(),self.frame_ids.detach())


class MemoryEWMModel(EWMModel):
    """Memory changes joint keys/values only, never queries or future patches.

    Training consumes explicit past frames and starts a fresh memory per sample.
    Deployment may instead provide an external EventMemoryStream. The model
    never stores a mutable online stream or a denoising-time memory update.
    """
    checkpoint_format="ewm_event_memory_latent_v1"
    _MEMORY_PREFIXES=('event_memory.','memory_projectors.','memory_key_norms.')

    def __init__(self, *, memory_enabled=True, memory_dim=128, memory_refined_cells=4,
                 memory_tau_fast=.08, memory_tau_slow=.40, memory_max_age=1.60, **kwargs):
        super().__init__(**kwargs)
        if self.training_mode not in ('joint','event_only'):
            raise ValueError('event memory requires joint or event_only mode')
        if self.latent_patch_size!=4 or self.event_codec_impl!='leanvae16_joint':
            raise ValueError('first memory integration requires full LeanVAE patch4 tokens')
        self.memory_enabled=bool(memory_enabled)
        self.event_memory=CompressedEventMemory(memory_dim,self.num_event_cameras,
            memory_refined_cells,tau_fast=memory_tau_fast,tau_slow=memory_tau_slow,max_age=memory_max_age)
        width=self.event_input.out_channels
        self.memory_projectors=nn.ModuleList([nn.Sequential(nn.LayerNorm(memory_dim),nn.Linear(memory_dim,2*width))
                                             for _ in self.blocks])
        self.memory_key_norms=nn.ModuleList([RMSNorm(width) for _ in self.blocks])
        if not self.memory_enabled:
            for module in (self.event_memory,self.memory_projectors,self.memory_key_norms):
                module.requires_grad_(False)

    def _fps(self,batch):
        b=batch['obs_voxel'].shape[0]
        fps=torch.as_tensor(batch.get('fps',30.),device=batch['obs_voxel'].device,dtype=torch.float32).expand(b)
        if not bool(torch.isfinite(fps).all() & (fps>0).all()):
            raise ValueError('fps must be finite positive per-sample physical rates')
        return fps

    def _history_read(self,batch):
        fps=self._fps(batch)
        valid_region=batch.get('valid_region_mask')
        current=batch['obs_voxel']
        # Query routing uses current activity, while content remains strictly past.
        current_activity=self.event_memory.frame_activity(current[:,None],valid_region)
        if 'memory_stream' in batch:
            stream=batch['memory_stream']
            if not isinstance(stream,EventMemoryStream):
                raise ValueError('memory_stream must be an EventMemoryStream')
            state=stream.state
            elapsed=torch.as_tensor(batch.get('memory_query_dt',1/fps),device=current.device,dtype=torch.float32).expand(current.shape[0])
            if not bool(torch.isfinite(elapsed).all() & (elapsed>=0).all()):
                raise ValueError('query elapsed time must be finite and nonnegative')
            state=self.event_memory.decay_to_query(state,elapsed)
        else:
            if 'memory_history_voxel' not in batch:
                raise ValueError('enabled event memory requires past frames or an explicit memory_stream')
            history=batch['memory_history_voxel']
            if history.ndim!=6 or history.shape[0]!=current.shape[0] or history.shape[1]<=0:
                raise ValueError('memory_history_voxel must contain positive-length [B,H,V,2,256,256] history')
            b,h=history.shape[:2]
            history_valid=batch.get('memory_history_valid',torch.ones((b,h),device=history.device,dtype=torch.bool))
            history_dt=batch.get('memory_history_dt',(1/fps)[:,None].expand(b,h))
            if history_valid.shape!=(b,h) or history_dt.shape!=(b,h):
                raise ValueError('history validity/dt must be [B,H]')
            if not bool(torch.isfinite(history_dt).all() & (history_dt>0).all()):
                raise ValueError('history dt must be finite positive seconds')
            if 'memory_history_indices' in batch and 'memory_anchor' in batch:
                indices=batch['memory_history_indices']
                anchors=batch['memory_anchor']
                if bool(((indices>=anchors[:,None]) & history_valid).any()):
                    raise ValueError('memory history must be strictly before the observation anchor')
            features,activity=self.event_memory.encode_frames(history,valid_region)
            state=None
            for index in range(h):
                state=self.event_memory.update(features[:,index],activity[:,index],history_dt[:,index],state,
                    frame_valid=history_valid[:,index])
            elapsed=torch.as_tensor(batch.get('memory_query_dt',1/fps),device=current.device,dtype=torch.float32).expand(current.shape[0])
            if not bool(torch.isfinite(elapsed).all() & (elapsed>=0).all()):
                raise ValueError('query elapsed time must be finite and nonnegative')
            state=self.event_memory.decay_to_query(state,elapsed)
        return self.event_memory.read(state,current_activity[:,0],fps,valid_region)

    def prepare_memory_kv(self,batch):
        if not self.memory_enabled:
            return None
        read=self._history_read(batch)
        return self.project_memory_kv(read)

    def project_memory_kv(self,read):
        b,m,_=read.tokens.shape
        width=self.event_input.out_channels
        heads=self.blocks[0].heads
        head_dim=width//heads
        phase=multimodal_rope_frequencies(read.coordinates.reshape(b*m,3),head_dim,dtype=read.tokens.dtype)
        phase=tuple((c.reshape(b,1,m,-1),s.reshape(b,1,m,-1)) for c,s in phase)
        result=[]
        for projection,key_norm in zip(self.memory_projectors,self.memory_key_norms):
            key,value=projection(read.tokens).chunk(2,-1)
            key=key_norm(key).unflatten(-1,(heads,head_dim)).transpose(1,2)
            key=multimodal_rope(key,frequencies=phase)
            value=value.unflatten(-1,(heads,head_dim)).transpose(1,2)
            result.append((key,value,read.valid))
        return tuple(result)

    @torch.no_grad()
    def observe_memory(self,voxel,*,frame_ids,dt,stream=None,reset=None,valid_region=None):
        """Explicit physical update. Call on every received frame, outside flow.

        dt and reset can differ by batch member; reset allows a new episode's
        frame counter to restart. Use an eval model for deployment.
        """
        frame_ids=torch.as_tensor(frame_ids,device=voxel.device,dtype=torch.long).reshape(voxel.shape[0])
        dt=torch.as_tensor(dt,device=voxel.device,dtype=torch.float32).expand(voxel.shape[0])
        if not bool(torch.isfinite(dt).all() & (dt>0).all()):
            raise ValueError('physical dt must be finite positive')
        reset=torch.zeros_like(frame_ids,dtype=torch.bool) if reset is None else reset.to(device=voxel.device,dtype=torch.bool)
        if stream is not None and bool(((frame_ids<=stream.frame_ids)&~reset).any()):
            raise ValueError('observe_memory requires new physical frame IDs or episode reset')
        features,activity=self.event_memory.encode_frames(voxel[:,None],valid_region)
        state=self.event_memory.update(features[:,0],activity[:,0],dt,
            None if stream is None else stream.state,reset=reset)
        return EventMemoryStream(state.detach(),frame_ids.clone())

    def token_counts(self,horizon=24):
        if horizon<=0 or horizon%4:
            raise ValueError('horizon must be positive and divisible by4')
        event=self.num_event_cameras*64*(1+horizon//4)
        action=0 if self.training_mode=='event_only' else horizon+1
        memory=self.num_event_cameras*self.event_memory.tokens_per_camera if self.memory_enabled else 0
        return dict(observed_per_camera=64,future_per_camera_per_slice=64,
                    query=event+action,key_value=event+action+memory,memory=memory)

    def initialize_from_dense_policy(self,state):
        expected={k for k in self.policy_state_dict() if not k.startswith(self._MEMORY_PREFIXES)}
        if set(state)!=expected:
            raise RuntimeError(f'invalid dense initialization: missing={expected-set(state)}, unexpected={set(state)-expected}')
        return self.load_state_dict(state,strict=False)

    def load_policy_state_dict(self,state):
        for key in ('event_memory.memory_spec','event_memory.time_spec'):
            expected=self.policy_state_dict()[key]
            if key not in state or not torch.equal(state[key].cpu(),expected.cpu()):
                raise RuntimeError('incompatible compressed-event-memory checkpoint contract')
        return super().load_policy_state_dict(state)
