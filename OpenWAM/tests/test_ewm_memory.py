from types import SimpleNamespace
import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from openwam.model.architectures.ewm_memory import MemoryEWMModel
from openwam.model.architectures.event_memory import CompressedEventMemory
from openwam.dataloader.ewm_memory import attach_history
from openwam.model.architectures.ewm import EWMModel as CandidateDense, full_mixed_self_attention
from openwam.model.architectures.ewm import EWMModel as OriginalDense, EWMContextEncoder
from openwam.model.architectures.lean_event_codec import LeanEventCodec
from tests.test_ewm_model import FakeLeanVAE, FakeDino, FakeText


def options(cameras=1,mode='joint',checkpoint=False):
    return dict(dim=16,action_hidden_dim=16,heads=2,layers=2,
        event_codec_impl='leanvae16_joint',event_temporal_compression=4,
        event_spatial_compression=32,event_patch_size=4,
        event_camera_mode='dual' if cameras==2 else 'wrist_only',
        event_codec=LeanEventCodec(vae=FakeLeanVAE(),batch_size=2),
        context_encoder=EWMContextEncoder(FakeDino(),FakeText(),16),
        training_mode=mode,gradient_checkpointing=checkpoint)


def model(cameras=1,mode='joint',enabled=True,checkpoint=False):
    return MemoryEWMModel(memory_dim=8,memory_enabled=enabled,**options(cameras,mode,checkpoint))


def batch(cameras=1,horizon=24,b=1,history=3):
    return dict(obs_voxel=torch.rand(b,cameras,2,256,256),
        future_voxel=torch.rand(b,horizon,cameras,2,256,256),
        proprio=torch.rand(b,8),action=torch.rand(b,horizon,7),
        obs_video=torch.zeros(b,2,3,256,256,dtype=torch.uint8),prompt=['move']*b,
        memory_history_voxel=torch.rand(b,history,cameras,2,256,256),
        memory_history_valid=torch.ones(b,history,dtype=torch.bool),
        memory_history_indices=torch.arange(history).expand(b,-1),memory_anchor=torch.full((b,),history),
        fps=torch.full((b,),30.))


def context(b=1):
    return torch.randn(b,3,16),torch.ones(b,3,dtype=torch.bool)


@pytest.mark.parametrize('cameras,mode',[(1,'joint'),(2,'joint'),(1,'event_only'),(2,'event_only')])
def test_dense_future_query_and_memory_kv_budget(monkeypatch,cameras,mode):
    m=model(cameras,mode);data=batch(cameras);observed=[]
    actual=F.scaled_dot_product_attention
    def inspect(q,k,v,*args,**kwargs):
        observed.append((q.shape[-2],k.shape[-2],v.shape[-2]))
        return actual(q,k,v,*args,**kwargs)
    monkeypatch.setattr(F,'scaled_dot_product_attention',inspect)
    expected=448*cameras+(25 if mode=='joint' else 0)
    loss=m(data,context=context())['loss']
    assert torch.isfinite(loss)
    assert (expected,expected+28*cameras,expected+28*cameras) in observed
    assert m.token_counts()['query']==expected
    assert m.token_counts()['key_value']==expected+28*cameras
    got=m.sample_policy(data,horizon=24,num_steps=2,context=context(),decode_future=False,
        normalizer=SimpleNamespace(denormalize_action=lambda x:x))
    assert got['future_latent'].shape==(1,6,cameras,16,32,32)
    if mode=='joint':assert got['action'].shape==(1,24,7)


@pytest.mark.parametrize('checkpoint',[False,True])
def test_memory_gradients_reach_encoder_projection_and_both_experts(checkpoint):
    m=model(checkpoint=checkpoint);data=batch(horizon=4)
    loss=m(data)['loss'];loss.backward()
    for p in m.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    assert m.event_memory.encoder.weight.grad.abs().sum()>0
    assert m.memory_projectors[0][1].weight.grad.abs().sum()>0
    assert m.blocks[0].event.qkv.weight.grad.abs().sum()>0
    assert m.blocks[0].action.qkv.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.event_codec.parameters())


def test_disabled_memory_matches_original_dense_bitwise():
    original=OriginalDense(**options());candidate=model(enabled=False)
    candidate.initialize_from_dense_policy(original.policy_state_dict())
    data=batch(horizon=4);ctx=context();t=torch.tensor([.37]);noise=torch.randn(1,1,1,16,32,32);an=torch.randn_like(data['action'])
    a=original(data,context=ctx,t=t,event_noise=noise,action_noise=an)
    b=candidate(data,context=ctx,t=t,event_noise=noise,action_noise=an)
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_history_changes_predictions_without_changing_future_inputs():
    m=model();data=batch(horizon=4);ctx=context();t=torch.tensor([.5])
    observed=m.event_codec.encode(data['obs_voxel'][:,None])[:,0]
    event=torch.randn(1,1,1,16,32,32);action=torch.randn_like(data['action'])
    kv=m.prepare_memory_kv(data)
    a=m.predict_velocity(observed,data['proprio'],event,action,t,ctx,memory_kv=kv)
    altered=dict(data);altered['memory_history_voxel']=torch.zeros_like(data['memory_history_voxel'])
    b=m.predict_velocity(observed,data['proprio'],event,action,t,ctx,memory_kv=m.prepare_memory_kv(altered))
    assert not torch.allclose(a[0],b[0]) and not torch.allclose(a[1],b[1])


def test_future_payload_does_not_enter_memory_and_anchor_guard():
    m=model();data=batch(horizon=4)
    a=m.prepare_memory_kv(data)
    changed=dict(data);changed['future_voxel']=data['future_voxel']*100
    b=m.prepare_memory_kv(changed)
    for x,y in zip(a,b):
        for u,v in zip(x,y):torch.testing.assert_close(u,v,rtol=0,atol=0)
    changed=dict(data);changed['memory_history_indices']=data['memory_history_indices']+1
    with pytest.raises(ValueError,match='strictly before'):m.prepare_memory_kv(changed)


def test_empty_history_masks_all_memory_and_matches_dense():
    m=model();data=batch(horizon=4);data['memory_history_valid'].zero_()
    kv=m.prepare_memory_kv(data)
    assert not kv[0][2].any()
    observed=m.event_codec.encode(data['obs_voxel'][:,None])[:,0]
    event=torch.randn(1,1,1,16,32,32);action=torch.randn_like(data['action']);ctx=context();t=torch.tensor([.3])
    a=m.predict_velocity(observed,data['proprio'],event,action,t,ctx,memory_kv=kv)
    b=m.predict_velocity(observed,data['proprio'],event,action,t,ctx)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)


def test_cache_prepared_once_per_sample_and_stream_is_read_only(monkeypatch):
    m=model().eval();data=batch(horizon=4)
    stream=m.observe_memory(data['memory_history_voxel'][:,0],frame_ids=[10],dt=1/30)
    online={k:v for k,v in data.items() if not k.startswith('memory_history')};online['memory_stream']=stream
    before=stream.state.traces.clone();calls=[]
    old=m.prepare_memory_kv
    def prepare(b):calls.append(1);return old(b)
    monkeypatch.setattr(m,'prepare_memory_kv',prepare)
    m.sample_policy(online,horizon=4,num_steps=4,context=context(),decode_future=False,
        normalizer=SimpleNamespace(denormalize_action=lambda x:x))
    assert len(calls)==1
    torch.testing.assert_close(stream.state.traces,before,rtol=0,atol=0)
    with pytest.raises(ValueError,match='new physical frame'):
        m.observe_memory(data['obs_voxel'],frame_ids=[10],dt=1/30,stream=stream)


def test_per_environment_reset_and_camera_isolation():
    mem=CompressedEventMemory(dim=4,cameras=2)
    feature=torch.ones(2,2,8,8,4);feature[:,1]*=2
    activity=torch.ones(2,2,8,8)
    state=mem.update(feature,activity,torch.tensor([1/30,1/25]))
    torch.testing.assert_close(state.traces[:,1],2*state.traces[:,0])
    new=mem.update(feature,torch.zeros_like(activity),1/30,state,reset=torch.tensor([True,False]))
    assert new.traces[0].abs().sum()==0 and not new.seen[0].any()
    assert new.traces[1].abs().sum()>0 and new.seen[1].all()
    assert new.traces.dtype==torch.float32


def test_padding_has_no_activity_or_valid_memory_and_fixed_budget():
    mem=CompressedEventMemory(dim=4)
    frames=torch.ones(1,1,1,2,256,256);valid=torch.zeros(1,1,256,256)
    valid[:,:,64:128,64:128]=1
    features,activity=mem.encode_frames(frames,valid)
    state=mem.update(features[:,0],activity[:,0],1/30)
    read=mem.read(state,activity[:,0],torch.tensor([30.]),valid)
    assert read.tokens.shape==(1,28,4)
    assert read.valid.sum()==4 and 5 in read.fine_cells[0,0].tolist()
    assert torch.isfinite(read.tokens).all()


def test_coarse_only_and_full_fine_read_budgets():
    for refined,n in [(0,16),(4,28),(8,40),(16,64)]:
        mem=CompressedEventMemory(dim=4,refined_cells=refined)
        x=torch.randn(1,1,8,8,4);activity=torch.ones(1,1,8,8)
        state=mem.update(x,activity,1/30)
        read=mem.read(state,activity,30.)
        assert read.tokens.shape==(1,n,4) and read.valid.all()


def test_decay_expiration_and_historical_routing():
    mem=CompressedEventMemory(dim=4,max_age=.1)
    x=torch.randn(1,1,8,8,4);a=torch.zeros(1,1,8,8);a[:,:,6:8,6:8]=1
    state=mem.update(x,a,1/30)
    read=mem.read(state,torch.zeros_like(a),30.)
    assert 15 in read.fine_cells[0,0].tolist()
    expired=mem.decay_to_query(state,.2)
    assert expired.traces.abs().sum()<state.traces.abs().sum()
    assert not mem.read(expired,torch.zeros_like(a),30.).valid.any()


def test_policy_contract_and_explicit_dense_initialization():
    m=model();dense=CandidateDense(**options());m.initialize_from_dense_policy(dense.policy_state_dict())
    state=m.policy_state_dict();restored=model();restored.load_policy_state_dict(state)
    assert all(not k.startswith('event_codec.') for k in state)
    assert not any('stream' in k for k in state)
    wrong=copy.copy(state);wrong['event_memory.time_spec']=state['event_memory.time_spec']*2
    with pytest.raises(RuntimeError,match='contract'):restored.load_policy_state_dict(wrong)
    with pytest.raises(RuntimeError):restored.load_policy_state_dict(dense.policy_state_dict())


def test_attach_history_pads_start_and_preserves_current_future():
    sample={'obs_voxel':torch.ones(1,2,256,256),'future_voxel':torch.ones(4,1,2,256,256)*3}
    future=sample['future_voxel'];past=torch.ones(2,1,2,256,256)*2
    out=attach_history(sample,past,[0,1],2,25.,8)
    assert out['future_voxel'] is future
    assert out['memory_history_valid'].tolist()==[False]*6+[True]*2
    assert out['memory_history_indices'].tolist()==[-1]*6+[0,1]
    assert (out['memory_history_indices'][out['memory_history_valid']]<out['memory_anchor']).all()
    assert out['memory_history_voxel'][:6].count_nonzero()==0
    assert out['memory_history_dt'].tolist()==pytest.approx([.04]*8)


def test_velocity_core_fullgraph_compile_with_memory():
    m=model().eval();data=batch(horizon=4);kv=m.prepare_memory_kv(data)
    observed=m.event_codec.encode(data['obs_voxel'][:,None])[:,0]
    args=(observed,data['proprio'],torch.randn(1,1,1,16,32,32),data['action'],torch.tensor([.4]),context())
    expected=m.predict_velocity(*args,memory_kv=kv)
    compiled=torch.compile(m.predict_velocity,backend='eager',fullgraph=True,dynamic=False)
    actual=compiled(*args,memory_kv=kv)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b)


def test_memory_history_gradient_is_causal():
    m=model();data=batch(horizon=4);data['memory_history_voxel'].requires_grad_();data['future_voxel'].requires_grad_()
    kv=m.prepare_memory_kv(data)
    kv[0][0].square().sum().backward()
    assert data['memory_history_voxel'].grad is not None and data['memory_history_voxel'].grad.abs().sum()>0
    assert data['future_voxel'].grad is None


@pytest.mark.parametrize('preset,reader',[('train_ewm_memory_piper','ewm_piper_memory'),
                                        ('train_ewm_memory_event_only','ewm_mixed_event_only_memory')])
def test_registered_memory_config_preserves_dense_geometry(preset,reader):
    from pathlib import Path
    from hydra import compose,initialize_config_dir
    from openwam.dataloader.registry import DATASET_REGISTRY
    path=Path(__file__).resolve().parents[1]/'configs'
    with initialize_config_dir(config_dir=str(path),version_base=None):
        cfg=compose(config_name=preset)
    assert cfg.model.type=='ewm_memory' and cfg.dataloader.type==reader
    assert cfg.model.event_patch_size==4 and cfg.dataloader.chunk_size==24
    assert cfg.model.memory_refined_cells==4
    assert not cfg.training.pure_noise_eval.enabled
    assert reader in DATASET_REGISTRY
