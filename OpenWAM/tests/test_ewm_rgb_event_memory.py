import torch
from openwam.model.architectures.hierarchical_memory import RGBEventMemory,event_difference_inputs

def test_pixel_difference_reconstructs_signed_polarities():
    e=torch.rand(2,10,2,2,128,128)
    x,valid=event_difference_inputs(e,torch.ones(2,10,dtype=torch.bool))
    last=x[:,0]; restored=[last]
    for delta in x[:,1:].flip(1).unbind(1):
        last=last-delta;restored.append(last)
    recovered=torch.stack(restored[::-1],1)
    torch.testing.assert_close(recovered[:,:,:,0],e[:,:,:,0])
    torch.testing.assert_close(recovered[:,:,:,2],e[:,:,:,1])
    assert valid.all() and (x[:,1:]<0).any()

def test_missing_bins_cannot_create_padding_motion():
    e=torch.ones(1,10,1,2,128,128);valid=torch.zeros(1,10,dtype=torch.bool);valid[:,-2:]=True
    x,v=event_difference_inputs(e,valid)
    assert x[:,1:].count_nonzero()==0 and v.sum()==2

def test_fusion_shapes_gradients_zero_event_and_padding():
    torch.set_num_threads(2)
    m=RGBEventMemory(rgb_dim=6,dim=16,cameras=2,heads=2)
    rgb=torch.randn(1,4,2,6,8,8,requires_grad=True)
    event=torch.zeros(1,10,2,2,128,128,requires_grad=True)
    valid=torch.ones(1,10,dtype=torch.bool)
    out=m(rgb,event,rgb_valid=valid[:,:4],event_valid=valid,fps=30.)
    assert out.tokens.shape==(1,128,16) and out.valid.all() and torch.isfinite(out.tokens).all()
    out.tokens.square().mean().backward()
    assert rgb.grad.abs().sum()>0 and event.grad.abs().sum()>0
    pad=torch.zeros(1,1,128,128)
    out=m(rgb,event,rgb_valid=valid[:,:4],event_valid=valid,fps=30.,valid_region=pad)
    assert not out.valid.any() and out.tokens.count_nonzero()==0

import copy
from types import SimpleNamespace
import pytest
import torch
from openwam.model.architectures.hierarchical_ewm import HierarchicalMemoryEWMModel
from tests.test_ewm_memory import options,batch,context
from openwam.model.architectures.ewm import EWMModel


def data(cameras=2):
    x=batch(cameras=cameras,horizon=4)
    x['memory_event']=torch.rand(1,10,cameras,2,256,256)
    x['memory_event'][:,-1]=x['obs_voxel']
    x['memory_event_valid']=torch.ones(1,10,dtype=torch.bool)
    x['memory_rgb_valid']=torch.ones(1,4,dtype=torch.bool)
    x['memory_rgb_features']=torch.rand(1,3,2,6,8,8)
    return x


def model(cameras=2,enabled=True):
    return HierarchicalMemoryEWMModel(memory_dim=16,memory_heads=2,memory_enabled=enabled,**options(cameras))


def test_cached_current_dino_reused_and_all_memory_gradients():
    torch.set_num_threads(2)
    m=model();x=data()
    loss=m(x)['loss'];loss.backward()
    assert m.context_encoder.dino.calls==1
    for name,p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(),name
    assert all(p.grad is None for p in m.context_encoder.dino.parameters())
    assert m.token_counts()==dict(query=921,key_value=1049,memory=128)


def test_disabled_matches_dense_exactly():
    m=model(enabled=False);dense=EWMModel(**options(2));m.initialize_from_dense_policy(dense.policy_state_dict())
    x=data();ctx=context();t=torch.tensor([.4]);en=torch.randn(1,1,2,16,32,32);an=torch.randn_like(x['action'])
    for a,b in zip(dense(x,context=ctx,t=t,event_noise=en,action_noise=an).values(),m(x,context=ctx,t=t,event_noise=en,action_noise=an).values()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_future_causality_and_history_effect():
    m=model();x=data();first=m.prepare_memory_kv(x)
    altered=dict(x,future_voxel=x['future_voxel']*100,action=x['action']*100)
    second=m.prepare_memory_kv(altered)
    for aa,bb in zip(first,second):
        for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=0,atol=0)
    altered=dict(x,memory_rgb_features=torch.zeros_like(x['memory_rgb_features']))
    assert not torch.allclose(first[0][0],m.prepare_memory_kv(altered)[0][0])
    altered=dict(x,memory_event=x['memory_event'].clone());altered['memory_event'][:,:-1]=0
    assert not torch.allclose(first[0][0],m.prepare_memory_kv(altered)[0][0])


def test_sampling_prepares_memory_once_and_checkpoint_contract(monkeypatch):
    m=model().eval();x=data();calls=[];real=m.prepare_memory_kv
    def prepare(batch):calls.append(1);return real(batch)
    monkeypatch.setattr(m,'prepare_memory_kv',prepare)
    out=m.sample_policy(x,horizon=4,num_steps=3,decode_future=False,
        normalizer=SimpleNamespace(denormalize_action=lambda x:x))
    assert len(calls)==1 and out['action'].shape==(1,4,7)
    state=m.policy_state_dict();model().load_policy_state_dict(state)
    bad=copy.copy(state);bad['rgb_event_memory.memory_spec']=state['rgb_event_memory.memory_spec']+1
    with pytest.raises(RuntimeError,match='contract'):model().load_policy_state_dict(bad)


def test_online_rgb_matches_cached_and_chunking():
    m=model();x=data();x.pop('memory_rgb_features');x['memory_rgb']=torch.randint(0,256,(1,3,2,3,256,256),dtype=torch.uint8)
    prepared=m._with_current_features(x)
    rgb=m._rgb_memory_features(prepared)
    cached=dict(prepared,memory_rgb_features=rgb[:,:3])
    a,b=m.prepare_memory_kv(prepared),m.prepare_memory_kv(cached)
    for aa,bb in zip(a,b):
        for u,v in zip(aa,bb):torch.testing.assert_close(u,v,rtol=0,atol=0)
    assert m.context_encoder.dino.calls==3


def test_invalid_past_payload_does_not_change_memory():
    m=model();x=data();x['memory_event_valid'][:,:5]=False;x['memory_rgb_valid'][:,:2]=False
    a=m.prepare_memory_kv(x)
    changed=dict(x,memory_event=x['memory_event'].clone(),memory_rgb_features=x['memory_rgb_features'].clone())
    changed['memory_event'][:,:5]=1000
    changed['memory_rgb_features'][:,:2]=1000
    b=m.prepare_memory_kv(changed)
    for aa,bb in zip(a,b):
        for u,v in zip(aa,bb):torch.testing.assert_close(u,v,rtol=0,atol=0)


def test_velocity_compiles_with_fused_memory():
    m=model().eval();x=data();kv=m.prepare_memory_kv(x)
    observed=m.event_codec.encode(x['obs_voxel'][:,None])[:,0]
    args=(observed,x['proprio'],torch.randn(1,1,2,16,32,32),x['action'],torch.tensor([.4]),context())
    expected=m.predict_velocity(*args,memory_kv=kv)
    compiled=torch.compile(m.predict_velocity,backend='eager',fullgraph=True,dynamic=False)
    for a,b in zip(expected,compiled(*args,memory_kv=kv)):torch.testing.assert_close(a,b)


def test_memory_history_ring_gaps_reset_and_snapshot():
    import numpy as np
    from scripts.eval_ewm.ewm_memory_history import RGBEventHistoryBuffer
    history=RGBEventHistoryBuffer()
    for step in range(12):
        r=np.full((2,3,4,4),step,dtype=np.uint8)
        e=np.full((2,2,4,4),step,dtype=np.float32)
        out=history.observe(r,e,step)
    assert out['memory_event'][:,0,0,0,0].tolist()==list(range(2,12))
    assert out['memory_rgb'][:,0,0,0,0].tolist()==[2,5,8]
    snapshot=out['memory_event'].copy();history.observe(r,e,13)
    np.testing.assert_equal(out['memory_event'],snapshot)
    with pytest.raises(ValueError,match='new physical'):history.observe(r,e,13)
    first=RGBEventHistoryBuffer().observe(r,e,0)
    assert first['memory_rgb_valid'].tolist()==[False,False,False,True]
    assert first['memory_event_valid'].sum()==1


def test_registered_preset_and_cache_loader(tmp_path,monkeypatch):
    import json
    import numpy as np
    import pandas as pd
    from hydra import compose,initialize_config_dir
    from pathlib import Path
    from openwam.dataloader.hierarchical_dataset import HierarchicalMemoryPiperDataset,dino_fingerprint
    from openwam.dataloader.ewm_dataset import EWMDataset
    from openwam.dataloader.registry import DATASET_REGISTRY
    config_dir=Path(__file__).resolve().parents[1]/'configs'
    with initialize_config_dir(config_dir=str(config_dir),version_base=None):
        cfg=compose(config_name='train_ewm_rgb_event_memory_piper')
    assert cfg.model.type=='ewm_rgb_event_memory' and cfg.model.event_camera_mode=='dual'
    assert cfg.dataloader.type in DATASET_REGISTRY and not cfg.training.pure_noise_eval.enabled
    d=HierarchicalMemoryPiperDataset.__new__(HierarchicalMemoryPiperDataset)
    d.first_anchor=0;d.window_stride=1;d.fps=30.;d._cumulative=np.array([0,12])
    d.episodes=pd.DataFrame([dict(episode_index=0,length=12,**{'data/chunk_index':0,'data/file_index':0,'_data_offset':0})])
    d.dataset_dir=tmp_path;d.data_path='{file_index}.parquet'
    def base(self,index):
        return dict(obs_voxel=torch.full((2,2,4,4),float(index)),obs_video=torch.full((2,3,4,4),index,dtype=torch.uint8))
    monkeypatch.setattr(EWMDataset,'__getitem__',base)
    d._direct_event_frames=lambda path,idx:torch.tensor(idx)[:,None,None,None,None].expand(-1,2,2,4,4).float()
    d._rgb=lambda row,idx:torch.tensor(idx,dtype=torch.uint8)[:,None,None,None,None].expand(-1,2,3,4,4)
    d.configure_memory_cache(None,None)
    first=d[0];assert first['memory_event_valid'].sum()==1
    online=d[10]
    assert online['memory_event'][:,0,0,0,0].tolist()==list(range(1,11))
    assert online['memory_rgb'][:,0,0,0,0].tolist()==[1,4,7]
    encoder=tmp_path/'encoder';encoder.mkdir();(encoder/'config.json').write_text('{}')
    cache=tmp_path/'cache';cache.mkdir()
    values=torch.arange(12,dtype=torch.float32)[:,None,None,None,None].expand(-1,2,6,8,8).to(torch.bfloat16).contiguous()
    np.save(cache/'episode.npy',values.view(torch.uint16).numpy())
    manifest=dict(format='ewm_dino_history_bf16_v1',grid=8,cameras=2,dim=6,
        dtype='bfloat16_uint16_bits',preprocessing='EWM ImageNet normalization; independent T1; area pool8',
        dino_fingerprint=dino_fingerprint(encoder),episodes={'0':dict(file='episode.npy',length=12,source={})})
    (cache/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr('openwam.dataloader.hierarchical_dataset.source_signature',lambda *args:{})
    d.configure_memory_cache(cache,encoder)
    cached=d[10]
    assert 'memory_rgb' not in cached and cached['memory_rgb_features'].dtype==torch.bfloat16
    torch.testing.assert_close(cached['memory_rgb_features'][:,0,0,0,0],torch.tensor([1,4,7],dtype=torch.bfloat16))
    (encoder/'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='fingerprint'):d.configure_memory_cache(cache,encoder)


def test_memory_history_survives_wire_roundtrip():
    import numpy as np
    from scripts.eval_ewm.ewm_eval_wire import encode_message,decode_message
    from scripts.eval_ewm.ewm_memory_history import RGBEventHistoryBuffer
    history=RGBEventHistoryBuffer().observe(np.zeros((2,3,4,4),dtype=np.uint8),np.zeros((2,2,4,4),dtype=np.float32),0)
    metadata,arrays=decode_message(encode_message({'op':'predict'},history))
    assert metadata=={'op':'predict'}
    for key,value in history.items():np.testing.assert_equal(arrays[key],value)


def test_profile_includes_shared_current_dino():
    m=model();result=m(data(),profile_timing=True)
    assert result['_profile']['memory_current_dino']>=0
    assert 'memory_encode' in result['_profile'] and m.context_encoder.dino.calls==1


def test_one_step_through_shared_trainer_without_checkpoint(tmp_path):
    from accelerate import Accelerator
    from tests.test_ewm_resume import config,TinyDataset
    from scripts.train_ewm.train_ewm import train
    class MemoryDataset(TinyDataset):
        def __len__(self):return 2
        def __getitem__(self,index):
            x=data()
            return {k:v[0] if isinstance(v,torch.Tensor) else v for k,v in x.items()}
    cfg=config(tmp_path,1)
    cfg.model.type='ewm_rgb_event_memory';cfg.dataloader.chunk_size=4
    cfg.training.batch_size=1;cfg.training.save_checkpoints=False
    cfg.training.save_full_states_for_resume=False
    m=model();before=m.rgb_event_memory.event_encoder[0].weight.detach().clone()
    train(cfg,dataset=MemoryDataset(),model=m,accelerator=Accelerator(cpu=True,mixed_precision='no'))
    assert not torch.equal(before,m.rgb_event_memory.event_encoder[0].weight)
    assert (tmp_path/'config.yaml').exists() and not (tmp_path/'checkpoint.pt').exists()


def test_cache_generation_cli_writes_bf16_artifact(tmp_path,monkeypatch):
    import json
    import sys
    import numpy as np
    import pandas as pd
    from scripts.train_ewm.cache_dino_memory import main
    from tests.test_ewm_model import FakeDino
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
    class CacheDataset:
        episodes=pd.DataFrame([dict(episode_index=0,length=4)])
        def _rgb(self,row,indices):
            return torch.tensor(indices,dtype=torch.uint8)[:,None,None,None,None].expand(-1,2,3,4,4).contiguous()
    encoder_path=tmp_path/'dino';encoder_path.mkdir();(encoder_path/'config.json').write_text('{}')
    target=tmp_path/'cache'
    dino=FakeDino()
    monkeypatch.setattr('openwam.dataloader.ewm_dataset.EWMDataset',lambda *args,**kwargs:CacheDataset())
    monkeypatch.setattr('openwam.dataloader.hierarchical_dataset.source_signature',lambda *args:{})
    monkeypatch.setattr(DinoV3VideoEncoder,'from_pretrained',lambda *args,**kwargs:dino)
    monkeypatch.setattr(sys,'argv',['cache_dino_memory','--dataset',str(tmp_path),'--output',str(target),
        '--dino-path',str(encoder_path),'--device','cpu','--batch-size','2'])
    main()
    manifest=json.loads((target/'manifest.json').read_text())
    array=np.load(target/'episode_000000.npy',allow_pickle=False)
    assert array.shape==(4,2,6,8,8) and array.dtype==np.uint16
    features=torch.from_numpy(array).view(torch.bfloat16)
    assert torch.isfinite(features).all() and dino.weight.dtype==torch.float32
    assert manifest['dtype']=='bfloat16_uint16_bits' and manifest['dim']==6
    with pytest.raises(FileExistsError,match='completed'):main()
