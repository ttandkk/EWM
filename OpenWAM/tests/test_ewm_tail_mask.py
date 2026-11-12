"""Terminal windows must never supervise padding or read the next episode."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from openwam.dataloader import ewm_dataset as loader
from openwam.model.architectures.temporal_supervision import latent_mask, masked_mean
from openwam.model.architectures.lean_event_codec import LeanEventCodec
from tests.test_ewm_model import FakeLeanVAE, small_model, small_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def toy_dataset(monkeypatch, tmp_path, *, pad=True, length=6, stride=1):
    features={key:{'shape':(256,256,3)} for key in loader.VIDEO_KEYS}
    features.update({f'observation.event_frame.{cam}':dict(dtype='image',shape=(256,256,3)) for cam in loader.CAMERAS})
    features.update(action={'shape':[7]},**{'observation.state':{'shape':[8]}})
    info=dict(fps=30,features=features,data_path='data/{file_index}.parquet',video_path='unused')
    episodes=pd.DataFrame([dict(episode_index=i,length=length,**{'data/chunk_index':0,'data/file_index':0}) for i in (0,1)])
    monkeypatch.setattr(loader,'parse_info_json',lambda p:info)
    monkeypatch.setattr(loader,'load_episodes_parquet',lambda p:episodes.copy())
    monkeypatch.setattr(loader,'compute_file_local_offsets',lambda *a:np.array([0,length]))
    monkeypatch.setattr(loader,'apply_info_splits',lambda eps,*a,**kw:eps)
    monkeypatch.setattr(loader.pd,'read_parquet',lambda p:pd.DataFrame({'task_index':[0]},index=['catch']))
    norm=SimpleNamespace(voxel_bins=1,normalize_proprio=lambda x:x,normalize_action=lambda x:x)
    monkeypatch.setattr(loader.EWMNormalizer,'from_dataset_dir',lambda *a,**kw:norm)
    table=pa.table({'task_index':[0]*(2*length),'observation.state':[[i]*8 for i in range(2*length)],
                    'action':[[i+1]*7 for i in range(2*length)]})
    monkeypatch.setattr(loader,'_read_table',lambda p:table)
    d=loader.EWMDataset(tmp_path,chunk_size=4,voxel_bins=1,pad_tail=pad,window_stride=stride)
    d._rgb_observation=lambda row,idx:{'obs_video':torch.tensor(idx)}
    d._direct_event_frames=lambda path,idx:torch.tensor(idx)[:,None,None,None,None].expand(-1,2,2,2,2).float()
    return d


def test_padded_tail_stays_within_episode_and_preserves_last_action(monkeypatch,tmp_path):
    d=toy_dataset(monkeypatch,tmp_path)
    assert len(d)==12
    x=d[4]
    assert x['action_valid_mask'].tolist()==[True,True,False,False]
    assert x['future_valid_mask'].tolist()==[True,False,False,False]
    assert x['action'][:,0].tolist()==[5,6,0,0]
    assert x['future_voxel'][:,0,0,0,0].tolist()==[5,5,5,5]
    x=d[5]
    assert x['action_valid_mask'].sum()==1 and x['future_valid_mask'].sum()==0
    assert x['action'][0,0]==6 and x['obs_voxel'][0,0,0,0]==5
    x=d[6]
    assert x['action'][0,0]==7 and x['obs_voxel'][0,0,0,0]==6
    with pytest.raises(IndexError):d[12]


def test_legacy_windows_and_stride(monkeypatch,tmp_path):
    old=toy_dataset(monkeypatch,tmp_path,pad=False)
    reference=old[1]; assert len(old)==4 and 'action_valid_mask' not in reference
    padded=toy_dataset(monkeypatch,tmp_path)
    for key in ('action','obs_video','obs_voxel','future_voxel','proprio'):
        torch.testing.assert_close(reference[key],padded[1][key],rtol=0,atol=0)
    assert len(toy_dataset(monkeypatch,tmp_path,stride=2))==6


def test_single_row_episode_is_valid_action_only_window(monkeypatch,tmp_path):
    d=toy_dataset(monkeypatch,tmp_path,length=1)
    assert len(d)==2
    assert d[0]['action_valid_mask'].sum()==1
    assert not d[0]['future_valid_mask'].any()


@pytest.mark.parametrize('count',[0,1,3,4,5,7,8])
def test_latent_masks_require_complete_groups(count):
    valid=torch.arange(8)<count
    assert latent_mask(valid[None],4).sum()==count//4


def make_model():
    return small_model(event_codec_impl='leanvae16_joint',event_temporal_compression=4,
        event_spatial_compression=32,event_patch_size=4,
        event_codec=LeanEventCodec(vae=FakeLeanVAE()))


def labeled(count):
    x=small_batch(8)
    x['action_valid_mask']=(torch.arange(8)<min(count+1,8))[None]
    x['future_valid_mask']=(torch.arange(8)<count)[None]
    return x


@pytest.mark.parametrize('count',[0,1,5,8])
def test_padded_labels_do_not_change_loss_or_gradients(count):
    torch.manual_seed(3);m=make_model();x=labeled(count)
    kw=dict(t=torch.tensor([.4]),event_noise=torch.randn(1,2,1,16,4,4),action_noise=torch.randn(1,8,7))
    a=m(x,diagnostics=True,**kw);a['loss'].backward()
    gradients={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    changed={k:v.clone() if torch.is_tensor(v) else v for k,v in x.items()}
    changed['action'][~x['action_valid_mask']]=float('nan')
    changed['future_voxel'][~x['future_valid_mask']]=float('nan')
    m.zero_grad(set_to_none=True);b=m(changed,diagnostics=True,**kw)
    # Diagnostic targets must also exclude padding without NaNs.
    for key in a:
        assert torch.isfinite(b[key]),key
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    b['loss'].backward()
    for n,p in m.named_parameters():
        if n in gradients:torch.testing.assert_close(p.grad,gradients[n],rtol=0,atol=0)
    if count<4:assert b['loss_event']==0
    assert m.action_head[-1].weight.grad.abs().sum()>0


def test_all_valid_matches_unmasked():
    torch.manual_seed(4);m=make_model();x=labeled(8)
    plain={k:v for k,v in x.items() if k not in ('action_valid_mask','future_valid_mask')}
    kw=dict(t=torch.tensor([.5]),event_noise=torch.randn(1,2,1,16,4,4),action_noise=torch.randn(1,8,7))
    for key,value in m(plain,**kw).items():torch.testing.assert_close(value,m(x,**kw)[key])




def test_masked_mean_empty_has_finite_zero_gradient():
    x=torch.ones(2,3,7,requires_grad=True)
    loss=masked_mean(x.square(),torch.zeros(2,3,1,dtype=torch.bool))
    loss.backward();assert loss==0 and not x.grad.any()


def test_compiled_core_matches_eager():
    torch.manual_seed(12);m=make_model();x=labeled(5)
    observed=m.event_codec.encode(x['obs_voxel'][:,None])[:,0]
    args=(observed,x['proprio'],torch.randn(1,2,1,16,4,4),x['action'],torch.tensor([.5]),m._context(x,None))
    expected=m.predict_velocity(*args)
    actual=torch.compile(m.predict_velocity,backend='eager',fullgraph=True)(*args)
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b)
