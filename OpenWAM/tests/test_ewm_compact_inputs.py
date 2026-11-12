"""Loss, gradients and time alignment must survive input-only optimizations."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openwam.model.architectures.ewm import EWMModel
from tests.ewm_helpers import options
from tests.test_ewm_model import small_batch


def model():
    return EWMModel(**options(cameras=2))


def data():
    return small_batch(4, cameras=2)


def test_compact_events_and_cached_current_preserve_loss_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(187)
    m = model()
    raw = data()
    for key,limit in (('obs_voxel',15),('future_voxel',15)):
        raw[key] = torch.randint(0,limit+1,raw[key].shape,dtype=torch.uint8)
    normalized = dict(raw)
    for key,limit in (('obs_voxel',15),('future_voxel',15)):
        normalized[key] = raw[key].float()/limit
    with torch.no_grad():
        features = m.context_encoder.encode_rgb_features(raw['obs_video'])
    cached = dict(raw,current_dino_features=features)
    cached.pop('obs_video')
    noise = dict(t=torch.tensor([.4]),event_noise=torch.randn(1,1,2,16,4,4),
                 action_noise=torch.randn_like(raw['action']))
    baseline = m(normalized,**noise)['loss']
    baseline.backward()
    gradients = {n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
    calls = m.context_encoder.dino.calls
    m.zero_grad(set_to_none=True)
    result = m(cached,**noise)['loss']
    result.backward()
    assert m.context_encoder.dino.calls == calls
    torch.testing.assert_close(result,baseline,rtol=0,atol=0)
    for n,p in m.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad,gradients[n],rtol=0,atol=0)
    prepared = m._normalize_event_counts(cached)
    again = m._normalize_event_counts(prepared)
    for key in ('obs_voxel','future_voxel'):
        assert raw[key].dtype == torch.uint8
        torch.testing.assert_close(prepared[key],normalized[key],rtol=0,atol=0)
        assert again[key] is prepared[key]  # Never normalize twice.


def test_cached_current_read_avoids_rgb_decode_and_future_rows(tmp_path,monkeypatch):
    from openwam.dataloader.ewm_dataset import EWMDataset
    from openwam.dataloader.current_dino_cache import CurrentDinoCache, dino_fingerprint
    d = EWMDataset.__new__(EWMDataset)
    d.context_frames = 1
    d.episodes = pd.DataFrame([dict(episode_index=0,length=4)])
    encoder = tmp_path/'encoder';encoder.mkdir();(encoder/'config.json').write_text('{}')
    target = tmp_path/'cache';target.mkdir()
    values = torch.arange(4,dtype=torch.bfloat16)[:,None,None,None,None].expand(-1,2,1024,16,16).contiguous()
    np.save(target/'episode.npy',values.view(torch.uint16).numpy())
    manifest = dict(format='ewm_dino_current_bf16_v1',grid=16,cameras=2,dim=1024,
        dtype='bfloat16_uint16_bits',preprocessing='EWM ImageNet normalization; independent T1; full grid',
        dino_fingerprint=dino_fingerprint(encoder),episodes={'0':dict(file='episode.npy',length=4,source={})})
    (target/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr('openwam.dataloader.current_dino_cache.source_signature',lambda *a:{})
    d.current_dino_cache = CurrentDinoCache(d,target,encoder)
    d._rgb = lambda *a: pytest.fail('cached training must not decode RGB')
    row = d.episodes.iloc[0]
    sample = d._rgb_observation(row,np.array([1]))
    assert set(sample) == {'current_dino_features'}
    assert torch.equal(sample['current_dino_features'],values[1])
    # Poison future data in a private RAM copy, leaving the source cache unchanged.
    changed = np.array(d.current_dino_cache.maps[0],copy=True);changed[2:]=0;d.current_dino_cache.maps[0]=changed
    assert torch.equal(sample['current_dino_features'],d._rgb_observation(row,np.array([1]))['current_dino_features'])
    (encoder/'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='fingerprint'):
        d.current_dino_cache = CurrentDinoCache(d,target,encoder)


def test_current_cache_builder_resume_and_source_guard(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from torch import nn
    from scripts.train_ewm.prepare_piper_current_dino import prepare_task
    class Dataset:
        episodes = pd.DataFrame([dict(episode_index=0,length=5)])
        def _rgb(self,row,indices):
            return torch.tensor(indices,dtype=torch.uint8)[:,None,None,None,None].expand(-1,2,3,256,256)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.weight = nn.Parameter(torch.ones(1)); self.calls = 0
        def batch_encode(self,pixels):
            self.calls += 1
            assert pixels.shape[0] == 4  # Two pairs, including the final one-frame tail.
            return pixels.mean((1,2,3,4))[:,None,None,None,None].expand(-1,1024,1,16,16)
    dino = tmp_path/'pretrained/dinov3-vitl16';dino.mkdir(parents=True)
    (dino/'config.json').write_text('{}')
    monkeypatch.setattr('openwam.dataloader.ewm_dataset.EWMDataset',lambda *a,**k:Dataset())
    monkeypatch.setattr('openwam.dataloader.current_dino_cache.source_signature',lambda *a:{'version':1})
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:nullcontext())
    encoder = Encoder();cache=tmp_path/'derived'
    prepare_task(tmp_path,cache,'unit',encoder,torch.device('cpu'),2)
    target = cache/'piper_unit/dino_current16'
    manifest = json.loads((target/'manifest.json').read_text())
    assert manifest['dtype']=='bfloat16_uint16_bits'
    assert manifest['encoding_batch_size']==2
    assert manifest['padding']=='repeat_last_frame_pair_discard_padding'
    array = np.load(target/'episode_000000.npy')
    assert array.shape==(5,2,1024,16,16) and array.dtype==np.uint16
    assert not np.array_equal(array[0],array[1])
    calls = encoder.calls
    prepare_task(tmp_path,cache,'unit',encoder,torch.device('cpu'),2)
    assert encoder.calls==calls
    with pytest.raises(ValueError,match='stale'):
        prepare_task(tmp_path,cache,'unit',encoder,torch.device('cpu'),3)
    monkeypatch.setattr('openwam.dataloader.current_dino_cache.source_signature',lambda *a:{'version':2})
    with pytest.raises(ValueError,match='stale'):
        prepare_task(tmp_path,cache,'unit',encoder,torch.device('cpu'),2)


def test_fixed_encoding_preserves_order_without_reading_padding_frames():
    from torch import nn
    from scripts.train_ewm.prepare_piper_current_dino import encode_fixed_batch
    class Dataset:
        reads = []
        def _rgb(self,row,indices):
            self.reads.extend(indices.tolist())
            return torch.tensor(indices,dtype=torch.uint8)[:,None,None,None,None].expand(-1,2,3,8,8)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.ones(1));self.inputs=[]
        def batch_encode(self,pixels):
            assert pixels.shape[0]==8
            self.inputs.append(pixels.clone())
            return pixels.mean((1,2,3,4))[:,None,None,None,None].expand(-1,1024,1,16,16)
    d=Dataset();m=Encoder();indices=np.array([0,3,9,12,17])
    output=encode_fixed_batch(d,None,indices,m,torch.device('cpu'),4)
    assert d.reads==indices.tolist()  # No synthetic later timestamps are read.
    assert output.shape==(5,2,1024,16,16)
    assert bool((output[1:,0,0,0,0]>output[:-1,0,0,0,0]).all())
    last=m.inputs[-1].reshape(4,2,3,1,8,8)
    torch.testing.assert_close(last,last[:1].expand_as(last),rtol=0,atol=0)
    # Separately encoded tail with identical shape must match the saved position.
    tail=encode_fixed_batch(d,None,indices[-1:],m,torch.device('cpu'),4)
    torch.testing.assert_close(output[-1:],tail,rtol=0,atol=0)


