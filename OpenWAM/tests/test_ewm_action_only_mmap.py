import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

import openwam.dataloader.ewm_action_only as loader
builder=load('action_only_mmap_builder',ROOT/'scripts/train_ewm/build_ewm_action_only_cache.py')

@pytest.fixture
def exported(tmp_path,monkeypatch):
    root=tmp_path/'dataset';cache=tmp_path/'cache';(root/'meta').mkdir(parents=True)
    (root/'meta/piper_temporal_contract.json').write_text(json.dumps(dict(temporal_contract='aligned_event_frame_pre_action_v1')))
    source=root/'data/file0.parquet';source.parent.mkdir()
    states=np.arange(8*8,dtype=np.float32).reshape(8,8)/10
    actions=np.arange(8*7,dtype=np.float32).reshape(8,7)/100
    pq.write_table(pa.table({'observation.state':states.tolist(),'action':actions.tolist(),'task_index':[0]*8,'observation.event_frame.cam0':[b'forbidden']*8}),source)
    for view in ['cam0','cam_wrist']:
        path=root/f'videos/observation.images.{view}/file0.mp4';path.parent.mkdir(parents=True)
        with av.open(str(path),'w') as container:
            stream=container.add_stream('libx264',rate=30);stream.width=stream.height=256;stream.pix_fmt='yuv420p';stream.thread_count=1
            for frame in range(8):
                rgb=np.zeros((256,256,3),dtype=np.uint8);rgb[:,:,0]=frame*20;rgb[:,:,1]=30 if view=='cam0' else 130
                rgb[:,:,2]=np.arange(256,dtype=np.uint8)[None,:]
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
    builder.build(root,cache)
    def init(self,dataset_dir,**kwargs):
        self.dataset_dir=Path(dataset_dir);self.first_anchor=0;self.window_stride=1;self.observation_row_offset=0;self.chunk_size=2
        self.data_path='data/file0.parquet';self.video_path='videos/{video_key}/file0.mp4'
        self.height=self.width=256;self.rgb_decode_backend='pyav'
        self._cumulative=np.array([0,2,4]);self.prompts={0:'conveyor'}
        self.normalizer=SimpleNamespace(normalize_action=lambda x:x*.5,normalize_proprio=lambda x:x*.25)
        records=[]
        for offset in [0,4]:
            row={'data/chunk_index':0,'data/file_index':0,'_data_offset':offset,'length':4}
            for key in loader.VIDEO_KEYS:
                row.update({f'videos/{key}/chunk_index':0,f'videos/{key}/file_index':0,f'_video_offset/{key}':offset})
            records.append(row)
        self.episodes=pd.DataFrame(records)
    monkeypatch.setattr(loader.EWMDataset,'__init__',init)
    yield root,cache
    loader._read_action_table.cache_clear();loader._read_action_mmap.cache_clear()


def test_cached_samples_match_baseline_across_shared_file_episode_offsets(exported,monkeypatch):
    root,cache=exported
    original=loader.PiperActionOnlyDataset(root)
    baseline=[original[i] for i in range(4)]
    cached=loader.PiperActionOnlyDataset(root,action_only_mmap_cache_dir=cache)
    def forbidden(*args,**kwargs):raise AssertionError('mmap sample decoded video or Parquet')
    monkeypatch.setattr(pq,'read_table',forbidden);monkeypatch.setattr(loader.EWMDataset,'_rgb',forbidden)
    for i,expected in enumerate(baseline):
        actual=cached[i]
        for name in ['obs_video','proprio','action']:torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
        assert actual['prompt']==expected['prompt']
    with pytest.raises(IndexError):cached[4]


def test_arrays_are_readonly_memory_maps(exported):
    root,cache=exported
    dataset=loader.PiperActionOnlyDataset(root,action_only_mmap_cache_dir=cache)
    paths=list(dataset._video_mmap_files.values())
    for item in dataset._action_mmap_files.values():paths.extend(item.values())
    for path in paths:
        array=loader._read_action_mmap(path)
        assert isinstance(array,np.memmap) and not array.flags.writeable


def test_stale_source_rejected(exported):
    root,cache=exported;path=root/'data/file0.parquet';stat=path.stat()
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='stale'):loader.PiperActionOnlyDataset(root,action_only_mmap_cache_dir=cache)


def test_invalid_action_array_rejected(exported):
    root,cache=exported
    np.save(cache/'data/file0/action.npy',np.zeros((8,8),dtype=np.float32))
    with pytest.raises(ValueError,match='invalid'):loader.PiperActionOnlyDataset(root,action_only_mmap_cache_dir=cache)


def test_config_selects_mmap_and_missing_completion_manifest_rejected(exported):
    root,cache=exported
    config=OmegaConf.create(dict(dataset_dir=str(root),chunk_size=2,voxel_bins=1,action_only_mmap_cache_dir=str(cache)))
    dataset=loader.PiperActionOnlyDataset.from_config(config)
    assert dataset.action_only_mmap_cache_dir==cache.resolve()
    (cache/'manifest.json').unlink()
    with pytest.raises(FileNotFoundError):loader.PiperActionOnlyDataset(root,action_only_mmap_cache_dir=cache)
