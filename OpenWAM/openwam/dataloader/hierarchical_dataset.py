"""Causal Piper RGB4/event10 windows and frozen DINO feature cache."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from openwam.dataloader.ewm_dataset import EWMDataset
from openwam.dataloader.utils import get_cfg


def dino_fingerprint(path):
    root=Path(path).resolve()
    names=sorted(p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.bin'))
    if not names:
        raise ValueError('DINO path has no config/weight files')
    specification=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in names]
    return hashlib.sha256(json.dumps([str(root),specification]).encode()).hexdigest()


def source_signature(dataset,row):
    result={}
    from openwam.dataloader.ewm_dataset import VIDEO_KEYS
    for key in VIDEO_KEYS:
        path=dataset.dataset_dir/dataset.video_path.format(video_key=key,
            chunk_index=int(row[f'videos/{key}/chunk_index']),file_index=int(row[f'videos/{key}/file_index']))
        stat=path.stat()
        result[key]=[str(path.resolve()),stat.st_size,stat.st_mtime_ns,int(row[f'_video_offset/{key}'])]
    return result


class HierarchicalMemoryPiperDataset(EWMDataset):
    def __init__(self,*args,memory_dino_cache_dir=None,memory_dino_path=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.gpu_voxelization or not self.direct_event_frames or self.observation_row_offset!=0:
            raise ValueError('RGB/event memory requires aligned direct event frames and CPU/direct reads')
        self.configure_memory_cache(memory_dino_cache_dir,memory_dino_path)

    def configure_memory_cache(self,path,dino_path):
        self.memory_cache_dir=Path(path) if path else None
        self.memory_cache_manifest=None
        if self.memory_cache_dir is None:
            return
        manifest=json.loads((self.memory_cache_dir/'manifest.json').read_text())
        if manifest.get('format')!='ewm_dino_history_bf16_v1' or manifest.get('grid')!=8 or manifest.get('cameras')!=2 or manifest.get('dtype')!='bfloat16_uint16_bits' or manifest.get('preprocessing')!='EWM ImageNet normalization; independent T1; area pool8':
            raise ValueError('incompatible DINO history cache')
        if not dino_path or manifest.get('dino_fingerprint')!=dino_fingerprint(dino_path):
            raise ValueError('DINO cache encoder fingerprint mismatch')
        for _,row in self.episodes.iterrows():
            entry=manifest['episodes'].get(str(int(row['episode_index'])))
            if entry is None or entry['source']!=source_signature(self,row) or entry['length']!=int(row['length']):
                raise ValueError('missing/stale episode in DINO history cache')
            shape=(int(row['length']),2,int(manifest['dim']),8,8)
            array=np.load(self.memory_cache_dir/entry['file'],mmap_mode='r',allow_pickle=False)
            if array.shape!=shape or array.dtype!=np.uint16:
                raise ValueError('invalid BF16 DINO cache shape/dtype')
        self.memory_cache_manifest=manifest
        self._memory_maps={}

    @classmethod
    def from_config(cls,config,split='train'):
        dataset=super().from_config(config,split)
        dataset.configure_memory_cache(get_cfg(config,'memory_dino_cache_dir',None),get_cfg(config,'memory_dino_path',None))
        return dataset

    def __getitem__(self,index):
        sample=dict(super().__getitem__(index))
        local=int(np.searchsorted(self._cumulative,index,side='right')-1)
        row=self.episodes.iloc[local]
        anchor=self.first_anchor+int(index-self._cumulative[local])*self.window_stride
        indices=anchor+np.arange(-9,1,dtype=np.int64)
        valid=indices>=self.first_anchor
        history=sample['obs_voxel'].new_zeros((10,*sample['obs_voxel'].shape))
        past=indices[(indices<anchor)&valid]
        if len(past):
            path=self.dataset_dir/self.data_path.format(chunk_index=int(row['data/chunk_index']),file_index=int(row['data/file_index']))
            history[:-1][valid[:-1]]=self._direct_event_frames(path,int(row['_data_offset'])+past)
        history[-1]=sample['obs_voxel']
        rindices=anchor+np.array([-9,-6,-3],dtype=np.int64)
        rvalid=rindices>=self.first_anchor
        if self.memory_cache_manifest is not None:
            episode=int(row['episode_index'])
            if episode not in self._memory_maps:
                entry=self.memory_cache_manifest['episodes'][str(episode)]
                self._memory_maps[episode]=np.load(self.memory_cache_dir/entry['file'],mmap_mode='r',allow_pickle=False)
            arr=self._memory_maps[episode]
            values=np.zeros((3,*arr.shape[1:]),dtype=np.uint16)
            values[rvalid]=arr[rindices[rvalid]]
            sample['memory_rgb_features']=torch.from_numpy(values).view(torch.bfloat16)
        else:
            rgb=sample['obs_video'].new_zeros((3,*sample['obs_video'].shape))
            if rvalid.any():rgb[torch.from_numpy(rvalid)]=self._rgb(row,rindices[rvalid])
            sample['memory_rgb']=rgb
        sample.update(memory_event=history,memory_event_valid=torch.from_numpy(valid),
            memory_rgb_valid=torch.tensor([*rvalid.tolist(),True]),memory_indices=torch.from_numpy(indices),
            memory_anchor=torch.tensor(anchor),fps=float(self.fps))
        return sample
