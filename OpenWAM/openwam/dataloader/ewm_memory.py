"""Explicit past-event wrappers; random samples never inherit stream state."""
import numpy as np
import torch
from openwam.dataloader.ewm_dataset import EWMDataset
from openwam.dataloader.ewm_event_only import EWMMixedEventOnlyDataset, _event_occupancy
from openwam.dataloader.utils import get_cfg


def attach_history(sample,frames,indices,anchor,fps,history_frames):
    sample=dict(sample)
    count=len(indices)
    history=torch.zeros((history_frames,*sample['obs_voxel'].shape),dtype=sample['obs_voxel'].dtype)
    valid=torch.zeros(history_frames,dtype=torch.bool)
    history_indices=torch.full((history_frames,),-1,dtype=torch.long)
    if count:
        history[-count:]=frames
        valid[-count:]=True
        history_indices[-count:]=torch.as_tensor(indices)
    sample.update(memory_history_voxel=history,memory_history_valid=valid,
        memory_history_indices=history_indices,memory_anchor=torch.tensor(anchor),
        memory_history_dt=torch.full((history_frames,),1/float(fps)),fps=float(fps),
        memory_query_dt=torch.tensor(1/float(fps)))
    return sample


class MemoryPiperDataset(EWMDataset):
    def __init__(self,*args,memory_history_frames=8,**kwargs):
        super().__init__(*args,**kwargs)
        if self.gpu_voxelization:
            raise ValueError('initial memory reader requires CPU/direct event frames')
        if memory_history_frames<=0:
            raise ValueError('memory_history_frames must be positive')
        self.memory_history_frames=int(memory_history_frames)

    @classmethod
    def from_config(cls,config,split='train'):
        dataset=super().from_config(config,split)
        dataset.memory_history_frames=int(get_cfg(config,'memory_history_frames',8))
        if dataset.memory_history_frames<=0:
            raise ValueError('memory_history_frames must be positive')
        return dataset

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        local=int(np.searchsorted(self._cumulative,index,side='right')-1)
        row=self.episodes.iloc[local]
        anchor=self.first_anchor+int(index-self._cumulative[local])*self.window_stride
        indices=np.arange(max(self.first_anchor,anchor-self.memory_history_frames),anchor,dtype=np.int64)
        if len(indices):
            if self.direct_event_frames:
                path=self.dataset_dir/self.data_path.format(chunk_index=int(row['data/chunk_index']),file_index=int(row['data/file_index']))
                frames=self._direct_event_frames(path,int(row['_data_offset'])+indices)
            else:
                frames=torch.stack([self._voxel(int(row['episode_index']),int(i)) for i in indices])
        else:
            frames=sample['obs_voxel'].new_empty((0,*sample['obs_voxel'].shape))
        return attach_history(sample,frames,indices,anchor,self.fps,self.memory_history_frames)


class MemoryEventOnlyDataset(EWMMixedEventOnlyDataset):
    def __init__(self,*,memory_history_frames=8,**kwargs):
        super().__init__(**kwargs)
        self.memory_history_frames=int(memory_history_frames)
        if self.memory_history_frames<=0:
            raise ValueError('memory_history_frames must be positive')

    @classmethod
    def from_config(cls,config,split='train'):
        dataset=super().from_config(config,split)
        dataset.memory_history_frames=int(get_cfg(config,'memory_history_frames',8))
        if dataset.memory_history_frames<=0:
            raise ValueError('memory_history_frames must be positive')
        return dataset

    def __getitem__(self,index):
        sample=super().__getitem__(index)
        local=int(np.searchsorted(self._cumulative,index,side='right')-1)
        row=self.episodes[local]
        anchor=int(index-self._cumulative[local])*self.window_stride
        indices=np.arange(max(0,anchor-self.memory_history_frames),anchor,dtype=np.int64)
        frames=_event_occupancy(self._event_frames(row,indices),source=row['source']) if len(indices) else sample['obs_voxel'].new_empty((0,*sample['obs_voxel'].shape))
        return attach_history(sample,frames,indices,anchor,row['fps'],self.memory_history_frames)
