"""Cached Piper event/action tasks with pooled normalization and balanced DDP batches."""
import json
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Sampler

from openwam.dataloader.ewm_dataset import EWMDataset
from openwam.dataloader.utils import get_cfg


from openwam.dataloader.piper_training_data import DYNAMIC_TASKS, source_signatures, validate_exports

# Retain old checkpoints' explicit task lists; the current preset uses DYNAMIC_TASKS.
PIPER_TASKS = (*DYNAMIC_TASKS, 'drawer_cup', 'push_t', 'square_nut', 'stack')


class BalancedPiperBatchSampler(Sampler):
    """Uniform windows per task, disjoint ranks within each optimizer step.

    Yield adjacent rank-sized batches for Accelerate's non-split sharding.
    Sampling is with replacement across steps, without replacement inside a
    global task group. An epoch is ceil(total_windows / global_batch) steps;
    seed + epoch enables deterministic replay without retaining sampled windows.
    """

    def __init__(self, lengths, batch_size, *, num_processes=1, seed=42):
        self.lengths = tuple(int(n) for n in lengths)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.seed = int(seed)
        self.epoch = 0
        self.drop_last = True
        if (not self.lengths or self.batch_size <= 0 or self.num_processes <= 0
                or self.batch_size % len(self.lengths)):
            raise ValueError('batch size must be a positive multiple of the task count')
        self.per_task = self.batch_size // len(self.lengths)
        if min(self.lengths) < self.num_processes * self.per_task:
            raise ValueError('each task needs enough windows for disjoint rank batches')
        self.steps = math.ceil(sum(self.lengths) / (self.batch_size * self.num_processes))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps * self.num_processes

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.steps):
            batches = [[] for _ in range(self.num_processes)]
            for task, length in enumerate(self.lengths):
                windows = rng.choice(length, self.num_processes * self.per_task, replace=False)
                for rank, batch in enumerate(batches):
                    start = rank * self.per_task
                    batch.extend((task, int(w)) for w in windows[start:start + self.per_task])
            for batch in batches:
                yield [batch[i] for i in rng.permutation(self.batch_size)]


class PiperMultiTaskDataset(Dataset):
    gpu_voxelization = False
    dataset_class = EWMDataset

    def __init__(self, config, split='train'):
        if split != 'train':
            raise ValueError('Piper multitask exports and pooled statistics are training-only')
        root = Path(get_cfg(config, 'project_root'))
        self.tasks = tuple(get_cfg(config, 'tasks'))
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or not set(self.tasks) <= set(PIPER_TASKS):
            raise ValueError('tasks must be distinct supported Piper tasks')
        dataset_root = Path(get_cfg(config, 'dataset_dir', root/'datasets')).resolve()
        if dataset_root != (root/'datasets').resolve():
            raise ValueError('multitask dataset_dir must match project_root/datasets')
        if tuple(self.tasks) == DYNAMIC_TASKS:
            validate_exports(root, self.tasks)
        stats = Path(get_cfg(config, 'normalization_stats_path'))
        payload = json.loads(stats.read_text())
        if payload.get('source_split') != 'train' or payload.get('sources') != source_signatures(root, self.tasks):
            raise ValueError('missing/stale pooled training normalization')
        self.voxel_bins = int(get_cfg(config, 'voxel_bins', 1))
        self.event_camera_mode = str(get_cfg(config, 'event_camera_mode', 'dual'))
        cache = Path(get_cfg(config, 'cache_root'))
        self.datasets = []
        for task in self.tasks:
            directory = root / 'datasets' / f'piper_{task}'
            info = json.loads((directory / 'meta/info.json').read_text())
            if info.get('splits', {}).get('train') != f"0:{info['total_episodes']}":
                raise ValueError(f'{task}: expected training-only export')
            child = self.dataset_class(
                directory, chunk_size=int(get_cfg(config, 'chunk_size', 24)),
                event_horizon=get_cfg(config, 'event_horizon', None),
                context_frames=int(get_cfg(config, 'context_frames', 1)),
                window_stride=int(get_cfg(config, 'window_stride', 1)),
                pad_tail=bool(get_cfg(config, 'pad_tail', False)),
                voxel_bins=self.voxel_bins, event_camera_mode=self.event_camera_mode,
                count_clip=int(get_cfg(config, 'count_clip', 15)),
                event_cache_size=int(get_cfg(config, 'event_cache_size', 8)),
                observation_row_offset=0, gpu_voxelization=False,
                normalization_stats_path=stats,
                event_frame_cache_dir=cache / f'piper_{task}' / 'event_frames',
                rgb_decode_backend=str(get_cfg(config, 'rgb_decode_backend', 'pyav')),
                current_dino_path=get_cfg(config, 'current_dino_path'),
                compact_events=bool(get_cfg(config, 'compact_events', False)),
                current_dino_cache_dir=(Path(get_cfg(config, 'current_dino_cache_root')) / f'piper_{task}' / 'dino_current16'
                    if get_cfg(config, 'current_dino_cache_root', None) else None),
            )
            self.datasets.append(child)
        self.normalizer = self.datasets[0].normalizer
        if any(d.normalizer.metadata != self.normalizer.metadata for d in self.datasets):
            raise ValueError('all tasks must share pooled normalization')
        self.lengths = tuple(len(d) for d in self.datasets)

    @classmethod
    def from_config(cls, config, split='train'):
        return cls(config, split)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        task, window = index
        sample = self.datasets[task][window]
        sample['task_name'] = self.tasks[task]
        return sample

    def make_batch_sampler(self, batch_size, *, num_processes, seed):
        return BalancedPiperBatchSampler(self.lengths, batch_size, num_processes=num_processes, seed=seed)
