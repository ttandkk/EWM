"""Balanced eight-task RGB/state/action training with optional causal RGB history."""
import json
from pathlib import Path

from torch.utils.data import Dataset

from .ewm_action_only import PiperActionOnlyDataset
from .ewm_piper_multitask import BalancedPiperBatchSampler
from .piper_training_data import DYNAMIC_TASKS, source_signatures, validate_exports
from .utils import get_cfg


class PiperMultiTaskActionDataset(Dataset):
    gpu_voxelization = False

    def __init__(self, config, split='train'):
        if split != 'train':
            raise ValueError('Piper pooled statistics and exports are training-only')
        root = Path(get_cfg(config, 'project_root')).resolve()
        self.tasks = tuple(get_cfg(config, 'tasks'))
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or not set(self.tasks) <= set(DYNAMIC_TASKS):
            raise ValueError('tasks must be distinct supported dynamic tasks')
        if Path(get_cfg(config, 'dataset_dir')).resolve() != root / 'datasets':
            raise ValueError('dataset_dir must match project_root/datasets')
        validate_exports(root, self.tasks)
        stats = Path(get_cfg(config, 'normalization_stats_path'))
        payload = json.loads(stats.read_text())
        if (payload.get('source_split') != 'train'
                or payload.get('sources') != source_signatures(root, self.tasks)
                or payload.get('action', {}).get('type') != 'minmax'):
            raise ValueError('missing/stale full-range pooled normalization')
        current_cache = get_cfg(config, 'current_dino_cache_root', None)
        self.datasets = []
        for task in self.tasks:
            directory = root / 'datasets' / f'piper_{task}'
            info = json.loads((directory / 'meta/info.json').read_text())
            if info.get('splits', {}).get('train') != f"0:{info['total_episodes']}":
                raise ValueError(f'{task}: expected training-only export')
            self.datasets.append(PiperActionOnlyDataset(
                directory, chunk_size=int(get_cfg(config, 'chunk_size', 24)),
                context_frames=1, window_stride=int(get_cfg(config, 'window_stride', 1)),
                pad_tail=bool(get_cfg(config, 'pad_tail', True)), observation_row_offset=0,
                voxel_bins=int(get_cfg(config, 'voxel_bins', 1)),
                normalization_stats_path=stats,
                rgb_decode_backend=str(get_cfg(config, 'rgb_decode_backend', 'pyav')),
                current_dino_cache_dir=(Path(current_cache) / f'piper_{task}' / 'dino_current16'
                                        if current_cache else None),
                current_dino_path=get_cfg(config, 'current_dino_path', None),
            ))
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
