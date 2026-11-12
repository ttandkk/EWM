"""Read current RGB rows from the independent-frame DINO cache."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

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


class CurrentDinoCache:
    def __init__(self, dataset, directory, dino_path):
        self.directory = Path(directory).resolve()
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        if (dataset.context_frames != 1
                or manifest.get('format') != 'ewm_dino_current_bf16_v1'
                or manifest.get('grid') != 16 or manifest.get('cameras') != 2
                or manifest.get('dim') != 1024 or manifest.get('dtype') != 'bfloat16_uint16_bits'
                or manifest.get('preprocessing') != 'EWM ImageNet normalization; independent T1; full grid'
                or not dino_path or manifest.get('dino_fingerprint') != dino_fingerprint(dino_path)):
            raise ValueError('incompatible current DINO cache or encoder fingerprint')
        self.files = {}
        self.maps = {}
        for _, row in dataset.episodes.iterrows():
            episode = int(row['episode_index'])
            entry = manifest['episodes'].get(str(episode))
            if (entry is None or entry['source'] != source_signature(dataset, row)
                    or entry['length'] != int(row['length'])):
                raise ValueError('missing/stale current DINO cache source')
            path = (self.directory / entry['file']).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError('current DINO array must stay inside cache directory')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (int(row['length']), 2, 1024, 16, 16) or array.dtype != np.uint16:
                raise ValueError('invalid current DINO cache shape/dtype')
            self.files[episode] = path

    def observation(self, row, anchor):
        episode = int(row['episode_index'])
        if episode not in self.maps:
            self.maps[episode] = np.load(self.files[episode], mmap_mode='r', allow_pickle=False)
        # Full-grid, independent per-frame features; never read future rows.
        value = self.maps[episode][anchor].copy()
        result = {'current_dino_features': torch.from_numpy(value).view(torch.bfloat16)}
        return result
