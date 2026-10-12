"""Predecode current RGB and raw state/action columns into read-only mmap arrays.

The published manifest is the completion marker. No events or normalized
latents are stored; normalization remains the training export's own contract.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

FORMAT = 'ewm_action_only_mmap_v1'


def source_signature(path):
    stat = path.stat()
    return dict(source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns)


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npy')
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, path)


def build(dataset_dir, cache_dir):
    dataset_dir, cache_dir = Path(dataset_dir).resolve(), Path(cache_dir).resolve()
    contract = json.loads((dataset_dir / 'meta/piper_temporal_contract.json').read_text())
    if contract.get('temporal_contract') != 'aligned_event_frame_pre_action_v1':
        raise ValueError('cache requires the aligned pre-action Piper contract')
    if (cache_dir / 'manifest.json').exists():
        raise FileExistsError('cache is already published; reuse it or choose a fresh path')
    sources = sorted((dataset_dir / 'data').rglob('*.parquet'))
    videos = sorted((dataset_dir / 'videos').rglob('*.mp4'))
    if not sources or not videos:
        raise ValueError('dataset has no action Parquet or RGB videos')
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(format=FORMAT, dataset_dir=str(dataset_dir), height=256, width=256,
                    rgb_layout='TCHW', data_files={}, video_files={})
    total_bytes = 0
    for source in sources:
        before = source_signature(source)
        relative = source.relative_to(dataset_dir)
        table = pq.read_table(source, memory_map=True,
                              columns=['observation.state', 'action', 'task_index'])
        arrays = dict(state=np.asarray(table['observation.state'].to_pylist(), dtype=np.float32),
                      action=np.asarray(table['action'].to_pylist(), dtype=np.float32),
                      task_index=np.asarray(table['task_index'].to_pylist(), dtype=np.int64))
        if arrays['state'].shape != (table.num_rows, 8) or arrays['action'].shape != (table.num_rows, 7):
            raise ValueError(f'invalid state/action dimensions: {source}')
        if not all(np.isfinite(array).all() for array in arrays.values()):
            raise ValueError(f'nonfinite state/action: {source}')
        entry = dict(rows=table.num_rows, **before)
        for name, array in arrays.items():
            target = cache_dir / relative.with_suffix('') / (name + '.npy')
            save_array(target, array)
            entry[name] = target.relative_to(cache_dir).as_posix()
            total_bytes += target.stat().st_size
        if source_signature(source) != before:
            raise RuntimeError(f'source changed while caching: {source}')
        manifest['data_files'][relative.as_posix()] = entry
    for number, source in enumerate(videos, 1):
        before = source_signature(source)
        relative = source.relative_to(dataset_dir)
        with av.open(str(source), options={'hwaccel': 'none'}) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            count = int(stream.frames)
            if count <= 0:
                count = sum(1 for _ in container.decode(stream))
        target = cache_dir / relative.with_suffix('.npy')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.stem + '.tmp.npy')
        array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                          shape=(count, 3, 256, 256))
        decoded = 0
        with av.open(str(source), options={'hwaccel': 'none'}) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            for frame in container.decode(stream):
                if decoded >= count:
                    raise ValueError(f'RGB frame count mismatch: {source}')
                # Match the original PyAV/PIL path, including resizing semantics.
                image = frame.to_image().resize((256, 256), Image.LANCZOS)
                array[decoded] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)
                decoded += 1
        if decoded != count or source_signature(source) != before:
            raise ValueError(f'RGB source count or metadata changed: {source}')
        array.flush()
        del array
        os.replace(temporary, target)
        manifest['video_files'][relative.as_posix()] = dict(
            rows=count, cache_file=target.relative_to(cache_dir).as_posix(), **before)
        total_bytes += target.stat().st_size
        print(f'RGB mmap {number}/{len(videos)}: {relative}, frames={count}', flush=True)
    manifest['cache_bytes'] = total_bytes
    temporary = cache_dir / 'manifest.tmp.json'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary, cache_dir / 'manifest.json')
    print(f'CACHE_READY {cache_dir}: {total_bytes} bytes', flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.dataset_dir, args.cache_dir)
