"""Experimental native-resolution REASSEMBLE event frames, with explicit clock status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/projects/hdd/myproject/CVPR2027')
DATASET = ROOT / 'datasets/reassemble'
FPS = 30
H, W = 260, 346
CLIP = 15


def episode_length(index: int) -> int:
    path = DATASET / 'meta/episodes/chunk-000/file-000.parquet'
    table = pq.read_table(path, columns=['episode_index', 'length']).to_pydict()
    positions = [i for i, value in enumerate(table['episode_index']) if value == index]
    if len(positions) != 1:
        raise ValueError(f'unknown or duplicate episode {index}')
    return int(table['length'][positions[0]])


def convert(index: int, output: Path, chunk_frames: int, absolute_t0: float | None) -> None:
    with np.load(DATASET / 'events' / f'episode_{index:06d}.npz') as src:
        events = src['events']
        timestamps = src['timestamps']
    if events.ndim != 2 or events.shape[1] != 3 or timestamps.shape != (len(events),):
        raise ValueError('invalid source arrays')
    if events.dtype != np.int64 or timestamps.dtype != np.float64:
        raise ValueError('unexpected source dtype')
    if len(events) == 0 or np.any(np.diff(timestamps) < 0):
        raise ValueError('empty or unsorted event timestamps')
    if np.any((events[:, 0] < 0) | (events[:, 0] >= W) | (events[:, 1] < 0) | (events[:, 1] >= H)):
        raise ValueError('event coordinate outside native resolution')
    if not np.all((events[:, 2] == 0) | (events[:, 2] == 1)):
        raise ValueError('expected polarity values 0/1')
    n = episode_length(index)
    origin = float(timestamps[0]) if absolute_t0 is None else float(absolute_t0)
    alignment = 'approximate_first_event' if absolute_t0 is None else 'supplied_absolute_origin'
    relative_t = timestamps - origin
    output.mkdir(parents=True, exist_ok=True)
    raw_active = 0
    saved = []
    for start in range(0, n, chunk_frames):
        stop = min(start + chunk_frames, n)
        counts = np.zeros((stop - start, 2, H, W), dtype=np.uint8)
        for i in range(start, stop):
            query_t = i / FPS
            begin = int(np.searchsorted(relative_t, query_t - 1 / FPS, side='right'))
            end = int(np.searchsorted(relative_t, query_t, side='right'))
            if begin == end:
                continue
            e = events[begin:end]
            bins = ((e[:, 2] * H + e[:, 1]) * W + e[:, 0]).astype(np.int64)
            hist = np.bincount(bins, minlength=2 * H * W).reshape(2, H, W)
            counts[i - start] = np.minimum(hist, CLIP).astype(np.uint8)
            raw_active += end - begin
        name = f'frames_{start:06d}_{stop:06d}.npz'
        np.savez_compressed(output / name, counts=counts)
        saved.append({'file': name, 'start': start, 'stop': stop, 'bytes': (output / name).stat().st_size})
        print(f'frames {start}:{stop}, {saved[-1]["bytes"]} bytes', flush=True)
    left = int(np.searchsorted(relative_t, -1 / FPS, side='right'))
    right = int(np.searchsorted(relative_t, (n - 1) / FPS, side='right'))
    report = {
        'episode': index, 'frames': n, 'fps': FPS, 'height': H, 'width': W,
        'tensor_layout': '[frames, polarity_0_then_1, height, width]',
        'dtype': 'uint8', 'count_clip': CLIP,
        'event_interval': 'E_i=(t0+(i-1)/30, t0+i/30] for i>=0',
        'absolute_t0': origin, 'alignment': alignment,
        'source_events': len(events), 'events_in_frame_windows': raw_active,
        'events_before_first_window': left, 'events_after_last_frame': len(events) - right,
        'source_timestamp_first': float(timestamps[0]),
        'source_timestamp_last': float(timestamps[-1]),
        'chunks': saved,
    }
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'chunks'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episode', type=int, default=21)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--chunk-frames', type=int, default=256)
    p.add_argument('--absolute-t0', type=float, default=None,
                   help='original port master-camera first timestamp (hand, or fallback camera)')
    p.add_argument('--timing-json', type=Path, default=None,
                   help='timing.json produced by extract_reassemble_timestamps.py')
    p.add_argument('--allow-approximate-origin', action='store_true',
                   help='use the first event as t0 for exploratory conversion only')
    a = p.parse_args()
    if a.episode < 0 or a.chunk_frames <= 0:
        p.error('episode must be nonnegative and chunk-frames positive')
    if sum((a.absolute_t0 is not None, a.timing_json is not None, a.allow_approximate_origin)) != 1:
        p.error('choose exactly one of --absolute-t0, --timing-json, or --allow-approximate-origin')
    if a.timing_json is not None:
        timing = json.loads(a.timing_json.read_text())
        if int(timing['episode']) != a.episode or int(timing['grid_fps']) != FPS:
            p.error('timing manifest episode or FPS does not match this conversion')
        a.absolute_t0 = float(timing['absolute_t0'])
    output = a.output or ROOT / 'tmp/reassemble_event_frames/results' / f'episode_{a.episode:06d}'
    convert(a.episode, output, a.chunk_frames, a.absolute_t0)
