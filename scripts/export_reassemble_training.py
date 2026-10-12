#!/usr/bin/env python3
"""Build a LeRobot-compatible REASSEMBLE export with filtered one-bin event frames.

The exported root contains hard-linked meta/data/videos and newly generated
event_frames. It never copies the source sparse events or audio. Each episode
is produced atomically, so an interrupted Slurm array task can be retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/projects/hdd/myproject/CVPR2027')
SOURCE = ROOT / 'datasets/reassemble'
STAGE = ROOT / 'datasets/reassemble_training_eventframes_10ms.partial'
FINAL = ROOT / 'datasets/reassemble_training_eventframes_10ms'
TIMING_ROOT = ROOT / 'tmp/reassemble_export_timing'
H, W, FPS, CLIP, CHUNK = 260, 346, 30, 15, 256
DELTA_S = 0.010
SOURCE_REPO = 'robot-lev/reassemble'


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_info() -> tuple[dict, dict, dict]:
    info = load_json(SOURCE / 'meta/info.json')
    splits = load_json(SOURCE / 'meta/splits.json')
    table = pq.read_table(SOURCE / 'meta/episodes/chunk-000/file-000.parquet',
                          columns=['episode_index', 'length']).to_pydict()
    lengths = {int(i): int(n) for i, n in zip(table['episode_index'], table['length'], strict=True)}
    count = int(info['total_episodes'])
    if len(lengths) != count or set(lengths) != set(range(count)):
        raise ValueError('source episode metadata has missing or duplicate indices')
    if len(splits) != count or sum(lengths.values()) != int(info['total_frames']):
        raise ValueError('source split or frame totals do not match info.json')
    if float(info['fps']) != FPS:
        raise ValueError('unexpected source FPS')
    for key, shape in {
        'observation.images.hand': [480, 640, 3],
        'observation.images.hama1': [480, 640, 3],
        'observation.images.hama2': [480, 640, 3],
        'observation.images.event_cam': [H, W, 3],
        'action': [9],
        'observation.state': [36],
    }.items():
        if info['features'][key]['shape'] != shape:
            raise ValueError(f'incompatible source feature {key}')
    return info, splits, lengths


def link_tree(source: Path, target: Path) -> int:
    linked = 0
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'unexpected symlink in source tree: {path}')
        relative = path.relative_to(source)
        output = target / relative
        if path.is_dir():
            output.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                a, b = path.stat(), output.stat()
                if (a.st_dev, a.st_ino, a.st_size) != (b.st_dev, b.st_ino, b.st_size):
                    raise ValueError(f'preexisting export file is not a source hard link: {output}')
            else:
                os.link(path, output)
            linked += 1
    return linked


def prepare() -> None:
    if FINAL.exists():
        raise FileExistsError(f'final export already exists: {FINAL}')
    if not (SOURCE / '.download_complete').is_file():
        raise FileNotFoundError('source download completion marker is missing')
    info, splits, lengths = source_info()
    STAGE.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in ('meta', 'data', 'videos'):
        counts[name] = link_tree(SOURCE / name, STAGE / name)
    (STAGE / 'event_frames').mkdir(exist_ok=True)
    schema = {
        'format': 'lerobot_v3_plus_event_frame_npz_v1',
        'source_repo': SOURCE_REPO,
        'source_revision': '37d242d3532fb17b6f749a26dad6d7aaacf810cd',
        'fps': FPS,
        'total_episodes': info['total_episodes'],
        'total_frames': info['total_frames'],
        'rgb_video_features': [
            'observation.images.hand',
            'observation.images.hama1',
            'observation.images.hama2',
        ],
        'davis_image_feature': 'observation.images.event_cam',
        'tabular_features': ['observation.state', 'action'],
        'event_frame_feature': {
            'path': 'event_frames/episode_{episode_index:06d}/frames_{start:06d}_{stop:06d}.npz',
            'array_key': 'counts',
            'shape': '[frames,2,260,346]',
            'dtype': 'uint8',
            'polarity_order': 'original_event_polarity_0_then_1',
            'physical_off_on_assignment_verified': False,
            'count_clip': CLIP,
            'window': '(t0+(i-1)/30,t0+i/30]',
            't0': 'original_hdf5_master_camera_first_timestamp',
            'filter': {
                'type': 'causal_8_neighbor_background_activity',
                'support_time_ms': 10,
                'minimum_supporting_neighbors': 1,
                'support_polarity_agnostic': True,
                'timestamp_surface_updates_on_all_raw_events': True,
                'refractory_enabled': False,
            },
        },
        'training_semantics': {
            'action': 'port action[i] is observed EE pose and gripper state at i+1, not a logged motor command',
            'rgb_timing': 'port RGB frames are nearest sampled to nominal 30 Hz and can be captured after the event-frame boundary',
            'direct_openwam_ewm_compatibility': False,
        },
        'source_data_preserved': True,
        'sparse_events_and_audio_in_export': False,
        'split_counts': {k: sum(1 for value in splits.values() if value['split'] == k)
                         for k in ('train', 'test', 'unassigned')},
    }
    metadata = STAGE / 'meta/event_frames.json'
    if metadata.exists() and load_json(metadata) != schema:
        raise ValueError(f'preexisting schema does not match: {metadata}')
    write_json(metadata, schema)
    print(f'prepared {STAGE}: hard links {counts}, episodes={len(lengths)}, frames={sum(lengths.values())}', flush=True)


def neighbor_indices() -> list[tuple[int, ...]]:
    table = []
    for y in range(H):
        for x in range(W):
            table.append(tuple(
                yy * W + xx
                for yy in range(max(0, y - 1), min(H, y + 2))
                for xx in range(max(0, x - 1), min(W, x + 2))
                if xx != x or yy != y
            ))
    return table


def causal_activity_mask(events: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    neighbors = neighbor_indices()
    last = [-1.0e30] * (H * W)
    keep = np.zeros(len(events), dtype=np.bool_)
    base = float(timestamps[0])
    for i in range(len(events)):
        x, y = int(events[i, 0]), int(events[i, 1])
        position = y * W + x
        relative_time = float(timestamps[i]) - base
        threshold = relative_time - DELTA_S
        for neighbor in neighbors[position]:
            if last[neighbor] >= threshold:
                keep[i] = True
                break
        last[position] = relative_time
    return keep


def validate_episode_manifest(directory: Path, index: int, frames: int) -> dict:
    manifest_path = directory / 'manifest.json'
    manifest = load_json(manifest_path)
    if manifest['episode'] != index or manifest['frames'] != frames:
        raise ValueError(f'manifest episode or length mismatch: {manifest_path}')
    if manifest['height'] != H or manifest['width'] != W or manifest['fps'] != FPS:
        raise ValueError(f'manifest image or FPS mismatch: {manifest_path}')
    if manifest['count_clip'] != CLIP or manifest['filter']['support_time_ms'] != 10:
        raise ValueError(f'manifest encoding/filter mismatch: {manifest_path}')
    covered = 0
    total_bytes = 0
    for chunk in manifest['chunks']:
        start, stop = int(chunk['start']), int(chunk['stop'])
        if start != covered or stop <= start or stop > frames:
            raise ValueError(f'manifest frame gap or overlap: {manifest_path}')
        path = directory / chunk['file']
        if path.stat().st_size != chunk['bytes'] or checksum(path) != chunk['sha256']:
            raise ValueError(f'chunk size/hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ['counts']:
                raise ValueError(f'unexpected chunk arrays: {path}')
            array = archive['counts']
        if array.shape != (stop - start, 2, H, W) or array.dtype != np.uint8 or array.max() > CLIP:
            raise ValueError(f'chunk shape/dtype/value mismatch: {path}')
        total_bytes += path.stat().st_size
        covered = stop
    if covered != frames:
        raise ValueError(f'episode incomplete: {directory}')
    manifest['compressed_bytes'] = total_bytes
    return manifest


def export_episode(index: int, timing_json: Path) -> None:
    _, splits, lengths = source_info()
    if index not in lengths:
        raise ValueError(f'unknown episode {index}')
    if not (STAGE / 'meta/event_frames.json').is_file():
        raise FileNotFoundError('run prepare before episode export')
    destination = STAGE / 'event_frames' / f'episode_{index:06d}'
    if destination.exists():
        manifest = validate_episode_manifest(destination, index, lengths[index])
        print(f'episode {index} already complete ({manifest["compressed_bytes"]:,} bytes)', flush=True)
        return
    timing = load_json(timing_json)
    if timing['episode'] != index or timing['grid_fps'] != FPS or timing['grid_frames'] != lengths[index]:
        raise ValueError('exact source timing does not match port episode')
    source = SOURCE / 'events' / f'episode_{index:06d}.npz'
    with np.load(source, allow_pickle=False) as archive:
        events, timestamps = archive['events'], archive['timestamps']
    if events.ndim != 2 or events.shape[1] != 3 or timestamps.shape != (len(events),):
        raise ValueError(f'invalid source event arrays: {source}')
    if len(events) == 0 or events.dtype != np.int64 or timestamps.dtype != np.float64:
        raise ValueError(f'invalid source event dtypes/length: {source}')
    if np.any(np.diff(timestamps) < 0) or not np.all(np.isfinite(timestamps)):
        raise ValueError(f'unsorted or nonfinite source timestamps: {source}')
    if np.any((events[:, 0] < 0) | (events[:, 0] >= W) |
              (events[:, 1] < 0) | (events[:, 1] >= H) |
              (events[:, 2] < 0) | (events[:, 2] > 1)):
        raise ValueError(f'invalid source event coordinate or polarity: {source}')
    if timing['events']['count'] != len(events):
        raise ValueError(f'original HDF5 and local sidecar event counts differ: {index}')
    if (timing['events']['first'] != float(timestamps[0]) or
            timing['events']['last'] != float(timestamps[-1])):
        raise ValueError(f'original HDF5 and local sidecar event timestamps differ: {index}')
    print(f'episode {index}: filtering {len(events):,} events', flush=True)
    start_time = time.monotonic()
    keep = causal_activity_mask(events, timestamps)
    origin = float(timing['absolute_t0'])
    relative_t = timestamps - origin
    episode_frames = lengths[index]
    working_root = STAGE / 'event_frames/.working'
    working_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'episode_{index:06d}_', dir=working_root))
    chunks = []
    accepted_in_windows = 0
    raw_in_windows = 0
    total_counts = 0
    try:
        for start in range(0, episode_frames, CHUNK):
            stop = min(start + CHUNK, episode_frames)
            counts = np.zeros((stop - start, 2, H, W), dtype=np.uint8)
            for i in range(start, stop):
                query = i / FPS
                begin = int(np.searchsorted(relative_t, query - 1 / FPS, side='right'))
                end = int(np.searchsorted(relative_t, query, side='right'))
                raw_in_windows += end - begin
                if begin == end:
                    continue
                selected = events[begin:end][keep[begin:end]]
                accepted_in_windows += len(selected)
                if not len(selected):
                    continue
                flat = (selected[:, 2] * H + selected[:, 1]) * W + selected[:, 0]
                hist = np.bincount(flat, minlength=2 * H * W).reshape(2, H, W)
                counts[i - start] = np.minimum(hist, CLIP).astype(np.uint8)
            name = f'frames_{start:06d}_{stop:06d}.npz'
            path = work / name
            np.savez_compressed(path, counts=counts)
            total_counts += int(counts.sum())
            chunks.append({
                'file': name, 'start': start, 'stop': stop,
                'bytes': path.stat().st_size, 'sha256': checksum(path),
            })
            print(f'episode {index}: {stop}/{episode_frames} frames', flush=True)
        manifest = {
            'episode': index, 'recording': splits[str(index)]['recording'],
            'split': splits[str(index)]['split'],
            'frames': episode_frames, 'fps': FPS, 'height': H, 'width': W,
            'dtype': 'uint8', 'shape': '[frames,2,260,346]', 'count_clip': CLIP,
            'channels': 'original_polarity_0_then_1',
            'event_interval': '(t0+(i-1)/30,t0+i/30]',
            'absolute_t0': origin,
            'filter': {'type': 'causal_8_neighbor_background_activity',
                       'support_time_ms': 10, 'polarity_agnostic': True,
                       'refractory_enabled': False},
            'source_events': int(len(events)),
            'retained_events': int(keep.sum()),
            'source_events_in_frame_windows': raw_in_windows,
            'retained_events_in_frame_windows': accepted_in_windows,
            'stored_clipped_count_sum': total_counts,
            'chunks': chunks,
        }
        write_json(work / 'manifest.json', manifest)
        validate_episode_manifest(work, index, episode_frames)
        work.rename(destination)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(f'episode {index} complete in {time.monotonic() - start_time:.1f}s: '
          f'{accepted_in_windows:,}/{raw_in_windows:,} window events', flush=True)


def finalize() -> None:
    if FINAL.exists():
        raise FileExistsError(f'final export already exists: {FINAL}')
    info, splits, lengths = source_info()
    if not STAGE.is_dir():
        raise FileNotFoundError(f'export staging directory missing: {STAGE}')
    schema = load_json(STAGE / 'meta/event_frames.json')
    if schema['total_episodes'] != len(lengths) or schema['total_frames'] != sum(lengths.values()):
        raise ValueError('export schema episode/frame totals mismatch')
    for name in ('meta', 'data', 'videos'):
        for source in (SOURCE / name).rglob('*'):
            if not source.is_file():
                continue
            target = STAGE / name / source.relative_to(SOURCE / name)
            if not target.is_file():
                raise FileNotFoundError(target)
            a, b = source.stat(), target.stat()
            if (a.st_dev, a.st_ino, a.st_size) != (b.st_dev, b.st_ino, b.st_size):
                raise ValueError(f'export RGB/table file differs from source: {target}')
    if (STAGE / 'events').exists() or (STAGE / 'audio').exists():
        raise ValueError('export unexpectedly contains original sparse events or audio')
    expected = {f'episode_{i:06d}' for i in lengths}
    existing = {p.name for p in (STAGE / 'event_frames').iterdir() if p.is_dir() and p.name != '.working'}
    if existing != expected:
        missing = sorted(expected - existing)
        extra = sorted(existing - expected)
        raise ValueError(f'episode directories missing={missing[:20]} extra={extra[:20]}')
    summaries = []
    for index in sorted(lengths):
        manifest = validate_episode_manifest(
            STAGE / 'event_frames' / f'episode_{index:06d}', index, lengths[index])
        if manifest['recording'] != splits[str(index)]['recording'] or manifest['split'] != splits[str(index)]['split']:
            raise ValueError(f'source recording/split mismatch: {index}')
        summaries.append(manifest)
        if (index + 1) % 10 == 0:
            print(f'verified {index + 1}/{len(lengths)} episodes', flush=True)
    summary = {
        'status': 'complete',
        'format': schema['format'],
        'source_repo': SOURCE_REPO,
        'episodes': len(summaries),
        'frames': sum(item['frames'] for item in summaries),
        'source_events': sum(item['source_events'] for item in summaries),
        'retained_events': sum(item['retained_events'] for item in summaries),
        'event_frame_compressed_bytes': sum(item['compressed_bytes'] for item in summaries),
        'event_frame_chunks': sum(len(item['chunks']) for item in summaries),
        'verified_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    if summary['episodes'] != info['total_episodes'] or summary['frames'] != info['total_frames']:
        raise ValueError('final frame/episode totals mismatch')
    working = STAGE / 'event_frames/.working'
    if working.exists():
        shutil.rmtree(working)
    write_json(STAGE / '.export_complete', summary)
    STAGE.rename(FINAL)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'published: {FINAL}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest='command', required=True)
    command.add_parser('prepare')
    episode = command.add_parser('episode')
    episode.add_argument('--index', type=int, required=True)
    episode.add_argument('--timing-json', type=Path, required=True)
    command.add_parser('finalize')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'episode':
        export_episode(args.index, args.timing_json)
    else:
        finalize()


if __name__ == '__main__':
    main()
