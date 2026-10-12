#!/usr/bin/env python3
"""Build a 25 FPS LeRobot derivative of the 250 FPS DVLA place7obj dataset.

RGB uses source indices 9,19,...; each matching event image covers 10 source
rendered event frames. Event sidecars store *active 4 ms subframe counts*, not
raw DVS event counts. Source H.264 polarity renderings cannot recover raw counts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path('/projects/hdd/myproject/CVPR2027')
DEFAULT_SOURCE = ROOT / 'datasets/dvla-place7obj-roll-250hz-events-250fps-delta-trim'
DEFAULT_DEST = ROOT / 'datasets/dvla-place7obj-roll-25fps-events-40ms-delta-trim.partial'
FINAL_DEST = ROOT / 'datasets/dvla-place7obj-roll-25fps-events-40ms-delta-trim'
SOURCE_REVISION = '8ded7c2eded1e3b68a3ea2de99999da55e468f14'
SOURCE_FPS = 250
TARGET_FPS = 25
FACTOR = SOURCE_FPS // TARGET_FPS
HEIGHT, WIDTH = 360, 480
COLOR_THRESHOLD = 25
COUNT_CLIP = 15


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    with temp.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(',', ':')) + '\n')
    temp.replace(path)


def read_source(source: Path) -> tuple[dict, list[dict]]:
    info = json.loads((source / 'meta/info.json').read_text())
    episodes = read_jsonl(source / 'meta/episodes.jsonl')
    if info['fps'] != SOURCE_FPS or len(episodes) != info['total_episodes']:
        raise ValueError('source metadata is not the expected 250 FPS DVLA dataset')
    if sorted(x['episode_index'] for x in episodes) != list(range(len(episodes))):
        raise ValueError('episode indices must be contiguous')
    if sum(x['length'] for x in episodes) != info['total_frames']:
        raise ValueError('source episode lengths do not sum to total_frames')
    if len([k for k, v in info['features'].items() if v['dtype'] == 'video']) != 6:
        raise ValueError('expected three RGB and three event video streams')
    return info, sorted(episodes, key=lambda x: x['episode_index'])


def target_lengths(episodes: list[dict]) -> np.ndarray:
    lengths = np.array([int(x['length']) // FACTOR for x in episodes], dtype=np.int64)
    if np.any(lengths < 1):
        raise ValueError('source episode shorter than one 40 ms window')
    return lengths


def prepare(source: Path, dest: Path, allow_incomplete: bool) -> None:
    if not allow_incomplete and not (source / '.download_complete').exists():
        raise RuntimeError('source download is not verified complete')
    if dest == DEFAULT_DEST and FINAL_DEST.exists():
        raise FileExistsError(f'published export already exists: {FINAL_DEST}')
    info, episodes = read_source(source)
    lengths = target_lengths(episodes)
    target_info = copy.deepcopy(info)
    target_info['fps'] = TARGET_FPS
    target_info['total_frames'] = int(lengths.sum())
    for feature in target_info['features'].values():
        if feature['dtype'] == 'video':
            feature['info']['video.fps'] = TARGET_FPS
    target_episodes = []
    for row, n in zip(episodes, lengths, strict=True):
        out = copy.deepcopy(row)
        out['length'] = int(n)
        target_episodes.append(out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'meta').mkdir(parents=True, exist_ok=True)
    (dest / 'event_frames').mkdir(exist_ok=True)
    for name in ('tasks.jsonl', 'camera.jsonl'):
        shutil.copy2(source / 'meta' / name, dest / 'meta' / name)
    if (source / 'assets').is_dir():
        for original in (source / 'assets').rglob('*'):
            relative = original.relative_to(source)
            target = dest / relative
            if original.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif original.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    os.link(original, target)
                elif original.stat().st_ino != target.stat().st_ino:
                    raise ValueError(f'asset hard link mismatch: {target}')
    if (source / 'README.md').is_file() and not (dest / 'SOURCE_README.md').exists():
        os.link(source / 'README.md', dest / 'SOURCE_README.md')
    write_json(dest / 'meta/info.json', target_info)
    write_jsonl(dest / 'meta/episodes.jsonl', target_episodes)
    write_json(dest / 'meta/event_frames.json', {
        'format': 'lerobot_v2_1_plus_event_frame_npz_v1',
        'source_dataset': 'mickeykang/dvla-place7obj-roll-250hz-events-250fps-delta-trim',
        'source_revision': SOURCE_REVISION,
        'fps': TARGET_FPS,
        'total_episodes': len(episodes),
        'total_frames': int(lengths.sum()),
        'rgb_video_features': [f'observation.images.{camera}' for camera in ('wrist_cam', 'side_cam', 'opst_cam')],
        'event_preview_video_features': [f'observation.images.{camera}_events' for camera in ('wrist_cam', 'side_cam', 'opst_cam')],
        'tabular_features_preserved': [key for key, feature in info['features'].items() if feature['dtype'] != 'video'],
        'event_frame_feature': {
            'path': 'event_frames/episode_{episode_index:06d}/{camera}/frames_{start:06d}_{stop:06d}.npz',
            'array_key': 'counts', 'shape': '[frames,2,360,480]', 'dtype': 'uint8',
            'channel_order': ['blue', 'red'], 'count_clip': COUNT_CLIP,
            'source': 'H.264 polarity renderings; count active 4 ms subframes, not raw DVS events',
            'classification': 'blue if B>G+25; red if R>G+25 after video decoding',
            'window': 'source event-video frames [10*j,10*j+9] (nominal 40 ms)',
        },
        'rgb_and_tabular_row': 'source row 10*j+9',
        'output_timestamp': 'j/25 seconds relative to the first selected source RGB frame',
        'action_semantics': 'source hindsight delta to EE pose reached 320 ms after source row; not executed 40 ms command',
        'direct_openwam_ewm_compatibility': False,
    })
    write_json(dest / 'meta/conversion.json', {
        'source_dataset': 'mickeykang/dvla-place7obj-roll-250hz-events-250fps-delta-trim',
        'source_revision': SOURCE_REVISION,
        'source_fps': SOURCE_FPS, 'target_fps': TARGET_FPS,
        'source_frame_indices': '9, 19, 29, ...; floor(source_length/10) output frames',
        'output_timestamp': 'j/25 seconds, relative to the first selected source RGB frame',
        'event_window': 'source event video frames [10*j, 10*j+9], 40 ms nominal',
        'event_counts': 'sum of red/blue active masks in ten rendered 4 ms frames; NOT raw DVS counts',
        'polarity_order': ['blue', 'red'],
        'classification': 'red if R>G+25; blue if B>G+25 after H.264 decoding',
        'count_clip': COUNT_CLIP, 'native_resolution': [HEIGHT, WIDTH],
        'action_note': 'preserved source hindsight delta to EE state reached 320 ms later',
    })
    (dest / 'README.md').write_text(
        '# DVLA place7obj 25 FPS training export\n\n'
        'The original six video features are downsampled to 25 FPS: three RGB and three '
        'accumulated-polarity preview videos. Every source Parquet field, including action, '
        'observation.state and observation.environment_state, is kept at source rows 10j+9.\n\n'
        'For each camera and output frame j, the uint8 `[2,360,480]` event frame counts '
        'blue/red activity in source rendered event frames 10j..10j+9. The count ceiling is 15. '
        'The input is H.264 polarity video, so these are active-subframe counts, not true '
        'sparse DVS event counts. The event preview videos are derived from these counts.\n\n'
        'Like the REASSEMBLE export, event frames are stored in episode-indexed NPZ files '
        'with manifests under `event_frames/`, and `meta/event_frames.json` defines the schema. '
        'Native 360x480 resolution is preserved. A shared training loader must handle the '
        'different resolution, camera names, action/state dimensions and event-count semantics.\n\n'
        'Source `action[j]` is a hindsight target for an EE state reached 320 ms later, '
        'not the control command executed during the 40 ms event interval. '
        'Do not treat it as a causal next-frame control input.\n'
    )
    print(f'prepared {len(episodes)} episodes, {int(lengths.sum())} frames at {TARGET_FPS} FPS', flush=True)


class ImageStats:
    def __init__(self) -> None:
        self.minimum = np.full(3, 255, dtype=np.uint8)
        self.maximum = np.zeros(3, dtype=np.uint8)
        self.sums = np.zeros(3, dtype=np.float64)
        self.sumsq = np.zeros(3, dtype=np.float64)
        self.frames = 0

    def add(self, rgb: np.ndarray) -> None:
        flat = rgb.reshape(-1, 3)
        self.minimum = np.minimum(self.minimum, flat.min(axis=0))
        self.maximum = np.maximum(self.maximum, flat.max(axis=0))
        self.sums += flat.sum(axis=0, dtype=np.float64)
        values = flat.astype(np.float32)
        self.sumsq += np.square(values).sum(axis=0, dtype=np.float64)
        self.frames += 1

    def to_dict(self) -> dict:
        pixels = self.frames * HEIGHT * WIDTH
        mean = self.sums / pixels
        std = np.sqrt(np.maximum(self.sumsq / pixels - mean**2, 0.0))
        def nested(x: np.ndarray) -> list:
            return [[[float(v) / 255.0]] for v in x]
        return {'min': nested(self.minimum), 'max': nested(self.maximum),
                'mean': nested(mean), 'std': nested(std), 'count': [self.frames]}


def numeric_stats(table: pa.Table, name: str) -> dict:
    values = np.asarray(table[name].to_pylist(), dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist(),
        'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0).tolist(),
        'count': [len(values)],
    }


def source_path(info: dict, episode: int, key: str) -> Path:
    return Path(info['video_path'].format(episode_chunk=episode // info['chunks_size'],
                                          episode_index=episode, video_key=key))


def data_path(info: dict, episode: int) -> Path:
    return Path(info['data_path'].format(episode_chunk=episode // info['chunks_size'],
                                         episode_index=episode))


def encode_stream(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.part.mp4')
    output = av.open(str(temp), mode='w')
    stream = output.add_stream('libx264', rate=TARGET_FPS)
    stream.width, stream.height = WIDTH, HEIGHT
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '17', 'preset': 'fast', 'threads': '1'}
    return output, stream, temp


def mux_rgb(output, stream, rgb: np.ndarray) -> None:
    frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
    for packet in stream.encode(frame):
        output.mux(packet)


def close_stream(output, stream) -> None:
    for packet in stream.encode():
        output.mux(packet)
    output.close()


def polarity_preview(counts: np.ndarray) -> np.ndarray:
    rgb = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    blue, red = counts[0] > 0, counts[1] > 0
    active = blue | red
    rgb[active, 1] = 0
    rgb[blue, 2] = 255
    rgb[red, 0] = 255
    rgb[blue & ~red, 0] = 0
    rgb[red & ~blue, 2] = 0
    return rgb


def convert_video(source: Path, dest: Path, expected_source_frames: int,
                  event_counts_path: Path | None) -> tuple[dict, int]:
    expected_target_frames = expected_source_frames // FACTOR
    stats = ImageStats()
    counts = None
    accumulator = None
    if event_counts_path is not None:
        counts = np.empty((expected_target_frames, 2, HEIGHT, WIDTH), dtype=np.uint8)
        accumulator = np.zeros((2, HEIGHT, WIDTH), dtype=np.uint16)
    output, stream, temp = encode_stream(dest)
    decoded = 0
    produced = 0
    try:
        with av.open(str(source)) as input_video:
            for decoded, frame in enumerate(input_video.decode(video=0), start=1):
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.shape != (HEIGHT, WIDTH, 3):
                    raise ValueError(f'unexpected video frame shape {rgb.shape}: {source}')
                if counts is not None:
                    values = rgb.astype(np.int16)
                    accumulator[0] += values[:, :, 2] > values[:, :, 1] + COLOR_THRESHOLD
                    accumulator[1] += values[:, :, 0] > values[:, :, 1] + COLOR_THRESHOLD
                if decoded % FACTOR != 0:
                    continue
                if produced >= expected_target_frames:
                    raise ValueError(f'too many frames: {source}')
                if counts is not None:
                    counts[produced] = np.minimum(accumulator, COUNT_CLIP).astype(np.uint8)
                    rgb = polarity_preview(counts[produced])
                    accumulator.fill(0)
                stats.add(rgb)
                mux_rgb(output, stream, rgb)
                produced += 1
        if decoded != expected_source_frames or produced != expected_target_frames:
            raise ValueError(f'frame count mismatch in {source}: got {decoded}, expected {expected_source_frames}')
        close_stream(output, stream)
        temp.replace(dest)
    except BaseException:
        output.close()
        temp.unlink(missing_ok=True)
        raise
    if counts is not None:
        event_counts_path.parent.mkdir(parents=True, exist_ok=True)
        temp_counts = event_counts_path.with_name(event_counts_path.stem + '.part.npz')
        np.savez_compressed(temp_counts, counts=counts)
        temp_counts.replace(event_counts_path)
    return stats.to_dict(), produced


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def event_frame_path(episode: int, camera: str, frames: int) -> Path:
    return Path('event_frames') / f'episode_{episode:06d}' / camera / f'frames_000000_{frames:06d}.npz'


def manifest_path(dest: Path, episode: int) -> Path:
    return dest / 'event_frames' / f'episode_{episode:06d}' / 'manifest.json'


def verify_npz_header(path: Path, frames: int) -> None:
    import zipfile
    from numpy.lib import format as fmt
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['counts.npy']:
            raise ValueError(f'unexpected event-frame NPZ members: {path}')
        with archive.open('counts.npy') as stream:
            version = fmt.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = fmt.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = fmt.read_array_header_2_0(stream)
            else:
                raise ValueError(f'unexpected NPY version {version}: {path}')
    if shape != (frames, 2, HEIGHT, WIDTH) or fortran or dtype != np.dtype('uint8'):
        raise ValueError(f'event-frame shape/dtype mismatch: {path}; {shape}, {dtype}')


def validate_episode(dest: Path, info: dict, episode: int, source_length: int) -> dict:
    path = manifest_path(dest, episode)
    manifest = json.loads(path.read_text())
    frames = source_length // FACTOR
    if manifest['episode'] != episode or manifest['frames'] != frames or manifest['source_frames'] != source_length:
        raise ValueError(f'episode manifest index or frame count mismatch: {path}')
    if manifest['fps'] != TARGET_FPS or manifest['count_clip'] != COUNT_CLIP:
        raise ValueError(f'episode manifest encoding mismatch: {path}')
    video_keys = [key for key, feature in info['features'].items() if feature['dtype'] == 'video']
    expected = {str(data_path(info, episode))}
    expected.update(str(source_path(info, episode, key)) for key in video_keys)
    expected.update(str(event_frame_path(episode, key.split('.')[-1].removesuffix('_events'), frames))
                    for key in video_keys if key.endswith('_events'))
    files = manifest['files']
    if len(files) != len(expected) or {item['path'] for item in files} != expected:
        raise ValueError(f'episode manifest file list mismatch: {path}')
    for item in files:
        file_path = dest / item['path']
        if not file_path.is_file() or file_path.stat().st_size != item['bytes'] or checksum(file_path) != item['sha256']:
            raise ValueError(f'episode file missing or hash mismatch: {file_path}')
        if item['path'].startswith('event_frames/'):
            verify_npz_header(file_path, frames)
    parquet = dest / data_path(info, episode)
    if pq.read_metadata(parquet).num_rows != frames:
        raise ValueError(f'Parquet row count mismatch: {parquet}')
    return manifest


def convert_episode(source: Path, dest: Path, info: dict, episode: int,
                    source_length: int, global_offset: int) -> dict:
    n = source_length // FACTOR
    table = pq.read_table(source / data_path(info, episode))
    if table.num_rows != source_length:
        raise ValueError(f'episode {episode} Parquet length mismatch')
    indices = pa.array(np.arange(FACTOR - 1, n * FACTOR, FACTOR, dtype=np.int64))
    result = table.take(indices)
    replacements = {
        'timestamp': pa.array(np.arange(n, dtype=np.float32) / TARGET_FPS, type=pa.float32()),
        'frame_index': pa.array(np.arange(n, dtype=np.int64)),
        'index': pa.array(global_offset + np.arange(n, dtype=np.int64)),
    }
    for name, values in replacements.items():
        column = result.schema.get_field_index(name)
        result = result.set_column(column, name, values)
    stats = {name: numeric_stats(result, name) for name in result.column_names}
    video_keys = [key for key, feature in info['features'].items() if feature['dtype'] == 'video']
    produced_paths = []
    for key in video_keys:
        in_video = source / source_path(info, episode, key)
        relative_video = source_path(info, episode, key)
        out_video = dest / relative_video
        counts_path = None
        if key.endswith('_events'):
            camera = key.split('.')[-1].removesuffix('_events')
            relative_counts = event_frame_path(episode, camera, n)
            counts_path = dest / relative_counts
            produced_paths.append(relative_counts)
        stats[key], frame_count = convert_video(in_video, out_video, source_length, counts_path)
        if frame_count != n:
            raise AssertionError('output frame count mismatch')
        produced_paths.append(relative_video)
    relative_parquet = data_path(info, episode)
    path = dest / relative_parquet
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.part')
    pq.write_table(result, temp, compression='zstd')
    temp.replace(path)
    produced_paths.append(relative_parquet)
    for relative in produced_paths:
        if str(relative).startswith('event_frames/'):
            verify_npz_header(dest / relative, n)
    manifest = {
        'episode': episode, 'frames': n, 'source_frames': source_length,
        'fps': TARGET_FPS, 'height': HEIGHT, 'width': WIDTH,
        'dtype': 'uint8', 'shape': '[frames,2,360,480]', 'count_clip': COUNT_CLIP,
        'channels': ['blue', 'red'],
        'source_rgb_and_tabular_rows': f'9,19,...,{n * FACTOR - 1}',
        'source_event_video_windows': '[10*j,10*j+9]',
        'event_count_semantics': 'active rendered 4ms subframes, not raw DVS counts',
        'files': [
            {'path': str(relative), 'bytes': (dest / relative).stat().st_size,
             'sha256': checksum(dest / relative)}
            for relative in produced_paths
        ],
        'stats': stats,
    }
    write_json(manifest_path(dest, episode), manifest)
    return {'episode_index': episode, 'stats': stats}


def convert_chunk(source: Path, dest: Path, chunk: int | None, episode: int | None,
                  allow_incomplete: bool) -> None:
    if not allow_incomplete and not (source / '.download_complete').exists():
        raise RuntimeError('source download is not verified complete')
    info, episodes = read_source(source)
    lengths = target_lengths(episodes)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    if episode is not None:
        ids = [episode]
        shard_name = f'pilot_episode_{episode:06d}.jsonl'
    else:
        if chunk is None or chunk < 0 or chunk >= math.ceil(len(episodes) / info['chunks_size']):
            raise ValueError('invalid chunk index')
        ids = list(range(chunk * info['chunks_size'], min((chunk + 1) * info['chunks_size'], len(episodes))))
        shard_name = f'chunk-{chunk:03d}.jsonl'
    rows = []
    for position, ep in enumerate(ids, 1):
        manifest = manifest_path(dest, ep)
        if manifest.exists():
            entry = validate_episode(dest, info, ep, int(episodes[ep]['length']))
            rows.append({'episode_index': ep, 'stats': entry['stats']})
            print(f'episode {ep:06d}: already complete ({position}/{len(ids)})', flush=True)
        else:
            rows.append(convert_episode(source, dest, info, ep, int(episodes[ep]['length']), int(offsets[ep])))
            print(f'episode {ep:06d}: {position}/{len(ids)}; {int(lengths[ep])} output frames', flush=True)
    write_jsonl(dest / 'meta/stats_shards' / shard_name, rows)
    print(f'wrote stats shard {shard_name}', flush=True)


def finalize(source: Path, dest: Path) -> None:
    if not (source / '.download_complete').exists():
        raise RuntimeError('source download completion marker is missing')
    info, episodes = read_source(source)
    lengths = target_lengths(episodes)
    target_info = json.loads((dest / 'meta/info.json').read_text())
    schema = json.loads((dest / 'meta/event_frames.json').read_text())
    if target_info['fps'] != TARGET_FPS or target_info['total_frames'] != int(lengths.sum()):
        raise ValueError('target info.json FPS or frame count mismatch')
    if schema['total_episodes'] != len(episodes) or schema['total_frames'] != int(lengths.sum()):
        raise ValueError('event-frame schema total mismatch')
    chunks = math.ceil(len(episodes) / info['chunks_size'])
    stats_path = dest / 'meta/episodes_stats.jsonl'
    temp = stats_path.with_name(stats_path.name + '.part')
    expected_episode = 0
    total_compressed_event_bytes = 0
    with temp.open('w') as output:
        for chunk in range(chunks):
            shard = dest / 'meta/stats_shards' / f'chunk-{chunk:03d}.jsonl'
            if not shard.is_file():
                raise RuntimeError(f'missing stats shard {shard}')
            with shard.open() as input_file:
                for line in input_file:
                    row = json.loads(line)
                    ep = int(row['episode_index'])
                    if ep != expected_episode:
                        raise RuntimeError(f'out-of-order stats: got episode {ep}, expected {expected_episode}')
                    manifest = validate_episode(dest, info, ep, int(episodes[ep]['length']))
                    if row['stats'] != manifest['stats']:
                        raise ValueError(f'episode stats mismatch: {ep}')
                    total_compressed_event_bytes += sum(item['bytes'] for item in manifest['files']
                                                        if item['path'].startswith('event_frames/'))
                    expected_episode += 1
                    output.write(line)
            print(f'verified chunk {chunk + 1}/{chunks}, episodes {expected_episode}/{len(episodes)}', flush=True)
    if expected_episode != len(episodes):
        raise RuntimeError(f'missing episodes: got {expected_episode}/{len(episodes)}')
    temp.replace(stats_path)
    summary = {
        'status': 'complete', 'format': schema['format'],
        'source_dataset': schema['source_dataset'],
        'episodes': len(episodes), 'frames': int(lengths.sum()),
        'fps': TARGET_FPS, 'event_cameras': 3,
        'event_frame_compressed_bytes': total_compressed_event_bytes,
    }
    write_json(dest / '.export_complete', summary)
    if dest == DEFAULT_DEST:
        if FINAL_DEST.exists():
            raise FileExistsError(f'published export already exists: {FINAL_DEST}')
        dest.rename(FINAL_DEST)
        print(f'published: {FINAL_DEST}', flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--dest', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--allow-incomplete-source', action='store_true', help='pilot only')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    conversion = sub.add_parser('convert')
    group = conversion.add_mutually_exclusive_group(required=True)
    group.add_argument('--chunk', type=int)
    group.add_argument('--episode', type=int)
    sub.add_parser('finalize')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.dest, args.allow_incomplete_source)
    elif args.command == 'convert':
        convert_chunk(args.source, args.dest, args.chunk, args.episode, args.allow_incomplete_source)
    else:
        finalize(args.source, args.dest)


if __name__ == '__main__':
    main()
