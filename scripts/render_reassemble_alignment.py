#!/usr/bin/env python3
"""Render a native-resolution DAVIS image/event/overlay comparison for one episode."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/projects/hdd/myproject/CVPR2027')
DATASET = ROOT / 'datasets/reassemble'
sys.path.insert(0, str(ROOT / 'OpenWAM'))
from openwam.dataloader.utils.video_io import decode_video_frames  # noqa: E402


def load_counts(directory: Path, start: int, stop: int) -> np.ndarray:
    parts = []
    covered = start
    for path in sorted(directory.glob('frames_*.npz')):
        first, last = [int(v) for v in path.stem.split('_')[1:]]
        if last <= start or first >= stop:
            continue
        if first > covered:
            raise ValueError(f'missing event frames {covered}:{first}')
        with np.load(path) as item:
            part = item['counts'][max(start, first) - first:min(stop, last) - first]
        parts.append(part)
        covered = min(stop, last)
    if covered < stop:
        raise ValueError(f'missing event frames {covered}:{stop}')
    return np.concatenate(parts)


def colorize(counts: np.ndarray) -> np.ndarray:
    zero, one = counts[0] > 0, counts[1] > 0
    rgb = np.full((counts.shape[1], counts.shape[2], 3), 255, dtype=np.uint8)
    rgb[zero & ~one] = (230, 40, 30)
    rgb[one & ~zero] = (0, 65, 230)
    rgb[zero & one] = (175, 40, 175)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, default=21)
    parser.add_argument('--start', type=int, default=150)
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.episode < 0 or args.start < 0 or args.frames <= 0:
        parser.error('episode/start must be nonnegative and frames positive')
    data = ROOT / 'tmp/reassemble_alignment' / f'episode_{args.episode:06d}'
    timing = json.loads((data / 'timing.json').read_text())
    if timing['episode'] != args.episode or timing['grid_fps'] != 30:
        raise ValueError('timing manifest does not match episode or frame rate')
    meta = pq.read_table(DATASET / 'meta/episodes/chunk-000/file-000.parquet',
                         columns=['episode_index', 'length',
                                  'videos/observation.images.event_cam/chunk_index',
                                  'videos/observation.images.event_cam/file_index',
                                  'videos/observation.images.event_cam/from_timestamp']).to_pandas()
    matches = meta.loc[meta.episode_index == args.episode]
    if len(matches) != 1:
        raise ValueError('unknown or duplicate episode index')
    row = matches.iloc[0]
    stop = min(args.start + args.frames, int(row['length']))
    if stop <= args.start:
        raise ValueError('start is past end of episode')
    counts = load_counts(data / 'event_frames', args.start, stop)
    feature = 'observation.images.event_cam'
    info = json.loads((DATASET / 'meta/info.json').read_text())
    video = DATASET / info['video_path'].format(
        video_key=feature,
        chunk_index=int(row[f'videos/{feature}/chunk_index']),
        file_index=int(row[f'videos/{feature}/file_index']),
    )
    first = round(float(row[f'videos/{feature}/from_timestamp']) * 30)
    frames = decode_video_frames(str(video), list(range(first + args.start, first + stop)), 260, 346)
    output = args.output or data / f'davis_event_overlay_{args.start:04d}_{stop:04d}.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + '.part.mp4')
    try:
        with av.open(str(temp), mode='w') as container:
            stream = container.add_stream('libx264', rate=30)
            stream.width, stream.height = 346 * 3, 260
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'fast', 'threads': '1'}
            for source_frame, activity in zip(frames, counts, strict=True):
                davis = np.asarray(source_frame, dtype=np.uint8)
                events = colorize(activity)
                active = (activity > 0).any(axis=0)
                overlay = davis.copy()
                overlay[active] = (0.4 * davis[active] + 0.6 * events[active]).astype(np.uint8)
                pair = np.concatenate((davis, events, overlay), axis=1)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pair, format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f'{output}: {stop - args.start} frames, 30 FPS; left DAVIS, middle events, right overlay')


if __name__ == '__main__':
    main()
