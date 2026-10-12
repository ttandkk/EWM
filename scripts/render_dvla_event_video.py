#!/usr/bin/env python3
"""Render a DVLA 25 FPS event-activity sidecar as a polarity-only video."""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np


def render(source: Path, output: Path, fps: int) -> None:
    with np.load(source) as archive:
        counts = archive['counts']
    if counts.ndim != 4 or counts.shape[1] != 2 or counts.dtype != np.uint8:
        raise ValueError(f'expected uint8 [T,2,H,W] event activity counts; got {counts.shape}, {counts.dtype}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + '.part.mp4')
    try:
        with av.open(str(temp), mode='w') as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width = counts.shape[3]
            stream.height = counts.shape[2]
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '17', 'preset': 'fast', 'threads': '1'}
            for frame_counts in counts:
                blue = frame_counts[0] > 0
                red = frame_counts[1] > 0
                rgb = np.full((counts.shape[2], counts.shape[3], 3), 255, dtype=np.uint8)
                rgb[blue & ~red] = (0, 75, 235)
                rgb[red & ~blue] = (235, 45, 30)
                rgb[blue & red] = (170, 40, 170)
                frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f'{output} ({counts.shape[0]} frames, {fps} FPS, {counts.shape[3]}x{counts.shape[2]})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='event_activity_counts/.../episode_*.npz')
    parser.add_argument('output', type=Path, help='rendered .mp4 path')
    parser.add_argument('--fps', type=int, default=25)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    render(args.source, args.output, args.fps)


if __name__ == '__main__':
    main()
