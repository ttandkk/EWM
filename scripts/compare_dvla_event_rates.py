#!/usr/bin/env python3
"""Compare original 250 FPS polarity render with 25 FPS 40 ms accumulations."""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np

SOURCE_FPS = 250
TARGET_FPS = 25
FACTOR = SOURCE_FPS // TARGET_FPS
THRESHOLD = 25


def colorize(blue: np.ndarray, red: np.ndarray) -> np.ndarray:
    height, width = blue.shape
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[blue & ~red] = (0, 75, 235)
    image[red & ~blue] = (235, 45, 30)
    image[blue & red] = (170, 40, 170)
    return image


def label_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, width = left.shape[:2]
    canvas = np.zeros((height + 30, width * 2, 3), dtype=np.uint8)
    canvas[30:, :width] = left
    canvas[30:, width:] = right
    cv2.putText(canvas, 'Original event: 250 FPS, 4 ms', (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Converted event: 25 FPS, 40 ms', (width + 8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render(source: Path, counts_path: Path, output: Path) -> None:
    with np.load(counts_path) as archive:
        counts = archive['counts']
    if counts.ndim != 4 or counts.shape[1] != 2 or counts.dtype != np.uint8:
        raise ValueError(f'expected uint8 [T,2,H,W], got {counts.shape} {counts.dtype}')
    frames, _, height, width = counts.shape
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + '.part.mp4')
    try:
        with av.open(str(source)) as input_video, av.open(str(temp), mode='w') as container:
            stream = container.add_stream('libx264', rate=SOURCE_FPS)
            stream.width, stream.height = 2 * width, height + 30
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'fast', 'threads': '1'}
            right = None
            produced = 0
            for index, frame in enumerate(input_video.decode(video=0)):
                if index == frames * FACTOR:
                    break  # Drop the source tail that has no complete 40 ms window.
                raw = frame.to_ndarray(format='rgb24')
                if raw.shape != (height, width, 3):
                    raise ValueError(f'unexpected source frame shape {raw.shape}')
                values = raw.astype(np.int16)
                left = colorize(values[:, :, 2] > values[:, :, 1] + THRESHOLD,
                                values[:, :, 0] > values[:, :, 1] + THRESHOLD)
                if index % FACTOR == 0:
                    right = colorize(counts[index // FACTOR, 0] > 0,
                                     counts[index // FACTOR, 1] > 0)
                pair = label_pair(left, right)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pair, format='rgb24')):
                    container.mux(packet)
                produced += 1
            if produced != frames * FACTOR:
                raise ValueError(f'source event video too short: {produced} frames')
            for packet in stream.encode():
                container.mux(packet)
        temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f'{output} ({produced} frames, {SOURCE_FPS} FPS, {width * 2}x{height + 30})')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='original 250 FPS event MP4')
    parser.add_argument('counts', type=Path, help='converted 25 FPS event_activity_counts NPZ')
    parser.add_argument('output', type=Path, help='real-time 250 FPS comparison MP4')
    args = parser.parse_args()
    render(args.source, args.counts, args.output)


if __name__ == '__main__':
    main()
