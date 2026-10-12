#!/usr/bin/env python3
"""Render a labeled RGB-and-event preview from a Piper HDF5 episode."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

import cv2
import h5py
import numpy as np


def _label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (256, 27), (0, 0, 0), -1)
    cv2.putText(image, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _events(x: np.ndarray, y: np.ndarray, p: np.ndarray, h: int, w: int) -> np.ndarray:
    """Accumulate events; positive is red and negative is blue."""
    pos, neg = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    np.add.at(pos, (y[p > 0], x[p > 0]), 1)
    np.add.at(neg, (y[p < 0], x[p < 0]), 1)
    pos, neg = (np.clip(np.log1p(v) / np.log(8.0), 0, 1) for v in (pos, neg))
    frame = np.zeros((h, w, 3), np.uint8)
    frame[..., 2] = np.round(255 * pos).astype(np.uint8)
    frame[..., 0] = np.round(255 * neg).astype(np.uint8)
    frame[..., 1] = np.round(80 * np.minimum(pos, neg)).astype(np.uint8)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.episode.with_name(args.episode.stem + "_preview.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.episode, "r") as h5:
        t = h5["observation/t"][:]
        cams = [str(v) for v in h5["meta"].attrs["camera_names"]]
        if cams != ["cam0", "cam_wrist"]:
            raise ValueError(f"Expected cam0/cam_wrist, got {cams}")
        h, w = h5["observation/rgb/cam0"].shape[1:3]
        fps = round(1 / np.median(np.diff(t)))
        event = {cam: tuple(h5[f"events/{cam}/{k}"][:] for k in ("t", "x", "y", "p")) for cam in cams}
        offset = {cam: np.searchsorted(event[cam][0], np.r_[t, t[-1] + 1 / fps]) for cam in cams}
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for H.264 preview output")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{2 * w}x{2 * h}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        video = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert video.stdin is not None
        try:
            for i, ts in enumerate(t):
                panes = []
                for cam in cams:
                    rgb = cv2.cvtColor(h5[f"observation/rgb/{cam}"][i], cv2.COLOR_RGB2BGR)
                    _, x, y, p = event[cam]
                    start, end = offset[cam][i : i + 2]
                    panes += [_label(rgb, f"RGB {cam}  t={ts:.2f}s"), _label(_events(x[start:end], y[start:end], p[start:end], h, w), f"Events {cam}  n={end-start}")]
                frame = np.vstack((np.hstack((panes[0], panes[2])), np.hstack((panes[1], panes[3]))))
                video.stdin.write(frame.tobytes())
        finally:
            video.stdin.close()
        if video.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with exit status {video.returncode}")
    print(output)


if __name__ == "__main__":
    main()
