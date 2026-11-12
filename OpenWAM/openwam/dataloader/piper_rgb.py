"""Native OpenWAM joint RGB/action windows from the Piper RGB-only export."""
from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.utils import get_cfg
from openwam.dataloader.utils.video_io import decode_video_frames


class PiperRGBDataset(BaseDataset):
    """Keep simulator commands as recorded; do not convert joint state to EEF.

    A window has current RGB plus future RGB, action[k:k+K], and state[k].
    Tail RGB is repeated, delta commands padded with zero and masked. Only the
    prepared RGB-only dataset is accepted, preventing event/GT feature leakage.
    """

    action_dim = 7
    state_dim = 8

    def __init__(self, dataset_dir, tasks, num_frames=25, video_stride=2,
                 height=256, width=512, window_stride=1, split="train"):
        if split != "train":
            raise ValueError("Piper export is training-only; evaluation uses independent simulator seeds")
        self.root = Path(dataset_dir)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != "piper_rgb_baselines_v1" or not manifest.get("rgb_only"):
            raise ValueError("Run scripts/prepare_piper_rgb_baselines.py first")
        self.num_frames = int(num_frames)
        self.horizon = self.num_frames - 1
        self.video_stride = int(video_stride)
        self.window_stride = int(window_stride)
        self.height, self.width = int(height), int(width)
        if self.horizon < 1 or self.video_stride < 1 or self.window_stride < 1:
            raise ValueError("Window and stride must be positive")
        if self.horizon % self.video_stride or (self.horizon // self.video_stride) % 4:
            raise ValueError("Wan requires 4n+1 sampled video frames")
        if self.height % 32 or self.width % 64:
            raise ValueError("Two-view canvas requires height % 32 == 0 and width % 64 == 0")
        self.stats = json.loads((self.root / "pooled_stats.json").read_text())
        self.normalization_stats_path = str(self.root / "openwam_normalization_stats.npy")
        self.episodes, self.ends = [], []
        total = 0
        if len(set(tasks)) != len(tasks) or not tasks:
            raise ValueError("Select distinct nonempty training tasks")
        for task in tasks:
            directory = self.root / f"piper_{task}"
            info = json.loads((directory / "meta/info.json").read_text())
            if any("event" in key or "object" in key for key in info["features"]):
                raise ValueError("Only RGB-only prepared datasets are supported")
            prompts = {r["task_index"]: r["task"] for r in self._jsonl(directory / "meta/tasks.jsonl")}
            for ep in self._jsonl(directory / "meta/episodes.jsonl"):
                n = int(ep["length"])
                self.episodes.append((directory, info, prompts, int(ep["episode_index"]), n))
                total += len(range(0, n - 1, self.window_stride))
                self.ends.append(total)

    @staticmethod
    def _jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @property
    def normalization_stats(self):
        return {k: np.asarray(v, dtype=np.float32) for k, v in self.stats["action"].items() if k != "count"}

    def normalize(self, values, key):
        lo = np.asarray(self.stats[key]["min"], dtype=np.float32)
        hi = np.asarray(self.stats[key]["max"], dtype=np.float32)
        span = hi - lo
        return np.where(span > 1e-6, 2 * (values - lo) / np.maximum(span, 1e-6) - 1, 0).astype(np.float32)

    def denormalize_action(self, values):
        lo = np.asarray(self.stats["action"]["min"], dtype=np.float32)
        hi = np.asarray(self.stats["action"]["max"], dtype=np.float32)
        return (np.asarray(values) + 1) * .5 * (hi - lo) + lo

    @staticmethod
    @lru_cache(maxsize=32)
    def _rows(path):
        table = pq.read_table(path, columns=["observation.state", "action", "task_index"])
        return (np.asarray(table["observation.state"].to_pylist(), dtype=np.float32),
                np.asarray(table["action"].to_pylist(), dtype=np.float32),
                np.asarray(table["task_index"].to_pylist()))

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect_right(self.ends, index)
        start = (index - (self.ends[position - 1] if position else 0)) * self.window_stride
        directory, info, prompts, ep, n = self.episodes[position]
        fields = {"episode_chunk": ep // info["chunks_size"], "episode_index": ep}
        states, actions, task_ids = self._rows(str(directory / info["data_path"].format(**fields)))
        valid = min(self.horizon, n - 1 - start)
        raw_action = np.repeat(actions[start:start + 1], self.horizon, axis=0)
        raw_action[:, :6] = 0
        raw_action[:valid] = actions[start:start + valid]
        offsets = np.arange(0, self.num_frames, self.video_stride) + start
        indices = np.minimum(offsets, n - 1).tolist()
        views = [decode_video_frames(str(directory / info["video_path"].format(
                    **fields, video_key=camera)), indices, self.height, self.width // 2)
                 for camera in ("observation.images.cam0", "observation.images.cam_wrist")]
        video = []
        for left, right in zip(*views):
            canvas = Image.new("RGB", (self.width, self.height))
            canvas.paste(left, (0, 0))
            canvas.paste(right, (self.width // 2, 0))
            video.append(canvas)
        return {"video": video, "first_frame_image": [video[0]], "vace_video": None,
                "action": torch.from_numpy(self.normalize(raw_action, "action")),
                "action_mask": torch.from_numpy(np.repeat((np.arange(self.horizon) < valid)[:, None], 7, axis=1)),
                "proprio": torch.from_numpy(self.normalize(states[start:start + 1], "observation.state")),
                "proprio_mask": torch.ones((1, 8), dtype=torch.bool),
                "video_mask": torch.from_numpy(offsets < n), "prompt": prompts[int(task_ids[start])]}

    @classmethod
    def from_config(cls, config, split="train"):
        return cls(**{key: get_cfg(config, key) for key in (
            "dataset_dir", "tasks", "num_frames", "video_stride", "height", "width", "window_stride")}, split=split)
