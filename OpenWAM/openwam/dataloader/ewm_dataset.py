"""Piper stack-cube EWM reader: current RGB/event condition to future targets."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data._utils.collate import default_collate

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.ewm_normalizer import EWMNormalizer
from openwam.dataloader.utils import get_cfg


def build_causal_event_voxel(events, query_t, *, window_s, num_bins, height, width, count_clip):
    """Return uint8 [2, num_bins, H, W] over the causal interval (q-window, q]."""
    if min(window_s, num_bins, height, width, count_clip) <= 0:
        raise ValueError("voxel parameters must be positive")
    if events.dtype.names is None or not {"x", "y", "t", "p"}.issubset(events.dtype.names):
        raise ValueError("events must be structured x/y/t/p records")
    start_t = float(query_t) - float(window_s)
    begin = int(np.searchsorted(events["t"], start_t, side="right"))
    end = int(np.searchsorted(events["t"], float(query_t), side="right"))
    out = np.zeros((2, num_bins, height, width), dtype=np.uint8)
    if begin == end:
        return out
    e = events[begin:end]
    x, y, p = e["x"].astype(np.int64), e["y"].astype(np.int64), e["p"]
    if np.any(x >= width) or np.any(y >= height):
        raise ValueError("event coordinates exceed voxel resolution")
    if not np.all((p == -1) | (p == 1)):
        raise ValueError("polarity must be -1/+1")
    bins = np.clip(np.floor((e["t"] - start_t) * num_bins / window_s).astype(np.int64), 0, num_bins - 1)
    polarity = (p > 0).astype(np.int64)
    flat = (((polarity * num_bins + bins) * height + y) * width + x)
    counts = np.bincount(flat, minlength=2 * num_bins * height * width)
    return np.minimum(counts.reshape(2, num_bins, height, width), count_clip).astype(np.uint8)


from openwam.dataloader.utils.lerobotv3 import apply_info_splits, compute_file_local_offsets, load_episodes_parquet, parse_info_json
from openwam.dataloader.utils.video_io import decode_video_frames, decode_video_frames_torchcodec

CAMERAS = ("cam0", "cam_wrist")
EVENT_CAMERA_MODES = {"dual": CAMERAS, "wrist_only": ("cam_wrist",), "global_only": ("cam0",)}
VIDEO_KEYS = tuple(f"observation.images.{camera}" for camera in CAMERAS)
ACTION_DIM, STATE_DIM = 7, 8


@lru_cache(maxsize=8)
def _read_table(path):
    return pq.read_table(path, columns=["task_index", "observation.state", "action"])


@lru_cache(maxsize=2)
def _read_event_frame_table(path, cameras):
    return pq.read_table(path, columns=[f"observation.event_frame.{camera}" for camera in cameras])


@lru_cache(maxsize=4)
def _read_event_frame_cache(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class EWMDataset(BaseDataset):
    """Returns current dual-view RGB/voxel and future action/voxel supervision."""

    def __init__(self, dataset_dir, *, chunk_size=8, context_frames=1, window_stride=1,
                 voxel_window_ms=None, voxel_bins=4, count_clip=15, event_cache_size=8,
                 normalization_stats_path=None, event_camera_mode="dual", gpu_voxelization=False,
                 observation_row_offset=0, event_frame_cache_dir=None,
                 rgb_decode_backend="pyav"):
        self.dataset_dir = Path(dataset_dir)
        self.chunk_size, self.context_frames = int(chunk_size), int(context_frames)
        self.window_stride, self.voxel_bins = int(window_stride), int(voxel_bins)
        self.count_clip, self.event_cache_size = int(count_clip), int(event_cache_size)
        # Direct event-frame datasets store RGB/state/event at the same row.
        # The -1 offset remains only for legacy sparse-event exports.
        self.observation_row_offset = int(observation_row_offset)
        if self.observation_row_offset not in (-1, 0):
            raise ValueError("observation_row_offset must be 0 (direct event-frame) or -1 (legacy sparse events)")
        # At exported time zero, the converter assigns keyframe events to
        # future[0] while the next causal loader window excludes that exact
        # boundary. Begin aligned sampling at k=2 to avoid this one-off bin.
        self.first_anchor = 2 if self.observation_row_offset == -1 else 0
        if event_camera_mode not in EVENT_CAMERA_MODES:
            raise ValueError(f"event_camera_mode must be one of {sorted(EVENT_CAMERA_MODES)}, got {event_camera_mode!r}")
        self.event_camera_mode = str(event_camera_mode)
        self.event_cameras = EVENT_CAMERA_MODES[self.event_camera_mode]
        self.event_frame_cache_dir = Path(event_frame_cache_dir) if event_frame_cache_dir else None
        if rgb_decode_backend not in ("pyav", "torchcodec"):
            raise ValueError("rgb_decode_backend must be pyav or torchcodec")
        self.rgb_decode_backend = rgb_decode_backend
        self.gpu_voxelization = bool(gpu_voxelization)
        if self.gpu_voxelization and self.voxel_bins != 1:
            raise ValueError("gpu_voxelization currently supports voxel_bins=1 only")
        if self.context_frames != 1:
            raise ValueError(
                "EWMDataset uses exactly one current observation. "
                "Set context_frames=1."
            )
        if min(self.chunk_size, self.context_frames, self.window_stride, self.voxel_bins, self.count_clip, self.event_cache_size) <= 0:
            raise ValueError("EWM loader sizes must be positive")
        self.normalizer = EWMNormalizer.from_dataset_dir(
            self.dataset_dir,
            stats_path=normalization_stats_path,
            expected_count_clip=self.count_clip,
        )
        if self.normalizer.voxel_bins != self.voxel_bins:
            raise ValueError(
                f"voxel bin mismatch: loader={self.voxel_bins}, stats={self.normalizer.voxel_bins}"
            )

        info = parse_info_json(self.dataset_dir)
        self.direct_event_frames = all(
            info["features"].get(f"observation.event_frame.{camera}", {}).get("dtype") == "image"
            and tuple(info["features"][f"observation.event_frame.{camera}"]["shape"]) == (256, 256, 3)
            for camera in self.event_cameras
        )
        if self.direct_event_frames:
            if self.observation_row_offset != 0:
                raise ValueError("direct event-frame datasets require observation_row_offset=0")
            if self.voxel_bins != 1:
                raise ValueError("direct event-frame datasets contain exactly one temporal bin")
            self.first_anchor = 0
        elif self.event_frame_cache_dir is not None:
            raise ValueError("event_frame_cache_dir requires direct aligned event frames")
        self.fps = float(info["fps"])
        if not np.isclose(self.fps, 30.0):
            raise ValueError(f"Piper EWM requires 30Hz data, got {self.fps}")
        self.height = self.width = 256
        for key in VIDEO_KEYS:
            if tuple(info["features"].get(key, {}).get("shape", ())) != (256, 256, 3):
                raise ValueError(f"missing or incompatible feature: {key}")
        if tuple(info["features"].get("action", {}).get("shape", ())) != (ACTION_DIM,):
            raise ValueError("Piper EWM requires action shape [7]")
        if tuple(info["features"].get("observation.state", {}).get("shape", ())) != (STATE_DIM,):
            raise ValueError("Piper EWM requires observation.state shape [8]")
        self.voxel_window_s = float(voxel_window_ms) / 1000 if voxel_window_ms is not None else 1 / self.fps
        if self.voxel_window_s <= 0:
            raise ValueError("voxel_window_ms must be positive")
        if self.direct_event_frames and not np.isclose(self.voxel_window_s, 1 / self.fps):
            raise ValueError("direct event-frame window is fixed at one 30 Hz control interval")

        eps = load_episodes_parquet(self.dataset_dir)
        eps["_data_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
        for key in VIDEO_KEYS:
            eps[f"_video_offset/{key}"] = compute_file_local_offsets(eps, f"videos/{key}/chunk_index", f"videos/{key}/file_index")
        self.episodes = apply_info_splits(eps, "train", info.get("splits", {}) or {}, source_name="PiperStackEWM").reset_index(drop=True)
        self.data_path, self.video_path = info["data_path"], info["video_path"]
        self._event_frame_cache = {}
        if self.event_frame_cache_dir is not None:
            manifest_path = self.event_frame_cache_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("format") != "ewm_raw_event_frames_v1" or manifest.get("cameras") != list(CAMERAS):
                raise ValueError(f"incompatible event-frame cache manifest: {manifest_path}")
            for _, episode in self.episodes.iterrows():
                source = self.dataset_dir / self.data_path.format(
                    chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"]))
                relative = source.relative_to(self.dataset_dir).as_posix()
                if str(source) in self._event_frame_cache:
                    continue
                entry = manifest["files"].get(relative)
                if entry is None or source.stat().st_size != entry["source_size"] or source.stat().st_mtime_ns != entry["source_mtime_ns"]:
                    raise ValueError(f"missing or stale event-frame cache for {relative}")
                cached = self.event_frame_cache_dir / entry["cache_file"]
                array = np.load(cached, mmap_mode="r", allow_pickle=False)
                if array.dtype != np.uint8 or tuple(array.shape) != (entry["rows"], 2, 2, self.height, self.width):
                    raise ValueError(f"invalid event-frame cache shape/dtype: {cached}")
                self._event_frame_cache[str(source)] = str(cached)
                del array
        tasks = pd.read_parquet(self.dataset_dir / "meta" / "tasks.parquet")
        self.prompts = {int(i): str(text).strip() for text, i in zip(tasks.index, tasks["task_index"], strict=True)}
        if not self.prompts or any(not prompt for prompt in self.prompts.values()):
            raise ValueError("tasks.parquet must provide nonempty prompts")
        self._event_cache = OrderedDict()
        self._event_offset_cache = OrderedDict()

        counts = []
        for _, row in self.episodes.iterrows():
            available = int(row["length"]) - self.chunk_size - self.first_anchor
            counts.append(max(0, (available - 1) // self.window_stride + 1))
        self._cumulative = np.concatenate(([0], np.cumsum(np.asarray(counts, dtype=np.int64))))
        if len(self) == 0:
            raise ValueError("no valid samples for context_frames/chunk_size")

    @property
    def action_dim(self):
        return ACTION_DIM

    @property
    def state_dim(self):
        return STATE_DIM

    @property
    def normalization_stats(self):
        """Persisted train-split stats shared with deployment and visualization."""
        return self.normalizer.metadata

    def __len__(self):
        return int(self._cumulative[-1])

    def _events_for(self, episode_index, camera):
        key = (int(episode_index), camera)
        events = self._event_cache.pop(key, None)
        if events is None:
            path = self.dataset_dir / "events" / f"episode_{key[0]:06d}" / camera / "events.npy"
            events = np.load(path, mmap_mode="r")
            if events.dtype.names is None or not {"x", "y", "t", "p"}.issubset(events.dtype.names):
                raise ValueError(f"invalid compact event file: {path}")
        self._event_cache[key] = events
        while len(self._event_cache) > self.event_cache_size:
            self._event_cache.popitem(last=False)
        return events

    def _voxel(self, episode_index, frame_index):
        query_t = float(frame_index) / self.fps
        views = [
            build_causal_event_voxel(
                self._events_for(episode_index, camera), query_t,
                window_s=self.voxel_window_s, num_bins=self.voxel_bins,
                height=self.height, width=self.width, count_clip=self.count_clip,
            ).reshape(2 * self.voxel_bins, self.height, self.width)
            for camera in self.event_cameras
        ]
        counts = torch.from_numpy(np.stack(views).copy())
        return self.normalizer.normalize_voxel(counts)

    def _event_offsets_for(self, episode_index, camera, episode_length):
        """Cache causal-window event offsets; avoids per-sample searchsorted."""
        key = (int(episode_index), camera)
        offsets = self._event_offset_cache.pop(key, None)
        if offsets is None:
            events = self._events_for(episode_index, camera)
            query = np.arange(int(episode_length), dtype=np.float64) / self.fps
            starts = np.searchsorted(events["t"], query - self.voxel_window_s, side="right")
            ends = np.searchsorted(events["t"], query, side="right")
            offsets = np.stack((starts, ends), axis=1).astype(np.int64, copy=False)
        self._event_offset_cache[key] = offsets
        while len(self._event_offset_cache) > self.event_cache_size:
            self._event_offset_cache.popitem(last=False)
        return offsets

    def _sparse_event_frames(self, episode_index, episode_length, frame_indices):
        """Return raw event slices, one per requested frame/camera, for GPU histogramming."""
        frames = []
        for frame_index in frame_indices:
            for camera in self.event_cameras:
                offsets = self._event_offsets_for(episode_index, camera, episode_length)
                begin, end = offsets[int(frame_index)]
                frames.append(self._events_for(episode_index, camera)[begin:end])
        return frames

    def _direct_event_frames(self, path, row_indices):
        if self.event_frame_cache_dir is not None:
            cache = _read_event_frame_cache(self._event_frame_cache[str(path)])
            indices = np.asarray(row_indices, dtype=np.int64)
            if len(indices) and np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))):
                frames = cache[int(indices[0]):int(indices[0]) + len(indices)]
            else:
                frames = cache[indices]
            camera_indices = tuple(CAMERAS.index(camera) for camera in self.event_cameras)
            return self.normalizer.normalize_voxel(torch.from_numpy(frames[:, camera_indices].copy()))
        table = _read_event_frame_table(str(path), self.event_cameras)
        views = []
        for camera in self.event_cameras:
            column = table[f"observation.event_frame.{camera}"]
            frames = []
            for row_index in row_indices:
                encoded = column[int(row_index)].as_py()
                if encoded["bytes"] is None:
                    raise ValueError(f"missing embedded event-frame PNG at row {row_index}")
                with Image.open(BytesIO(encoded["bytes"])) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if rgb.shape != (self.height, self.width, 3) or np.any(rgb[..., 2]):
                    raise ValueError(f"invalid event-frame payload for {camera} at row {row_index}")
                frames.append(rgb[..., :2].transpose(2, 0, 1))
            views.append(np.stack(frames))
        # [time, camera, OFF/ON, H, W], with no event histogramming in the loader.
        return self.normalizer.normalize_voxel(
            torch.from_numpy(np.stack(views, axis=1).copy())
        )

    def _rgb(self, row, indices):
        views = []
        for key in VIDEO_KEYS:
            path = self.dataset_dir / self.video_path.format(
                video_key=key,
                chunk_index=int(row[f"videos/{key}/chunk_index"]),
                file_index=int(row[f"videos/{key}/file_index"]),
            )
            base = int(row[f"_video_offset/{key}"])
            if self.rgb_decode_backend == "torchcodec":
                frames = decode_video_frames_torchcodec(str(path), (base + indices).tolist(), self.height, self.width)
                views.append(frames)
            else:
                frames = decode_video_frames(str(path), (base + indices).tolist(), self.height, self.width)
                views.append(np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames]))
        return torch.from_numpy(np.stack(views).transpose(1, 0, 4, 2, 3).copy())

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep_local = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        row = self.episodes.iloc[ep_local]
        k = self.first_anchor + int(index - self._cumulative[ep_local]) * self.window_stride
        episode_index = int(row["episode_index"])
        context = np.arange(k - self.context_frames + 1, k + 1, dtype=np.int64)
        future = np.arange(k + 1, k + self.chunk_size + 1, dtype=np.int64)
        path = self.dataset_dir / self.data_path.format(chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        start = int(row["_data_offset"]) + k
        table = _read_table(str(path))
        current = table.slice(start + self.observation_row_offset, 1).to_pandas().iloc[0]
        targets = table.slice(start, self.chunk_size).to_pandas()
        task_index = int(current["task_index"])
        if task_index not in self.prompts:
            raise KeyError(f"missing prompt for task_index={task_index}")
        sample = {
            "obs_video": self._rgb(row, context + self.observation_row_offset)[0],
            "proprio": self.normalizer.normalize_proprio(
                torch.from_numpy(np.asarray(current["observation.state"], dtype=np.float32).copy())
            ),
            "prompt": self.prompts[task_index],
            "action": self.normalizer.normalize_action(
                torch.from_numpy(np.stack(targets["action"].to_numpy()).astype(np.float32, copy=False))
            ),
        }
        if self.direct_event_frames:
            frames = self._direct_event_frames(path, start + np.concatenate(([0], future - k)))
            sample["obs_voxel"] = frames[0]
            sample["future_voxel"] = frames[1:]
        elif self.gpu_voxelization:
            sample["event_sparse_frames"] = self._sparse_event_frames(
                episode_index, int(row["length"]), np.concatenate(([k], future))
            )
        else:
            sample["obs_voxel"] = self._voxel(episode_index, k)
            sample["future_voxel"] = torch.stack([self._voxel(episode_index, i) for i in future])
        return sample

    @classmethod
    def from_config(cls, config, split="train"):
        if split != "train":
            raise ValueError("EWMDataset supports only train")
        root = get_cfg(config, "dataset_dir")
        if root is None:
            raise ValueError("EWMDataset requires dataset_dir")
        return cls(root, chunk_size=int(get_cfg(config, "chunk_size", 8)),
                   context_frames=int(get_cfg(config, "context_frames", 1)),
                   window_stride=int(get_cfg(config, "window_stride", 1)),
                   voxel_window_ms=get_cfg(config, "voxel_window_ms", None),
                   voxel_bins=int(get_cfg(config, "voxel_bins", 4)),
                   count_clip=int(get_cfg(config, "count_clip", 15)),
                   event_cache_size=int(get_cfg(config, "event_cache_size", 8)),
                   normalization_stats_path=get_cfg(config, "normalization_stats_path", None),
                   event_camera_mode=str(get_cfg(config, "event_camera_mode", "dual")),
                   gpu_voxelization=bool(get_cfg(config, "gpu_voxelization", False)),
                   observation_row_offset=int(get_cfg(config, "observation_row_offset", 0)),
                   event_frame_cache_dir=get_cfg(config, "event_frame_cache_dir", None),
                   rgb_decode_backend=str(get_cfg(config, "rgb_decode_backend", "pyav")))


def collate_ewm_sparse_events(samples):
    """Collate sparse event records without materializing dense CPU voxels."""
    if not samples:
        raise ValueError("cannot collate an empty EWM batch")
    if "event_sparse_frames" not in samples[0]:
        return default_collate(samples)
    frames_per_sample = len(samples[0]["event_sparse_frames"])
    if any(len(sample["event_sparse_frames"]) != frames_per_sample for sample in samples):
        raise ValueError("inconsistent sparse event frame count")
    payload = tuple(key for key in samples[0] if key != "event_sparse_frames")
    batch = default_collate([{key: sample[key] for key in payload} for sample in samples])
    frame_ids, xs, ys, polarities = [], [], [], []
    for sample_index, sample in enumerate(samples):
        for frame_index, events in enumerate(sample["event_sparse_frames"]):
            if not len(events):
                continue
            frame_ids.append(np.full(len(events), sample_index * frames_per_sample + frame_index, dtype=np.int64))
            xs.append(events["x"].astype(np.int64, copy=False))
            ys.append(events["y"].astype(np.int64, copy=False))
            polarities.append((events["p"] > 0).astype(np.int64, copy=False))
    def _concat(parts):
        return torch.from_numpy(np.concatenate(parts) if parts else np.empty(0, dtype=np.int64))
    batch["event_sparse"] = {"frame_id": _concat(frame_ids), "x": _concat(xs), "y": _concat(ys), "polarity": _concat(polarities)}
    return batch
