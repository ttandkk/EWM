"""Native-rate REASSEMBLE/DVLA event-frame reader for event-only EWM pretraining."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from openwam.dataloader.bases import BaseDataset

from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, load_episodes_parquet
from openwam.dataloader.utils.video_io import decode_video_frames


class OccupancyNormalizer:
    """Records the event-only input convention for the generic EWM trainer."""

    count_clip = 15.0
    metadata = {"event_representation": "binary_native_activity_then_area_resize",
                "event_range": [0.0, 1.0], "event_channel_order": ["OFF", "ON"],
                "source_channel_indices": {"reassemble": [0, 1], "dvla": [1, 0]},
                "action": None, "proprio": None}


@lru_cache(maxsize=6)
def _counts(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        counts = archive["counts"]
    if counts.dtype != np.uint8 or counts.ndim != 4 or counts.shape[1] != 2:
        raise ValueError(f"invalid event frames: {path}")
    return counts


def _event_occupancy(frames: np.ndarray, *, source: str = "reassemble") -> torch.Tensor:
    # Resize at native aspect ratio. The two event sources have different count
    # semantics, so retain only native activity before area downsampling.
    # REASSEMBLE source 0/1 is OFF/ON. DVLA's rendered blue/red is ON/OFF
    # (same-view RGB brightness audit: tmp/event_only_contract_audit/REPORT.md).
    if source not in ("reassemble", "dvla"):
        raise ValueError(f"unknown event source: {source}")
    if source == "dvla":
        frames = frames[:, [1, 0]]
    activity = torch.from_numpy((frames > 0).astype(np.float32))
    activity = F.interpolate(activity, size=(192, 256), mode="area")
    return F.pad(activity, (0, 0, 32, 32))[:, None]  # [time,camera,2,256,256]


def _rgb_frame(path: Path, index: int) -> torch.Tensor:
    image = decode_video_frames(str(path), [index], 192, 256)[0]
    frame = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)
    return F.pad(frame, (0, 0, 32, 32))


class EWMMixedEventOnlyDataset(BaseDataset):
    """One current RGB/event frame and K future event frames from either export.

    The official REASSEMBLE train split is honored. DVLA uses its train split.
    Event accumulations are disjoint at each source rate; training windows
    slide by window_stride frames, matching Piper's current1 + futureK layout.
    Neither source action nor state is opened. A source-specific neutral prompt
    tells the shared model which count process/camera produced the sample.
    """

    voxel_bins = 1
    event_camera_mode = "wrist_only"  # One event view; camera role is source-specific.
    gpu_voxelization = False
    normalizer = OccupancyNormalizer()

    def __init__(self, *, reassemble_dir=None, dvla_dir=None, chunk_size=24,
                 window_stride=1, dvla_event_camera="opst_cam", split="train"):
        if split != "train":
            raise ValueError("joint event-only reader currently supports the training split only")
        self.chunk_size, self.window_stride = int(chunk_size), int(window_stride)
        if self.chunk_size <= 0 or self.window_stride <= 0:
            raise ValueError("chunk_size and window_stride must be positive")
        if dvla_event_camera not in ("wrist_cam", "side_cam", "opst_cam"):
            raise ValueError("invalid DVLA event camera")
        self.dvla_event_camera = dvla_event_camera
        self.episodes = []
        if reassemble_dir:
            self._add_reassemble(Path(reassemble_dir))
        if dvla_dir:
            self._add_dvla(Path(dvla_dir))
        if not self.episodes:
            raise ValueError("no training episodes; provide at least one completed export")
        counts = [max(0, (row["length"] - self.chunk_size - 1) // self.window_stride + 1)
                  for row in self.episodes]
        self._cumulative = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        if not len(self):
            raise ValueError("no episode is long enough for the requested horizon")
        self.source_ranges = {}
        for i, row in enumerate(self.episodes):
            start, stop = int(self._cumulative[i]), int(self._cumulative[i + 1])
            if stop > start:
                self.source_ranges.setdefault(row["source"], []).append((start, stop))

    @staticmethod
    def _schema(root: Path, fps: int) -> dict:
        if not (root / ".export_complete").is_file() or not (root / "meta/event_frames.json").is_file():
            raise ValueError(f"completed event-frame export required: {root}")
        schema = json.loads((root / "meta/event_frames.json").read_text())
        if schema.get("fps") != fps or schema.get("event_frame_feature", {}).get("array_key") != "counts":
            raise ValueError(f"incompatible event-frame export: {root}")
        return schema

    def _add_reassemble(self, root: Path) -> None:
        self._schema(root, 30)
        episodes = load_episodes_parquet(root)
        for camera in ("observation.images.hama1", "observation.images.hand"):
            episodes[f"_offset/{camera}"] = compute_file_local_offsets(
                episodes, f"videos/{camera}/chunk_index", f"videos/{camera}/file_index")
        splits = json.loads((root / "meta/splits.json").read_text())
        info = json.loads((root / "meta/info.json").read_text())
        for _, row in episodes.iterrows():
            episode = int(row["episode_index"])
            if splits[str(episode)]["split"] != "train":
                continue
            videos = []
            for camera in ("observation.images.hama1", "observation.images.hand"):
                videos.append((root / info["video_path"].format(
                    video_key=camera,
                    chunk_index=int(row[f"videos/{camera}/chunk_index"]),
                    file_index=int(row[f"videos/{camera}/file_index"])),
                    int(row[f"_offset/{camera}"])))
            self.episodes.append(dict(source="reassemble", root=root, episode=episode,
                                      length=int(row["length"]), fps=30, videos=videos,
                                      prompt="REASSEMBLE robot manipulation"))

    def _add_dvla(self, root: Path) -> None:
        self._schema(root, 25)
        # The exporter only renames the .partial directory after final verification.
        if not (root / "meta/episodes.jsonl").is_file():
            raise ValueError(f"DVLA export is incomplete: {root}")
        info = json.loads((root / "meta/info.json").read_text())
        with (root / "meta/episodes.jsonl").open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != int(info["total_episodes"]):
            raise ValueError("DVLA episode table length does not match info.json")
        for row in rows:
            episode = int(row["episode_index"])
            videos = []
            for camera in ("opst_cam", "wrist_cam"):
                path = root / info["video_path"].format(
                    episode_chunk=episode // int(info["chunks_size"]),
                    episode_index=episode, video_key=f"observation.images.{camera}")
                videos.append((path, 0))
            self.episodes.append(dict(source="dvla", root=root, episode=episode,
                                      length=int(row["length"]), fps=25, videos=videos,
                                      prompt="DVLA robot manipulation"))

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def _event_frames(self, row: dict, indices: np.ndarray) -> np.ndarray:
        root, episode = row["root"], row["episode"]
        if row["source"] == "dvla":
            name = (root / "event_frames" / f"episode_{episode:06d}" / self.dvla_event_camera /
                    f"frames_000000_{row['length']:06d}.npz")
            return _counts(str(name))[indices]
        chunks = []
        for index in indices:
            start = (int(index) // 256) * 256
            stop = min(start + 256, row["length"])
            name = root / "event_frames" / f"episode_{episode:06d}" / f"frames_{start:06d}_{stop:06d}.npz"
            chunks.append(_counts(str(name))[int(index) - start])
        return np.stack(chunks)

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        pos = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        row = self.episodes[pos]
        anchor = int(index - self._cumulative[pos]) * self.window_stride
        indices = np.arange(anchor, anchor + self.chunk_size + 1)
        events = _event_occupancy(self._event_frames(row, indices), source=row["source"])
        rgb = torch.stack([_rgb_frame(path, offset + anchor) for path, offset in row["videos"]])
        valid = torch.zeros((1, 256, 256), dtype=torch.bool)
        valid[:, 32:224] = True
        return {"obs_video": rgb, "obs_voxel": events[0], "future_voxel": events[1:],
                "valid_region_mask": valid, "prompt": row["prompt"],
                "source": row["source"], "fps": float(row["fps"])}

    @classmethod
    def from_config(cls, config, split="train"):
        from openwam.dataloader.utils import get_cfg
        return cls(reassemble_dir=get_cfg(config, "reassemble_dir"),
                   dvla_dir=get_cfg(config, "dvla_dir"),
                   chunk_size=get_cfg(config, "chunk_size", 24),
                   window_stride=get_cfg(config, "window_stride", 1),
                   dvla_event_camera=get_cfg(config, "dvla_event_camera", "opst_cam"), split=split)
