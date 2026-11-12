"""Piper current RGB/state -> aligned actions, optionally using raw mmap caches."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .ewm_dataset import EWMDataset, VIDEO_KEYS
from .utils import get_cfg

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _read_action_table(path):
    # Column projection excludes the event PNG payloads in the same Parquet.
    return pq.read_table(path, memory_map=True, columns=["observation.state", "action", "task_index"])


@lru_cache(maxsize=512)
def _read_action_mmap(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


class PiperActionOnlyDataset(EWMDataset):
    """obs[k] conditions action[k:k+K]; no event payloads or future RGB targets."""

    def __init__(self, dataset_dir, *, action_only_mmap_cache_dir=None, current_dino_cache_dir=None,
                 current_dino_path=None, **kwargs):
        kwargs["gpu_voxelization"] = False
        kwargs["event_frame_cache_dir"] = None
        super().__init__(dataset_dir, **kwargs)
        contract = json.loads((Path(dataset_dir) / "meta" / "piper_temporal_contract.json").read_text())
        if (contract.get("temporal_contract") != "aligned_event_frame_pre_action_v1" or
                self.observation_row_offset != 0):
            raise ValueError("Piper action-only requires aligned pre-action RGB/state/action rows")
        self._configure_action_mmap_cache(action_only_mmap_cache_dir)
        from .current_dino_cache import CurrentDinoCache
        self.current_dino_cache = (CurrentDinoCache(self, current_dino_cache_dir, current_dino_path)
                                   if current_dino_cache_dir else None)

    @classmethod
    def from_config(cls, config, split="train"):
        dataset = super().from_config(config, split)
        dataset._configure_action_mmap_cache(get_cfg(config, "action_only_mmap_cache_dir", None))
        from .current_dino_cache import CurrentDinoCache
        cache = get_cfg(config, "current_dino_cache_dir", None)
        dataset.current_dino_cache = (CurrentDinoCache(dataset, cache, get_cfg(config, "current_dino_path"))
                                      if cache else None)
        return dataset

    def _configure_action_mmap_cache(self, cache_dir):
        self.action_only_mmap_cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self._action_mmap_files, self._video_mmap_files = {}, {}
        if self.action_only_mmap_cache_dir is None:
            return
        root = self.dataset_dir.resolve()
        cache = self.action_only_mmap_cache_dir
        manifest = json.loads((cache / "manifest.json").read_text())
        if (manifest.get("format") != "ewm_action_only_mmap_v1" or
                Path(manifest["dataset_dir"]).resolve() != root or
                manifest.get("rgb_layout") != "TCHW" or
                (manifest.get("height"), manifest.get("width")) != (256, 256)):
            raise ValueError("incompatible action-only mmap cache")

        def check_source(source, entries):
            relative = source.resolve().relative_to(root).as_posix()
            entry = entries.get(relative)
            stat = source.stat()
            if entry is None or (entry["source_size"], entry["source_mtime_ns"]) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError(f"missing or stale action-only mmap source: {relative}")
            return entry

        def check_array(relative, dtype, shape):
            path = (cache / relative).resolve()
            if not path.is_relative_to(cache):
                raise ValueError("mmap array must stay inside its cache directory")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if not isinstance(array, np.memmap) or array.dtype != dtype or array.shape != shape:
                raise ValueError(f"invalid action-only mmap array: {path}")
            return str(path)

        for _, row in self.episodes.iterrows():
            source = self.dataset_dir / self.data_path.format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
            if str(source) not in self._action_mmap_files:
                entry = check_source(source, manifest["data_files"])
                rows = entry["rows"]
                self._action_mmap_files[str(source)] = {
                    name: check_array(entry[name], dtype, shape) for name, dtype, shape in (
                        ("state", np.dtype("float32"), (rows, 8)),
                        ("action", np.dtype("float32"), (rows, 7)),
                        ("task_index", np.dtype("int64"), (rows,)))}
            if int(row["_data_offset"]) + int(row["length"]) > entry_rows(self._action_mmap_files[str(source)]["action"]):
                raise ValueError("action mmap does not cover the episode")
            for key in VIDEO_KEYS:
                video = self.dataset_dir / self.video_path.format(
                    video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                    file_index=int(row[f"videos/{key}/file_index"]))
                if str(video) not in self._video_mmap_files:
                    entry = check_source(video, manifest["video_files"])
                    self._video_mmap_files[str(video)] = check_array(
                        entry["cache_file"], np.dtype("uint8"), (entry["rows"], 3, 256, 256))
                if int(row[f"_video_offset/{key}"]) + int(row["length"]) > entry_rows(self._video_mmap_files[str(video)]):
                    raise ValueError("RGB mmap does not cover the episode")
        logger.info("Action-only mmap enabled: cache=%s data_files=%d rgb_files=%d",
                    cache, len(self._action_mmap_files), len(self._video_mmap_files))

    def _rgb(self, row, indices):
        if self.action_only_mmap_cache_dir is None:
            return super()._rgb(row, indices)
        views = []
        for key in VIDEO_KEYS:
            source = self.dataset_dir / self.video_path.format(
                video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
                file_index=int(row[f"videos/{key}/file_index"]))
            base = int(row[f"_video_offset/{key}"])
            views.append(_read_action_mmap(self._video_mmap_files[str(source)])[base + indices])
        # stack copies the read-only maps into writable, sample-sized tensors.
        return torch.from_numpy(np.stack(views, axis=1))

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep_local = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        row = self.episodes.iloc[ep_local]
        k = self.first_anchor + int(index - self._cumulative[ep_local]) * self.window_stride
        data_path = self.dataset_dir / self.data_path.format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        start = int(row["_data_offset"]) + k
        count = min(self.chunk_size, int(row["length"]) - k) if self.pad_tail else self.chunk_size
        if self.action_only_mmap_cache_dir is None:
            table = _read_action_table(str(data_path))
            current = table.slice(start, 1).to_pandas().iloc[0]
            targets = table.slice(start, count).to_pandas()
            state = np.asarray(current["observation.state"], dtype=np.float32).copy()
            actions = np.stack(targets["action"].to_numpy()).astype(np.float32, copy=False)
            task_index = int(current["task_index"])
        else:
            files = self._action_mmap_files[str(data_path)]
            state = _read_action_mmap(files["state"])[start].copy()
            actions = _read_action_mmap(files["action"])[start:start + count].copy()
            task_index = int(_read_action_mmap(files["task_index"])[start])
        if task_index not in self.prompts:
            raise KeyError(f"missing prompt for task_index={task_index}")
        normalized = self.normalizer.normalize_action(torch.from_numpy(actions))
        if self.pad_tail:
            normalized = torch.nn.functional.pad(normalized, (0, 0, 0, self.chunk_size - count))
        observation = (self.current_dino_cache.observation(
                           row, k)
                       if self.current_dino_cache else
                       self._rgb_observation(row, np.array([k], dtype=np.int64)))
        sample = {
            **observation,
            "proprio": self.normalizer.normalize_proprio(torch.from_numpy(state)),
            "action": normalized,
            "prompt": self.prompts[task_index],
        }
        if self.pad_tail:
            sample["action_valid_mask"] = torch.arange(self.chunk_size) < count
        return sample


def entry_rows(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return array.shape[0]
