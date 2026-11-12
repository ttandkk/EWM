"""Piper pre-action RGB/action EWM reader; no event files are opened per sample."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .ewm_dataset import EWMDataset, _read_table


class PiperRGBActionDataset(EWMDataset):
    """Current dual RGB/state -> future selected RGB view(s) and aligned actions."""

    def __init__(self, dataset_dir, **kwargs):
        kwargs["gpu_voxelization"] = False
        # Ignore stale event-cache options: this mode never reads event targets.
        kwargs["event_frame_cache_dir"] = None
        super().__init__(dataset_dir, **kwargs)
        contract_path = Path(dataset_dir) / "meta" / "piper_temporal_contract.json"
        contract = json.loads(contract_path.read_text())
        if (contract.get("temporal_contract") != "aligned_event_frame_pre_action_v1" or
                self.observation_row_offset != 0):
            raise ValueError("Piper RGB-action requires aligned pre-action RGB/action rows")

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep_local = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        row = self.episodes.iloc[ep_local]
        k = self.first_anchor + int(index - self._cumulative[ep_local]) * self.window_stride
        padded = getattr(self, "pad_tail", False)
        future = np.arange(k + 1, k + self.future_horizon + 1, dtype=np.int64)
        if padded:
            length = int(row["length"])
            action_valid = np.arange(k, k + self.chunk_size) < length
            future_valid = future < length
            future = np.minimum(future, length - 1)
        video = self._rgb(row, np.concatenate(([k], future)))
        camera_indices = {"dual": (0, 1), "wrist_only": (1,), "global_only": (0,)}[self.event_camera_mode]
        selected = video[:, camera_indices].contiguous()
        data_path = self.dataset_dir / self.data_path.format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        start = int(row["_data_offset"]) + k
        table = _read_table(str(data_path))
        current = table.slice(start, 1).to_pandas().iloc[0]
        count = min(self.chunk_size, length - k) if padded else self.chunk_size
        targets = table.slice(start, count).to_pandas()
        task_index = int(current["task_index"])
        if task_index not in self.prompts:
            raise KeyError(f"missing prompt for task_index={task_index}")
        actions = np.zeros((self.chunk_size, 7), dtype=np.float32)
        actions[:count] = np.stack(targets["action"].to_numpy()).astype(np.float32, copy=False)
        normalized = self.normalizer.normalize_action(torch.from_numpy(actions))
        cache = getattr(self, "current_dino_cache", None)
        observation = cache.observation(row, k) if cache is not None else {"obs_video": video[0]}
        sample = {
            **observation,
            "obs_rgb": selected[0],
            "future_rgb": selected[1:],
            "proprio": self.normalizer.normalize_proprio(
                torch.from_numpy(np.asarray(current["observation.state"], dtype=np.float32).copy())),
            "action": normalized,
            "prompt": self.prompts[task_index],
        }
        if padded:
            sample["action_valid_mask"] = torch.from_numpy(action_valid)
            sample["future_valid_mask"] = torch.from_numpy(future_valid)
            sample["action"][~sample["action_valid_mask"]] = 0
        return sample
