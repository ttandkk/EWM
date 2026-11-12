"""Shared EWM normalization for training, deployment, and visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

NEGATIVE_BINS = slice(0, 4)
POSITIVE_BINS = slice(4, 8)
NORM_EPS = 1e-6


class EWMNormalizer:
    """Normalize EWM event, state, and action tensors from persisted train stats."""

    def __init__(self, stats_path: str | Path) -> None:
        self.stats_path = Path(stats_path)
        payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"unsupported EWM stats version in {self.stats_path}")
        voxel = payload.get("voxel", {})
        self.count_clip = float(voxel["count_clip"])
        self.voxel_bins = int(voxel["num_bins"])
        if self.count_clip <= 0 or self.voxel_bins <= 0:
            raise ValueError("EWM voxel stats require count_clip > 0 and a positive temporal-bin count")

        proprio = payload["proprio"]
        action = payload["action"]
        self.state_mean = self._vector(proprio, "mean", 8)
        self.state_std = self._vector(proprio, "std", 8)
        self.action_type = action.get("type", "quantile")
        if self.action_type == "quantile":
            # Preserve the exact mapping used by existing checkpoints.
            self.action_low = self._vector(action, "q01", 7)
            self.action_high = self._vector(action, "q99", 7)
        elif self.action_type == "minmax":
            self.action_low = self._vector(action, "min", 7)
            self.action_high = self._vector(action, "max", 7)
        else:
            raise ValueError(f"unsupported EWM action normalization: {self.action_type}")
        if np.any(self.action_low > self.action_high):
            raise ValueError("action normalization lower bound exceeds upper bound")
        # Quantiles remain useful diagnostics; they are not the minmax bounds.
        self.action_q01 = self._vector(action, "q01", 7) if "q01" in action else self.action_low
        self.action_q99 = self._vector(action, "q99", 7) if "q99" in action else self.action_high
        self.action_q50 = (self._vector(action, "q50", 7) if "q50" in action
                           else (self.action_low + self.action_high) * 0.5)
        self.degenerate_threshold = float(action.get("degenerate_threshold", NORM_EPS))
        self._degenerate_action = (self.action_high - self.action_low) < self.degenerate_threshold
        self.metadata = payload

    @staticmethod
    def _vector(group: dict[str, Any], key: str, dim: int) -> np.ndarray:
        value = np.asarray(group[key], dtype=np.float32)
        if value.shape != (dim,):
            raise ValueError(f"normalization {key} must have shape ({dim},), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"normalization {key} must be finite")
        return value

    @classmethod
    def from_dataset_dir(
        cls,
        dataset_dir: str | Path,
        *,
        stats_path: str | Path | None = None,
        expected_count_clip: int | None = None,
    ) -> "EWMNormalizer":
        resolved = Path(stats_path) if stats_path is not None else Path(dataset_dir) / "meta" / "ewm_normalization_stats.json"
        normalizer = cls(resolved)
        if expected_count_clip is not None and normalizer.count_clip != float(expected_count_clip):
            raise ValueError(
                f"count_clip mismatch: loader={expected_count_clip}, stats={normalizer.count_clip}"
            )
        return normalizer

    @staticmethod
    def _require_tensor(value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"EWMNormalizer expects torch.Tensor, got {type(value).__name__}")
        return value

    @staticmethod
    def _stats_on(vector: np.ndarray, value: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(vector, dtype=value.dtype, device=value.device)

    @property
    def degenerate_action_mask(self) -> np.ndarray:
        return self._degenerate_action.copy()

    def normalize_voxel(self, voxel: torch.Tensor) -> torch.Tensor:
        voxel = self._require_tensor(voxel)
        return voxel.to(torch.float32).clamp(0.0, self.count_clip) / self.count_clip

    def denormalize_voxel(
        self, voxel_normalized: torch.Tensor, *, round_counts: bool = False
    ) -> torch.Tensor:
        """Map normalized voxel values to clipped count space [0, count_clip].

        The default retains floating values so a predicted event density can be
        visualized continuously. Set round_counts=True for discrete-count views.
        """
        voxel_normalized = self._require_tensor(voxel_normalized)
        counts = voxel_normalized.to(torch.float32).clamp(0.0, 1.0) * self.count_clip
        return counts.round() if round_counts else counts

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        proprio = self._require_tensor(proprio).to(torch.float32)
        mean = self._stats_on(self.state_mean, proprio)
        std = self._stats_on(self.state_std, proprio).clamp_min(NORM_EPS)
        return (proprio - mean) / std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action = self._require_tensor(action).to(torch.float32)
        low = self._stats_on(self.action_low, action)
        high = self._stats_on(self.action_high, action)
        scale = high - low
        normalized = (2.0 * (action - low) / scale.clamp_min(NORM_EPS) - 1.0).clamp(-1.0, 1.0)
        mask = self._stats_on(self._degenerate_action, action).to(torch.bool)
        return torch.where(mask, torch.zeros_like(normalized), normalized)

    def denormalize_action(self, action_normalized: torch.Tensor) -> torch.Tensor:
        """Map policy output in [-1, 1] to raw 7-D Piper action coordinates."""
        action_normalized = self._require_tensor(action_normalized).to(torch.float32)
        low = self._stats_on(self.action_low, action_normalized)
        q50 = self._stats_on(self.action_q50, action_normalized)
        high = self._stats_on(self.action_high, action_normalized)
        raw = (action_normalized.clamp(-1.0, 1.0) + 1.0) * 0.5 * (high - low) + low
        mask = self._stats_on(self._degenerate_action, action_normalized).to(torch.bool)
        return torch.where(mask, q50, raw)


__all__ = ["EWMNormalizer", "NEGATIVE_BINS", "POSITIVE_BINS", "NORM_EPS"]
