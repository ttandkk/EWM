"""Vectorized sparse-event histogramming for EWM GPU training."""

from __future__ import annotations

import torch


def materialize_sparse_event_voxels(batch, *, num_cameras: int, count_clip: float, height=256, width=256):
    """Replace ``event_sparse`` with normalized one-bin dense voxel tensors.

    ``frame_id`` enumerates sample-major ``(current, future...) x camera``
    frames. One global bincount replaces per-window CPU ``searchsorted`` and
    dense histogram allocation.
    """
    sparse = batch.pop("event_sparse", None)
    if sparse is None:
        return batch
    if num_cameras <= 0 or count_clip <= 0 or min(height, width) <= 0:
        raise ValueError("invalid GPU voxelization geometry")
    action = batch["action"]
    if action.ndim != 3:
        raise ValueError("action must be [B,K,7] before GPU voxelization")
    batch_size, horizon = action.shape[:2]
    frame_count = horizon + 1
    expected_frames = batch_size * frame_count * num_cameras
    frame_id, x, y, polarity = (sparse[key].to(torch.long) for key in ("frame_id", "x", "y", "polarity"))
    if not (frame_id.numel() == x.numel() == y.numel() == polarity.numel()):
        raise ValueError("sparse event fields must have equal length")
    if x.numel() and (x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height):
        raise ValueError("sparse event coordinates exceed voxel resolution")
    if polarity.numel() and (polarity.min() < 0 or polarity.max() > 1):
        raise ValueError("sparse event polarity must be zero or one")
    flat = (((frame_id * 2 + polarity) * height + y) * width + x)
    counts = torch.bincount(flat, minlength=expected_frames * 2 * height * width)
    dense = counts.reshape(batch_size, frame_count, num_cameras, 2, height, width)
    dense = dense.clamp_max_(int(count_clip)).to(torch.float32).div_(count_clip)
    batch["obs_voxel"] = dense[:, 0]
    batch["future_voxel"] = dense[:, 1:]
    return batch


__all__ = ["materialize_sparse_event_voxels"]
