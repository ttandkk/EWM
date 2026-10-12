"""Postprocess normalized EWM predictions for count-like event-frame display."""
from __future__ import annotations

import cv2
import numpy as np


def predicted_event_counts(
    voxel: np.ndarray,
    *,
    count_clip: int,
    occupancy_threshold: float = 1 / 30,
    min_neighbors: int = 2,
) -> np.ndarray:
    """Convert [time,camera,OFF/ON,H,W] VAE output to clipped uint8 counts.

    Thresholding precedes rounding, as in the pure-noise evaluator. A small
    3x3 support filter removes isolated decoder noise without blurring counts
    or mixing OFF and ON polarities. The filter affects display only.
    """
    voxel = np.asarray(voxel)
    if voxel.ndim != 5 or voxel.shape[2] != 2:
        raise ValueError("predicted events must be [time,camera,2,H,W]")
    if not np.isfinite(voxel).all():
        raise ValueError("predicted events contain nonfinite values")
    if not 1 <= count_clip <= 255 or not 0 < occupancy_threshold <= 1:
        raise ValueError("invalid count_clip or occupancy_threshold")
    if not 1 <= min_neighbors <= 9:
        raise ValueError("min_neighbors must be in [1,9]")
    counts = np.rint(np.clip(voxel, 0, 1) * count_clip).astype(np.uint8)
    active = voxel > occupancy_threshold
    counts = np.where(active, np.maximum(counts, 1), 0).astype(np.uint8)
    if min_neighbors > 1:
        kernel = np.ones((3, 3), dtype=np.uint8)
        for index in np.ndindex(counts.shape[:3]):
            support = cv2.filter2D(
                (counts[index] > 0).astype(np.uint8), cv2.CV_8U, kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            counts[index][support < min_neighbors] = 0
    return counts
