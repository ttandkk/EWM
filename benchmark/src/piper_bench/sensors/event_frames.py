"""Lossless one-interval EVIS event frames, aligned to 30 Hz observations."""

from __future__ import annotations

import numpy as np

EVENT_FRAME_KEY = "observation.event_frame"
COUNT_CLIP = 15


def events_to_frame(chunks, start_t: float, end_t: float, height: int = 256, width: int = 256) -> np.ndarray:
    """Return HWC RGB PNG payload: OFF, ON, zero over (start_t, end_t]."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if not chunks:
        return frame
    flat_parts = []
    for x, y, t, p in chunks:
        x, y, t, p = (np.asarray(v) for v in (x, y, t, p))
        mask = (t > start_t) & (t <= end_t)
        if not np.any(mask):
            continue
        xx, yy, pp = x[mask], y[mask], p[mask]
        if np.any(xx >= width) or np.any(yy >= height) or np.any(xx < 0) or np.any(yy < 0):
            raise ValueError("EVIS event coordinates exceed event-frame resolution")
        if not np.all((pp == -1) | (pp == 1)):
            raise ValueError("EVIS polarity must be -1/+1")
        flat_parts.append(((yy.astype(np.int64) * width + xx.astype(np.int64)) * 3 + (pp > 0)).astype(np.int64))
    if flat_parts:
        counts = np.bincount(np.concatenate(flat_parts), minlength=height * width * 3)
        frame[:] = np.minimum(counts.reshape(height, width, 3), COUNT_CLIP).astype(np.uint8)
    return frame
