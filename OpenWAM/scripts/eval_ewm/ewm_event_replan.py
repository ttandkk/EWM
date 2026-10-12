"""CPU-only occupancy comparison and execution statistics for adaptive replanning."""
from collections import Counter

import numpy as np


def event_error(observed, predicted, *, min_pixels=4):
    """Polarity-preserving tolerant F1 on 64x64 max-pooled occupancy.

    Inputs are [camera, OFF/ON, 256, 256] counts. A one-cell Chebyshev
    tolerance absorbs small codec spatial errors; sparse noise is ignored.
    """
    def pool(value):
        value = np.asarray(value)
        if value.ndim != 4 or value.shape[1:] != (2, 256, 256):
            raise ValueError("events must have shape [camera,2,256,256]")
        if not np.isfinite(value).all():
            raise ValueError("nonfinite events")
        return (value > 0).reshape(value.shape[0], 2, 64, 4, 64, 4).any(axis=(3, 5))

    def dilate(mask):
        padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)))
        return np.logical_or.reduce([padded[:, y:y+64, x:x+64]
                                     for y in range(3) for x in range(3)])

    real, pred = pool(observed), pool(predicted)
    if real.shape != pred.shape:
        raise ValueError("camera counts differ")
    scores = []
    for r, p in zip(real, pred):
        nr, npred = int(r.sum()), int(p.sum())
        if max(nr, npred) < min_pixels or nr + npred == 0:
            scores.append(0.0)
        elif not nr or not npred:
            scores.append(1.0)
        else:
            precision = float((p & dilate(r)).sum()) / npred
            recall = float((r & dilate(p)).sum()) / nr
            scores.append(1.0 - 2 * precision * recall / (precision + recall)
                          if precision + recall else 1.0)
    return float(np.mean(scores)), scores


class EventReplanGate:
    def __init__(self, threshold=0.6, patience=2, min_horizon=4):
        self.threshold, self.patience, self.min_horizon = threshold, patience, min_horizon
        self.streak = 0

    def update(self, score, executed):
        self.streak = self.streak + 1 if score > self.threshold else 0
        return executed >= self.min_horizon and self.streak >= self.patience


def horizon_statistics(chunks):
    def stats(items):
        lengths = [c["executed_horizon"] for c in items]
        return {"count": len(lengths), "mean": sum(lengths) / len(lengths) if lengths else None,
                "min": min(lengths) if lengths else None, "max": max(lengths) if lengths else None}
    return {"all_chunks": stats(chunks),
            "replan_chunks": stats([c for c in chunks if c["end_reason"] in
                                    ("event_mismatch", "max_horizon")]),
            "end_reasons": dict(Counter(c["end_reason"] for c in chunks))}
