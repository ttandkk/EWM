"""Validate recorded EVIS timing, polarity and reset behavior."""
from __future__ import annotations

import json
from pathlib import Path


def validate_recording(directory: Path) -> dict:
    import numpy as np

    metadata = json.loads((directory / "events.json").read_text())
    checks = {}
    with np.load(directory / "event_frames.npz", allow_pickle=False) as data:
        times = data["time_s"]
        checks["timestamps"] = bool(times.ndim == 1 and len(times) > 1 and np.all(np.isfinite(times))
                                     and np.allclose(np.diff(times), 1 / 30, rtol=0, atol=1e-7))
        for camera in ("cam0", "cam_wrist"):
            frames, initial = data[camera], data["initial_" + camera]
            checks[camera + "_shape"] = frames.shape == (len(times), 256, 256, 2)
            checks[camera + "_reset_empty"] = bool(not np.any(initial))
            checks[camera + "_finite_nonnegative"] = bool(np.all(np.isfinite(frames)) and np.all(frames >= 0))
            checks[camera + "_both_polarities"] = bool(np.any(frames[..., 0]) and np.any(frames[..., 1]))
    checks["video_decoded"] = metadata.get("video_decoded") is True
    return {"passed": all(checks.values()), "checks": checks, "frames": len(times),
            "evidence": [str(directory / "events.json"), str(directory / "event_frames.npz")]}
