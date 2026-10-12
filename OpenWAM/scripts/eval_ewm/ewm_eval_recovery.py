"""Small, Isaac-independent helpers for resumable EWM simulator evaluation."""
from __future__ import annotations

import json
from pathlib import Path


def load_completed_results(output: Path, requested_trials: int) -> list[dict]:
    """Load only a contiguous prefix whose video and step log are present."""
    results = []
    while len(results) < requested_trials:
        index = len(results)
        path = output / f"trial_{index:03d}_result.json"
        if not path.is_file():
            break
        result = json.loads(path.read_text(encoding="utf-8"))
        video = Path(result.get("video", ""))
        steps = output / f"trial_{index:03d}_steps.json"
        if result.get("trial") != index or not video.is_file() or not steps.is_file():
            raise ValueError(f"trial {index} has incomplete or inconsistent saved results")
        if video.resolve().parent != output.resolve():
            raise ValueError(f"trial {index} video is outside its output directory")
        results.append(result)
    unexpected = list(output.glob("trial_*_result.json"))
    if len(unexpected) != len(results):
        raise ValueError("saved trial results are not a contiguous prefix")
    return results


def archive_interrupted_trial(output: Path, trial: int) -> list[Path]:
    """Move an incomplete trial's artifacts aside before replaying its seed."""
    prefix = f"trial_{trial:03d}_"
    paths = sorted(
        path for path in output.glob(prefix + "*")
        if path.is_file() and "_interrupted_" not in path.name
    )
    if not paths:
        return []
    attempt = 1
    while any((output / f"{path.stem}_interrupted_{attempt:02d}{path.suffix}").exists()
              for path in paths):
        attempt += 1
    moved = []
    for path in paths:
        target = output / f"{path.stem}_interrupted_{attempt:02d}{path.suffix}"
        path.rename(target)
        moved.append(target)
    return moved
