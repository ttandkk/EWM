"""Persist scene-reset RNG state so stream-seeded trials can resume exactly."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def state_path(output: Path, trial: int) -> Path:
    return output / f"scene_rng_before_trial_{trial:03d}.pt"


def save_scene_rng(output: Path, trial: int) -> Path:
    """Atomically save all RNGs used by Piper's cube and joint reset terms."""
    path = state_path(output, trial)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, temporary)
    temporary.replace(path)
    return path


def restore_scene_rng(output: Path, trial: int) -> None:
    """Restore a trusted checkpoint written by this evaluator before reset."""
    path = state_path(output, trial)
    state = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
