"""Strict RGB/state-only live policy contract, independent of the event server."""
from __future__ import annotations

import math

import numpy as np
import torch

OBSERVATION_CONTRACT = "rgb_state_pre_action_v1"


def validate_timing(timing: dict) -> None:
    """Current RGB/state at row k precede the first executed command action[k]."""
    if not isinstance(timing, dict):
        raise ValueError("observation timing must be a dictionary")
    try:
        video_t = float(timing["obs_video_time_s"])
        state_t = float(timing["proprio_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("RGB/state observation needs explicit finite timestamps") from exc
    if not all(map(math.isfinite, (video_t, state_t))):
        raise ValueError("RGB/state timestamps must be finite")
    if abs(video_t - state_t) > 0.001:
        raise ValueError("RGB and proprioception timestamps differ")


def prepare_request(request: dict, arrays: dict, normalizer, *, horizon: int,
                    device="cuda") -> tuple[dict, int, int, int]:
    """Validate before sampling; reject event arrays rather than silently ignore them."""
    if request.get("observation_contract") != OBSERVATION_CONTRACT:
        raise ValueError("missing or incompatible action-only observation contract")
    if request.get("observation_row_offset") != 0:
        raise ValueError("action-only inference requires same-row RGB/state")
    if request.get("raw_observation") is not True:
        raise ValueError("action-only simulation expects raw RGB/state observations")
    if request.get("decode_future_events", False) is not False:
        raise ValueError("action-only policies do not predict events")
    expected = {"obs_video", "proprio"}
    if set(arrays) != expected:
        raise ValueError(f"action-only input accepts exactly {sorted(expected)}; no events")
    validate_timing(request.get("timing", {}))
    values = {}
    fields = [("obs_video", (2, 3, 256, 256), "uint8"), ("proprio", (8,), "float32")]
    for name, shape, dtype in fields:
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.name != dtype:
            raise ValueError(f"{name} must be {dtype} {shape}")
        if dtype == "float32" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
        values[name] = torch.from_numpy(value).to(device)
    seed, steps = int(request["seed"]), int(request["sampling_steps"])
    chunk = int(request.get("action_chunk_size", 1))
    if steps <= 0 or not 0 <= seed < 2**63 or not 1 <= chunk <= horizon:
        raise ValueError("invalid seed, sampling_steps or action_chunk_size")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    batch = {"obs_video": values["obs_video"][None],
             "proprio": normalizer.normalize_proprio(values["proprio"])[None],
             "prompt": [prompt]}
    return batch, seed, steps, chunk
