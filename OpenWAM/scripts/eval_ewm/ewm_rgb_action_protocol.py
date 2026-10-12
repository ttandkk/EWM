"""RGB-action wire contract; no event input or adaptive replan."""
from __future__ import annotations

import torch

from scripts.eval_ewm.ewm_action_only_protocol import (
    OBSERVATION_CONTRACT as ACTION_CONTRACT,
    prepare_request as prepare_action_request,
)

OBSERVATION_CONTRACT = "rgb_state_pre_action_rgb_latent_v1"


def prepare_request(request, arrays, normalizer, *, horizon, device="cuda"):
    if request.get("observation_contract") != OBSERVATION_CONTRACT:
        raise ValueError("requires the RGB-action observation contract")
    if request.get("decode_future_rgb") is not True:
        raise ValueError("this evaluator requires RGB video decoding")
    header = dict(request, observation_contract=ACTION_CONTRACT)
    batch, seed, steps, chunk = prepare_action_request(
        header, arrays, normalizer, horizon=horizon, device=device)
    # Both RGB views are current conditions; no future or event arrays accepted.
    batch["obs_rgb"] = batch["obs_video"]
    return batch, seed, steps, chunk


def prediction_array(result, horizon, camera_count):
    rgb = result.get("future_rgb")
    if (rgb is None or rgb.shape != (1, horizon, camera_count, 3, 256, 256)
            or not torch.isfinite(rgb).all()):
        raise ValueError("invalid decoded future RGB")
    # Byte conversion is for saved videos only; native sampling remains float32.
    return (rgb[0].float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
