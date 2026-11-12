"""Raw timestamped observations and physical Piper action plans."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VERSION = "piper_policy_v1"
RGB_CONTRACT = "rgb_state_pre_action_v1"
RGB_LATENT_CONTRACT = "rgb_state_pre_action_rgb_latent_v1"
EVENT_CONTRACT = "aligned_event_frame_pre_action_v1"
ACTION_CONTRACT = "piper_delta_eef_gripper_v1"


@dataclass(frozen=True)
class Observation:
    video: np.ndarray
    proprio: np.ndarray
    voxel: np.ndarray | None
    prompt: str
    time_s: float
    step: int | None
    seed: int
    sampling_steps: int
    chunk_size: int
    timing: dict


def validate_request(request, arrays, descriptor):
    if request.get("raw", True) is not True or request.get("raw_observation", True) is not True:
        raise ValueError("policy services accept raw observations only")
    if request.get("observation_row_offset", 0) != 0:
        raise ValueError("current observations require observation_row_offset=0")
    if request.get("decode_future_events") or request.get("decode_future_rgb") or request.get("defer_visualization"):
        raise ValueError("use post-trial visualization; inline future decoding is unsupported")
    if request.get("step", descriptor["checkpoint_step"]) != descriptor["checkpoint_step"]:
        raise ValueError("checkpoint step differs; start a separate service for each checkpoint")
    events = descriptor["mode"] == "joint"
    key = "event_time_contract" if events else "observation_contract"
    expected = descriptor[key]
    if request.get(key) != expected:
        raise ValueError(f"{key} differs: expected {expected}")
    expected_keys = {"obs_video", "proprio"} | ({"obs_voxel"} if events else set())
    if set(arrays) != expected_keys:
        raise ValueError(f"observation arrays must be exactly {sorted(expected_keys)}")

    def array(name, shape, dtype):
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite {dtype} {shape}")
        return value

    video = array("obs_video", (2, 3, 256, 256), np.dtype("uint8"))
    proprio = array("proprio", (8,), np.dtype("float32"))
    voxel = None
    if events:
        voxel = array("obs_voxel", (descriptor["event_cameras"], 2 * descriptor["voxel_bins"], 256, 256), np.dtype("float32"))
        if np.any(voxel < 0):
            raise ValueError("raw event counts must be nonnegative")
    timing = request.get("timing", {})
    names = ["obs_video_time_s", "proprio_time_s"]
    if events:
        names.extend(("obs_event_start_s", "obs_event_end_s"))
    if any(name not in timing for name in names):
        raise ValueError("observation requires explicit RGB/state/event timestamps")
    times = [float(timing[name]) for name in names]
    if not all(math.isfinite(t) for t in times) or abs(times[0] - times[1]) > .001:
        raise ValueError("observation timestamps must be finite and aligned")
    if events and (abs(times[3] - times[2] - 1 / descriptor["control_hz"]) > .001 or abs(times[3] - times[0]) > .001):
        raise ValueError("event interval must end at RGB/state time and span one control step")
    time_s = times[0]
    if "observation_time_s" in request:
        declared_time = float(request["observation_time_s"])
        if not math.isfinite(declared_time) or abs(declared_time - time_s) > .001:
            raise ValueError("observation_time_s differs from timing")
    step = request.get("observation_step")
    if step is not None and (type(step) is not int or step < 0 or abs(step / descriptor["control_hz"] - time_s) > .001):
        raise ValueError("observation_step differs from timestamp/control_hz")
    if float(request.get("control_hz", descriptor["control_hz"])) != descriptor["control_hz"]:
        raise ValueError("control_hz differs from trained policy contract")
    seed = request.get("seed", 0)
    sampling_steps = request.get("sampling_steps", 20)
    chunk = request.get("action_chunk_size", descriptor["action_horizon"])
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if type(sampling_steps) is not int or sampling_steps <= 0:
        raise ValueError("sampling_steps must be a positive integer")
    if type(chunk) is not int or not 1 <= chunk <= descriptor["action_horizon"]:
        raise ValueError("action_chunk_size exceeds the trained horizon")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty instruction")
    if request.get("record_prediction") is not None and not descriptor["post_trial_visualization_supported"]:
        raise ValueError("this policy does not support prediction recording")
    return Observation(video, proprio, voxel, prompt, time_s, step, seed, sampling_steps, chunk, dict(timing))


def action_reply(actions, observation, descriptor, inference_seconds, **metadata):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < observation.chunk_size or not np.isfinite(actions).all():
        raise ValueError("model must return a finite physical [horizon,7] action plan")
    if not math.isfinite(inference_seconds) or inference_seconds < 0:
        raise ValueError("invalid inference duration")
    result = np.ascontiguousarray(actions[:observation.chunk_size])
    arrays = {"executed_actions": result}
    if observation.chunk_size == 1:
        arrays["executed_action"] = result[0]
    return dict(ok=True, step=descriptor["checkpoint_step"], seed=observation.seed,
                sampling_steps=observation.sampling_steps, action_chunk_size=len(result),
                inference_seconds=inference_seconds, observation_time_s=observation.time_s,
                observation_step=observation.step, control_hz=descriptor["control_hz"],
                action_contract=ACTION_CONTRACT, event_horizon=descriptor["event_horizon"],
                future_decoded=False, visualization_pending=False, **metadata), arrays
