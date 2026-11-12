"""Raw-observation policy transport and action-plan validation."""
import math
import numpy as np
from .constants import CAMERAS, CONTROL_HZ, EVENT_TIME_CONTRACT, OBSERVATION_CONTRACT, RGB_ACTION_CONTRACT
from .wire import MAX_MESSAGE, encode_message, decode_message
from .event_visualization import predicted_event_counts

def request(conn, header, arrays=None):
    conn.send(encode_message(header, arrays))
    reply, values = decode_message(conn.recv(timeout=600))
    if not reply.get("ok"):
        raise RuntimeError(f"EWM server: {reply.get('error', reply)}")
    return reply, values


def policy_request(conn, args, header, arrays):
    header = dict(header)
    descriptor = getattr(args, "_policy_descriptor", None)
    if descriptor:
        timestamp = float(header["timing"]["obs_video_time_s"])
        header.update(observation_time_s=timestamp, observation_step=round(timestamp * CONTROL_HZ),
                      control_hz=CONTROL_HZ, defer_visualization=False)
    session = getattr(args, "_prediction_session", None)
    if session:
        header["record_prediction"] = dict(session=session,
            observation_step=round(header["timing"]["obs_video_time_s"] * CONTROL_HZ))
        header["decode_future_events"] = False
        header["defer_visualization"] = False
    if args.policy_mode in ("action_only", "rgb_action"):
        header = dict(header)
        header.pop("event_time_contract", None)
        header.pop("defer_visualization", None)
        header["observation_contract"] = (RGB_ACTION_CONTRACT if args.policy_mode == "rgb_action" else OBSERVATION_CONTRACT)
        if args.policy_mode == "rgb_action":
            header["decode_future_rgb"] = False
        header["decode_future_events"] = False
        header["timing"] = {name: header["timing"][name]
                            for name in ("obs_video_time_s", "proprio_time_s")
                            if name in header['timing']}
        arrays = {name: arrays[name] for name in ("obs_video", "proprio") if name in arrays}
    reply, values = request(conn, header, arrays)
    if descriptor:
        if (reply.get("action_contract") != "piper_delta_eef_gripper_v1"
                or reply.get("control_hz") != CONTROL_HZ
                or reply.get("observation_step") != header["observation_step"]
                or not math.isclose(float(reply.get("observation_time_s", math.nan)),
                                    header["observation_time_s"], rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Policy action plan is not anchored to the submitted observation")
    if session and reply.get("prediction_recorded") is not True:
        raise RuntimeError("server did not record prediction latents")
    return reply, values


def validate_hello(hello, args):
    if "policy_descriptor" in hello:
        return validate_descriptor(hello, args)
    if args.policy_mode in ("action_only", "rgb_action"):
        rgb_action = args.policy_mode == "rgb_action"
        expected_inputs = (['obs_video', 'obs_rgb', 'proprio', 'prompt'] if rgb_action
                           else ['obs_video', 'proprio', 'prompt'])
        formats = ("ewm_rgb_action_latent_v1",) if rgb_action else ("ewm_action_only_v1", "ewm_action_only_v2")
        if (hello.get("training_mode") != args.policy_mode
                or hello.get("checkpoint_format") not in formats
                or hello.get("observation_contract") != (RGB_ACTION_CONTRACT if rgb_action else OBSERVATION_CONTRACT)
                or hello.get("inference_observation_row_offset") != 0
                or hello.get("training_observation_row_offset") != 0
                or hello.get("input_keys") != expected_inputs):
            raise ValueError("Server RGB/state-only protocol differs from the training contract")
        mode = "dual"  # Observed event panels only; no events are sent to the policy.
    else:
        if hello.get("event_time_contract") != EVENT_TIME_CONTRACT or hello.get("inference_observation_row_offset") != 0:
            raise ValueError("Server event/time protocol differs from the training export")
        mode = hello.get("event_camera_mode")
        if (mode, hello.get("event_cameras")) not in (("dual", 2), ("wrist_only", 1)):
            raise ValueError("Unsupported event camera mode/count")
        if hello.get("voxel_bins") != 1 or hello.get("count_clip") != 15:
            raise ValueError("Conveyor event frames require one bin and count_clip=15")
        event_horizon = hello.get('event_horizon', hello['horizon'])
        if type(event_horizon) is not int or event_horizon <= 0:
            raise ValueError('invalid server event_horizon')
        args.event_prediction_horizon = event_horizon
    checkpoints = hello.get("checkpoints", [])
    step = args.checkpoint_step if args.checkpoint_step is not None else max(checkpoints)
    if step not in checkpoints:
        raise ValueError(f"Checkpoint step {step} unavailable; available: {checkpoints}")
    if args.latency_controller != "continuous" and args.execution_horizon > hello["horizon"]:
        raise ValueError("Execution horizon exceeds the trained action horizon")
    if (args.latency_controller == "continuous" and
            getattr(args, "min_execution_steps", 4) > hello["horizon"]):
        raise ValueError("Minimum continuous execution exceeds the trained action horizon")
    if args.latency_aware:
        if args.policy_mode == "joint" and not hello.get("deferred_visualization_supported"):
            raise ValueError("Latency evaluation requires action replies before deferred visualization")
        if args.latency_controller == "buffered" and args.execution_horizon >= hello["horizon"]:
            raise ValueError("Buffered control needs a forecast tail beyond execution-horizon")
    return step, CAMERAS if mode == "dual" else ("cam_wrist",)


def validate_reply(reply, values, horizon, event_cameras, args, count_clip):
    inference_seconds = float(reply.get("inference_seconds", math.nan))
    if not math.isfinite(inference_seconds) or inference_seconds < 0:
        raise ValueError("Server returned invalid inference timing")
    actions = values.get("executed_actions")
    if (actions is None or actions.shape != (horizon, 7) or actions.dtype != np.float32
            or not np.isfinite(actions).all()):
        raise ValueError("Server returned invalid physical action chunk")
    predicted = None
    if args.decode_predicted_events and not getattr(args, "_prediction_session", None):
        voxel = values.get("future_voxel")
        configured = getattr(args, 'event_prediction_horizon', None)
        event_horizon = reply.get('event_horizon', horizon if configured is None else configured)
        if (type(event_horizon) is not int or event_horizon <= 0 or
                (configured is not None and event_horizon != configured)):
            raise ValueError('reply event horizon differs from the server contract')
        expected = (min(horizon, event_horizon), len(event_cameras), 2, 256, 256)
        if reply.get("future_decoded") is not True or voxel is None or voxel.shape != expected:
            raise ValueError(f"Server did not return decoded future event frames {expected}")
        predicted = predicted_event_counts(voxel, count_clip=count_clip,
                                           occupancy_threshold=args.predicted_event_threshold,
                                           min_neighbors=args.predicted_event_min_neighbors)
    return actions, predicted, inference_seconds


def validate_descriptor(hello, args):
    """Validate the simulator-facing contract without opening model files."""
    descriptor = hello["policy_descriptor"]
    if descriptor.get("version") != "piper_policy_v1":
        raise ValueError("Unsupported policy descriptor version")
    expected = dict(action_dim=7, state_dim=8, control_hz=30, rgb_shape=[2, 3, 256, 256],
                    action_contract="piper_delta_eef_gripper_v1",
                    training_observation_row_offset=0, inference_observation_row_offset=0)
    if any(descriptor.get(key) != value for key, value in expected.items()):
        raise ValueError("Policy observation/action geometry or timing contract differs")
    mode = descriptor.get("mode")
    if mode not in ("joint", "rgb_action", "action_only"):
        raise ValueError("Unsupported policy input mode")
    horizon = descriptor.get("action_horizon")
    if type(horizon) is not int or horizon <= 0 or hello.get("horizon") != horizon:
        raise ValueError("Invalid policy action horizon")
    cameras = CAMERAS
    if mode == "joint":
        if (descriptor.get("event_time_contract") != EVENT_TIME_CONTRACT
                or descriptor.get("voxel_bins") != 1 or descriptor.get("count_clip") != 15):
            raise ValueError("Policy event representation differs from one-bin/count-clip-15 observations")
        selection = descriptor.get("event_camera_mode")
        options = {"dual": CAMERAS, "wrist_only": ("cam_wrist",)}
        if selection not in options or descriptor.get("event_cameras") != len(options[selection]):
            raise ValueError("Unsupported event camera contract")
        cameras = options[selection]
        event_horizon = descriptor.get("event_horizon")
        if type(event_horizon) is not int or event_horizon <= 0:
            raise ValueError("Invalid event prediction horizon")
        args.event_prediction_horizon = event_horizon
    elif descriptor.get("observation_contract") != (RGB_ACTION_CONTRACT if mode == "rgb_action" else OBSERVATION_CONTRACT):
        raise ValueError("Policy RGB/state contract differs")
    requested = args.min_execution_steps if args.latency_controller == "continuous" else args.execution_horizon
    if requested > horizon or (args.latency_controller == "buffered" and requested >= horizon):
        raise ValueError("Execution profile exceeds the policy action horizon")
    step = descriptor.get("checkpoint_step")
    if type(step) is not int or step < 0 or (args.checkpoint_step is not None and args.checkpoint_step != step):
        raise ValueError("Requested checkpoint differs from the policy service")
    for key in ("checkpoint_sha256", "config_sha256", "normalization_sha256"):
        value = descriptor.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Policy descriptor requires a SHA256 {key}")
    supported = descriptor.get("post_trial_visualization_supported") is True
    if args.record_predictions is True and not supported:
        raise ValueError("This policy does not support prediction recording; use --no-record-predictions")
    if args.decode_predicted_events:
        raise ValueError("Versioned policy services use post-trial prediction decoding")
    if args.record_predictions is None:
        args.record_predictions = supported
    args.policy_mode = mode
    args._policy_descriptor = descriptor
    return step, cameras
