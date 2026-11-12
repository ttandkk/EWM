"""Shared rollout timing and observation handling for all Piper tasks."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from .constants import *
from .controllers import LatencyAwareActionStream, BufferedLatencyActionStream, ContinuousLatencyActionStream
from .policy_client import request, policy_request, validate_hello, validate_reply
from .wire import MAX_MESSAGE, decode_message, encode_message
from .recording import TrialVideo, preview
from .recovery import archive_interrupted_trial, load_completed_results
from .reporting import write_summary, horizon_statistics
from .prediction_recording import begin_recording, end_recording, finish_recording, recording_enabled, recording_metadata

def simulation_clock(args):
    return LATENCY_CLOCK if args.latency_aware else PAUSED_CLOCK


def latency_source(args):
    return "action_reply_round_trip_seconds"


def timing_stats(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(calls=len(values), total_seconds=float(array.sum()),
                mean_seconds=float(array.mean()), p50_seconds=float(np.percentile(array, 50)),
                p95_seconds=float(np.percentile(array, 95)))

def run_latency_loop(args, runtime, dvs, conn, hello, checkpoint_step, event_cameras,
                      obs, video, rows, chunks, step_timings, heartbeat, output, trial,
                      *, capture=None, prompt="", stable_steps=None):
    import torch

    # Other Piper scenes share the action scheduler, but own physics and success.
    if capture is None:
        raise ValueError("A task capture callback is required")
    stable_steps = (lambda runtime: runtime.tracker.steps) if stable_steps is None else stable_steps

    def predict(copied, step, horizon):
        started = time.perf_counter()
        reply, values = policy_request(conn, args, dict(
            op="predict", step=checkpoint_step, seed=args.seed + trial * args.max_steps + step,
            sampling_steps=args.sampling_steps, action_chunk_size=horizon,
            prompt=prompt, event_time_contract=EVENT_TIME_CONTRACT,
            observation_row_offset=0, raw_observation=True, timing=copied["timing"],
            decode_future_events=args.decode_predicted_events,
            defer_visualization=True,
        ), {key: copied[key] for key in ("obs_video", "obs_voxel", "proprio")})
        action_options = argparse.Namespace(**vars(args))
        action_options.decode_predicted_events = False
        actions, predicted, inference_seconds = validate_reply(
            reply, values, horizon, event_cameras, action_options, int(hello.get("count_clip", 15)))
        round_trip_seconds = time.perf_counter() - started
        visualization_started = time.perf_counter()
        if args.decode_predicted_events and not getattr(args, "_prediction_session", None):
            if reply.get("visualization_pending") is not True:
                raise RuntimeError("Server must defer visualization until after the action reply")
            visual_reply, visual_values = decode_message(conn.recv(timeout=600))
            if (visual_reply.get("ok") is not True or visual_reply.get("op") != "predict_visualization"
                    or visual_reply.get("step") != checkpoint_step
                    or visual_reply.get("seed") != args.seed + trial * args.max_steps + step):
                raise RuntimeError(f"Invalid deferred visualization reply: {visual_reply}")
            visual_values["executed_actions"] = actions
            visual_reply["inference_seconds"] = inference_seconds
            _, predicted, _ = validate_reply(visual_reply, visual_values, horizon, event_cameras, args,
                                            int(hello.get("count_clip", 15)))
        visualization_seconds = time.perf_counter() - visualization_started
        # Action delivery and control-relevant decoding have separate clocks.
        # Actions can run while the forecast decoder is still logically pending.
        action_reply_seconds = round_trip_seconds
        event_ready_seconds = time.perf_counter() - started
        return dict(actions=actions, predicted=predicted, inference_seconds=inference_seconds,
                    round_trip_seconds=round_trip_seconds, visualization_seconds=visualization_seconds,
                    action_reply_round_trip_seconds=action_reply_seconds,
                    event_forecast_ready_round_trip_seconds=event_ready_seconds,
                    charge_event_forecast_latency=(args.latency_controller == "continuous" and args.decode_predicted_events
                                                   and not getattr(args, "_prediction_session", None)))

    if args.latency_controller == "continuous":
        stream = ContinuousLatencyActionStream(predict, plan_horizon=hello["horizon"], max_steps=args.max_steps,
                                               min_execution_steps=getattr(args, "min_execution_steps", 4))
    elif args.latency_controller == "buffered":
        stream = BufferedLatencyActionStream(predict, horizon=args.execution_horizon,
                                             plan_horizon=hello["horizon"], max_steps=args.max_steps)
    else:
        stream = LatencyAwareActionStream(predict, horizon=args.execution_horizon, max_steps=args.max_steps)
    stream.gripper = float(getattr(runtime, "initial_gripper_action", 1.))
    trial_started = time.perf_counter()
    success, reason, step = False, None, 0
    try:
        while step < args.max_steps:
            stream.ensure_request(obs, step)
            decision = stream.decision(step)
            # A zero-latency request created on this tick can replace a plan
            # without being predictable on the previous tick.
            if (args.latency_controller == "continuous" and rows
                    and decision["action_source"] == "policy"
                    and rows[-1]["action_source"] == "policy"
                    and rows[-1]["replan"] != decision["replan"]
                    and rows[-1]["chunk_end_reason"] is None):
                previous = rows[-1]["replan"]
                rows[-1]["chunk_end_reason"] = "new_prediction"
                chunks.append(dict(replan=previous,
                                   executed_horizon=stream.requests[previous]["executed_actions"],
                                   end_reason="new_prediction"))
            started = time.perf_counter()
            command = torch.from_numpy(decision["action"].copy())[None].to(runtime.env.device)
            obs, success, reason = capture(runtime, dvs, command, event_cameras)
            capture_seconds = time.perf_counter() - started
            age_steps = step - decision["observation_step"]
            step += 1
            end_reason = ("success" if success else reason if reason else
                          "step_budget" if step >= args.max_steps else
                          decision.get("planned_end_reason", "max_horizon") if decision["chunk_last"] else None)
            started = time.perf_counter()
            video.write(obs, step, stable_steps(runtime),
                        predicted_counts=decision["predicted_counts"],
                        replan_label=f"latency {decision['action_source']} obs_age={age_steps} ticks")
            video_seconds = time.perf_counter() - started
            row = dict(step=step, simulation_time_s=runtime.time, replan=decision["replan"],
                       chunk_index=decision["chunk_index"], action=decision["action"].tolist(),
                       action_source=decision["action_source"],
                       source_observation_step=decision["observation_step"],
                       observation_age_steps=age_steps, observation_age_seconds=age_steps / CONTROL_HZ,
                       event_count=obs["event_counts"], timing=obs["timing"],
                       stable_success_steps=stable_steps(runtime), success=bool(success),
                       failure_reason=reason, chunk_end_reason=end_reason,
                       inference_seconds=decision["inference_seconds"],
                       round_trip_seconds=decision["round_trip_seconds"])
            diagnostics = getattr(getattr(runtime, "tracker", None), "last_diagnostics", None)
            if diagnostics is not None:
                row["success_diagnostics"] = diagnostics
            if args.profile_timing:
                row["timing_seconds"] = dict(capture_total=capture_seconds, video_write=video_seconds)
                step_timings.append(row["timing_seconds"])
            rows.append(row)
            if step % 10 == 0 or end_reason is not None:
                (output / f"trial_{trial:03d}_steps.json").write_text(json.dumps(rows, indent=2) + "\n")
                heartbeat.touch()
                print(f"[ewm-conveyor] latency trial={trial + 1}/{args.trials} step={step} time={runtime.time:.2f} source={decision['action_source']} obs_age={age_steps} end={end_reason}", flush=True)
            if end_reason is not None and decision["action_source"] == "policy":
                chunks.append(dict(replan=decision["replan"], executed_horizon=decision.get("executed_in_chunk", decision["chunk_index"] + 1),
                                   end_reason=end_reason))
            if success or reason:
                break
    finally:
        elapsed = max(time.perf_counter() - trial_started, 1e-12)
        timing = dict(simulation_control_hz=CONTROL_HZ, simulation_seconds=step / CONTROL_HZ,
                      offline_wall_seconds=elapsed, offline_throughput_hz=step / elapsed,
                      latency_source=latency_source(args), rounding="ceil_to_30hz_tick",
                      wall_clock_realtime=False)
        stream.close(step)
    timing.update(policy_wait_steps=stream.wait_steps, policy_wait_simulation_seconds=stream.wait_steps / CONTROL_HZ,
                  controller=args.latency_controller,
                  min_execution_steps=(stream.min_execution_steps if args.latency_controller == "continuous" else None),
                  bootstrap_wait_steps=getattr(stream, "bootstrap_wait_steps", 0),
                  buffer_underflow_steps=getattr(stream, "underflow_steps", 0),
                  inference_requests=[{key: value for key, value in request.items()
                                       if key not in ("submitted_at", "received_at")} for request in stream.requests])
    return (obs, success, reason, step, len(stream.requests),
            [request["inference_seconds"] for request in stream.requests],
            [request["round_trip_seconds"] for request in stream.requests], timing)

from piper_bench.sensors.capture import configure_renderer, snapshot, refresh_spawn_views
from piper_bench.sensors.event_frames import events_to_frame

def observation(runtime, dvs, event_cameras, *, rgb_transform=None):
    """Flush EVIS at t; current RGB/state and events over (t-1/30,t] align."""
    dvs.warp_and_process(dvs._conveyor_snapshot, dvs._conveyor_snapshot,
                         1, runtime.time, 1 / (CONTROL_HZ * WARP))
    chunks = dvs.recorder._events.pop(0, {})
    tick = round(runtime.time * CONTROL_HZ)
    counts = np.stack([
        events_to_frame(chunks.get("DVS/" + name, []), (tick-1) / CONTROL_HZ,
                        tick / CONTROL_HZ)[:, :, :2].transpose(2, 0, 1)
        for name in event_cameras
    ]).astype(np.float32)
    recorder = getattr(runtime, "_eval_event_recorder", None)
    display_events = recorder.observe(chunks, tick, event_cameras, counts) if recorder else None
    rgb = np.stack([
        runtime.env.scene[name].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        for name in CAMERAS
    ]).transpose(0, 3, 1, 2).astype(np.uint8)
    if rgb_transform is not None:
        rgb = rgb_transform(rgb)
    state = runtime.env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    result = dict(obs_video=rgb, obs_voxel=counts, proprio=state,
                event_counts={name: int(value.sum()) for name, value in zip(event_cameras, counts)},
                timing=dict(obs_video_time_s=runtime.time, proprio_time_s=runtime.time,
                            obs_event_start_s=runtime.time - 1 / CONTROL_HZ,
                            obs_event_end_s=runtime.time))
    if display_events is not None:
        result["display_event_frames"] = display_events
        result["event_counts"].update({name:int(value.sum()) for name,value in display_events.items()})
    return result


def validate_args(args):
    from piper_bench.protocols.execution import load_execution
    profile = load_execution(getattr(args, "execution_profile", "continuous_min8_v1"))
    if getattr(args, "min_execution_steps", None) is None:
        args.min_execution_steps = profile.min_execution_steps
    if getattr(args, "latency_controller", None) is None:
        args.latency_controller = profile.latency_controller
    if args.latency_controller == "continuous" and not args.latency_aware:
        raise ValueError("Continuous control requires --latency-aware")
    if args.policy_mode not in ("joint", "action_only", "rgb_action"):
        raise ValueError("Unsupported conveyor policy mode")
    if args.policy_mode in ("action_only", "rgb_action") and args.decode_predicted_events:
        raise ValueError("RGB/state-only policies do not predict events")
    if args.device != "cpu":
        raise ValueError("Use --device cpu for validated conveyor contacts; RTX/EVIS still use the GPU")
    if min(args.trials, args.max_steps, args.sampling_steps, args.execution_horizon) <= 0:
        raise ValueError("trial/step/sampling/horizon limits must be positive")
    if getattr(args, "min_execution_steps", 4) <= 0:
        raise ValueError("min-execution-steps must be positive")
    if not 0 <= args.scene_seed < 2**63 or not 0 <= args.seed < 2**63:
        raise ValueError("seeds must be in [0, 2**63)")
    if args.scene_seed + args.trials >= 2**63 or args.seed + args.trials * args.max_steps >= 2**63:
        raise ValueError("trial seed range exceeds the server's supported range")
    if args.checkpoint_step is not None and args.checkpoint_step <= 0:
        raise ValueError("checkpoint-step must be positive")
    if args.pathtracing_spp is not None and args.pathtracing_spp <= 0:
        raise ValueError("pathtracing-spp must be positive")
    if not 0 < args.predicted_event_threshold <= 1 or not 1 <= args.predicted_event_min_neighbors <= 9:
        raise ValueError("invalid predicted-event threshold or neighbor filter")
