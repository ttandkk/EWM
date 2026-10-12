"""Closed-loop, in-distribution EWM event-action evaluation in the Piper conveyor scene.

Run with Isaac Sim Python and the shared OpenWAM policy server. Scene, cameras,
lighting, and event capture match the collector. Default inference pauses the
simulation.
--latency-aware reproduces measured request latency in simulation ticks offline,
retaining the collector's renderer and sensor rate even on slower hardware.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT = next(root for root in Path(__file__).resolve().parents
               if (root / "Event-DataCollection/scripts/piper_conveyor.py").is_file())
sys.path.insert(0, str(PROJECT / "OpenWAM"))
sys.path.insert(0, str(PROJECT / "Event-DataCollection/scripts"))

from piper_conveyor import (
    CAMERAS, CONTROL_HZ, TASK, PROMPT, ConveyorConfig, ConveyorRuntime,
    make_env_cfg, set_views, camera_metadata, BASKET_GEOMETRY_TOLERANCE_M, BASKET_SUCCESS_CRITERION,
)
from piper_conveyor_speed import uniform_speed_spec, sample_episode_speed, set_episode_speed
from piper_conveyor_io import configure_renderer, snapshot, refresh_spawn_views
from piper_event_frames import events_to_frame
from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, encode_message, decode_message
from scripts.eval_ewm.ewm_eval_time_audit import EVENT_TIME_CONTRACT
from scripts.eval_ewm.ewm_action_only_protocol import OBSERVATION_CONTRACT
from scripts.eval_ewm.ewm_eval_recovery import archive_interrupted_trial, load_completed_results
from scripts.eval_ewm.ewm_eval_summary import write_summary
from scripts.eval_ewm.ewm_event_visualization import predicted_event_counts
from scripts.eval_ewm.ewm_event_replan import horizon_statistics, event_error, EventReplanGate

DEFAULT_DATASET = PROJECT / "datasets/piper_conveyor_front_wrist_v3_random_50/shards/job181776"
WARP = 8
RESET_WARMUP_STEPS = 60
PAUSED_CLOCK = "fixed_simulation_time_inference_paused"
LATENCY_CLOCK = "offline_30hz_simulation_with_measured_policy_latency"
SPEED_OOD_SCALES = {"none": 1., "slow": 2 / 3, "fast": 4 / 3}


def simulation_clock(args):
    return LATENCY_CLOCK if args.latency_aware else PAUSED_CLOCK


def build_parser(app_launcher, *, policy_mode="joint"):
    parser = argparse.ArgumentParser(description=__doc__)
    ready_argument = parser.add_argument("--ready-file", type=Path)
    output_argument = parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Training run whose dataset metadata defines the ID scene")
    parser.add_argument("--dataset-dir", type=Path,
                        help="Collector export; defaults to the checkpoint's training dataset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1050)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--execution-horizon", "--action-chunk-size", dest="execution_horizon",
                        type=int, default=16)
    parser.add_argument("--checkpoint-step", type=int, help="Defaults to latest available policy")
    parser.add_argument("--seed", type=int, default=10000, help="Model sampling seed")
    parser.add_argument("--scene-seed", type=int, default=20280002,
                        help="Independent scene seed; each trial uses scene-seed + trial")
    parser.add_argument("--decode-predicted-events", action="store_true")
    parser.add_argument("--predicted-event-threshold", type=float, default=1 / 30)
    parser.add_argument("--predicted-event-min-neighbors", type=int, default=2)
    parser.add_argument("--profile-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latency-aware", action=argparse.BooleanOptionalAction, default=False,
                        help="Offline evaluation with measured action-reply latency and full physics/RGB/EVIS ticks")
    parser.add_argument("--latency-controller", choices=("serial", "buffered"), default="buffered",
                        help="buffered replans every execution-horizon ticks while consuming the forecast tail; serial holds between chunks")
    parser.add_argument("--event-guided-replan", action="store_true")
    parser.add_argument("--event-replan-record-only", action="store_true")
    parser.add_argument("--event-replan-threshold", type=float, default=0.6)
    parser.add_argument("--event-replan-patience", type=int, default=2)
    parser.add_argument("--event-replan-min-horizon", type=int, default=4)
    parser.add_argument("--event-replan-min-pixels", type=int, default=4)
    parser.add_argument("--speed-ood", choices=tuple(SPEED_OOD_SCALES), default="none",
                        help="Change only belt surface speed: slow=2/3, fast=4/3 of the training speed")
    parser.add_argument("--conveyor-speed-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Uniform episode-constant speed; otherwise inherit the training range when present")
    parser.add_argument("--conveyor-speed", type=float,
                        help="Exact belt speed in m/s; mutually exclusive with slow/fast presets")
    parser.add_argument("--pathtracing-spp", type=int,
                        help="Must match the collector; omitted uses the export's setting")
    parser.add_argument("--interface-only", action="store_true",
                        help="Short interface diagnostic, excluded from formal success rates")
    # Isaac Lab probes parse_known_args with help temporarily disabled.
    # Enforce required paths after that probe so --help also works on its own.
    app_launcher.add_app_launcher_args(parser)
    ready_argument.required = output_argument.required = True
    parser.set_defaults(device="cpu", policy_mode=policy_mode)
    return parser


def event_replan_metadata(args):
    return dict(enabled=args.event_guided_replan, record_only=args.event_replan_record_only,
                threshold=args.event_replan_threshold, patience=args.event_replan_patience,
                min_horizon=args.event_replan_min_horizon, min_pixels=args.event_replan_min_pixels)


def latency_source(args):
    return ("action_reply_and_event_forecast_ready_round_trip_seconds" if args.event_guided_replan else
            "action_reply_round_trip_seconds")


def validate_args(args):
    monitor = args.event_guided_replan or args.event_replan_record_only
    if args.event_guided_replan and args.event_replan_record_only:
        raise ValueError("Choose event-guided-replan or record-only")
    if (not 0 <= args.event_replan_threshold <= 1 or args.event_replan_patience < 1
            or args.event_replan_min_horizon < 1 or args.event_replan_min_pixels < 1):
        raise ValueError("Invalid event replan threshold, patience, horizon or pixel gate")
    if monitor:
        if args.policy_mode != "joint":
            raise ValueError("Event-guided replanning requires an event-action policy")
        if args.event_replan_min_horizon > args.execution_horizon:
            raise ValueError("Event replan minimum horizon exceeds execution horizon")
        if args.latency_aware and args.latency_controller != "buffered":
            raise ValueError("Latency-aware event monitoring requires buffered control for temporal alignment")
        args.decode_predicted_events = True
    if args.speed_ood not in SPEED_OOD_SCALES:
        raise ValueError("Unknown conveyor speed OOD preset")
    bounds = getattr(args, "conveyor_speed_range", None)
    uniform_speed_spec(bounds)
    if bounds is not None and (args.conveyor_speed is not None or args.speed_ood != "none"):
        raise ValueError("speed range cannot be combined with fixed speed or OOD presets")
    if args.conveyor_speed is not None:
        if not math.isfinite(args.conveyor_speed) or not 0 < args.conveyor_speed <= .2:
            raise ValueError("conveyor-speed must be finite and in (0, 0.2] m/s")
        if args.speed_ood != "none":
            raise ValueError("conveyor-speed cannot be combined with a speed-ood preset")
    if args.policy_mode not in ("joint", "action_only"):
        raise ValueError("Unsupported conveyor policy mode")
    if args.policy_mode == "action_only" and args.decode_predicted_events:
        raise ValueError("Action-only policies do not predict events")
    if args.device != "cpu":
        raise ValueError("Use --device cpu for validated conveyor contacts; RTX/EVIS still use the GPU")
    if min(args.trials, args.max_steps, args.sampling_steps, args.execution_horizon) <= 0:
        raise ValueError("trial/step/sampling/horizon limits must be positive")
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


def load_collection(args):
    """Use recorded collector settings rather than the narrower class defaults."""
    training = None
    if args.checkpoint_dir is not None:
        training = yaml.safe_load((args.checkpoint_dir / "config.yaml").read_text())
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        dataset_dir = Path(training["dataloader"]["dataset_dir"]) if training else DEFAULT_DATASET
    dataset_dir = dataset_dir.resolve()
    if training and not args.interface_only:
        if dataset_dir != Path(training["dataloader"]["dataset_dir"]).resolve():
            raise ValueError("Evaluation must use the checkpoint's training scene metadata")
        if training["model"].get("training_mode", "joint") != args.policy_mode:
            raise ValueError("Checkpoint training mode differs from the evaluator")
        if args.policy_mode == "action_only" and training["dataloader"]["type"] != "ewm_piper_action_only":
            raise ValueError("Action-only evaluation requires the RGB/state training contract")
    metadata = dataset_dir / "meta"
    collection = json.loads((metadata / "piper_collection_config.json").read_text())
    config = ConveyorConfig(**collection["scene_config"])
    if collection.get("probe_only") or collection.get("renderer_mode") != "pathtracing":
        raise ValueError("Use a completed pathtraced conveyor training export")
    if json.loads((metadata / "piper_camera_calibration.json").read_text()) != camera_metadata():
        raise ValueError("Simulator camera calibration differs from the training export")
    contract = json.loads((metadata / "piper_temporal_contract.json").read_text())
    if contract.get("temporal_contract") != EVENT_TIME_CONTRACT:
        raise ValueError("Dataset event-time contract differs")
    spp = int(collection["pathtracing_spp"])
    if args.pathtracing_spp is not None and args.pathtracing_spp != spp:
        raise ValueError("ID evaluation pathtracing-spp must match the collector")
    if collection.get("speed_sampling") and args.speed_ood != "none":
        raise ValueError("range-trained data requires an explicit fixed speed or range instead of OOD scale presets")
    if args.conveyor_speed is not None:
        config = replace(config, speed=args.conveyor_speed)
    elif args.speed_ood != "none":
        config = replace(config, speed=round(config.speed * SPEED_OOD_SCALES[args.speed_ood], 12))
    return dataset_dir, collection, config, spp


def evaluation_speed_spec(args, collection):
    bounds = getattr(args, "conveyor_speed_range", None)
    if bounds is not None:
        return uniform_speed_spec(bounds)
    if args.conveyor_speed is None and args.speed_ood == "none":
        return collection.get("speed_sampling")
    return None


def speed_ood_metadata(args, collection, config):
    training_spec=collection.get("speed_sampling")
    eval_spec=evaluation_speed_spec(args, collection)
    training_speed=float(collection["scene_config"]["speed"])
    if training_spec is not None or eval_spec is not None:
        lo,hi=(eval_spec['min_m_s'],eval_spec['max_m_s']) if eval_spec else (config.speed,config.speed)
        in_range=(training_spec['min_m_s'] <= lo <= hi <= training_spec['max_m_s']) if training_spec else lo==hi==training_speed
        return dict(distribution="in_distribution" if in_range else "out_of_distribution",
                    speed_ood="uniform" if eval_spec else "custom",
                    training_speed_sampling=training_spec, evaluation_speed_sampling=eval_spec,
                    training_conveyor_speed_m_s=None if training_spec else training_speed,
                    evaluation_conveyor_speed_m_s=None if eval_spec else float(config.speed),
                    conveyor_speed_scale=None)
    training_speed = float(collection["scene_config"]["speed"])
    custom = args.conveyor_speed is not None
    shifted = not math.isclose(config.speed, training_speed, rel_tol=0, abs_tol=1e-12)
    return dict(distribution="out_of_distribution" if shifted else "in_distribution",
                speed_ood="custom" if custom else args.speed_ood,
                training_conveyor_speed_m_s=training_speed,
                evaluation_conveyor_speed_m_s=float(config.speed),
                conveyor_speed_scale=config.speed / training_speed if custom else SPEED_OOD_SCALES[args.speed_ood])


def request(conn, header, arrays=None):
    conn.send(encode_message(header, arrays))
    reply, values = decode_message(conn.recv(timeout=600))
    if not reply.get("ok"):
        raise RuntimeError(f"EWM server: {reply.get('error', reply)}")
    return reply, values


def policy_request(conn, args, header, arrays):
    if args.policy_mode == "action_only":
        header = dict(header)
        header.pop("event_time_contract", None)
        header.pop("defer_visualization", None)
        header["observation_contract"] = OBSERVATION_CONTRACT
        header["decode_future_events"] = False
        header["timing"] = {name: header["timing"][name]
                            for name in ("obs_video_time_s", "proprio_time_s")}
        arrays = {name: arrays[name] for name in ("obs_video", "proprio")}
    return request(conn, header, arrays)


def observation(runtime, dvs, event_cameras):
    """Flush EVIS at t; current RGB/state and events over (t-1/30,t] align."""
    dvs.warp_and_process(dvs._conveyor_snapshot, dvs._conveyor_snapshot,
                         1, runtime.time, 1 / (CONTROL_HZ * WARP))
    chunks = dvs.recorder._events.pop(0, {})
    counts = np.stack([
        events_to_frame(chunks.get("DVS/" + name, []), runtime.time - 1 / CONTROL_HZ,
                        runtime.time)[:, :, :2].transpose(2, 0, 1)
        for name in event_cameras
    ]).astype(np.float32)
    rgb = np.stack([
        runtime.env.scene[name].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        for name in CAMERAS
    ]).transpose(0, 3, 1, 2).astype(np.uint8)
    state = runtime.env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
    result = dict(obs_video=rgb, obs_voxel=counts, proprio=state,
                event_counts={name: int(value.sum()) for name, value in zip(event_cameras, counts)},
                timing=dict(obs_video_time_s=runtime.time, proprio_time_s=runtime.time,
                            obs_event_start_s=runtime.time - 1 / CONTROL_HZ,
                            obs_event_end_s=runtime.time))
    history = getattr(runtime, "_ewm_memory_history", None)
    if history is not None:
        result.update(history.observe(rgb,counts,round(runtime.time*CONTROL_HZ)))
    return result


def capture_step(runtime, dvs, action, event_cameras):
    previous, previous_time = dvs._conveyor_snapshot, runtime.time
    set_views(runtime.env)
    runtime.env.step(action)
    success, reason = runtime.after_step()
    if runtime.supply():
        refresh_spawn_views(runtime.env)
    current = snapshot(dvs, runtime.env)
    dvs.warp_and_process(previous, current, WARP, previous_time, 1 / (CONTROL_HZ * WARP))
    dvs._conveyor_snapshot = current
    return observation(runtime, dvs, event_cameras), success, reason


def preview(obs, step):
    image = np.hstack(obs["obs_video"].transpose(0, 2, 3, 1))
    for index, name in enumerate(CAMERAS):
        cv2.putText(image, f"{name} step={step}", (index * 256 + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def timing_stats(values):
    array = np.asarray(values, dtype=np.float64)
    return dict(calls=len(values), total_seconds=float(array.sum()),
                mean_seconds=float(array.mean()), p50_seconds=float(np.percentile(array, 50)),
                p95_seconds=float(np.percentile(array, 95)))


class LatencyAwareActionStream:
    """Deliver measured replies on their 30 Hz simulation deadlines, offline.

    Full physics/RGB/EVIS ticks are charged before applying the returned chunk.
    Holds retain the last gripper command and never repeat relative EE motion.
    """

    def __init__(self, predict, *, horizon, max_steps, now=time.perf_counter):
        self.predict, self.horizon, self.max_steps, self.now = predict, horizon, max_steps, now
        self.pending = self.packet = None
        self.index = self.wait_steps = 0
        self.requests = []
        self.gripper = 1.

    def ensure_request(self, observation, step):
        if self.pending is not None or self.packet is not None:
            return
        horizon = min(self.horizon, self.max_steps - step)
        copied = {key: observation[key].copy() for key in ("obs_video", "obs_voxel", "proprio") + tuple(k for k in observation if k.startswith("memory_"))}
        copied["timing"] = dict(observation["timing"])
        submitted_at = self.now()
        packet = self.predict(copied, step, horizon)
        latency = packet["round_trip_seconds"]
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Request latency must be finite and nonnegative")
        delay = math.ceil(latency * CONTROL_HZ)
        metadata = dict(replan=len(self.requests), observation_step=step,
                        request_horizon=horizon, submitted_at=submitted_at,
                        measured_round_trip_seconds=latency, latency_delay_steps=delay,
                        latency_delay_simulation_seconds=delay / CONTROL_HZ,
                        latency_quantization_seconds=delay / CONTROL_HZ - latency,
                        scheduled_ready_step=step + delay,
                        inference_seconds=packet["inference_seconds"],
                        round_trip_seconds=latency)
        self.requests.append(metadata)
        packet["metadata"] = metadata
        self.pending = packet

    def receive(self, step):
        if self.pending is None:
            return
        metadata = self.pending["metadata"]
        if step < metadata["scheduled_ready_step"]:
            return
        metadata.update(received_step=step,
                        simulation_steps_during_inference=step - metadata["observation_step"])
        self.packet, self.index, self.pending = self.pending, 0, None

    def decision(self, step):
        self.receive(step)
        if self.packet is None:
            self.wait_steps += 1
            metadata = self.pending["metadata"]
            return dict(action=np.array([0.,0.,0.,0.,0.,0.,self.gripper], np.float32),
                        action_source="policy_wait_hold", replan=metadata["replan"],
                        chunk_index=None, chunk_last=False, predicted_counts=None,
                        observation_step=metadata["observation_step"],
                        inference_seconds=None, round_trip_seconds=None)
        packet = self.packet
        metadata = packet["metadata"]
        if self.index == 0:
            metadata.update(first_action_step=step,
                            first_action_observation_age_steps=step - metadata["observation_step"])
        index = self.index
        action = packet["actions"][index]
        self.gripper = float(action[6])
        self.index += 1
        last = self.index == len(packet["actions"])
        if last:
            self.packet = None
        predicted = packet["predicted"]
        return dict(action=action, action_source="policy", replan=metadata["replan"],
                    chunk_index=index, chunk_last=last,
                    predicted_counts=None if predicted is None else predicted[index],
                    observation_step=metadata["observation_step"],
                    inference_seconds=packet["inference_seconds"],
                    round_trip_seconds=packet["round_trip_seconds"])

    def close(self, step):
        if self.pending is not None:
            metadata = self.pending["metadata"]
            metadata.update(reply_discarded_at_terminal=True, terminal_step=step,
                            simulation_steps_during_inference=step - metadata["observation_step"])
        self.pending = None


class BufferedLatencyActionStream:
    """Single-request asynchronous controller emulated on simulation time.

    Request a full forecast; replan after execution_horizon ticks from its
    observation anchor. Continue consuming the prior forecast while waiting.
    Later replies start at their current prediction index, so elapsed actions
    are not repeated. The bootstrap plan starts at zero after its initial wait:
    dropping its relative approach commands would change the initial pose path.
    """

    def __init__(self, predict, *, horizon, plan_horizon, max_steps):
        self.predict, self.horizon, self.plan_horizon, self.max_steps = predict, horizon, plan_horizon, max_steps
        self.pending = self.packet = None
        self.requests = []
        self.next_request_step = 0
        self.bootstrapped = False
        self.gripper = 1.
        self.wait_steps = self.bootstrap_wait_steps = self.underflow_steps = 0

    def receive(self, step):
        if self.pending is None:
            return
        packet = self.pending
        metadata = packet["metadata"]
        if step < metadata["scheduled_ready_step"]:
            return
        offset = step - metadata["observation_step"] if self.bootstrapped else 0
        self.pending = None
        if offset >= len(packet["actions"]):
            metadata.update(reply_expired=True, discarded_at_step=step)
            self.next_request_step = step
            return
        anchor = metadata["observation_step"] if self.bootstrapped else step
        metadata.update(received_step=step, first_action_step=step,
                        first_action_observation_age_steps=step - metadata["observation_step"],
                        discarded_prefix_steps=offset, action_anchor_step=anchor,
                        bootstrap=not self.bootstrapped, executed_actions=0)
        self.packet, self.bootstrapped = packet, True
        self.next_request_step = anchor + self.horizon

    def ensure_request(self, observation, step):
        self.receive(step)
        if self.pending is not None or step < self.next_request_step:
            return
        if self.packet is not None and step < self.packet["metadata"]["request_pipeline_ready_step"]:
            return
        horizon = min(self.plan_horizon, self.max_steps - step)
        copied = {key: observation[key].copy() for key in ("obs_video", "obs_voxel", "proprio") + tuple(k for k in observation if k.startswith("memory_"))}
        copied["timing"] = dict(observation["timing"])
        packet = self.predict(copied, step, horizon)
        latency = packet["round_trip_seconds"]
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Request latency must be finite and nonnegative")
        delay = math.ceil(latency * CONTROL_HZ)
        metadata = dict(replan=len(self.requests), observation_step=step, request_horizon=horizon,
                        measured_round_trip_seconds=latency, latency_delay_steps=delay,
                        latency_delay_simulation_seconds=delay / CONTROL_HZ,
                        latency_quantization_seconds=delay / CONTROL_HZ - latency,
                        scheduled_ready_step=step + delay,
                        inference_seconds=packet["inference_seconds"], round_trip_seconds=latency,
                        visualization_seconds=packet.get("visualization_seconds", 0.),
                        action_reply_round_trip_seconds=packet.get("action_reply_round_trip_seconds", latency),
                        event_forecast_ready_round_trip_seconds=packet.get("event_forecast_ready_round_trip_seconds", latency))
        forecast_latency = packet.get("event_forecast_ready_round_trip_seconds", latency)
        if not math.isfinite(forecast_latency) or forecast_latency < latency:
            raise ValueError("Event forecast cannot precede action delivery")
        forecast_delay = math.ceil(forecast_latency * CONTROL_HZ)
        metadata.update(event_forecast_ready_step=step + forecast_delay,
                        event_forecast_delay_steps=forecast_delay,
                        event_forecast_quantization_seconds=forecast_delay / CONTROL_HZ - forecast_latency,
                        request_pipeline_ready_step=step + (forecast_delay if packet.get("charge_event_forecast_latency") else delay))
        self.requests.append(metadata)
        packet["metadata"] = metadata
        self.pending = packet

    def decision(self, step):
        self.receive(step)
        packet = self.packet
        if packet is not None:
            metadata = packet["metadata"]
            index = step - metadata["action_anchor_step"]
            if index >= len(packet["actions"]):
                self.packet = packet = None
        if packet is None:
            self.wait_steps += 1
            if self.bootstrapped:
                self.underflow_steps += 1
            else:
                self.bootstrap_wait_steps += 1
            metadata = self.pending["metadata"] if self.pending is not None else self.requests[-1]
            return dict(action=np.array([0.,0.,0.,0.,0.,0.,self.gripper], np.float32),
                        action_source="buffer_underflow_hold" if self.bootstrapped else "bootstrap_wait_hold",
                        replan=metadata["replan"], chunk_index=None, chunk_last=False,
                        executed_in_chunk=0, predicted_counts=None,
                        observation_step=metadata["observation_step"],
                        inference_seconds=None, round_trip_seconds=None)
        action = packet["actions"][index]
        self.gripper = float(action[6])
        metadata["executed_actions"] += 1
        switching_next = (self.pending is not None and step + 1 >= self.pending["metadata"]["scheduled_ready_step"]
                          and step + 1 - self.pending["metadata"]["observation_step"] < len(self.pending["actions"]))
        predicted = packet["predicted"]
        return dict(action=action, action_source="policy", replan=metadata["replan"], chunk_index=index,
                    chunk_last=switching_next or index + 1 == len(packet["actions"]),
                    executed_in_chunk=metadata["executed_actions"],
                    predicted_counts=None if predicted is None else predicted[index],
                    prediction_target_step=metadata["observation_step"] + index + 1,
                    event_prediction_ready_step=metadata["event_forecast_ready_step"],
                    planned_end_reason=metadata.get("next_plan_reason", "max_horizon"),
                    observation_step=metadata["observation_step"],
                    inference_seconds=packet["inference_seconds"], round_trip_seconds=packet["round_trip_seconds"])

    def request_replan(self, step):
        # One in-flight request: never cancel a pending forecast or reset its clock.
        # Fixed-horizon requests due at this step retain precedence.
        if (self.pending is not None or self.packet is None
                or self.packet["metadata"]["bootstrap"] or step >= self.next_request_step
                or step < self.packet["metadata"]["request_pipeline_ready_step"]):
            return False
        self.packet["metadata"].update(next_plan_reason="event_mismatch", event_trigger_step=step)
        self.next_request_step = step
        return True

    def close(self, step):
        if self.pending is not None:
            metadata = self.pending["metadata"]
            metadata.update(reply_discarded_at_terminal=True, terminal_step=step)
            self.pending = None


def validate_hello(hello, args):
    if args.policy_mode == "action_only":
        if (hello.get("training_mode") != "action_only"
                or hello.get("checkpoint_format") != "ewm_action_only_v1"
                or hello.get("observation_contract") != OBSERVATION_CONTRACT
                or hello.get("inference_observation_row_offset") != 0
                or hello.get("training_observation_row_offset") != 0
                or hello.get("input_keys") != ["obs_video", "proprio", "prompt"]):
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
    checkpoints = hello.get("checkpoints", [])
    step = args.checkpoint_step if args.checkpoint_step is not None else max(checkpoints)
    if step not in checkpoints:
        raise ValueError(f"Checkpoint step {step} unavailable; available: {checkpoints}")
    if args.execution_horizon > hello["horizon"]:
        raise ValueError("Execution horizon exceeds the trained action horizon")
    if args.latency_aware:
        if args.policy_mode == "joint" and not hello.get("deferred_visualization_supported"):
            raise ValueError("Latency evaluation requires action replies before deferred visualization")
        if args.latency_controller == "buffered" and args.execution_horizon >= hello["horizon"]:
            raise ValueError("Buffered control needs a forecast tail beyond execution-horizon")
    if args.checkpoint_dir is not None:
        digest = hashlib.sha256((args.checkpoint_dir / "config.yaml").read_bytes()).hexdigest()
        if digest != hello.get("config_sha256"):
            raise ValueError("Server checkpoint config differs from --checkpoint-dir")
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
    if args.decode_predicted_events:
        voxel = values.get("future_voxel")
        expected = (horizon, len(event_cameras), 2, 256, 256)
        if reply.get("future_decoded") is not True or voxel is None or voxel.shape != expected:
            raise ValueError(f"Server did not return decoded future event frames {expected}")
        predicted = predicted_event_counts(voxel, count_clip=count_clip,
                                           occupancy_threshold=args.predicted_event_threshold,
                                           min_neighbors=args.predicted_event_min_neighbors)
    return actions, predicted, inference_seconds


def prepare_results(output, manifest, args):
    path = output / "manifest.json"
    if args.resume:
        saved = json.loads(path.read_text())
        # Existing paused evaluations predate the optional latency switch.
        if "latency_aware" in manifest:
            saved.setdefault("latency_aware", False)
        saved.setdefault("speed_ood", "none")
        saved.setdefault("training_conveyor_speed_m_s", saved.get("scene_config", {}).get("speed"))
        saved.setdefault("evaluation_conveyor_speed_m_s", saved.get("scene_config", {}).get("speed"))
        saved.setdefault("conveyor_speed_scale", 1.)
        saved.setdefault("basket_geometry_tolerance_m", 0.)
        saved.setdefault("basket_success_criterion_version", "full_volume_v1")
        saved.setdefault("training_mode", "joint")
        saved.setdefault("event_input", True)
        saved.setdefault("policy_observation_contract", EVENT_TIME_CONTRACT)
        saved.setdefault("latency_controller", "serial" if saved.get("latency_aware") else None)
        saved.setdefault("latency_source", "client_round_trip_seconds" if saved.get("latency_aware") else None)
        if "event_replan" in manifest:
            saved.setdefault("event_replan", dict(enabled=False, record_only=False, threshold=0.6,
                                                  patience=2, min_horizon=4, min_pixels=4))
        immutable = set(manifest) - {"server_optimization"}
        mismatches = [key for key in sorted(immutable) if saved.get(key) != manifest[key]]
        if mismatches:
            raise ValueError(f"Resume settings differ from saved manifest: {mismatches}")
        results = load_completed_results(output, args.trials)
        archived = archive_interrupted_trial(output, len(results))
        print(f"[ewm-conveyor] resume completed={len(results)} archived={archived}", flush=True)
        (output / "failure.json").unlink(missing_ok=True)
        return results
    if path.exists() or list(output.glob("trial_*")):
        raise FileExistsError(f"Existing evaluation requires --resume: {output}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return []


class ConveyorEventMonitor:
    def __init__(self, args):
        self.args = args
        self.enabled = args.event_guided_replan or args.event_replan_record_only
        self.plan = None
        self.gate = None

    def inspect(self, observation, decision, step):
        if not self.enabled:
            return {}, False
        if decision["replan"] != self.plan:
            self.plan = decision["replan"]
            self.gate = EventReplanGate(self.args.event_replan_threshold,
                                       self.args.event_replan_patience, self.args.event_replan_min_horizon)
        target = decision.get("prediction_target_step")
        ready = step >= decision.get("event_prediction_ready_step", 0)
        aligned = (ready and decision["action_source"] == "policy"
                   and decision["predicted_counts"] is not None and target == step)
        # Bootstrap actions are intentionally delayed without prefix dropping.
        # Their forecast timestamps are therefore not eligible for error triggers.
        score = cameras = None
        trigger = False
        if aligned:
            score, cameras = event_error(observation["obs_voxel"], decision["predicted_counts"],
                                         min_pixels=self.args.event_replan_min_pixels)
            trigger = self.gate.update(score, decision["executed_in_chunk"])
        else:
            self.gate.streak = 0
        return dict(event_error=score, event_error_per_camera=cameras,
                    event_error_streak=self.gate.streak, event_prediction_target_step=target,
                    event_prediction_aligned=aligned, event_prediction_ready=ready,
                    event_replan_eligible=trigger,
                    event_replan_triggered=False), trigger and self.args.event_guided_replan


def run_latency_loop(args, runtime, dvs, conn, hello, checkpoint_step, event_cameras,
                      obs, video, rows, chunks, step_timings, heartbeat, output, trial):
    import torch

    def predict(copied, step, horizon):
        started = time.perf_counter()
        reply, values = policy_request(conn, args, dict(
            op="predict", step=checkpoint_step, seed=args.seed + trial * args.max_steps + step,
            sampling_steps=args.sampling_steps, action_chunk_size=horizon,
            prompt=PROMPT, event_time_contract=EVENT_TIME_CONTRACT,
            observation_row_offset=0, raw_observation=True, timing=copied["timing"],
            decode_future_events=args.decode_predicted_events,
            defer_visualization=True,
        ), {key: copied[key] for key in ("obs_video", "obs_voxel", "proprio") + tuple(k for k in copied if k.startswith("memory_"))})
        action_options = argparse.Namespace(**vars(args))
        action_options.decode_predicted_events = False
        actions, predicted, inference_seconds = validate_reply(
            reply, values, horizon, event_cameras, action_options, int(hello.get("count_clip", 15)))
        round_trip_seconds = time.perf_counter() - started
        visualization_started = time.perf_counter()
        if args.decode_predicted_events:
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
                    charge_event_forecast_latency=args.event_guided_replan)

    if args.latency_controller == "buffered":
        stream = BufferedLatencyActionStream(predict, horizon=args.execution_horizon,
                                             plan_horizon=hello["horizon"], max_steps=args.max_steps)
    else:
        stream = LatencyAwareActionStream(predict, horizon=args.execution_horizon, max_steps=args.max_steps)
    monitor = ConveyorEventMonitor(args)
    trial_started = time.perf_counter()
    success, reason, step = False, None, 0
    try:
        while step < args.max_steps:
            stream.ensure_request(obs, step)
            decision = stream.decision(step)
            started = time.perf_counter()
            command = torch.from_numpy(decision["action"].copy())[None].to(runtime.env.device)
            obs, success, reason = capture_step(runtime, dvs, command, event_cameras)
            capture_seconds = time.perf_counter() - started
            age_steps = step - decision["observation_step"]
            step += 1
            monitor_started = time.perf_counter()
            event_fields, trigger = monitor.inspect(obs, decision, step)
            terminal = success or reason or step >= args.max_steps
            if trigger and not terminal:
                event_fields["event_replan_triggered"] = stream.request_replan(step)
            monitor_seconds = time.perf_counter() - monitor_started
            end_reason = ("success" if success else reason if reason else
                          "step_budget" if step >= args.max_steps else
                          decision.get("planned_end_reason", "max_horizon") if decision["chunk_last"] else None)
            started = time.perf_counter()
            video.write(obs, step, runtime.tracker.steps,
                        predicted_counts=decision["predicted_counts"],
                        replan_label=(f"latency {decision['action_source']} obs_age={age_steps} ticks"
                                      + (f" error={event_fields['event_error']:.3f}"
                                         if event_fields.get("event_error") is not None else "")
                                      + (" event_replan" if event_fields.get("event_replan_triggered") else "")))
            video_seconds = time.perf_counter() - started
            row = dict(step=step, simulation_time_s=runtime.time, replan=decision["replan"],
                       chunk_index=decision["chunk_index"], action=decision["action"].tolist(),
                       action_source=decision["action_source"],
                       source_observation_step=decision["observation_step"],
                       observation_age_steps=age_steps, observation_age_seconds=age_steps / CONTROL_HZ,
                       event_count=obs["event_counts"], timing=obs["timing"],
                       stable_success_steps=runtime.tracker.steps, success=bool(success),
                       failure_reason=reason, chunk_end_reason=end_reason,
                       inference_seconds=decision["inference_seconds"],
                       round_trip_seconds=decision["round_trip_seconds"])
            row.update(event_fields)
            if args.profile_timing:
                row["timing_seconds"] = dict(capture_total=capture_seconds, video_write=video_seconds,
                                           event_monitor=monitor_seconds)
                step_timings.append(row["timing_seconds"])
            rows.append(row)
            if step % 10 == 0 or end_reason is not None or event_fields.get("event_replan_triggered"):
                (output / f"trial_{trial:03d}_steps.json").write_text(json.dumps(rows, indent=2) + "\n")
                heartbeat.touch()
                print(f"[ewm-conveyor] latency trial={trial + 1}/{args.trials} step={step} time={runtime.time:.2f} source={decision['action_source']} obs_age={age_steps} end={end_reason} event_trigger={event_fields.get('event_replan_triggered', False)}", flush=True)
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
                  bootstrap_wait_steps=getattr(stream, "bootstrap_wait_steps", 0),
                  buffer_underflow_steps=getattr(stream, "underflow_steps", 0),
                  inference_requests=[{key: value for key, value in request.items()
                                       if key not in ("submitted_at", "received_at")} for request in stream.requests])
    return (obs, success, reason, step, len(stream.requests),
            [request["inference_seconds"] for request in stream.requests],
            [request["round_trip_seconds"] for request in stream.requests], timing)


class TrialVideo:
    """Stream RGB, observed events, and optionally predicted events to H.264."""

    def __init__(self, path: Path, event_cameras: tuple[str, ...], *, show_prediction: bool = False, scene_label: str = ""):
        self.scene_label = scene_label
        self.event_cameras = event_cameras
        self.show_prediction = show_prediction
        bundled = sorted((PROJECT / "envs/openwam/lib").glob(
            "python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        ))
        system_ffmpeg = Path("/usr/bin/ffmpeg")
        if bundled:
            ffmpeg = str(bundled[0])
        elif system_ffmpeg.is_file():
            ffmpeg = str(system_ffmpeg)
        else:
            ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for evaluation video output")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.ffmpeg = ffmpeg
        self.process = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-video_size", "512x768" if show_prediction else "512x512",
                "-framerate", str(CONTROL_HZ),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")

    @staticmethod
    def _label(image: np.ndarray, text: str) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
        cv2.putText(image, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def write(self, observation: dict, step: int, success_steps: int,
              *, predicted_counts: np.ndarray | None = None, replan_label: str = "") -> None:
        rgb = observation["obs_video"].transpose(0, 2, 3, 1)
        rgb_row = np.hstack((
            self._label(rgb[0], f"cam0 step={step}"),
            self._label(rgb[1], f"wrist success={success_steps}"),
        ))
        event_images = {}
        for name, counts in zip(self.event_cameras, observation["obs_voxel"], strict=True):
            image = np.full((256, 256, 3), 255, dtype=np.uint8)
            image[counts[0] > 0] = (255, 0, 0)  # OFF: blue in BGR.
            image[counts[1] > 0] = (0, 0, 255)  # ON: red in BGR.
            cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
            cv2.putText(image, f"{name} events={observation['event_counts'][name]}",
                        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            event_images[name] = image
        event_row = np.hstack(tuple(
            event_images[name] if name in event_images else np.full((256, 256, 3), 255, dtype=np.uint8)
            for name in CAMERAS
        ))
        rows = [rgb_row, event_row]
        if self.show_prediction:
            if predicted_counts is not None and predicted_counts.shape != (len(self.event_cameras), 2, 256, 256):
                raise ValueError("predicted counts must be [camera,2,256,256]")
            predicted_images = {}
            if predicted_counts is not None:
                for name, counts in zip(self.event_cameras, predicted_counts, strict=True):
                    image = np.full((256, 256, 3), 255, dtype=np.uint8)
                    image[counts[0] > 0] = (255, 0, 0)  # OFF: blue in BGR.
                    image[counts[1] > 0] = (0, 0, 255)  # ON: red in BGR.
                    cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
                    count = int((counts > 0).sum())
                    cv2.putText(image, f"pred {name} active={count}",
                                (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    predicted_images[name] = image
            predicted_row = np.hstack(tuple(
                predicted_images[name] if name in predicted_images else self._blank_prediction(name, step)
                for name in CAMERAS
            ))
            rows.append(predicted_row)
        frame = np.vstack(rows)
        replan_label = " | ".join(label for label in (self.scene_label, replan_label) if label)
        if replan_label:
            cv2.rectangle(frame, (0, frame.shape[0] - 24), (512, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, replan_label, (7, frame.shape[0] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    @staticmethod
    def _blank_prediction(name: str, step: int) -> np.ndarray:
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
        label = f"pred {name} pending" if step == 0 else f"pred {name} unavailable"
        cv2.putText(image, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with exit status {self.process.returncode}")


def run(args, dataset_dir, collection, cfg, spp):
    import gymnasium as gym
    import torch
    from dvs_gen.sensors import DVSCamera
    from websockets.sync.client import connect

    speed_meta = speed_ood_metadata(args, collection, cfg)
    speed_spec = evaluation_speed_spec(args, collection)
    output = args.output_dir.resolve()
    if (output / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    heartbeat = output / "progress.heartbeat"
    heartbeat.touch()
    configure_renderer(spp)
    env = gym.make(TASK, cfg=make_env_cfg(cfg, device=args.device, seed=args.scene_seed)).unwrapped
    results, attempted_trials = [], 0

    def summarize(error=None):
        write_summary(output, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                      requested_trials=args.trials, attempted_trials=attempted_trials,
                      results=results, task_name=TASK,
                      decode_predicted_events=args.decode_predicted_events, error=error)
        with (output / "summary.txt").open("a") as stream:
            for key, value in speed_meta.items():
                stream.write(f"{key}: {value}\n")
            stream.write(f"basket_geometry_tolerance_m: {BASKET_GEOMETRY_TOLERANCE_M}\n")
            stream.write(f"basket_success_criterion_version: {BASKET_SUCCESS_CRITERION}\n")
            stream.write(f"training_mode: {args.policy_mode}\nevent_input: {str(args.policy_mode == 'joint').lower()}\n")
            stream.write(f"latency_aware: {str(args.latency_aware).lower()}\n")
            stream.write(f"latency_controller: {args.latency_controller if args.latency_aware else 'none'}\n")
            stream.write(f"simulation_clock: {simulation_clock(args)}\n")
            stream.write(f"event_guided_replan: {str(args.event_guided_replan).lower()}\n")
            if args.event_guided_replan or args.event_replan_record_only:
                stream.write(f"event_replan_config: {json.dumps(event_replan_metadata(args))}\n")
                triggers = sum(row.get("event_replan_trigger_count", 0) for row in results)
                stream.write(f"event_replan_trigger_count: {triggers}\n")
            if args.latency_aware:
                stream.write(f"latency_source: {latency_source(args)}\n")
            if args.latency_aware and results:
                waits = sum(result["latency_timing"]["policy_wait_steps"] for result in results)
                stream.write(f"latency_hold_steps: {waits}\nlatency_hold_simulation_seconds: {waits / CONTROL_HZ:.6f}\nwall_clock_realtime: false\n")
            if args.interface_only:
                stream.write("interface_only: true\nformal_success_rate: not_applicable\n")

    try:
        dvs = DVSCamera.from_scene(env.scene, list(CAMERAS), out_dir=str(output / "dvs_tmp"),
                                  compression="lzf", event_source="hdr", antialiasing=None)
        runtime = ConveyorRuntime(env, cfg)
        zero = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=env.device)
        ready = json.loads(args.ready_file.read_text())
        with connect(ready["url"], open_timeout=30, max_size=MAX_MESSAGE, compression=None) as conn:
            hello, _ = request(conn, dict(op="hello"))
            checkpoint_step, event_cameras = validate_hello(hello, args)
            manifest = dict(
                sim_task=TASK, prompt=PROMPT, **speed_meta,
                basket_geometry_tolerance_m=BASKET_GEOMETRY_TOLERANCE_M,
                basket_success_criterion_version=BASKET_SUCCESS_CRITERION,
                training_mode=args.policy_mode, event_input=args.policy_mode == "joint",
                policy_observation_contract=OBSERVATION_CONTRACT if args.policy_mode == "action_only" else EVENT_TIME_CONTRACT,
                dataset_dir=str(dataset_dir), scene_config=cfg.metadata(),
                camera_calibration=camera_metadata(), checkpoint_step=checkpoint_step,
                checkpoint_dir=str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
                config_sha256=hello.get("config_sha256"), normalization_sha256=hello.get("normalization_sha256"),
                event_time_contract=EVENT_TIME_CONTRACT, inference_observation_row_offset=0,
                training_observation_row_offset=hello.get("training_observation_row_offset"),
                trials=args.trials, max_steps=args.max_steps, sampling_steps=args.sampling_steps,
                execution_horizon=args.execution_horizon, seed=args.seed, scene_seed=args.scene_seed,
                scene_seed_mode="per_trial", control_hz=CONTROL_HZ, event_warp=WARP,
                event_cameras=list(event_cameras), renderer_mode="pathtracing", pathtracing_spp=spp,
                latency_aware=args.latency_aware, simulation_clock=simulation_clock(args), physics_backend="CPU PhysX, GPU RTX rendering",
                latency_controller=args.latency_controller if args.latency_aware else None,
                latency_source=latency_source(args) if args.latency_aware else None,
                event_replan=event_replan_metadata(args),
                decode_predicted_events=args.decode_predicted_events,
                predicted_event_threshold=args.predicted_event_threshold,
                predicted_event_min_neighbors=args.predicted_event_min_neighbors,
                profile_timing=args.profile_timing, interface_only=args.interface_only,
                trial_seed_rule="scene_seed + trial; model_seed = seed + trial * max_steps + control_step",
                success_criterion="Both objects intersect basket interior below rim with all clipped vertices contained; protrusion/stacking allowed; velocities below thresholds, TCP retreated, stable for collector stable_seconds",
                server_optimization={key: hello.get(key) for key in (
                    "compile_core_requested", "compile_core_active", "cache_context_kv", "warmup_iters")},
            )
            results = prepare_results(output, manifest, args)
            attempted_trials = len(results)
            summarize()
            for trial in range(len(results), args.trials):
                attempted_trials += 1
                scene_seed = args.scene_seed + trial
                print(f"[ewm-conveyor] trial={trial + 1}/{args.trials} reset seed={scene_seed}", flush=True)
                heartbeat.touch()
                env.reset(seed=scene_seed)
                for warmup_index in range(RESET_WARMUP_STEPS):
                    set_views(env)
                    env.step(zero)
                    if warmup_index % 10 == 9:
                        heartbeat.touch()
                runtime.reset(scene_seed)
                runtime._ewm_memory_history = None
                if hello.get("memory_history_required", False):
                    from scripts.eval_ewm.ewm_memory_history import RGBEventHistoryBuffer
                    runtime._ewm_memory_history = RGBEventHistoryBuffer()
                if speed_spec is not None:
                    set_episode_speed(runtime, sample_episode_speed(speed_spec, args.scene_seed, trial))
                runtime.supply()
                refresh_spawn_views(env)
                dvs.reset(torch.tensor([0], dtype=torch.long, device="cuda:0"))
                dvs.recorder._events.pop(0, None)
                dvs._conveyor_snapshot = snapshot(dvs, env)
                obs = observation(runtime, dvs, event_cameras)
                if np.any(obs["obs_voxel"]):
                    raise ValueError("First event frame must be zero after reset")
                poses, _, _ = runtime.state()
                initial = dict(trial=trial, scene_seed=scene_seed, scene=runtime.metadata(),
                               object_pose=poses.tolist(), joint_state=obs["proprio"].tolist())
                initial["scene"]["clock"] = simulation_clock(args)
                (output / f"trial_{trial:03d}_initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
                cv2.imwrite(str(output / f"trial_{trial:03d}_initial.png"), preview(obs, 0))
                rows, chunks, inference_times, round_trip_times, step_timings = [], [], [], [], []
                success, reason, control_step, replans = False, None, 0, 0
                suffix = "_pred_events" if args.decode_predicted_events else ""
                video_path = output / f"trial_{trial:03d}_h264{suffix}_recording.mp4"
                scene_label = (f"belt speed: {runtime.cfg.speed:.3f} m/s" if args.conveyor_speed is not None or speed_spec is not None else
                               "" if args.speed_ood == "none" else f"speed OOD {args.speed_ood}: {cfg.speed:.3f} m/s")
                video = TrialVideo(video_path, event_cameras, show_prediction=args.decode_predicted_events,
                                   scene_label=scene_label)
                trial_started = time.perf_counter()
                try:
                    video.write(obs, 0, 0)
                    if args.latency_aware:
                        (obs, success, reason, control_step, replans, inference_times,
                         round_trip_times, latency_timing) = run_latency_loop(
                            args, runtime, dvs, conn, hello, checkpoint_step, event_cameras,
                            obs, video, rows, chunks, step_timings, heartbeat, output, trial)
                    else:
                        while control_step < args.max_steps:
                            horizon = min(args.execution_horizon, args.max_steps - control_step)
                            simulation_time = runtime.time
                            request_started = time.perf_counter()
                            reply, values = policy_request(conn, args, dict(
                                op="predict", step=checkpoint_step,
                                seed=args.seed + trial * args.max_steps + control_step,
                                sampling_steps=args.sampling_steps, action_chunk_size=horizon,
                                prompt=PROMPT, event_time_contract=EVENT_TIME_CONTRACT,
                                observation_row_offset=0, raw_observation=True, timing=obs["timing"],
                                decode_future_events=args.decode_predicted_events,
                            ), {key: obs[key] for key in ("obs_video", "obs_voxel", "proprio") + tuple(k for k in obs if k.startswith("memory_"))})
                            round_trip_seconds = time.perf_counter() - request_started
                            if runtime.time != simulation_time:
                                raise RuntimeError("Simulation advanced during model inference")
                            actions, predicted, inference_seconds = validate_reply(
                                reply, values, horizon, event_cameras, args, int(hello.get("count_clip", 15)))
                            inference_times.append(inference_seconds)
                            round_trip_times.append(round_trip_seconds)
                            replans += 1
                            monitor = ConveyorEventMonitor(args)
                            observation_step = control_step
                            for index, action in enumerate(actions):
                                started = time.perf_counter()
                                command = torch.from_numpy(action.copy())[None].to(env.device)
                                obs, success, reason = capture_step(runtime, dvs, command, event_cameras)
                                capture_seconds = time.perf_counter() - started
                                control_step += 1
                                decision = dict(replan=replans - 1, action_source="policy",
                                                predicted_counts=None if predicted is None else predicted[index],
                                                prediction_target_step=observation_step + index + 1,
                                                executed_in_chunk=index + 1)
                                event_fields, trigger = monitor.inspect(obs, decision, control_step)
                                end_reason = ("success" if success else reason if reason else
                                              "step_budget" if control_step >= args.max_steps else
                                              "max_horizon" if index + 1 == horizon else
                                              "event_mismatch" if trigger else None)
                                if event_fields:
                                    event_fields["event_replan_triggered"] = end_reason == "event_mismatch"
                                started = time.perf_counter()
                                video.write(obs, control_step, runtime.tracker.steps,
                                            predicted_counts=None if predicted is None else predicted[index],
                                            replan_label=(f"error={event_fields['event_error']:.3f} {end_reason or ''}"
                                                          if event_fields.get("event_error") is not None else ""))
                                video_seconds = time.perf_counter() - started
                                row = dict(step=control_step, simulation_time_s=runtime.time,
                                           replan=replans - 1, chunk_index=index, action=action.tolist(),
                                           event_count=obs["event_counts"], timing=obs["timing"],
                                           stable_success_steps=runtime.tracker.steps, success=bool(success),
                                           failure_reason=reason, chunk_end_reason=end_reason,
                                           inference_seconds=inference_seconds, round_trip_seconds=round_trip_seconds)
                                row.update(event_fields)
                                if args.profile_timing:
                                    row["timing_seconds"] = dict(capture_total=capture_seconds, video_write=video_seconds)
                                    step_timings.append(row["timing_seconds"])
                                rows.append(row)
                                if control_step % 10 == 0 or end_reason is not None:
                                    (output / f"trial_{trial:03d}_steps.json").write_text(json.dumps(rows, indent=2) + "\n")
                                    heartbeat.touch()
                                    print(f"[ewm-conveyor] trial={trial + 1}/{args.trials} step={control_step} time={runtime.time:.2f} replan={replans} stable={runtime.tracker.steps} end={end_reason}", flush=True)
                                if end_reason is not None:
                                    chunks.append(dict(replan=replans - 1, executed_horizon=index + 1, end_reason=end_reason))
                                    break
                            if success or reason:
                                break
                finally:
                    video.close()
                final_video = output / f"trial_{trial:03d}_h264{suffix}_{'success' if success else 'failure'}.mp4"
                video_path.rename(final_video)
                subprocess.run([video.ffmpeg, "-v", "error", "-i", str(final_video), "-f", "null", "-"], check=True)
                cv2.imwrite(str(output / f"trial_{trial:03d}_terminal.png"), preview(obs, control_step))
                poses, velocities, tcp = runtime.state()
                result = dict(
                    trial=trial, sim_seed=scene_seed, scene_seed_mode="per_trial", steps=len(rows), frames=len(rows),
                    replans=replans, success=bool(success), reason=reason or ("success" if success else "step_limit"),
                    video=str(final_video), video_decoded=True, scene=runtime.metadata(),
                    initial_object_pose=initial["object_pose"], initial_joint_state=initial["joint_state"],
                    terminal_poses=poses.tolist(), terminal_velocities=velocities.tolist(), terminal_tcp=tcp.tolist(),
                    stability_steps=runtime.tracker.steps, trial_wall_seconds=time.perf_counter() - trial_started,
                    inference_timing=timing_stats(inference_times), round_trip_timing=timing_stats(round_trip_times),
                    execution_chunks=chunks, execution_horizon_statistics=horizon_statistics(chunks),
                )
                result.update(speed_meta)
                result["evaluation_conveyor_speed_m_s"] = runtime.cfg.speed
                result["basket_geometry_tolerance_m"] = BASKET_GEOMETRY_TOLERANCE_M
                result["basket_success_criterion_version"] = BASKET_SUCCESS_CRITERION
                result["training_mode"] = args.policy_mode
                result["event_input"] = args.policy_mode == "joint"
                result["event_replan"] = event_replan_metadata(args)
                result["event_replan_trigger_count"] = sum(bool(row.get("event_replan_triggered")) for row in rows)
                result["latency_aware"] = args.latency_aware
                result["latency_controller"] = args.latency_controller if args.latency_aware else None
                result["simulation_clock"] = simulation_clock(args)
                result["scene"]["clock"] = result["simulation_clock"]
                if args.latency_aware:
                    result["latency_timing"] = latency_timing
                if args.profile_timing:
                    result["step_timing"] = {name: timing_stats([item[name] for item in step_timings])
                                             for name in step_timings[0]}
                (output / f"trial_{trial:03d}_steps.json").write_text(json.dumps(rows, indent=2) + "\n")
                (output / f"trial_{trial:03d}_result.json").write_text(json.dumps(result, indent=2) + "\n")
                results.append(result)
                heartbeat.touch()
                summarize()
                print(f"[ewm-conveyor] result trial={trial} success={success} reason={result['reason']}", flush=True)
            successes = sum(result["success"] for result in results)
            report = dict(complete=True, task=TASK, prompt=PROMPT, **speed_meta,
                          basket_geometry_tolerance_m=BASKET_GEOMETRY_TOLERANCE_M,
                          basket_success_criterion_version=BASKET_SUCCESS_CRITERION,
                          training_mode=args.policy_mode, event_input=args.policy_mode == "joint",
                          interface_only=args.interface_only, interface_passed=True,
                          requested_trials=args.trials, trials=len(results), successes=successes,
                          success_rate=None if args.interface_only else successes / args.trials,
                          success_rate_denominator="requested_trials", checkpoint_step=checkpoint_step,
                          camera_calibration=camera_metadata(), scene_config=cfg.metadata(),
                          event_replan=event_replan_metadata(args),
                          event_replan_trigger_count=sum(r.get("event_replan_trigger_count", 0) for r in results),
                          latency_aware=args.latency_aware, simulation_clock=simulation_clock(args),
                          latency_controller=args.latency_controller if args.latency_aware else None,
                          video_decoded=all(result["video_decoded"] for result in results), results=results)
            (output / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
            # The submission wrapper owns server lifetime, including watchdog retries.
    except BaseException as exc:
        summarize(error=str(exc))
        raise
    finally:
        env.close()


def main(*, policy_mode="joint"):
    from isaaclab.app import AppLauncher
    parser = build_parser(AppLauncher, policy_mode=policy_mode)
    args = parser.parse_args()
    try:
        validate_args(args)
        dataset_dir, collection, cfg, spp = load_collection(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    app = AppLauncher(args).app
    try:
        run(args, dataset_dir, collection, cfg, spp)
    except BaseException as exc:
        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "failure.json").write_text(json.dumps(dict(
            type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()), indent=2) + "\n")
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
