"""Measured-latency controllers; simulation clock and expired-action semantics are preserved."""
import math
import time
import numpy as np
from .constants import CONTROL_HZ

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
        copied = {key: observation[key].copy() for key in ("obs_video", "obs_voxel", "proprio")}
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
                    predicted_counts=None if predicted is None or index >= len(predicted) else predicted[index],
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
        copied = {key: observation[key].copy() for key in ("obs_video", "obs_voxel", "proprio")}
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
        switching_next = self._switching_next(step + 1)
        predicted = packet["predicted"]
        return dict(action=action, action_source="policy", replan=metadata["replan"], chunk_index=index,
                    chunk_last=switching_next or index + 1 == len(packet["actions"]),
                    executed_in_chunk=metadata["executed_actions"],
                    predicted_counts=None if predicted is None or index >= len(predicted) else predicted[index],
                    prediction_target_step=metadata["observation_step"] + index + 1,
                    event_prediction_ready_step=metadata["event_forecast_ready_step"],
                    planned_end_reason=metadata.get("next_plan_reason", "max_horizon"),
                    observation_step=metadata["observation_step"],
                    inference_seconds=packet["inference_seconds"], round_trip_seconds=packet["round_trip_seconds"])

    def _switching_next(self, step):
        return (self.pending is not None and step >= self.pending["metadata"]["scheduled_ready_step"]
                and step - self.pending["metadata"]["observation_step"] < len(self.pending["actions"]))

    def close(self, step):
        if self.pending is not None:
            metadata = self.pending["metadata"]
            metadata.update(reply_discarded_at_terminal=True, terminal_step=step)
            self.pending = None


class ContinuousLatencyActionStream(BufferedLatencyActionStream):
    """Keep inferring while each active plan executes a minimum number of actions."""

    def __init__(self, predict, *, plan_horizon, max_steps, min_execution_steps=4):
        if type(min_execution_steps) is not int or min_execution_steps < 1:
            raise ValueError("min_execution_steps must be a positive integer")
        super().__init__(predict, horizon=0, plan_horizon=plan_horizon, max_steps=max_steps)
        self.min_execution_steps = min_execution_steps
        self.candidate = None

    def ensure_request(self, observation, step):
        if step < self.max_steps:
            super().ensure_request(observation, step)

    def receive(self, step):
        if self.pending is not None and step >= self.pending["metadata"]["scheduled_ready_step"]:
            packet, self.pending = self.pending, None
            metadata = packet["metadata"]
            offset = step - metadata["observation_step"]
            metadata.update(received_step=step, discarded_prefix_steps=min(offset, len(packet["actions"])),
                            simulation_steps_during_inference=offset, executed_actions=0)
            # A deferred visualization occupies the worker after action delivery.
            self.next_request_step = max(step, metadata["observation_step"] + 1,
                                         metadata["request_pipeline_ready_step"])
            if offset >= len(packet["actions"]):
                metadata.update(reply_expired=True, discarded_at_step=step)
            else:
                if self.candidate is not None:
                    self.candidate["metadata"].update(reply_superseded=True, superseded_at_step=step,
                                                      superseded_by=metadata["replan"])
                self.candidate = packet

        if self.packet is not None:
            active = self.packet["metadata"]
            if step - active["action_anchor_step"] >= len(self.packet["actions"]):
                if active["executed_actions"] < self.min_execution_steps:
                    active["min_execution_unmet_reason"] = "plan_exhausted"
                self.packet = None
        if self.candidate is not None:
            candidate = self.candidate["metadata"]
            offset = step - candidate["observation_step"]
            if offset >= len(self.candidate["actions"]):
                candidate.update(reply_expired=True, discarded_at_step=step,
                                 discarded_prefix_steps=len(self.candidate["actions"]))
                self.candidate = None
            elif (self.packet is None or
                  self.packet["metadata"]["executed_actions"] >= self.min_execution_steps):
                candidate.update(action_anchor_step=candidate["observation_step"],
                                 discarded_prefix_steps=offset, bootstrap=not self.bootstrapped,
                                 next_plan_reason="new_prediction", activated_step=step)
                self.packet, self.candidate, self.bootstrapped = self.candidate, None, True

    def _switching_next(self, step):
        if self.packet is None or self.packet["metadata"]["executed_actions"] < self.min_execution_steps:
            return False
        candidates = (self.candidate, self.pending)
        return any(p is not None and step >= p["metadata"]["scheduled_ready_step"]
                   and step - p["metadata"]["observation_step"] < len(p["actions"])
                   for p in candidates)

    def decision(self, step):
        decision = super().decision(step)
        if decision["action_source"] == "policy" and decision["executed_in_chunk"] == 1:
            self.packet["metadata"].update(first_action_step=step,
                first_action_observation_age_steps=step - decision["observation_step"])
        if (decision["action_source"] == "policy" and decision["chunk_last"]
                and decision["chunk_index"] + 1 >= len(self.packet["actions"])
                and not self._switching_next(step + 1)):
            decision["planned_end_reason"] = "max_horizon"
        return decision

    def close(self, step):
        if self.packet is not None:
            active = self.packet["metadata"]
            if 0 < active["executed_actions"] < self.min_execution_steps:
                active.setdefault("min_execution_unmet_reason", "terminal")
        if self.candidate is not None:
            self.candidate["metadata"].update(reply_discarded_at_terminal=True, terminal_step=step)
            self.candidate = None
        super().close(step)
