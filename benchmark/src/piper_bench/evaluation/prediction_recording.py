"""Bounded per-trial latent recording and post-trial prediction video rendering.

Actions never wait for a decoder. Only small latent copies are charged to the
measured action reply; decoding and video encoding happen after control stops.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np


class PredictionArchive:
    """One evaluator owns this server; a new trial also clears an interrupted one."""

    def __init__(self, max_bytes=1024**3):
        self.max_bytes = max_bytes
        self.clear()

    def clear(self):
        self.session = None
        self.records = {}
        self.bytes = self.peak_bytes = self.count = 0

    def _check(self, request):
        if not self.session or request.get("session") != self.session:
            raise ValueError("prediction recording session differs")

    def record(self, request, result):
        spec = request.get("record_prediction")
        if spec is None:
            return False
        self._check(spec)
        anchor = spec.get("observation_step")
        if type(anchor) is not int or anchor < 0 or anchor in self.records:
            raise ValueError("invalid or duplicate prediction observation step")
        tensors = [result[k] for k in ("observed_latent", "future_latent")]
        size = sum(t.numel() * t.element_size() for t in tensors)
        if self.bytes + size > self.max_bytes:
            raise MemoryError("prediction archive exceeds 1 GiB; shorten diagnostic trials or disable recording")
        # CPU copies own their storage, including when the next CUDA Graph replay
        # reuses a tensor. Do not retain a GPU tensor across requests.
        observed, future = [t.detach().to(device="cpu", copy=True) for t in tensors]
        self.records[anchor] = (observed, future, size, dict(
            source_observation_step=anchor, seed=request["seed"], checkpoint_step=request["step"],
            sampling_steps=request["sampling_steps"], observation_time_s=request["timing"]["obs_video_time_s"]))
        self.bytes += size
        self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.count += 1
        return True

    def handle(self, request, model):
        import torch

        op = request["op"]
        if op == "visualization_begin":
            session = request.get("session")
            if not isinstance(session, str) or not 1 <= len(session) <= 128:
                raise ValueError("invalid prediction recording session")
            self.clear()
            self.session = session
            return dict(ok=True, session=session), {}
        self._check(request)
        if op == "visualization_end":
            stats = dict(ok=True, recorded_predictions=self.count, peak_archive_bytes=self.peak_bytes)
            self.clear()
            return stats, {}
        if op != "visualization_decode":
            raise ValueError(f"unknown prediction recording operation: {op}")
        anchor = request.get("observation_step")
        if type(anchor) is not int or anchor not in self.records:
            raise ValueError("prediction observation step was not recorded")
        observed, future, size, metadata = self.records[anchor]
        device = next(model.parameters()).device
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                   enabled=device.type == "cuda"):
            prediction = model._decode_future(observed.to(device), future.to(device))[0]
        if prediction.ndim != 5 or not torch.isfinite(prediction).all():
            raise ValueError("invalid post-trial prediction")
        rgb = model.training_mode == "rgb_action"
        if rgb:
            value = (prediction.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        else:
            value = prediction.float().cpu().numpy()
        del self.records[anchor]
        self.bytes -= size
        return dict(ok=True, **metadata, prediction_kind="rgb" if rgb else "events",
                    prediction_frames=len(value), decode_seconds=time.perf_counter()-started), {
                        "future_rgb" if rgb else "future_voxel": value}


def recording_enabled(args):
    return bool(getattr(args, "record_predictions", False) and
                getattr(args, "policy_mode", "joint") in ("joint", "rgb_action"))


def recording_metadata(args):
    enabled = recording_enabled(args)
    inline = bool(not enabled and getattr(args, "decode_predicted_events", False))
    return dict(record_predictions=enabled,
                prediction_decode_phase="post_trial" if enabled else "inline" if inline else "disabled",
                prediction_kind=("rgb" if args.policy_mode == "rgb_action" else "events") if enabled or inline else None,
                decode_predicted_rgb=enabled and args.policy_mode == "rgb_action",
                prediction_decode_in_equivalent_latency=inline)


def begin_recording(args, conn, hello, request):
    if not recording_enabled(args):
        return
    if hello.get("post_trial_visualization_supported") is not True:
        raise ValueError("server lacks post-trial prediction recording; update it or use --no-record-predictions")
    session = uuid.uuid4().hex
    request(conn, dict(op="visualization_begin", session=session))
    args._prediction_session = session


def end_recording(args, conn, request):
    session = vars(args).pop("_prediction_session", None)
    if session is None:
        return {}
    reply, _ = request(conn, dict(op="visualization_end", session=session))
    return {k: reply[k] for k in ("recorded_predictions", "peak_archive_bytes")}


def prediction_target(row):
    """Align to elapsed observation time, even when serial control delayed an action."""
    if row.get("action_source", "policy") != "policy" or row.get("chunk_index") is None:
        return None
    action_index = row["chunk_index"]
    anchor = row.get("source_observation_step", row["step"]-action_index-1)
    index = row["step"]-anchor-1
    if type(action_index) is not int or action_index < 0 or type(anchor) is not int or anchor < 0 or index < 0:
        raise ValueError("prediction/video time alignment differs from action timing")
    return anchor, index


def prediction_row(prediction, *, kind, cameras, label):
    import cv2

    images = {}
    for name, value in zip(cameras, prediction if prediction is not None else (), strict=prediction is not None):
        if kind == "rgb":
            image = cv2.cvtColor(value.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        else:
            image = np.full((256, 256, 3), 255, np.uint8)
            image[value[0] > 0] = (255, 0, 0)
            image[value[1] > 0] = (0, 0, 255)
        images[name] = image
    panels = []
    for name in ("cam0", "cam_wrist"):
        image = images.get(name, np.full((256, 256, 3), 255, np.uint8))
        cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
        title = f"pred {name} {label}" if name in images else f"pred {name}: unavailable"
        cv2.putText(image, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)
    return np.hstack(panels)


def append_prediction_video(path, rows, *, ffmpeg, decode, kind, cameras, heartbeat):
    """Append one prediction row atomically; the existing real video stays on error."""
    import cv2
    import subprocess

    path = Path(path)
    target = path.with_name(path.stem + "_visualization.mp4")
    if target.exists():
        raise FileExistsError(target)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"cannot read evaluation video {path}")
    writer = None
    plans, current_anchor, forecast = [], None, None
    started = time.perf_counter()
    try:
        writer = subprocess.Popen([ffmpeg, "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", "512x768", "-framerate", "30", "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            stdin=subprocess.PIPE)
        for step, row in enumerate([None, *rows]):
            heartbeat.touch()
            ok, frame = capture.read()
            if not ok or frame.shape != (512, 512, 3):
                raise ValueError("real video frame count or dimensions differ from trajectory")
            if row is not None and row["step"] != step:
                raise ValueError("trajectory steps are not contiguous")
            mapping = None if row is None else prediction_target(row)
            prediction, label = None, "pending"
            if mapping is not None:
                anchor, index = mapping
                if anchor != current_anchor:
                    forecast, metadata = decode(anchor)
                    plans.append(metadata)
                    current_anchor = anchor
                label = f"obs={anchor} t={step}"
                # Short asymmetric event horizons must not repeat their last frame.
                if index < len(forecast):
                    prediction = forecast[index]
            panel = prediction_row(prediction, kind=kind, cameras=cameras, label=label)
            writer.stdin.write(np.ascontiguousarray(np.vstack((frame, panel))).tobytes())
        if capture.read()[0]:
            raise ValueError("real video has more frames than trajectory")
        writer.stdin.close()
        if writer.wait(timeout=120) != 0:
            raise RuntimeError("prediction video encoder failed")
        subprocess.run([ffmpeg, "-v", "error", "-i", str(target), "-f", "null", "-"], check=True, timeout=120)
        target.replace(path)
        return dict(decoded_predictions=len(plans), plans=plans,
                    post_trial_wall_seconds=time.perf_counter()-started,
                    video_layout="real_rgb_over_observed_events_over_predicted_"+kind)
    finally:
        capture.release()
        if writer is not None:
            if writer.stdin and not writer.stdin.closed:
                writer.stdin.close()
            if writer.poll() is None:
                writer.terminate()
                try:
                    writer.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    writer.kill()
                    writer.wait()


def finish_recording(args, conn, hello, request, video, rows, heartbeat):
    if not getattr(args, "_prediction_session", None):
        return None
    from .event_visualization import predicted_event_counts

    kind = "rgb" if args.policy_mode == "rgb_action" else "events"
    count = hello.get("rgb_cameras") if kind == "rgb" else hello.get("event_cameras")
    cameras = ("cam0", "cam_wrist") if count == 2 else ("cam_wrist",)
    session = args._prediction_session

    def decode(anchor):
        reply, arrays = request(conn, dict(op="visualization_decode", session=session, observation_step=anchor))
        if reply.get("source_observation_step") != anchor or reply.get("prediction_kind") != kind:
            raise ValueError("post-trial prediction metadata differs")
        value = arrays["future_rgb" if kind == "rgb" else "future_voxel"]
        if (value.ndim != 5 or value.shape[1:] != (len(cameras), 3 if kind == "rgb" else 2, 256, 256)
                or not len(value) or not np.isfinite(value).all()):
            raise ValueError("invalid post-trial prediction shape or values")
        if kind == "events":
            value = predicted_event_counts(value, count_clip=int(hello.get("count_clip", 15)),
                occupancy_threshold=args.predicted_event_threshold, min_neighbors=args.predicted_event_min_neighbors)
        elif value.dtype != np.uint8:
            raise ValueError("post-trial RGB must be uint8")
        return value, {k: v for k, v in reply.items() if k != "ok"}

    try:
        result = append_prediction_video(video.path, rows, ffmpeg=video.ffmpeg, decode=decode,
            kind=kind, cameras=cameras, heartbeat=heartbeat)
        result.update(recording_metadata(args))
    finally:
        stats = end_recording(args, conn, request)
    result.update(stats)
    return result
