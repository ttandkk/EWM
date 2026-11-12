"""Unclipped EVIS events and causal 30 Hz visualization, alongside LeRobot rows."""
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np

from piper_bench.sensors.event_frames import events_to_frame

CAMERAS = ("cam0", "cam_wrist")
DTYPES = (np.uint16, np.uint16, np.float64, np.int8)
CONTRACT = dict(
    version="raw_xypt_with_aligned_frames_v1", raw_timestamp_unit="seconds",
    raw_fields=["x", "y", "t", "p"], raw_polarity=[-1, 1], raw_count_clipping=False,
    frame_hz=30, frame_window="(t_(k-1), t_k]", first_frame="empty at t=0",
    raw_path="events/{camera}/episode_{episode:06d}.npz",
    video_path="videos/observation.event_frame.{camera}/episode_{episode:06d}.mp4",
    video_encoding="H264 visualization: white background, OFF blue, ON red, overlap purple",
    training_frames="lossless OFF/ON count PNG in Parquet, clipped at 15",
    terminal_tail="raw events after the last recorded observation through final action end; not framed",
)


def checked_events(chunks, start, end):
    """Preserve every event; reject stale/future events instead of silently filtering."""
    if not chunks:
        return tuple(np.empty(0, dtype=d) for d in DTYPES)
    for chunk in chunks:
        if len(chunk) != 4 or any(np.asarray(a).ndim != 1 for a in chunk):
            raise ValueError("raw events require four 1-D arrays")
        x, y, t, p = map(np.asarray, chunk)
        if len({len(a) for a in chunk}) != 1:
            raise ValueError("raw event array lengths differ")
        if (not all(np.isfinite(a).all() for a in (x, y, t, p))
                or np.any(x < 0) or np.any(x >= 256) or np.any(y < 0) or np.any(y >= 256)
                or np.any(x != np.floor(x)) or np.any(y != np.floor(y))
                or not np.all((p == -1) | (p == 1))):
            raise ValueError("invalid raw coordinates, timestamps or polarity")
        if not np.all((t > start) & (t <= end)):
            raise ValueError(f"raw event outside causal window ({start}, {end}]")
    values = tuple(np.concatenate([c[i] for c in chunks]).astype(d, copy=False)
                   for i, d in enumerate(DTYPES))
    order = np.argsort(values[2], kind="stable")
    return tuple(a[order] for a in values)


def visualization(counts):
    off, on = counts[:, :, 0] > 0, counts[:, :, 1] > 0
    image = np.full(counts.shape, 255, dtype=np.uint8)
    image[off], image[on], image[off & on] = (45, 65, 220), (225, 45, 45), (160, 45, 180)
    return image


def paths(root, episode, camera):
    return (root / CONTRACT["raw_path"].format(camera=camera, episode=episode),
            root / CONTRACT["video_path"].format(camera=camera, episode=episode))


def write_episode(root, episode, frames, terminal_time):
    """Stage sidecars before the LeRobot save/ACK; resume can replace an unsaved episode."""
    root = Path(root)
    from piper_bench.backend.video import find_ffmpeg
    ffmpeg = find_ffmpeg()
    count = len(frames)
    if not count or abs(terminal_time-count/30) > 1e-9:
        raise ValueError("raw tail must end exactly one action after the last observation")
    report = dict(version=CONTRACT["version"], frames=count, raw_end_time_s=terminal_time, cameras={})
    for cam in CAMERAS:
        raw_path, video_path = paths(root, episode, cam)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        intervals, offsets = [], [0]
        for k, frame in enumerate(frames):
            if abs(float(np.asarray(frame["observation.sim_time"]).reshape(-1)[0])-k/30) > 1e-5:
                raise ValueError("event archive/RGB observation clock mismatch")
            raw = checked_events([frame["_raw_events"][cam]], (k-1)/30, k/30)
            if k == 0 and len(raw[2]):
                raise ValueError("initial event frame must be empty")
            expected = events_to_frame([raw], (k-1)/30, k/30)
            if not np.array_equal(expected, frame["observation.event_frame."+cam]):
                raise ValueError("raw events do not reconstruct the saved count frame")
            intervals.append(raw)
            offsets.append(offsets[-1]+len(raw[2]))
        tail = checked_events([frames[-1]["_raw_event_tail"][cam]], (count-1)/30, terminal_time)
        intervals.append(tail)
        arrays = dict(zip(("x", "y", "t", "p"), (
            np.concatenate([part[i] for part in intervals]) for i in range(4))))
        raw_temp = raw_path.with_suffix(".partial.npz")
        np.savez_compressed(raw_temp, **arrays, frame_offsets=np.asarray(offsets, dtype=np.int64),
                            frame_times_s=np.arange(count, dtype=np.float64)/30,
                            raw_end_time_s=np.asarray(terminal_time, dtype=np.float64))
        raw_temp.replace(raw_path)
        video_temp = video_path.with_suffix(".partial.mp4")
        proc = subprocess.Popen([
            ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "256x256", "-r", "30", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_temp),
        ], stdin=subprocess.PIPE)
        try:
            for frame in frames:
                proc.stdin.write(visualization(frame["observation.event_frame."+cam]).tobytes())
            proc.stdin.close()
            if proc.wait(timeout=60):
                raise RuntimeError("event video encoding failed")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        video_temp.replace(video_path)
        report["cameras"][cam] = dict(raw_events=len(arrays["t"]), framed_events=offsets[-1],
                                     terminal_tail_events=len(tail[2]))
    meta = root / "meta/event_archives" / f"episode_{episode:06d}.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    temp = meta.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2)+"\n")
    temp.replace(meta)
    return report


class ArchiveReader:
    """Validate exact integer-tick windows and reconstruct frames from raw events."""
    def __init__(self, root, episode, count):
        root = Path(root)
        self.count = count
        self.data = {}
        report = json.loads((root / "meta/event_archives" / f"episode_{episode:06d}.json").read_text())
        if report["version"] != CONTRACT["version"] or report["frames"] != count:
            raise ValueError("event archive metadata mismatch")
        for cam in CAMERAS:
            raw_path, video = paths(root, episode, cam)
            with np.load(raw_path, allow_pickle=False) as source:
                d = {key: source[key] for key in source.files}
            arrays = tuple(d[k] for k in ("x", "y", "t", "p"))
            if any(a.dtype != dtype for a, dtype in zip(arrays, DTYPES)):
                raise ValueError("raw event dtypes differ")
            checked_events([arrays], 0., count/30)
            if np.any(np.diff(d["t"]) < 0):
                raise ValueError("raw events are not time sorted")
            offsets = d["frame_offsets"]
            if (offsets.dtype != np.int64 or offsets.shape != (count+1,)
                    or offsets[0] != 0 or offsets[1] != 0
                    or np.any(np.diff(offsets) < 0) or offsets[-1] > len(d["t"])
                    or not np.array_equal(d["frame_times_s"], np.arange(count)/30)
                    or float(d["raw_end_time_s"]) != count/30
                    or report["raw_end_time_s"] != count/30):
                raise ValueError("raw frame offsets or timestamps differ")
            # searchsorted enforces (previous tick, current tick], including exact boundaries.
            expected = np.r_[0, np.searchsorted(d["t"], np.arange(count)/30, side="right")]
            if not np.array_equal(offsets, expected):
                raise ValueError("events assigned to the wrong observation window")
            stats = report["cameras"][cam]
            if (stats["raw_events"] != len(d["t"]) or stats["framed_events"] != offsets[-1]
                    or stats["terminal_tail_events"] != len(d["t"])-offsets[-1]):
                raise ValueError("raw event counts differ")
            self.data[cam] = d
            import av
            with av.open(str(video)) as container:
                seen = 0
                for frame in container.decode(video=0):
                    if frame.pts is None or abs(float(frame.pts*frame.time_base)-seen/30) > 1e-6:
                        raise ValueError("event video PTS differs from RGB/observation clock")
                    if (frame.width, frame.height) != (256, 256):
                        raise ValueError("event video dimensions differ")
                    seen += 1
                if seen != count:
                    raise ValueError("event video frame count differs from observations")

    def frame(self, camera, k):
        d = self.data[camera]
        start, end = d["frame_offsets"][k:k+2]
        return events_to_frame([tuple(d[key][start:end] for key in ("x", "y", "t", "p"))],
                               (k-1)/30, k/30)
