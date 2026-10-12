"""Convert single-view Piper EVIS HDF5 episodes into a local LeRobot dataset.

The raw event stream intentionally remains outside LeRobot's tabular/video
store.  For each exported frame ``i`` (the source's first and last frame are
discarded), ``event_past_offsets[i]`` selects ``[t_i-dt, t_i)`` and
``event_future_offsets[i]`` selects ``[t_i, t_i+dt)`` from ``events.npy``.
This supports past-event to future-event prediction without duplicating raw
events per frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK_PROMPT = "Stack the blue-and-yellow patterned block on the red-and-white block, then stack the green-and-white block on top."
EVENT_DTYPE = np.dtype([( "x", np.uint8), ("y", np.uint8), ("t", np.float64), ("p", np.int8)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source Piper HDF5 episode or directory of episode_*.h5 files.")
    parser.add_argument("--output", type=Path, required=True, help="New local LeRobot dataset directory.")
    parser.add_argument("--repo-id", default="local/piper_stack_evis", help="Local LeRobot repository identifier.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def features(height: int, width: int) -> dict:
    return {
        "observation.images.cam0": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "observation.joint_vel": {"dtype": "float32", "shape": (8,), "names": None},
        "observation.ee_pose": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.cube_pose": {"dtype": "float32", "shape": (21,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }


def event_offsets(event_t: np.ndarray, frame_t: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return past and future [start, end) offsets for every retained frame."""
    past = np.empty((len(frame_t), 2), dtype=np.int64)
    future = np.empty((len(frame_t), 2), dtype=np.int64)
    for row, timestamp in enumerate(frame_t):
        past[row] = np.searchsorted(event_t, (timestamp - dt, timestamp), side="left")
        future[row] = np.searchsorted(event_t, (timestamp, timestamp + dt), side="left")
    return past, future


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    sources = [input_path] if input_path.is_file() else sorted(input_path.glob("episode_*.h5"))
    if not sources:
        raise FileNotFoundError(f"No episode_*.h5 files found at {input_path}")
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    with h5py.File(sources[0], "r") as first:
        control_hz = int(first["meta"].attrs["control_hz"])
        height, width = first["observation/rgb/cam0"].shape[1:3]
        if first["observation/rgb/cam_wrist"].shape[1:3] != (height, width):
            raise ValueError("cam0 and cam_wrist resolutions must match")
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=output, fps=control_hz, robot_type="agilex_piper", features=features(height, width), use_videos=True, image_writer_threads=4, video_files_size_in_mb=1e-6)
    try:
        for episode_id, source in enumerate(sources):
            with h5py.File(source, "r") as h5:
                meta = h5["meta"].attrs
                timestamps = h5["observation/t"][:].astype(np.float64)
                if len(timestamps) < 3 or int(meta["control_hz"]) != control_hz:
                    raise ValueError(f"{source.name}: incompatible timestamps or control_hz")
                dt = 1.0 / control_hz
                event_hz = int(meta["event_hz"])
                if not np.allclose(np.diff(timestamps), dt, rtol=0.0, atol=1e-5):
                    raise ValueError(f"{source.name}: timestamps are not uniformly sampled")
                retained = np.arange(1, len(timestamps) - 1, dtype=np.int64)
                retained_t = timestamps[retained]
                rgb_cam0 = h5["observation/rgb/cam0"]
                rgb_wrist = h5["observation/rgb/cam_wrist"]
                if rgb_cam0.shape[1:3] != (height, width) or rgb_wrist.shape[1:3] != (height, width):
                    raise ValueError(f"{source.name}: camera resolution differs")
                event_payloads = {}
                for camera_name in ("cam0", "cam_wrist"):
                    event_x = h5[f"events/{camera_name}/x"][:]
                    event_y = h5[f"events/{camera_name}/y"][:]
                    event_t = h5[f"events/{camera_name}/t"][:].astype(np.float64)
                    event_p = h5[f"events/{camera_name}/p"][:]
                    if np.any(np.diff(event_t) < 0):
                        raise ValueError(f"{source.name}: {camera_name} event timestamps are not sorted")
                    event_payloads[camera_name] = (event_x, event_y, event_t, event_p, *event_offsets(event_t, retained_t, dt))
                for source_index in retained:
                    dataset.add_frame({"observation.images.cam0": rgb_cam0[source_index], "observation.images.cam_wrist": rgb_wrist[source_index], "observation.state": h5["observation/joint_pos"][source_index], "observation.joint_vel": h5["observation/joint_vel"][source_index], "observation.ee_pose": h5["observation/ee_pose"][source_index], "observation.cube_pose": h5["observation/cube_pose"][source_index].reshape(-1), "action": h5["action"][source_index], "task": TASK_PROMPT})
                dataset.save_episode()
            for camera_name, (event_x, event_y, event_t, event_p, past_offsets, future_offsets) in event_payloads.items():
                event_dir = output / "events" / f"episode_{episode_id:06d}" / camera_name
                event_dir.mkdir(parents=True, exist_ok=True)
                event_metadata = {"source_hdf5": str(source), "episode_id": episode_id, "camera_name": camera_name, "resolution": [int(width), int(height)], "control_hz": control_hz, "event_hz": event_hz, "window_ms": 1000.0 * dt, "event_columns": ["x", "y", "timestamp_s_relative_to_first_exported_frame", "polarity"], "source_event_dtypes": {"x": "uint16", "y": "uint16", "t": "float64", "p": "int8"}, "stored_event_dtype": {"x": "uint8", "y": "uint8", "t": "float64", "p": "int8"}, "retained_source_indices": [int(retained[0]), int(retained[-1])], "event_past_window": "[t_i - dt, t_i)", "event_future_window": "[t_i, t_i + dt)"}
                if event_x.max(initial=0) > 255 or event_y.max(initial=0) > 255:
                    raise ValueError(f"{source.name}: {camera_name} event coordinates exceed uint8 range")
                compact_events = np.empty(len(event_t), dtype=EVENT_DTYPE)
                compact_events["x"] = event_x
                compact_events["y"] = event_y
                compact_events["t"] = event_t - retained_t[0]
                compact_events["p"] = event_p
                np.save(event_dir / "events.npy", compact_events)
                np.save(event_dir / "event_past_offsets.npy", past_offsets)
                np.save(event_dir / "event_future_offsets.npy", future_offsets)
                (event_dir / "metadata.json").write_text(json.dumps(event_metadata, indent=2) + "\n")
            event_counts = ", ".join(f"{name}={len(payload[2])}" for name, payload in event_payloads.items())
            print(f"Converted {episode_id + 1}/{len(sources)}: {source.name} ({len(retained)} frames; {event_counts})", flush=True)
        dataset.finalize()
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    print(f"Converted {len(sources)} episodes -> {output}")


if __name__ == "__main__":
    main()
