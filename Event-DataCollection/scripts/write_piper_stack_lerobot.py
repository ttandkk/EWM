"""LeRobot writer subprocess for Isaac Sim's separate Python environment.

Protocol: 8-byte little-endian pickle length, followed by one trusted local
pickle payload. Only the collector launches this process and writes its stdin.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from piper_event_frames import COUNT_CLIP, EVENT_FRAME_KEY

TASK_PROMPT = "Stack the blue-and-yellow patterned block on the red-and-white block, then stack the green-and-white block on top."
CAMERAS = ("cam0", "cam_wrist")


def features() -> dict:
    result = {}
    for camera in CAMERAS:
        result[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"],
        }
        result[f"{EVENT_FRAME_KEY}.{camera}"] = {
            "dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"],
        }
    result.update({
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "observation.joint_vel": {"dtype": "float32", "shape": (8,), "names": None},
        "observation.ee_pose": {"dtype": "float32", "shape": (7,), "names": None},
        "observation.cube_pose": {"dtype": "float32", "shape": (21,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    })
    return result


def read_message():
    header = sys.stdin.buffer.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise EOFError("truncated writer message header")
    length = int.from_bytes(header, "little")
    if not 0 < length < 10_000_000:
        raise ValueError(f"invalid writer message length {length}")
    payload = sys.stdin.buffer.read(length)
    if len(payload) != length:
        raise EOFError("truncated writer message payload")
    return pickle.loads(payload)


def write_normalization_stats(dataset: LeRobotDataset) -> None:
    """Compute small state/action vectors from saved Parquet shards, including resumed episodes."""
    states, actions = [], []
    for path in sorted((dataset.root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        states.append(np.stack(table["observation.state"].to_pylist()).astype(np.float64))
        actions.append(np.stack(table["action"].to_pylist()).astype(np.float64))
    if not states:
        return
    state = np.concatenate(states)
    action = np.concatenate(actions)
    payload = {
        "version": 1,
        "source_split": "train",
        "source_stats": "data/*.parquet",
        "rgb": {"type": "backbone", "backbone": "dinov3"},
        "voxel": {
            "type": "fixed_count_clip", "count_clip": COUNT_CLIP, "num_bins": 1,
            "channel_layout": {"negative_bins": [0], "positive_bins": [1]},
        },
        "proprio": {"type": "z-score", "mean": state.mean(0).tolist(), "std": state.std(0).tolist()},
        "action": {
            "type": "quantile",
            "q01": np.quantile(action, 0.01, axis=0).tolist(),
            "q50": np.quantile(action, 0.50, axis=0).tolist(),
            "q99": np.quantile(action, 0.99, axis=0).tolist(),
            "degenerate_threshold": 1e-6,
        },
    }
    path = dataset.root / "meta" / "ewm_normalization_stats.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(*, task_prompt=TASK_PROMPT, feature_builder=features, contract_overrides=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/piper_stack_evis_event_frames")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    rgb_encoder = RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=20, g=30)
    contract = {
        "temporal_contract": "aligned_event_frame_pre_action_v1",
        "control_hz": 30,
        "event_hz": 240,
        "warp": 8,
        "event_source": "hdr",
        "cameras": list(CAMERAS),
        "resolution": [256, 256],
        "row": "RGB, state, action, event-frame at observation time t_i",
        "event_frame_window": "(t_(i-1), t_i]",
        "action_interval": "(t_i, t_(i+1)]",
        "event_frame_encoding": "lossless RGB PNG: R=OFF count, G=ON count, B=0; clipped at 15",
        "first_event_frame": "all zero after DVS reset",
    }
    if contract_overrides:
        contract.update(contract_overrides)
    if args.resume:
        contract_path = output / "meta" / "piper_temporal_contract.json"
        if not contract_path.is_file() or json.loads(contract_path.read_text()).get("temporal_contract") != contract["temporal_contract"]:
            raise ValueError(f"{output} is not an aligned event-frame dataset and cannot be resumed")
        if contract_overrides:
            saved_contract = json.loads(contract_path.read_text())
            for key, value in contract_overrides.items():
                if saved_contract.get(key) != value:
                    raise ValueError(f"resume contract differs for {key}: {saved_contract.get(key)!r} != {value!r}")
        dataset = LeRobotDataset.resume(args.repo_id, root=output, image_writer_threads=4,
                                        rgb_encoder=rgb_encoder)
        if any(dataset.meta.features.get(key, {}).get("dtype") != "image" for key in
               (f"{EVENT_FRAME_KEY}.{camera}" for camera in CAMERAS)):
            raise ValueError("existing LeRobot dataset lacks aligned event-frame image features")
    else:
        if output.exists():
            raise FileExistsError(f"{output} exists; use a fresh path or --resume")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, root=output, fps=30, robot_type="agilex_piper",
            features=feature_builder(), use_videos=True, image_writer_threads=4,
            video_files_size_in_mb=1e-6, rgb_encoder=rgb_encoder,
        )
    (output / "meta" / "piper_temporal_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"READY {dataset.meta.total_episodes}", flush=True)
    try:
        while (message := read_message()) is not None:
            if message["command"] == "frame":
                frame = message["frame"]
                frame["task"] = task_prompt
                dataset.add_frame(frame)
            elif message["command"] == "save":
                dataset.save_episode()
                print(f"SAVED {dataset.meta.total_episodes}", flush=True)
            else:
                raise ValueError(f"unknown writer command {message['command']!r}")
    finally:
        dataset.finalize()
        write_normalization_stats(dataset)


if __name__ == "__main__":
    main()
