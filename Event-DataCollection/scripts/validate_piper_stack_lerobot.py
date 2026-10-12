"""Validate a completed four-view, pre-action Piper stack LeRobot dataset."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from piper_wrist_camera import CALIBRATION_ID


CAMERAS = ("cam0", "cam_wrist")
CONTRACT = "aligned_event_frame_pre_action_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--require-collection-config", action="store_true")
    args = parser.parse_args()
    root = args.dataset.resolve()
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    timing = json.loads((meta / "piper_temporal_contract.json").read_text(encoding="utf-8"))
    calibration = json.loads((meta / "piper_camera_calibration.json").read_text(encoding="utf-8"))
    collection_path = meta / "piper_collection_config.json"
    require(not args.require_collection_config or collection_path.is_file(),
            "missing collection/randomization configuration")
    collection = json.loads(collection_path.read_text(encoding="utf-8")) if collection_path.is_file() else None
    require(info["total_episodes"] == args.episodes, "wrong episode count")
    require(info["fps"] == 30, "expected 30 Hz dataset")
    require(timing["temporal_contract"] == CONTRACT, "wrong pre-action temporal contract")
    require(timing["cameras"] == list(CAMERAS), "wrong temporal-contract camera list")
    require(calibration["calibration_id"] == CALIBRATION_ID, "wrong camera calibration")
    if collection is not None:
        require(collection["requested_successful_episodes"] == args.episodes, "wrong collection target")
    for camera in CAMERAS:
        require(info["features"][f"observation.images.{camera}"]["dtype"] == "video",
                f"missing {camera} RGB video feature")
        require(info["features"][f"observation.event_frame.{camera}"]["dtype"] == "image",
                f"missing {camera} lossless event-frame feature")

    episode_files = sorted((meta / "episodes").rglob("*.parquet"))
    require(bool(episode_files), "missing episode metadata")
    episode_columns = ["episode_index", "length"]
    for camera in CAMERAS:
        episode_columns += [f"videos/observation.images.{camera}/chunk_index",
                            f"videos/observation.images.{camera}/file_index"]
    episodes = {}
    for path in episode_files:
        for item in pq.read_table(path, columns=episode_columns).to_pylist():
            index = item["episode_index"]
            require(index not in episodes, f"duplicate episode metadata {index}")
            episodes[index] = item
            for camera in CAMERAS:
                prefix = f"videos/observation.images.{camera}"
                video = root / prefix / f"chunk-{item[f'{prefix}/chunk_index']:03d}" / f"file-{item[f'{prefix}/file_index']:03d}.mp4"
                require(video.is_file() and video.stat().st_size > 0, f"missing RGB video: {video}")
    require(set(episodes) == set(range(args.episodes)), "episode indices are incomplete")

    data_files = sorted((root / "data").rglob("*.parquet"))
    require(bool(data_files), "missing Parquet data")
    columns = ["episode_index", "frame_index", "timestamp", "observation.cube_pose",
               "observation.event_frame.cam0", "observation.event_frame.cam_wrist"]
    counts = {index: 0 for index in episodes}
    initial_layouts = set()
    total_rows = 0
    for path in data_files:
        for row in pq.read_table(path, columns=columns).to_pylist():
            index = row["episode_index"]
            require(index in counts, f"unknown episode {index}")
            frame = counts[index]
            require(row["frame_index"] == frame, f"episode {index}: unexpected frame index")
            require(abs(row["timestamp"] - frame / 30.0) < 1e-3,
                    f"episode {index}: timestamp is not aligned at 30 Hz")
            for camera in CAMERAS:
                payload = row[f"observation.event_frame.{camera}"]
                require(payload is not None and bool(payload["bytes"]),
                        f"episode {index}, frame {frame}: missing {camera} event-frame")
                if frame == 0:
                    event = np.asarray(Image.open(BytesIO(payload["bytes"])).convert("RGB"))
                    require(event.shape == (256, 256, 3) and not np.any(event),
                            f"episode {index}: {camera} initial event-frame must be zero")
            if frame == 0:
                cubes = np.asarray(row["observation.cube_pose"], dtype=np.float32).reshape(3, 7)
                initial_layouts.add(tuple(np.round(cubes[:, :3].reshape(-1), 3)))
            counts[index] += 1
            total_rows += 1
    for index, count in counts.items():
        require(count == episodes[index]["length"], f"episode {index}: metadata length mismatch")
        require(count >= 25, f"episode {index}: too short for EWM K=24")
    require(total_rows == info["total_frames"], "total frame count mismatch")
    require(len(initial_layouts) == args.episodes,
            "initial cube layouts are not unique at 1 mm resolution")
    report = {
        "status": "passed", "episodes": args.episodes, "frames": total_rows,
        "unique_initial_cube_layouts_1mm": len(initial_layouts),
        "rgb_cameras": list(CAMERAS), "event_frame_cameras": list(CAMERAS),
        "temporal_contract": CONTRACT, "camera_calibration_id": CALIBRATION_ID,
        "seed": collection["seed"] if collection is not None else None,
    }
    (meta / "piper_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
