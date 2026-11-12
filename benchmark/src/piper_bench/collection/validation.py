"""Audit task episodes, launch timing, physical terminal truth, RGB and events."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from piper_bench.tasks import load_task
from piper_bench.collection.protocol import OBJECT_COUNTS
from piper_bench.collection.protocol import check_attempts
from piper_bench.collection.protocol import episode_spec
from piper_bench.collection.protocol import make_plan
from piper_bench.collection.protocol import write_json
from piper_bench.collection.runtime import terminal_valid
from piper_bench.collection.sampling import TENNIS_TASKS
from piper_bench.collection.sampling import physics_hz


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_sampling(config, episodes, contract):
    name, plan = config["task"], config["plan"]
    require(config["version"] == plan["version"], "wrong collection protocol")
    require(plan == make_plan(name, episodes, plan["seed"], plan["split"], version=config["version"]),
            "sampling plan differs")
    require(contract["collection_protocol"] == config["version"], "writer protocol mismatch")


def validate(root, episodes):
    import pyarrow.parquet as pq
    root = Path(root)
    def read(name):
        return json.loads((root / name).read_text())
    info = read("meta/info.json")
    config = read("meta/piper_collection_config.json")
    contract = read("meta/piper_temporal_contract.json")
    attempts = read("meta/piper_task_attempts.json")
    name, plan = config["task"], config["plan"]
    task = load_task(name)
    validate_sampling(config, episodes, contract)
    require(info["total_episodes"] == episodes and info["fps"] == 30, "episode count or FPS mismatch")
    require(read(".collection_complete")["episodes"] == episodes, "collection incomplete")
    acceptance = read("meta/piper_acceptance.json")
    require(acceptance["passed"] and len(acceptance["results"]) == config["acceptance_trials"], "expert acceptance missing")
    require(sum(r["success"] for r in acceptance["results"]) >= config["acceptance_rate"]*config["acceptance_trials"],
            "expert acceptance rate below threshold")
    if config.get("boundary_preflight"):
        from piper_bench.tasks.push_t.protocol import boundary_plan
        boundary = read("meta/piper_boundary_acceptance.json")
        expected = boundary_plan(plan["seed"])
        require(name == "push_t" and boundary["plan"] == expected, "wrong boundary plan")
        require(boundary["passed"] and len(boundary["results"]) == 35
                and all(r["success"] for r in boundary["results"]), "boundary acceptance failed")
        require(all(r["spec"] == episode_spec(expected, i) for i, r in enumerate(boundary["results"])),
                "boundary initial states differ")
    require(contract["temporal_contract"] == "aligned_event_frame_pre_action_v1", "wrong temporal contract")
    require(contract["task_id"] == task.TASK and contract["object_count"] == OBJECT_COUNTS[name], "wrong task")
    require(config["physics_hz"] == contract["physics_hz"] == physics_hz(name), "wrong physics frequency")
    from piper_bench.collection.protocol import protocol_camera
    from piper_bench.protocols.dynamic_evaluation import numerical_equal
    recorded_camera = protocol_camera(name, plan, calibration_id=config["camera"]["calibration_id"])
    require(read("meta/piper_camera_calibration.json") == config["camera"]
            and numerical_equal(recorded_camera, config["camera"]), "camera mismatch")
    check_attempts(plan, attempts, episodes)
    saved = {a["saved_episode"]: a for a in attempts if a["success"]}
    for a in saved.values():
        require(terminal_valid(name, a["spec"]["config"], a["terminal"]), "physical terminal predicate failed")
        require(np.allclose(a["initial"]["object_poses"], a["spec"]["object_poses"], atol=1e-5), "initial pose mismatch")
        if name in TENNIS_TASKS:
            launch = a["initial"]["launch"]
            require(launch["launch_time_s"] == a["spec"]["launch_time_s"], "launch schedule mismatch")
            for key in ("launch_position_m", "launch_velocity"):
                require(np.allclose(launch[key], a["spec"][key], atol=1e-12, rtol=0),
                        "launch state mismatch: "+key)
    n = OBJECT_COUNTS[name]
    dims = {"state": 8, "joint_vel": 8, "ee_pose": 7, "object_pose": n*7,
            "object_velocity": n*6, "object_active": n, "sim_time": 1}
    if name == "drawer_cup":
        dims["drawer_state"] = 2
    if name == "tennis_container":
        dims.update(container_pose=7, container_velocity=6)
    for key, size in dims.items():
        require(info["features"]["observation."+key]["shape"] == [size], "invalid feature: " + key)
    require(info["features"]["action"]["shape"] == [7], "wrong action shape")
    cameras = ("cam0", "cam_wrist")
    archive = None
    archive_episode = None
    if config.get("event_archive"):
        from piper_bench.sensors.event_archive import CONTRACT
        from piper_bench.sensors.event_archive import ArchiveReader
        require(config["event_archive"] == contract.get("event_archive") == CONTRACT,
                "raw event temporal contract mismatch")
    counts = {i: 0 for i in range(episodes)}
    mass = {i: {cam: 0 for cam in cameras} for i in counts}
    for camera in cameras:
        require(info["features"]["observation.images."+camera]["dtype"] == "video", "RGB must be video")
        require(info["features"]["observation.event_frame."+camera]["dtype"] == "image", "events must be lossless images")
    for path in sorted((root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                require(episode in counts, "unknown episode index")
                k = counts[episode]
                if config.get("event_archive") and archive_episode != episode:
                    archive = ArchiveReader(root, episode, saved[episode]["frames"])
                    archive_episode = episode
                require(row["frame_index"] == k and abs(row["timestamp"]-k/30) < 1e-5, "frame timing mismatch")
                require(abs(float(np.asarray(row["observation.sim_time"]).reshape(-1)[0])-k/30) < 1e-5,
                        "sim time mismatch")
                for key, size in dims.items():
                    a = np.asarray(row["observation."+key])
                    require(a.size == size and np.isfinite(a).all(), "invalid values: " + key)
                require(len(row["action"]) == 7 and np.isfinite(row["action"]).all(), "invalid action")
                if name in TENNIS_TASKS:
                    delay = saved[episode]["spec"]["launch_time_s"]
                    expected_active = float(delay == 0 or k/30 > delay+1e-9)
                    require(np.all(np.asarray(row["observation.object_active"]) == expected_active),
                            "ball activation differs from scheduled launch")
                    if name == "tennis_container":
                        require(row["action"][-1] < 0, "container gripper opened")
                else:
                    require(np.all(np.asarray(row["observation.object_active"]) == 1.),
                            "objects not active from initial frame")
                if k == 0:
                    require(np.allclose(np.array(row["observation.object_pose"]).reshape(n, 7),
                                        saved[episode]["initial"]["object_poses"], atol=1e-5), "first observation differs from initial state")
                for cam in cameras:
                    payload = row["observation.event_frame."+cam]
                    with Image.open(BytesIO(payload["bytes"])) as image:
                        require(image.format == "PNG", "event payload must be PNG")
                        event = np.asarray(image.convert("RGB"))
                    require(event.shape == (256, 256, 3) and event.dtype == np.uint8, "invalid event dimensions")
                    require(event[:, :, :2].max() <= 15 and not event[:, :, 2].any(), "invalid OFF/ON counts")
                    if k == 0:
                        require(not event.any(), "first event frame not empty")
                    if archive is not None:
                        require(np.array_equal(event, archive.frame(cam, k)),
                                "raw events do not reconstruct stored observation frame")
                    mass[episode][cam] += int(event[:, :, :2].sum())
                counts[episode] += 1
    require(sum(counts.values()) == info["total_frames"], "total frames mismatch")
    for episode, count in counts.items():
        minimum = 16 if name in TENNIS_TASKS else 25
        require(count == saved[episode]["frames"] and count >= minimum, "episode length mismatch")
        require(mass[episode] == saved[episode]["event_mass"] and all(mass[episode].values()), "event mass mismatch")
    normal = read("meta/ewm_normalization_stats.json")
    require(normal["voxel"]["count_clip"] == 15 and normal["voxel"]["num_bins"] == 1, "event normalization mismatch")
    for group in ("action", "proprio"):
        for values in normal[group].values():
            if isinstance(values, list):
                require(np.isfinite(values).all(), "invalid normalization")
    from piper_bench.backend.video import find_ffmpeg
    ffmpeg = find_ffmpeg()
    videos = sorted((root / "videos").rglob("*.mp4"))
    require(len(videos) >= 2, "missing RGB videos")
    for video in videos:
        subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-f", "null", "-"], check=True, timeout=60)
    report = dict(passed=True, task=name, episodes=episodes, frames=sum(counts.values()),
                  videos_decoded=len(videos), failed_attempts=sum(not a["success"] for a in attempts),
                  speeds=[saved[i]["spec"]["speed"] for i in counts], event_mass=mass)
    report["raw_events_validated"] = bool(config.get("event_archive"))
    write_json(root / "meta/piper_validation.json", report)
    write_json(root / ".export_complete", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--episodes", type=int, required=True)
    args = p.parse_args(argv)
    print(json.dumps(validate(args.dataset, args.episodes)), flush=True)


if __name__ == "__main__":
    main()
