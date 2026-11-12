#!/usr/bin/env python3
"""Export the eight immutable Piper v3 datasets as RGB-only LeRobot v2.1.

Only low-dimensional whitelisted columns are read. Per-episode RGB files are
symlinked, never re-encoded. Run with envs/lerobot/bin/python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TASKS = ('conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube', 'tennis_rolling', 'tennis_falling', 'tennis_parabolic', 'tennis_container')
COLUMNS = ("observation.state", "action", "timestamp", "frame_index",
           "episode_index", "index", "task_index")
CAMERAS = ("observation.images.cam0", "observation.images.cam_wrist")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def statistics(table):
    result = {}
    for key in COLUMNS:
        values = np.asarray(table[key].to_pylist(), dtype=np.float64).reshape(len(table), -1)
        result[key] = {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
                       "mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                       "q01": np.quantile(values, .01, axis=0).tolist(),
                       "q99": np.quantile(values, .99, axis=0).tolist(), "count": [len(values)]}
    return result


def fingerprint(source):
    # Include every source data/video/metadata file, without reading event payloads.
    return {str(p.relative_to(source)): [p.stat().st_size, p.stat().st_mtime_ns]
            for folder in ("meta", "data", "videos")
            for p in sorted((source / folder).rglob("*")) if p.is_file()}


def export_task(source, destination):
    import pyarrow as pa
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError(f"Unexpected source format/fps: {source}")
    if info["total_episodes"] != 50:
        raise ValueError(f"Expected 50 complete training episodes: {source}")
    source_fingerprint = fingerprint(source)
    records = []
    for file in sorted((source / "meta/episodes").rglob("*.parquet")):
        records.extend(pq.read_table(file).to_pylist())
    records.sort(key=lambda row: row["episode_index"])
    assert [r["episode_index"] for r in records] == list(range(50))
    tasks = pq.read_table(source / "meta/tasks.parquet").to_pylist()
    tables, episodes, episodes_stats, video_links = [], [], [], []
    # Group by source parquet file to avoid repeated I/O on event-heavy files.
    groups = {}
    for row in records:
        key = (row["data/chunk_index"], row["data/file_index"])
        groups.setdefault(key, []).append(row)
    for (chunk, file_index), group in groups.items():
        file = source / info["data_path"].format(chunk_index=chunk, file_index=file_index)
        table = pq.read_table(file, columns=list(COLUMNS))
        ids = np.asarray(table["episode_index"].to_pylist())
        for row in group:
            ep = int(row["episode_index"])
            indices = np.flatnonzero(ids == ep)
            part = table.take(pa.array(indices)).replace_schema_metadata(None)
            n = int(row["length"])
            assert len(part) == n and n > 24
            assert np.array_equal(part["frame_index"].to_pylist(), np.arange(n))
            assert np.asarray(part["action"].to_pylist()).shape == (n, 7)
            assert np.asarray(part["observation.state"].to_pylist()).shape == (n, 8)
            np.testing.assert_allclose(part["timestamp"].to_pylist(), np.arange(n) / 30, atol=1e-4)
            path = destination / f"data/chunk-000/episode_{ep:06d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(part, path)
            # Verify the actual persisted action/state/indices, including every row.
            assert pq.read_table(path).equals(part)
            tables.append((ep, part))
            episodes.append({"episode_index": ep, "tasks": row["tasks"], "length": n})
            episodes_stats.append({"episode_index": ep, "stats": statistics(part)})
            for camera in CAMERAS:
                prefix = f"videos/{camera}"
                start = float(row[f"{prefix}/from_timestamp"])
                end = float(row[f"{prefix}/to_timestamp"])
                if abs(start) > 1e-8 or abs(end - n / 30) > 1e-4:
                    raise ValueError(f"Video is not episode-local; refusing unsafe link: {source} ep={ep}")
                video = source / info["video_path"].format(
                    video_key=camera, chunk_index=row[f"{prefix}/chunk_index"],
                    file_index=row[f"{prefix}/file_index"])
                if not video.is_file():
                    raise FileNotFoundError(video)
                link = destination / f"videos/chunk-000/{camera}/episode_{ep:06d}.mp4"
                link.parent.mkdir(parents=True, exist_ok=True)
                # Both staging and final destination are at the same depth.
                link.symlink_to(os.path.relpath(video, link.parent))
                video_links.append(str(video))
    assert len(set(video_links)) == 100, "Shared videos require splitting, not symlinking"
    table = pa.concat_tables([t for _, t in sorted(tables)])
    assert len(table) == info["total_frames"]
    derived = {k: v for k, v in info.items() if k not in ("features", "data_files_size_in_mb", "video_files_size_in_mb")}
    derived.update(codebase_version="v2.1", total_chunks=1, total_videos=100,
                   data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                   video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                   features={k: info["features"][k] for k in (*COLUMNS, *CAMERAS)})
    write_json(destination / "meta/info.json", derived)
    write_jsonl(destination / "meta/episodes.jsonl", sorted(episodes, key=lambda r: r["episode_index"]))
    write_jsonl(destination / "meta/episodes_stats.jsonl", sorted(episodes_stats, key=lambda r: r["episode_index"]))
    write_jsonl(destination / "meta/tasks.jsonl", tasks)
    write_json(destination / "meta/stats.json", statistics(table))
    write_json(destination / "meta/modality.json", {
        "state": {"joints": {"start": 0, "end": 8}},
        "action": {"command": {"start": 0, "end": 7}},
        "video": {"front": {"original_key": CAMERAS[0]}, "wrist": {"original_key": CAMERAS[1]}},
        "annotation": {"human.task_description": {"original_key": "task_index"}}})
    assert fingerprint(source) == source_fingerprint, f"Source changed during export: {source}"
    write_json(destination / "meta/source_manifest.json", {
        "source": str(source), "source_files": source_fingerprint,
        "retained_columns": COLUMNS, "cameras": CAMERAS,
        "action_semantics": "Unmodified simulator IK-relative command (6) + gripper command (1).",
        "temporal_alignment": "observation[k] -> action[k:k+24]; RGB targets k+1 onward",
        "event_and_privileged_state_excluded": True})
    return table, {"task": source.name, "episodes": 50, "frames": len(table), "rgb_links": 100}


def main():
    import pyarrow as pa
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", nargs="+", default=TASKS,
                        help="Explicit task names; default is the current dynamic eight-task protocol")
    args = parser.parse_args()
    root = args.project_root.resolve()
    tasks = tuple(args.tasks)
    if len(tasks) != 8 or len(set(tasks)) != 8 or any('/' in t or t in ('.', '..') for t in tasks):
        raise ValueError("Expected eight distinct task directory suffixes")
    output = (args.output or root / "tmp/training_data_cache/piper_rgb_dynamic_v1").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    staging = output.with_name(output.name + ".building")
    staging.mkdir(parents=True, exist_ok=False)
    tables, reports = [], []
    for task in tasks:
        table, report = export_task(root / "datasets" / f"piper_{task}", staging / f"piper_{task}")
        tables.append(table)
        reports.append(report)
        print(json.dumps(report), flush=True)
    pooled = statistics(pa.concat_tables(tables))
    write_json(staging / "pooled_stats.json", pooled)
    # Native FastWAM uses this nested schema and global_* fields.
    fastwam = {name: {"default": {f"global_{k}": v for k, v in pooled[column].items() if k != "count"}}
               for name, column in (("state", "observation.state"), ("action", "action"))}
    fastwam.update(num_episodes=400, num_transition=sum(r["frames"] for r in reports))
    write_json(staging / "fastwam_stats.json", fastwam)
    # OpenWAM checkpoint exporter copies this artifact for inference normalization.
    np.save(staging / "openwam_normalization_stats.npy", {
        "eef": {k: np.asarray(v, dtype=np.float32) for k, v in pooled["action"].items() if k != "count"},
        "state": {k: np.asarray(v, dtype=np.float32) for k, v in pooled["observation.state"].items() if k != "count"}})
    write_json(staging / "manifest.json", {
        "format": "piper_rgb_baselines_v1", "tasks": reports, "episodes": 400,
        "frames": sum(r["frames"] for r in reports), "rgb_only": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    staging.rename(output)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
