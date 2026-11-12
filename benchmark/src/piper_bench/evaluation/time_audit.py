"""Check the exported Piper event-frame and legacy sparse-event time contracts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVENT_TIME_CONTRACT = "aligned_event_frame_pre_action_v1"


def verify_exported_event_alignment(dataset, indices, horizon: int) -> list[dict]:
    if horizon != getattr(dataset, 'future_horizon', dataset.chunk_size) or horizon <= 0:
        raise ValueError("event alignment audit requires the training horizon")
    if dataset.direct_event_frames:
        contract_path = Path(dataset.dataset_dir) / "meta" / "piper_temporal_contract.json"
        contract = json.loads(contract_path.read_text())
        if contract.get("temporal_contract") != EVENT_TIME_CONTRACT:
            raise ValueError(f"unsupported event-frame time contract: {contract_path}")
        if dataset.observation_row_offset != 0:
            raise ValueError("direct event-frame observations must use row offset 0")
        return _verify_direct_event_frames(dataset, indices, horizon)

    records = []
    for index in indices:
        ep_local = int(np.searchsorted(dataset._cumulative, index, side="right") - 1)
        if not 0 <= ep_local < len(dataset.episodes):
            raise IndexError(index)
        row = dataset.episodes.iloc[ep_local]
        episode = int(row["episode_index"])
        k = dataset.first_anchor + int(index - dataset._cumulative[ep_local]) * dataset.window_stride
        for camera in dataset.event_cameras:
            path = (Path(dataset.dataset_dir) / "events" / f"episode_{episode:06d}" /
                    camera / "event_future_offsets.npy")
            exported = np.load(path, mmap_mode="r", allow_pickle=False)
            if exported.ndim != 2 or exported.shape[1] != 2:
                raise ValueError(f"invalid event offsets: {path}")
            actual = dataset._event_offsets_for(episode, camera, int(row["length"]))
            # Frame 0 has no preceding exported future interval; all other
            # observation and target frames must match exactly.
            first = max(k, 1)
            last = k + horizon
            if last >= len(actual) or last - 1 >= len(exported):
                raise IndexError(f"event alignment indices exceed episode {episode}")
            if not np.array_equal(actual[first:last + 1], exported[first - 1:last]):
                bad = np.flatnonzero(np.any(actual[first:last + 1] != exported[first - 1:last], axis=1))[0]
                frame = first + int(bad)
                raise ValueError(
                    f"event time mismatch: dataset index {index}, episode {episode}, "
                    f"camera {camera}, frame {frame}: loader={actual[frame].tolist()} "
                    f"export={exported[frame - 1].tolist()}"
                )
            records.append({"index": int(index), "episode": episode, "anchor": k,
                            "observation_row": k + dataset.observation_row_offset,
                            "camera": camera, "verified_frames": last - first + 1,
                            "obs_verified": k > 0})
    return records


def _verify_direct_event_frames(dataset, indices, horizon: int) -> list[dict]:
    """Check the declared contract and actual Parquet row/timestamp/payload mapping."""
    import pyarrow.parquet as pq
    records = []
    tables = {}
    for index in indices:
        if not 0 <= index < len(dataset):
            raise IndexError(index)
        ep_local = int(np.searchsorted(dataset._cumulative, index, side="right") - 1)
        episode_row = dataset.episodes.iloc[ep_local]
        episode = int(episode_row["episode_index"])
        k = dataset.first_anchor + int(index - dataset._cumulative[ep_local]) * dataset.window_stride
        path = Path(dataset.dataset_dir) / dataset.data_path.format(
            chunk_index=int(episode_row["data/chunk_index"]),
            file_index=int(episode_row["data/file_index"]),
        )
        if path not in tables:
            tables[path] = pq.read_table(path, columns=[
                "episode_index", "frame_index", "timestamp",
                *(f"observation.event_frame.{camera}" for camera in dataset.event_cameras),
            ])
        table = tables[path]
        start = int(episode_row["_data_offset"]) + k
        window = table.slice(start, horizon + 1)
        if len(window) != horizon + 1:
            raise ValueError(f"truncated event-frame window: {path}, row {start}")
        expected_frames = np.arange(k, k + horizon + 1)
        frames = window["frame_index"].to_numpy()
        timestamps = window["timestamp"].to_numpy()
        episodes = window["episode_index"].to_numpy()
        if not np.array_equal(frames, expected_frames) or not np.all(episodes == episode):
            raise ValueError(f"event-frame row mapping mismatch: {path}, row {start}")
        if not np.allclose(timestamps, expected_frames / dataset.fps, atol=1e-5, rtol=0):
            raise ValueError(f"event-frame timestamps mismatch: {path}, row {start}")
        for camera in dataset.event_cameras:
            column = window[f"observation.event_frame.{camera}"]
            if any(not (payload := value.as_py().get("bytes")) or
                   not payload.startswith(b"\x89PNG\r\n\x1a\n") for value in column):
                raise ValueError(f"missing/invalid event-frame PNG: {path}, {camera}, row {start}")
            records.append({
                "index": int(index), "episode": episode, "anchor": k,
                "observation_row": k, "camera": camera,
                "verified_frames": horizon + 1, "obs_verified": True,
            })
    return records
