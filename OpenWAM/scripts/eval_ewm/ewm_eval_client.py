"""Training-window EWM evaluator that sends observations to a local GPU server."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.dataloader.registry import build_dataset
from websockets.sync.client import connect as ws_connect

from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message
from scripts.eval_ewm.ewm_eval_time_audit import EVENT_TIME_CONTRACT, verify_exported_event_alignment
from scripts.eval_ewm.eval_ewm_pure_noise import (
    DEFAULT_SEEDS, METRIC_KEYS, aligned_default_indices, file_hash, metric,
    render_comparisons, save_json,
)


def call(conn, request: dict, arrays: dict | None = None) -> tuple[dict, dict]:
    conn.send(encode_message(request, arrays))
    reply, values = decode_message(conn.recv(timeout=600))
    if not reply.get("ok"):
        raise RuntimeError(f"inference server: {reply.get('error', 'unknown failure')}")
    return reply, values


def connect_ready(path: Path, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        try:
            url = json.loads(path.read_text())["url"]
            return ws_connect(url, open_timeout=3, max_size=MAX_MESSAGE,
                              compression=None, ping_interval=None, proxy=None)
        except (OSError, KeyError, ValueError, TimeoutError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"inference server did not start; readiness file: {path}")
            time.sleep(0.5)


def require_prediction(values: dict, name: str, shape: tuple) -> torch.Tensor:
    value = values.get(name)
    if value is None or value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(f"invalid server {name}: expected finite float32 {shape}")
    return torch.from_numpy(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, help="Defaults to latest saved checkpoint")
    parser.add_argument("--indices", nargs="+", type=int,
                        help="Dataset indices; default retains the historical episode/target anchors")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--occupancy-threshold", type=float, default=1 / 30)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    if args.indices is not None and (not args.indices or min(args.indices) < 0 or len(args.indices) != len(set(args.indices))):
        parser.error("--indices must be distinct nonnegative indices")
    if not args.seeds or min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must be distinct nonnegative seeds")
    if args.steps is not None and (min(args.steps) <= 0 or len(args.steps) != len(set(args.steps))):
        parser.error("--steps must be distinct positive checkpoint steps")
    if args.sampling_steps <= 0 or not 0 < args.occupancy_threshold <= 1:
        parser.error("invalid sampling steps or occupancy threshold")
    torch.set_num_threads(4)
    directory = args.checkpoint_dir.resolve()
    output = args.output_dir.resolve()
    if (output / "complete.json").exists():
        raise FileExistsError(f"refusing to overwrite completed evaluation: {output}")
    config_path = directory / "config.yaml"
    normalization_path = directory / "normalization.json"
    cfg = OmegaConf.load(config_path)
    cfg.dataloader.gpu_voxelization = False
    cfg.dataloader.observation_row_offset = 0  # Same-row RGB/state/event-frame.
    cfg.dataloader.normalization_stats_path = str(normalization_path)
    dataset = build_dataset(cfg.dataloader, split="train")
    if not dataset.direct_event_frames:
        raise ValueError("evaluation requires an aligned direct event-frame dataset")
    indices = list(args.indices) if args.indices is not None else aligned_default_indices(dataset)
    seeds = list(args.seeds)
    if max(indices) >= len(dataset):
        raise IndexError(f"evaluation index exceeds dataset length {len(dataset)}")
    alignment = verify_exported_event_alignment(dataset, indices, dataset.future_horizon)
    samples = {index: dataset[index] for index in indices}
    with connect_ready(args.ready_file, 300) as conn:
        hello, _ = call(conn, {"op": "hello"})
        if hello.get("event_time_contract") != EVENT_TIME_CONTRACT:
            raise ValueError("client/server event time contracts differ")
        available = hello["checkpoints"]
        steps = list(args.steps) if args.steps else [max(available)]
        if any(step not in available for step in steps):
            raise ValueError(f"requested checkpoint unavailable; server has {available}")
        if hello["config_sha256"] != file_hash(config_path) or hello["normalization_sha256"] != file_hash(normalization_path):
            raise ValueError("client/server training config or normalization hashes differ")
        if hello.get("inference_observation_row_offset") != dataset.observation_row_offset:
            raise ValueError("client/server observation-row time alignment differs")
        if hello["horizon"] != dataset.chunk_size or hello["event_camera_mode"] != dataset.event_camera_mode:
            raise ValueError("client/server horizon or event camera mode differ")
        if hello["voxel_bins"] != dataset.voxel_bins or hello["count_clip"] != dataset.count_clip:
            raise ValueError("client/server voxel normalization differs")
        event_horizon = hello.get('event_horizon', hello['horizon'])
        if event_horizon != dataset.future_horizon:
            raise ValueError('server and dataset event horizons differ')
        if not args.no_images and event_horizon != 24:
            raise ValueError("comparison image layout requires horizon=24; use --no-images")
        output.mkdir(parents=True, exist_ok=True)
        mapping = []
        for index in indices:
            episode_local = int(np.searchsorted(dataset._cumulative, index, side="right") - 1)
            mapping.append({"index": index,
                            "episode": int(dataset.episodes.iloc[episode_local]["episode_index"]),
                            "anchor": int(dataset.first_anchor +
                                          (index - dataset._cumulative[episode_local]) * dataset.window_stride)})
        save_json(output / "manifest.json", {
            "checkpoint_dir": str(directory), "checkpoint_steps": steps,
            "checkpoint_sha256": {str(step): file_hash(directory / f"checkpoint_step_{step:06d}.pt") for step in steps},
            "config_sha256": hello["config_sha256"],
            "normalization_sha256": hello["normalization_sha256"],
            "dataset_dir": str(Path(cfg.dataloader.dataset_dir).resolve()),
            "dataset_loader": "openwam.dataloader.ewm_dataset.EWMDataset",
            "event_time_contract": EVENT_TIME_CONTRACT,
            "observation_row_offset": dataset.observation_row_offset,
            "training_observation_row_offset": hello["training_observation_row_offset"],
            "event_offset_audit": alignment,
            "ids": mapping, "seeds": seeds, "sampling_steps": args.sampling_steps,
            "sampling_method": "reverse Euler, shift=1, sigma 1 to 0",
            "dtype": "BF16 autocast; FP32 flow state", "transport": "local WebSocket binary frames, JSON + raw NumPy arrays",
            "input_keys": ["obs_video", "obs_voxel", "proprio", "prompt"],
            "occupancy_threshold": args.occupancy_threshold,
            "visualization": {"enabled": not args.no_images, "count_clip": dataset.count_clip,
                              "prediction_threshold": args.occupancy_threshold, "gt_threshold": 0},
            "dataset_split": "training episodes; no held-out evaluation",
            "metrics": "macro mean over windows and seeds; target > 0",
        })
        rows, pictures = [], {}
        for step in steps:
            for index, sample in samples.items():
                target = sample["future_voxel"]
                action = sample["action"]
                obs = sample["obs_voxel"]
                obs_arrays = {
                    "obs_video": sample["obs_video"].numpy(),
                    "obs_voxel": obs.numpy().astype(np.float32, copy=False),
                    "proprio": sample["proprio"].numpy().astype(np.float32, copy=False),
                }
                if step == steps[0]:
                    _, oracle_values = call(conn, {"op": "clean_codec"},
                                            {"obs_voxel": obs_arrays["obs_voxel"],
                                             "future_voxel": target.numpy().astype(np.float32, copy=False)})
                    oracle = require_prediction(oracle_values, "future_voxel", tuple(target.shape))
                    persistence = obs[None].expand_as(target)
                    for name, prediction in (("persistence", persistence),
                                             ("clean_codec", oracle),
                                             ("zero", torch.zeros_like(target))):
                        rows.append({"method": name, "index": index,
                                     **metric(prediction, target, args.occupancy_threshold)})
                        if not args.no_images and index in indices[:2]:
                            pictures[index, name] = prediction[:, 0].cpu()
                    if not args.no_images and index in indices[:2]:
                        pictures[index, "GT"] = target[:, 0].cpu()
                for seed in seeds:
                    reply, values = call(conn, {"op": "predict", "step": step,
                                                "seed": seed + index, "sampling_steps": args.sampling_steps,
                                                "prompt": sample["prompt"],
                                                "event_time_contract": EVENT_TIME_CONTRACT,
                                                "observation_row_offset": dataset.observation_row_offset}, obs_arrays)
                    if reply["step"] != step or reply["seed"] != seed + index:
                        raise ValueError("server reply step/seed mismatch")
                    prediction = require_prediction(values, "future_voxel", tuple(target.shape))
                    predicted_action = require_prediction(values, "action_normalized", tuple(action.shape))
                    row = {"method": f"ckpt{step}", "index": index, "seed": seed,
                           "action_mse": float((predicted_action - action).square().mean()),
                           **metric(prediction, target, args.occupancy_threshold),
                           "frame_f1": [metric(prediction[frame:frame + 1], target[frame:frame + 1],
                                               args.occupancy_threshold)["f1"]
                                        for frame in range(event_horizon)]}
                    rows.append(row)
                    print(json.dumps({key: value for key, value in row.items() if key != "frame_f1"}), flush=True)
                    save_json(output / "rows.json", rows)
                    if not args.no_images and index in indices[:2] and seed == seeds[0]:
                        pictures[index, f"ckpt{step}"] = prediction[:, 0]
        summary = {}
        for method in dict.fromkeys(row["method"] for row in rows):
            selected = [row for row in rows if row["method"] == method]
            values = {key: statistics.mean(row[key] for row in selected) for key in METRIC_KEYS}
            if "action_mse" in selected[0]:
                values["action_mse"] = statistics.mean(row["action_mse"] for row in selected)
                values["per_seed_f1"] = {str(seed): statistics.mean(row["f1"] for row in selected
                                                             if row["seed"] == seed) for seed in seeds}
                values["frame_f1"] = [statistics.mean(row["frame_f1"][frame] for row in selected)
                                      for frame in range(event_horizon)]
            summary[method] = values
        if not args.no_images:
            render_comparisons(output, pictures, indices, steps, args.occupancy_threshold, dataset.count_clip)
        save_json(output / "summary.json", summary)
        save_json(output / "complete.json", {"complete": True, "rows": len(rows)})
        print(json.dumps(summary), flush=True)
        call(conn, {"op": "shutdown"})


if __name__ == "__main__":
    main()
