"""Evaluate EWM policy checkpoints from pure event/action noise.

Run from the project root, for example:
  envs/openwam/bin/python OpenWAM/scripts/eval_ewm/eval_ewm_pure_noise.py \
    --checkpoint-dir OpenWAM/outputs/<run>/job<id> \
    --output-dir OpenWAM/outputs/<run>/job<id>/eval_manual \
    --steps 80000

Uses the checkpoint run's saved model config and normalization statistics,
with physically aligned RGB/state and observed event at inference.
The LeRobot dataset stores aligned event-frame PNGs in the data Parquet.
All default windows are training-distribution windows. Metrics are macro means
over windows and, for generated rollouts, seeds. No pixel/frame is a replicate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "OpenWAM"))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from openwam.dataloader.registry import build_dataset
from scripts.eval_ewm.eval_ewm_policy import load_policy
from scripts.eval_ewm.ewm_eval_time_audit import EVENT_TIME_CONTRACT, verify_exported_event_alignment

DEFAULT_SEEDS = (1234, 5678)
METRIC_KEYS = ("f1", "precision", "recall", "mse", "active_mse",
               "inactive_mse", "pred_nz", "target_nz")


def aligned_default_indices(dataset) -> list[int]:
    """Sample up to eight valid windows spread across the selected dataset."""
    return np.linspace(0, len(dataset) - 1, num=min(8, len(dataset)),
                       dtype=int).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="Checkpoint steps; defaults to every saved policy checkpoint")
    parser.add_argument("--indices", nargs="+", type=int,
                        help="Dataset indices; default spreads eight windows across the dataset")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--occupancy-threshold", type=float, default=1 / 30)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and (len(set(args.steps)) != len(args.steps) or min(args.steps) <= 0):
        parser.error("--steps must contain distinct positive checkpoint steps")
    if args.indices is not None and (not args.indices or len(set(args.indices)) != len(args.indices) or min(args.indices) < 0):
        parser.error("--indices must contain distinct nonnegative dataset indices")
    if not args.seeds or args.sampling_steps <= 0:
        parser.error("--seeds must be nonempty and --sampling-steps positive")
    if not 0 < args.occupancy_threshold <= 1:
        parser.error("--occupancy-threshold must lie in (0, 1]")
    return args


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> dict:
    active, support = target > 0, pred > threshold
    true_positive = (active & support).sum().float()
    precision = true_positive / support.sum().clamp_min(1)
    recall = true_positive / active.sum().clamp_min(1)
    error = (pred - target).square()
    return {
        "f1": float(2 * precision * recall / (precision + recall).clamp_min(1e-12)),
        "precision": float(precision),
        "recall": float(recall),
        "mse": float(error.mean()),
        "active_mse": float(error[active].mean()),
        "inactive_mse": float(error[~active].mean()),
        "pred_nz": float(support.float().mean()),
        "target_nz": float(active.float().mean()),
    }


def quantize_event_for_display(voxel: torch.Tensor, count_clip: int,
                               threshold: float) -> np.ndarray:
    """Project a two-polarity decoded voxel onto the GT count grid for display."""
    counts = (voxel.float().clamp(0, 1) * count_clip).round().to(torch.int16)
    counts = torch.where(voxel > threshold, counts.clamp_min(1), 0)
    return counts.numpy()


def render_comparisons(output_dir: Path, pictures: dict, indices, steps,
                       occupancy_threshold: float, count_clip: int) -> None:
    if count_clip <= 0:
        raise ValueError("visualization count_clip must be positive")
    names = ["GT", "persistence", "clean_codec", *(f"ckpt{step}" for step in steps)]
    off_color = np.array((220, 38, 38), dtype=np.float32)
    on_color = np.array((35, 95, 191), dtype=np.float32)
    for index in indices[:2]:
        binary_canvas = Image.new("RGB", (1024, 280 * len(names)), "white")
        count_canvas = Image.new("RGB", (1024, 280 * len(names)), "white")
        binary_draw, count_draw = ImageDraw.Draw(binary_canvas), ImageDraw.Draw(count_canvas)
        quantized = {}
        for row, name in enumerate(names):
            cutoff = 0 if name == "GT" else occupancy_threshold
            counts = quantize_event_for_display(pictures[index, name], count_clip, cutoff)
            quantized[name] = counts
            off_count, on_count = counts[:, 0], counts[:, 1]
            off_active, on_active = off_count > 0, on_count > 0

            binary = np.full((*off_active.shape, 3), 255, dtype=np.uint8)
            binary[off_active] = (220, 38, 38)     # OFF: red
            binary[on_active] = (35, 95, 191)     # ON: blue
            binary[off_active & on_active] = (105, 55, 139)  # both polarities

            total = off_count.astype(np.float32) + on_count.astype(np.float32)
            tint = (off_count[..., None] * off_color + on_count[..., None] * on_color) / np.maximum(total[..., None], 1)
            strength = np.where(total > 0, 0.45 + 0.55 * np.maximum(off_count, on_count) / count_clip, 0)
            count_rgb = np.rint(255 * (1 - strength[..., None]) + tint * strength[..., None]).clip(0, 255).astype(np.uint8)
            for column, frame in enumerate((0, 7, 15, 23)):
                xy = (column * 256, row * 280 + 24)
                binary_canvas.paste(Image.fromarray(binary[frame]), xy)
                count_canvas.paste(Image.fromarray(count_rgb[frame]), xy)
                label = f"{name} future{frame + 1}"
                binary_draw.text((xy[0] + 4, row * 280 + 4), label, fill="black")
                count_draw.text((xy[0] + 4, row * 280 + 4), label, fill="black")
        binary_canvas.save(output_dir / f"comparison_{index}.png")
        count_canvas.save(output_dir / f"comparison_{index}_counts.png")
        np.savez_compressed(output_dir / f"comparison_{index}_counts.npz", **quantized)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.steps is None:
        args.steps = sorted(int(path.stem.removeprefix("checkpoint_step_"))
                            for path in checkpoint_dir.glob("checkpoint_step_*.pt")
                            if path.stem.removeprefix("checkpoint_step_").isdigit())
        if not args.steps:
            raise FileNotFoundError(f"no policy checkpoints found in {checkpoint_dir}")
    paths = [checkpoint_dir / f"checkpoint_step_{step:06d}.pt" for step in args.steps]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint(s): {missing}")
    if (output_dir / "complete.json").exists():
        raise FileExistsError(f"refusing to overwrite a completed evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = checkpoint_dir / "config.yaml"
    normalization_path = checkpoint_dir / "normalization.json"
    cfg = OmegaConf.load(config_path)
    training_observation_row_offset = int(cfg.dataloader.get("observation_row_offset", 0))
    cfg.dataloader.gpu_voxelization = False
    cfg.dataloader.observation_row_offset = 0  # Same-row RGB/state/event-frame.
    cfg.dataloader.normalization_stats_path = str(normalization_path)
    dataset = build_dataset(cfg.dataloader, split="train")
    if not dataset.direct_event_frames:
        raise ValueError("evaluation requires an aligned direct event-frame dataset")
    indices = list(args.indices) if args.indices is not None else aligned_default_indices(dataset)
    seeds, steps = list(args.seeds), list(args.steps)
    alignment = verify_exported_event_alignment(dataset, indices, dataset.chunk_size)
    samples = {index: dataset[index] for index in indices}
    mapping = []
    for index in indices:
        episode_local = int(np.searchsorted(dataset._cumulative, index, side="right") - 1)
        mapping.append({
            "index": index,
            "episode": int(dataset.episodes.iloc[episode_local]["episode_index"]),
            "anchor": int(dataset.first_anchor + (index - dataset._cumulative[episode_local]) * dataset.window_stride),
        })
    save_json(output_dir / "manifest.json", {
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_dir": str(Path(cfg.dataloader.dataset_dir).resolve()),
        "dataset_loader": "openwam.dataloader.ewm_dataset.EWMDataset",
        "normalization_path": str(normalization_path),
        "checkpoint_steps": steps,
        "checkpoint_sha256": {str(step): file_hash(path) for step, path in zip(steps, paths)},
        "config_sha256": file_hash(config_path),
        "normalization_sha256": file_hash(normalization_path),
        "ids": mapping,
        "event_time_contract": EVENT_TIME_CONTRACT,
        "observation_row_offset": dataset.observation_row_offset,
        "training_observation_row_offset": training_observation_row_offset,
        "event_offset_audit": alignment,
        "seeds": seeds,
        "sampling_steps": args.sampling_steps,
        "sampling_method": "reverse Euler, shift=1, sigma 1 to 0",
        "dtype": "BF16 autocast; FP32 flow state",
        "input_keys": ["obs_video", "obs_voxel", "proprio", "prompt"],
        "occupancy_threshold": args.occupancy_threshold,
        "visualization": ({"count_clip": int(cfg.dataloader.count_clip),
                           "prediction_threshold": args.occupancy_threshold,
                           "gt_threshold": 0,
                           "files": "binary comparison, count-shaded comparison, exact two-channel count NPZ"}
                          if not args.no_images else {"enabled": False}),
        "dataset_split": "training episodes; no held-out evaluation",
        "metrics": "macro mean over windows and seeds; target > 0",
    })

    records, pictures = [], {}
    with torch.inference_mode():
        model, normalizer, horizon = load_policy(paths[0], device="cuda")
        if dataset.chunk_size != horizon or int(cfg.dataloader.chunk_size) != horizon:
            raise ValueError("checkpoint horizon and training dataset chunk_size differ")
        if model.event_camera_mode != dataset.event_camera_mode or model.event_camera_mode != cfg.model.event_camera_mode:
            raise ValueError("model, dataset and training config event_camera_mode differ")
        if model.voxel_bins != dataset.voxel_bins or model.voxel_bins != int(cfg.model.voxel_bins):
            raise ValueError("model, dataset and training config voxel_bins differ")
        if normalizer.voxel_bins != dataset.voxel_bins or normalizer.count_clip != dataset.count_clip:
            raise ValueError("checkpoint normalization and training dataset voxel settings differ")
        if dataset.normalizer.metadata != normalizer.metadata:
            raise ValueError("dataset and checkpoint normalization metadata differ")
        if horizon != 24 and not args.no_images:
            raise ValueError("comparison image layout requires horizon=24; use --no-images")
        for step, path in zip(steps, paths):
            if step != steps[0]:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if int(payload["step"]) != step:
                    raise ValueError(f"{path}: saved step disagrees with filename")
                model.load_policy_state_dict(payload["model"])
                del payload
            model.eval()
            for index, sample in samples.items():
                batch = {key: sample[key][None].cuda() for key in
                         ("obs_video", "obs_voxel", "proprio")}
                batch["prompt"] = [sample["prompt"]]
                target = sample["future_voxel"].cuda()
                action = sample["action"].cuda()
                if step == steps[0]:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        full = torch.cat((batch["obs_voxel"][:, None], target[None]), 1)
                        oracle = model.event_codec.decode(model.event_codec.encode(full))[0, 1:]
                    persistence = batch["obs_voxel"][0, None].expand_as(target)
                    baselines = (("persistence", persistence), ("clean_codec", oracle),
                                 ("zero", torch.zeros_like(target)))
                    for name, pred in baselines:
                        records.append({"method": name, "index": index,
                                        **metric(pred, target, args.occupancy_threshold)})
                        if not args.no_images and index in indices[:2]:
                            pictures[index, name] = pred[:, 0].cpu()
                    if not args.no_images and index in indices[:2]:
                        pictures[index, "GT"] = target[:, 0].cpu()
                for seed in seeds:
                    generator = torch.Generator(device="cuda").manual_seed(seed + index)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = model.sample_policy(
                            batch, horizon=horizon, normalizer=normalizer,
                            num_steps=args.sampling_steps, generator=generator,
                        )
                    pred = result["future_voxel"][0]
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError(f"nonfinite rollout: step={step}, index={index}, seed={seed}")
                    row = {
                        "method": f"ckpt{step}", "index": index, "seed": seed,
                        "action_mse": float((result["action_normalized"][0] - action).square().mean()),
                        **metric(pred, target, args.occupancy_threshold),
                        "frame_f1": [
                            metric(pred[frame:frame + 1], target[frame:frame + 1],
                                   args.occupancy_threshold)["f1"]
                            for frame in range(horizon)
                        ],
                    }
                    records.append(row)
                    print(json.dumps({key: value for key, value in row.items() if key != "frame_f1"}),
                          flush=True)
                    save_json(output_dir / "rows.json", records)
                    if not args.no_images and index in indices[:2] and seed == seeds[0]:
                        pictures[index, f"ckpt{step}"] = pred[:, 0].cpu()

    summary = {}
    for method in dict.fromkeys(row["method"] for row in records):
        selected = [row for row in records if row["method"] == method]
        values = {key: statistics.mean(row[key] for row in selected) for key in METRIC_KEYS}
        if "action_mse" in selected[0]:
            values["action_mse"] = statistics.mean(row["action_mse"] for row in selected)
            values["per_seed_f1"] = {
                str(seed): statistics.mean(row["f1"] for row in selected if row["seed"] == seed)
                for seed in seeds
            }
            values["frame_f1"] = [
                statistics.mean(row["frame_f1"][frame] for row in selected)
                for frame in range(horizon)
            ]
        summary[method] = values
    if not args.no_images:
        render_comparisons(output_dir, pictures, indices, steps, args.occupancy_threshold, int(cfg.dataloader.count_clip))
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "complete.json", {"complete": True, "rows": len(records)})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
