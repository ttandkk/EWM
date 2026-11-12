#!/usr/bin/env python3
"""CPU checks with real native loaders; use each baseline's own Python env."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "OpenWAM"))
sys.path.insert(0, str(PROJECT / "FastWAM/src"))
sys.path.insert(0, str(PROJECT / "Isaac-GR00T"))
sys.path.insert(0, str(PROJECT / "Isaac-GR00T/examples/Piper"))


def raw_episode(root, task, ep):
    table = pq.read_table(root / task / f"data/chunk-000/episode_{ep:06d}.parquet")
    assert set(table.column_names) == {"observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"}
    return np.asarray(table["action"].to_pylist(), np.float32), np.asarray(table["observation.state"].to_pylist(), np.float32)


def check_export(root):
    from prepare_piper_rgb_baselines import fingerprint
    manifest = json.loads((root / "manifest.json").read_text())
    for record in manifest["tasks"]:
        directory = root / record["task"]
        source = json.loads((directory / "meta/source_manifest.json").read_text())
        assert fingerprint(Path(source["source"])) == source["source_files"], "Original dataset changed"
        info = json.loads((directory / "meta/info.json").read_text())
        assert len(info["features"]) == 9
        assert not any("event" in k or "object" in k for k in info["features"])
        for ep in range(50):
            action, state = raw_episode(root, record["task"], ep)
            assert np.isfinite(action).all() and np.isfinite(state).all()
        videos = list((directory / "videos").rglob("*.mp4"))
        assert len(videos) == 100 and all(p.is_symlink() and p.is_file() for p in videos)
    return manifest


def openwam(root, report_dir):
    import torch
    from hydra import compose, initialize_config_dir
    from openwam.dataloader.registry import build_dataset
    with initialize_config_dir(config_dir=str(PROJECT / "OpenWAM/configs"), version_base="1.3"):
        cfg = compose(config_name="train_openwam_piper_rgb")
    assert cfg.model.architecture.action_dim == 7 and cfg.model.architecture.state_dim == 8
    dataset = build_dataset(cfg.dataloader)
    report = []
    for task_i, task in enumerate(cfg.dataloader.tasks):
        for ep, anchor in ((0, 0), (25, 100), (49, 0)):
            position = task_i * 50 + ep
            n = dataset.episodes[position][-1]
            anchor = min(anchor, max(0, (n - 2) // 2))
            if ep == 49:
                anchor = n - 2  # tail padding must be masked, never leak next episode
            offset = dataset.ends[position - 1] if position else 0
            sample = dataset[offset + anchor]
            action, state = raw_episode(root, f"piper_{task}", ep)
            valid = min(24, n - 1 - anchor)
            np.testing.assert_allclose(dataset.denormalize_action(sample["action"][:valid]), action[anchor:anchor + valid], atol=2e-6)
            np.testing.assert_allclose(sample["proprio"], dataset.normalize(state[anchor:anchor + 1], "observation.state"), atol=1e-6)
            assert sample["action"].shape == (24, 7) and sample["proprio"].shape == (1, 8)
            assert sample["action_mask"].sum().item() == valid * 7
            assert len(sample["video"]) == 13 and all(image.size == (512, 256) for image in sample["video"])
            assert sample["first_frame_image"][0] is sample["video"][0]
            assert sample["prompt"] and all(torch.isfinite(sample[k]).all() for k in ("action", "proprio"))
            assert not any("event" in k for k in sample)
            report.append({"task": task, "episode": ep, "anchor": anchor, "valid_actions": valid})
    return {"samples": report, "windows": len(dataset), "batch_contract": "native list[dict], action(24,7), proprio(1,8), 13 RGB frames(256,512)"}


def fastwam(root, report_dir):
    import torch
    from hydra import compose, initialize_config_dir
    from fastwam.runtime import build_datasets
    from fastwam.utils.misc import register_work_dir
    register_work_dir(report_dir / "fastwam_runtime")
    with initialize_config_dir(config_dir=str(PROJECT / "FastWAM/configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=piper_rgb"])
    dataset, validation_dataset = build_datasets(cfg.data)
    # Native factory aliases train as val when no val source is configured.
    # Keep its behavior, but require online evaluation to be disabled so it
    # cannot be mistaken for an independent validation set.
    assert validation_dataset is dataset and cfg.eval_every == 0
    report = []
    for task_i, directory in enumerate(cfg.data.train.dataset_dirs):
        task = Path(directory).name
        for ep, anchor in ((0, 0), (25, 100), (49, 0)):
            action, _ = raw_episode(root, task, ep)
            anchor = min(anchor, max(0, (len(action) - 2) // 2))
            if ep == 49:
                anchor = len(action) - 2
            index = int(dataset.lerobot_dataset.episode_data_index["from"][task_i * 50 + ep]) + anchor
            # Native __getitem__ retries a random sample on failure. _get must
            # succeed at the requested index so corruption cannot be concealed.
            sample = dataset._get(index)
            valid = min(24, len(action) - anchor)
            norm = dataset.lerobot_dataset.processor.normalizer.normalizers["action"]["default"]
            restored = norm.backward(sample["action"][:valid]).numpy()
            np.testing.assert_allclose(restored, action[anchor:anchor + valid], atol=2e-6)
            assert sample["video"].shape == (3, 13, 256, 512)
            assert sample["action"].shape == (24, 7) and sample["proprio"].shape == (24, 8)
            assert sample["context"].shape == (128, 4096)
            assert int(sample["action_is_pad"].sum()) == 24 - valid
            assert not any("event" in k for k in sample)
            batch = torch.utils.data.default_collate([sample])
            assert batch["video"].shape == (1, 3, 13, 256, 512)
            assert all(torch.isfinite(batch[k]).all() for k in ("action", "proprio", "video", "context"))
            report.append({"task": task, "episode": ep, "anchor": anchor, "valid_actions": valid})
    return {"samples": report, "windows": len(dataset), "batch_contract": "native collated RGB/action/proprio/T5 context"}


def groot(root, report_dir):
    import torch
    from piper_config import piper_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.dataset.sharded_mixture_dataset import merge_statistics
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    tag = EmbodimentTag.NEW_EMBODIMENT
    manifest = json.loads((root / "manifest.json").read_text())
    loaders = [LeRobotEpisodeLoader(root / r["task"], piper_config) for r in manifest["tasks"]]
    stats = [loader.get_dataset_statistics() for loader in loaders]
    merged = {kind: merge_statistics([s[kind] for s in stats], [r["frames"] for r in manifest["tasks"]]) for kind in ("state", "action")}
    processor = Gr00tN1d7Processor(
        modality_configs={tag.value: piper_config}, statistics={tag.value: merged},
        model_name=str(PROJECT / "pretrained/nvidia/Cosmos-Reason2-2B"),
        max_state_dim=132, max_action_dim=132, max_action_horizon=40,
        shortest_image_edge=256, crop_fraction=1.0,
        image_target_size=None, image_crop_size=None, use_albumentations=True,
        state_dropout_prob=0.0, use_relative_action=True, use_percentiles=False,
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True})
    report = []
    for loader, record in zip(loaders, manifest["tasks"]):
        for ep, anchor in ((0, 0), (25, 100), (49, 0)):
            data = loader[ep]
            if ep == 49:
                anchor = len(data) - 24
            raw_action, state = raw_episode(root, record["task"], ep)
            anchor = min(anchor, len(raw_action) - 24)
            step = extract_step_data(data, anchor, piper_config, tag, allow_padding=False)
            np.testing.assert_array_equal(step.actions["command"], raw_action[anchor:anchor + 24])
            np.testing.assert_array_equal(step.states["joints"], state[anchor:anchor + 1])
            assert len(step.images) == 2 and step.images["front"][0].shape == (256, 256, 3)
            sample = processor([{"type": "episode_step", "content": step}])
            assert sample["action"].shape == (40, 132)
            assert sample["action_mask"].sum() == 24 * 7
            restored = processor.state_action_processor.unapply_action(
                {"command": sample["action"][:24, :7].numpy()}, tag.value, state=step.states)["command"]
            np.testing.assert_allclose(restored, raw_action[anchor:anchor + 24], atol=2e-6)
            batch = processor.collator([sample])
            assert batch["inputs"]["action"].shape == (1, 40, 132)
            assert torch.isfinite(batch["inputs"]["action"]).all()
            report.append({"task": record["task"], "episode": ep, "anchor": anchor, "valid_actions": 24})
            del data, step, sample, batch
    return {"samples": report, "batch_contract": "native GR00T processor+collator; 24x7 valid actions padded to 40x132"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["export", "openwam", "fastwam", "groot"])
    parser.add_argument("--data-root", type=Path, default=PROJECT / "tmp/training_data_cache/piper_rgb_dynamic_v1")
    parser.add_argument("--report-dir", type=Path, default=PROJECT / "tmp/wam_lora_alignment_20261110")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PIPER_RGB_DATA_ROOT"] = str(args.data_root)
    print(f"Starting {args.backend} validation", flush=True)
    result = check_export(args.data_root) if args.backend == "export" else globals()[args.backend](args.data_root, args.report_dir)
    (args.report_dir / f"{args.backend}_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS {args.backend}: {len(result.get('samples', []))} native samples", flush=True)


if __name__ == "__main__":
    main()
