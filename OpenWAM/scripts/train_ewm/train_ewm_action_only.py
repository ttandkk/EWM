"""Reproduce action-only job180957 using event-action job179463's saved config.

Promoted from tmp/ewm_action_only_formal/train_from_179463.py.
DRY_RUN=true validates and prints the configuration without training or writes.
"""
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/projects/hdd/myproject/CVPR2027")
sys.path.insert(0, str(PROJECT_ROOT / "OpenWAM"))

from omegaconf import OmegaConf

BASELINE = PROJECT_ROOT / "OpenWAM/outputs/ewm_lean16_p4_d768_a512_dual_ada2_80k/job179463/config.yaml"


def prepare(job_id):
    baseline = OmegaConf.load(BASELINE)
    cfg = baseline.copy()
    output = PROJECT_ROOT / f"OpenWAM/outputs/ewm_action_only_d768_a512_ada2_80k/job{job_id}"
    updates = {
        "model.training_mode": "action_only",
        "model.event_vae_path": None,
        "dataloader.type": "ewm_piper_action_only",
        "dataloader.gpu_voxelization": False,
        "dataloader.event_frame_cache_dir": None,
        "training.output_path": str(output),
        "project.wandb.run_name": f"ewm-action-only-d768-a512-ada2-80k-{job_id}",
    }
    differences = []
    for key, value in updates.items():
        previous = OmegaConf.select(baseline, key)
        OmegaConf.update(cfg, key, value, force_add=True)
        differences.append({"key": key, "baseline": previous, "action_only": value})
    # Prove that resetting only those seven fields recovers the original config.
    restored = cfg.copy()
    for key in updates:
        if key == "model.training_mode" and "training_mode" not in baseline.model:
            del restored.model.training_mode
        else:
            OmegaConf.update(restored, key, OmegaConf.select(baseline, key), force_add=True)
    if OmegaConf.to_container(restored, resolve=True) != OmegaConf.to_container(baseline, resolve=True):
        raise ValueError("non-ablation configuration differs from job179463")
    if not cfg.project.wandb.project or cfg.project.wandb.project != "CVPR2027-EWM":
        raise ValueError("online W&B project must match baseline")
    return cfg, differences


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    dry_run = os.environ.get("DRY_RUN", "false") == "true"
    job_id = os.environ.get("SLURM_JOB_ID", "dryrun") if dry_run else os.environ["SLURM_JOB_ID"]
    if os.environ.get("WANDB_MODE") != "online":
        raise ValueError("formal action-only training requires WANDB_MODE=online")
    cfg, differences = prepare(job_id)
    if dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
        return
    if int(os.environ.get("RANK", "0")) == 0:
        output = Path(cfg.training.output_path)
        if (output / "config.yaml").exists():
            raise FileExistsError(f"refusing to overwrite an existing training run: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "baseline_comparison.json").write_text(json.dumps({
            "baseline": str(BASELINE), "differences": differences,
            "all_other_fields_equal": True, "wandb_mode": "online",
        }, indent=2) + "\n")
        print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    from scripts.train_ewm import train
    train(cfg)


if __name__ == "__main__":
    main()
