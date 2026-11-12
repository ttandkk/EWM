"""Full training-state checkpoints for the standalone EWM trainer."""

import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def resume_contract(cfg, dataset, loader, accelerator, lr_schedule_steps):
    """Fields that must match before restoring optimizer and sampler state."""
    t = cfg.training
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    # Adding the identity sampler must not invalidate legacy full-state resumes.
    if model_config.get('event_temporal_downsample', 1) == 1:
        model_config.pop('event_temporal_downsample', None)
    # The disabled RGB-history option must preserve legacy single-frame resumes.
    loader_config = OmegaConf.to_container(cfg.dataloader, resolve=True)
    # Explicit null retains legacy equal-horizon resume fingerprints.
    for options in (model_config, loader_config):
        if options.get('event_horizon') is None:
            options.pop('event_horizon', None)
    contract = {
        "model": _fingerprint(model_config),
        "dataloader": _fingerprint(loader_config),
        "normalization": _fingerprint(dataset.normalizer.metadata),
        "dataset_length": len(dataset),
        "loader_length": len(loader),
        "world_size": accelerator.num_processes,
        "batch_size": int(t.batch_size),
        "gradient_accumulation_steps": int(t.gradient_accumulation_steps),
        "seed": int(cfg.project.seed),
        "learning_rate": float(t.learning_rate),
        "adam_betas": list(t.adam_betas),
        "weight_decay": float(t.weight_decay),
        "lr_scheduler": str(t.lr_scheduler),
        "lr_schedule_steps": int(lr_schedule_steps),
        "warmup_ratio": float(t.warmup_ratio),
        "lr_min_ratio": float(t.lr_min_ratio),
        "mixed_precision": str(t.mixed_precision),
    }
    return contract


def capture_rng():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload, path):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_resume_state(accelerator, output, step, epoch, batches_consumed,
                      epoch_generator_state, optimizer, scheduler, contract):
    output = Path(output)
    state_dir = output / f"resume_state_step_{step:06d}"
    state_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    rank = accelerator.process_index
    _atomic_save({"rng": capture_rng(), "epoch_generator_state": epoch_generator_state},
                 state_dir / f"rank_{rank:05d}.pt")
    if accelerator.is_main_process:
        _atomic_save({"optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict() if scheduler is not None else None},
                     state_dir / "optimizer.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metadata = {"format": "ewm_resume_v1", "step": step, "epoch": epoch,
                    "batches_consumed": batches_consumed, "contract": contract}
        marker = state_dir / "complete.json"
        temporary = marker.with_name("complete.json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(marker)
    accelerator.wait_for_everyone()
    return state_dir


def find_latest_resume_state(output, world_size):
    output = Path(output)
    for state_dir in sorted(output.glob("resume_state_step_*"), reverse=True):
        marker = state_dir / "complete.json"
        if not marker.is_file() or not (state_dir / "optimizer.pt").is_file():
            continue
        try:
            metadata = json.loads(marker.read_text())
            step = int(metadata["step"])
        except (ValueError, KeyError, TypeError):
            continue
        if (state_dir.name != f"resume_state_step_{step:06d}"
                or not (output / f"checkpoint_step_{step:06d}.pt").is_file()):
            continue
        if all((state_dir / f"rank_{rank:05d}.pt").is_file() for rank in range(world_size)):
            return state_dir, metadata
    raise FileNotFoundError(
        f"No complete EWM resume state in {output}; policy-only checkpoints cannot restore optimizer/RNG."
    )


def load_resume_state(state_dir, metadata, accelerator, model, optimizer, scheduler, contract):
    if metadata.get("format") != "ewm_resume_v1":
        raise ValueError(f"Unsupported EWM resume format in {state_dir}")
    old = metadata.get("contract", {})
    mismatches = [key for key, value in contract.items() if old.get(key) != value]
    if mismatches:
        raise ValueError(f"EWM resume configuration differs in: {', '.join(mismatches)}")
    step = int(metadata["step"])
    policy = torch.load(Path(state_dir).parent / f"checkpoint_step_{step:06d}.pt",
                        map_location="cpu", weights_only=True)
    policy_model = accelerator.unwrap_model(model)
    training_mode = getattr(policy_model, "training_mode", "joint")
    expected_format = {"joint": "ewm_wan_latent_v2", "event_only": "ewm_wan_latent_v2",
                       "rgb_action": "ewm_rgb_action_latent_v1",
                       "action_only": "ewm_action_only_v2"}.get(training_mode)
    if expected_format is None or policy.get("format") != expected_format or int(policy.get("step", -1)) != step:
        raise ValueError("EWM resume policy checkpoint does not match training state")
    policy_model.load_policy_state_dict(policy["model"])
    training = torch.load(Path(state_dir) / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(training["optimizer"])
    if (scheduler is None) != (training["scheduler"] is None):
        raise ValueError("EWM resume scheduler presence differs from saved state")
    if scheduler is not None:
        scheduler.load_state_dict(training["scheduler"])
    rank_state = torch.load(Path(state_dir) / f"rank_{accelerator.process_index:05d}.pt",
                            map_location="cpu", weights_only=True)
    restore_rng(rank_state["rng"])
    return step, int(metadata["epoch"]), int(metadata["batches_consumed"]), rank_state["epoch_generator_state"]


def prune_resume_states(output, keep_last_k):
    if keep_last_k <= 0:
        raise ValueError("keep_last_k_resume_states must be positive")
    complete = [path for path in sorted(Path(output).glob("resume_state_step_*"))
                if (path / "complete.json").is_file()]
    for path in complete[:-keep_last_k]:
        shutil.rmtree(path)


def prune_policy_checkpoints(output, keep_last_k):
    """Retain the newest versioned policy snapshots; leave checkpoint.pt intact."""
    if keep_last_k <= 0:
        raise ValueError("keep_last_k_policy_checkpoints must be positive")
    output = Path(output)
    snapshots = []
    for path in output.glob("checkpoint_step_*.pt"):
        step = path.stem.removeprefix("checkpoint_step_")
        if step.isdigit():
            snapshots.append((int(step), path))
    for _, path in sorted(snapshots)[:-keep_last_k]:
        path.unlink()
