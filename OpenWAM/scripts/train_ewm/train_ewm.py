"""EWM training via OpenWAM Hydra configs, dataset registry and logging helpers.

Run from OpenWAM: python scripts/train.py --config-name train_ewm
The separate entrypoint also works: python scripts/train_ewm/train_ewm.py
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from openwam.model.architectures.ewm import EWMModel
from openwam.dataloader.ewm_event_only import EWMMixedEventOnlyDataset
from openwam.dataloader.ewm_source_batch_sampler import SourceHomogeneousBatchSampler
from openwam.dataloader.ewm_rgb_action import PiperRGBActionDataset
from openwam.dataloader.ewm_action_only import PiperActionOnlyDataset
from openwam.train.utils.seeding import seed_everything
from openwam.train.utils.training_utils import build_cosine_scheduler, init_wandb, write_debug_loss_row
from openwam.train.utils.pure_noise_eval import eval_due, submit_pure_noise_eval
from openwam.train.utils.ewm_resume import (
    find_latest_resume_state, load_resume_state, prune_policy_checkpoints, prune_resume_states,
    resume_contract, save_resume_state,
)

logger = logging.getLogger(__name__)


def _publish_policy_snapshot(output, step):
    """Replace a stale snapshot from an interrupted save without deleting it."""
    snapshot = output / f"checkpoint_step_{step:06d}.pt"
    if snapshot.exists():
        archive = output / "incomplete_policy_checkpoints"
        archive.mkdir(exist_ok=True)
        snapshot.replace(archive / f"checkpoint_step_{step:06d}_{time.time_ns()}.pt")
    snapshot.hardlink_to(output / "checkpoint.pt")


def _repoint_latest_policy(output, step):
    """Make checkpoint.pt match the last complete training state on resume."""
    temporary = output / f"checkpoint_resume_{time.time_ns()}.tmp"
    temporary.hardlink_to(output / f"checkpoint_step_{step:06d}.pt")
    temporary.replace(output / "checkpoint.pt")


def train(cfg, *, dataset=None, model=None, accelerator=None):
    from accelerate import Accelerator

    from openwam.dataloader.registry import build_dataset
    from openwam.dataloader.ewm_dataset import collate_ewm_sparse_events
    from openwam.dataloader.ewm_gpu_voxel import materialize_sparse_event_voxels

    t = cfg.training
    log_every = int(getattr(t, "log_every", 1))
    diagnostic_every = int(getattr(t, "diagnostic_every", log_every))
    profile_segments = bool(getattr(t, "profile_segments", False))
    profile_warmup_steps = int(getattr(t, "profile_warmup_steps", 20))
    profile_log_every = int(getattr(t, "profile_log_every", 20))
    if profile_segments and (profile_warmup_steps < 0 or profile_log_every <= 0):
        raise ValueError("profile_warmup_steps must be nonnegative and profile_log_every positive")
    if min(int(t.max_steps), int(t.batch_size), int(t.gradient_accumulation_steps), log_every, diagnostic_every) <= 0:
        raise ValueError("max_steps, batch_size, accumulation, log_every and diagnostic_every must be positive")
    save_checkpoints = bool(getattr(t, "save_checkpoints", True))
    save_full_states = bool(getattr(t, "save_full_states_for_resume", False))
    resume_path = getattr(t, "resume_ckpt_path", None)
    keep_resume_states = int(getattr(t, "keep_last_k_resume_states", 2))
    keep_policy_checkpoints = getattr(t, "keep_last_k_policy_checkpoints", None)
    keep_policy_checkpoints = int(keep_policy_checkpoints) if keep_policy_checkpoints is not None else None
    if save_full_states and not save_checkpoints:
        raise ValueError("training.save_full_states_for_resume requires save_checkpoints=true")
    if save_full_states and keep_resume_states <= 0:
        raise ValueError("training.keep_last_k_resume_states must be positive")
    if keep_policy_checkpoints is not None:
        if keep_policy_checkpoints <= 0:
            raise ValueError("training.keep_last_k_policy_checkpoints must be positive")
        if save_full_states and keep_policy_checkpoints < keep_resume_states:
            raise ValueError("policy retention must cover every retained full resume state")
    lr_schedule_steps = getattr(t, "lr_schedule_steps", None)
    lr_schedule_steps = int(lr_schedule_steps) if lr_schedule_steps is not None else int(t.max_steps)
    if lr_schedule_steps < int(t.max_steps):
        raise ValueError("training.lr_schedule_steps must be at least training.max_steps")
    pure_noise_eval = getattr(t, "pure_noise_eval", None)
    if pure_noise_eval is not None and bool(pure_noise_eval.enabled):
        if not save_checkpoints:
            raise ValueError("training.pure_noise_eval requires training.save_checkpoints=true")
        if int(pure_noise_eval.every_steps) <= 0:
            raise ValueError("training.pure_noise_eval.every_steps must be positive")
        if not all(str(getattr(pure_noise_eval, key, "")) for key in ("partition", "gres", "time")):
            raise ValueError("training.pure_noise_eval requires partition, gres and time")
    accelerator = accelerator or Accelerator(
        gradient_accumulation_steps=t.gradient_accumulation_steps, mixed_precision=str(t.mixed_precision)
    )
    resume_state = find_latest_resume_state(resume_path, accelerator.num_processes) if resume_path else None
    seed_everything(int(cfg.project.seed), rank=accelerator.process_index)
    # Stable CPU diagnostics on hosts affected by oneDNN Conv3D backward.
    if accelerator.device.type == "cpu":
        torch.backends.mkldnn.enabled = False
    dataset = build_dataset(cfg.dataloader, split="train") if dataset is None else dataset
    if not len(dataset):
        raise ValueError("training dataset is empty")
    if model is None:
        if str(getattr(cfg.model, "type", "ewm")) == "ewm_sparse":
            raise ValueError("sparse-token training is retired; use ewm or ewm_rgb_event_memory")
        if str(getattr(cfg.model, "type", "ewm")) == "ewm_rgb_event_memory":
            from openwam.model.architectures.hierarchical_ewm import HierarchicalMemoryEWMModel
            model = HierarchicalMemoryEWMModel.from_config(cfg.model)
            logger.info("RGB/event memory Q/KV budget: %s", model.token_counts(horizon=int(cfg.dataloader.chunk_size)))
        elif str(getattr(cfg.model, "type", "ewm")) == "ewm_memory":
            from openwam.model.architectures.ewm_memory import MemoryEWMModel
            model = MemoryEWMModel.from_config(cfg.model)
            logger.info("Memory EWM Q/KV budget: %s", model.token_counts(horizon=int(cfg.dataloader.chunk_size)))
        else:
            model = EWMModel.from_config(cfg.model)
    init_policy = getattr(t, "init_policy_checkpoint", None)
    if init_policy:
        if resume_path:
            raise ValueError("init_policy_checkpoint and resume_ckpt_path are mutually exclusive")
        if not hasattr(model, "initialize_from_dense_policy"):
            raise ValueError("dense initialization requires a memory model")
        initial = torch.load(init_policy, map_location="cpu", weights_only=True)
        if initial.get("format") != "ewm_wan_latent_v2":
            raise ValueError("init_policy_checkpoint must be a dense latent EWM policy")
        model.initialize_from_dense_policy(initial["model"])
        del initial
    if isinstance(dataset, EWMMixedEventOnlyDataset) and getattr(model, "training_mode", "joint") != "event_only":
        raise ValueError("ewm_mixed_event_only requires model.training_mode=event_only")
    rgb_mode = getattr(model, "training_mode", "joint") == "rgb_action"
    action_only = getattr(model, "training_mode", "joint") == "action_only"
    if rgb_mode != isinstance(dataset, PiperRGBActionDataset):
        raise ValueError("rgb_action mode requires the Piper RGB-action dataloader, and vice versa")
    if action_only != isinstance(dataset, PiperActionOnlyDataset):
        raise ValueError("action_only mode requires the Piper action-only dataloader, and vice versa")
    if getattr(model, "training_mode", "joint") != "joint" and pure_noise_eval is not None and bool(pure_noise_eval.enabled):
        raise ValueError("the pure_noise_eval launcher requires event-action training")
    if getattr(model, "memory_enabled", False) and pure_noise_eval is not None and bool(pure_noise_eval.enabled):
        raise ValueError("memory EWM requires history-aware evaluation; disable the legacy pure_noise_eval launcher")
    for name in ("voxel_bins", "event_camera_mode"):
        if not action_only and hasattr(dataset, name) and getattr(dataset, name) != getattr(model, name):
            raise ValueError(f"model and dataloader {name} must match")
    compile_core = bool(getattr(t, "compile_core", False))
    if compile_core and accelerator.num_processes > 1:
        # The RTX 6000 Ada DDP profiling run is stable with this compiled core.
        # Keep the guard for the RTX Pro 6000 path, where Inductor + DDP
        # previously caused a CUDA illegal-memory-access / NCCL watchdog abort.
        device_name = torch.cuda.get_device_name(accelerator.device).lower() if torch.cuda.is_available() else ""
        # Conveyor action-only L40 launches are gated by a dual-GPU compile smoke.
        l40_action_only = action_only and device_name in {"nvidia l40", "l40", "nvidia l40s", "l40s"}
        if "6000 ada" not in device_name and not l40_action_only:
            logger.warning(
                "Disabling training.compile_core for multi-GPU device %r: only RTX 6000 Ada DDP is validated; "
                "other tested hardware produced a CUDA illegal-memory-access.",
                device_name,
            )
            compile_core = False
    if compile_core:
        if not hasattr(torch, "compile"):
            raise RuntimeError("training.compile_core requires torch.compile support")
        # The denoising core has fixed B/K/token shapes in production. Keep
        # DINO/T5 and the dataloader eager, where Python inputs are intentional.
        model.predict_velocity = torch.compile(model.predict_velocity, dynamic=False)
        logger.info("Enabled training core compile: mode=%s device=%s", model.training_mode, torch.cuda.get_device_name(accelerator.device))
    gpu_voxelization = bool(getattr(dataset, "gpu_voxelization", False))
    voxel_count_clip = float(dataset.normalizer.count_clip)
    num_event_cameras = int(model.num_event_cameras)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(cfg.project.seed))
    source_batch_sampler = None
    homogeneous_sources = isinstance(dataset, EWMMixedEventOnlyDataset) and bool(
        getattr(cfg.dataloader, "homogeneous_source_batches", True)
    )
    if isinstance(dataset, EWMMixedEventOnlyDataset):
        # Record the effective default for checkpoint/resume compatibility even
        # when an older caller omits the newly introduced configuration field.
        OmegaConf.update(cfg, "dataloader.homogeneous_source_batches", homogeneous_sources, force_add=True)
    if homogeneous_sources:
        if accelerator.split_batches:
            raise ValueError("homogeneous_source_batches requires Accelerate split_batches=false")
        source_batch_sampler = SourceHomogeneousBatchSampler(
            dataset.source_ranges, t.batch_size,
            num_processes=accelerator.num_processes, seed=cfg.project.seed,
        )
        loader = DataLoader(
            dataset, batch_sampler=source_batch_sampler, num_workers=t.dataset_num_workers,
            generator=loader_generator,
        )
        logger.info("Same-source batches across %d ranks: windows=%s, groups=%s",
                    accelerator.num_processes, source_batch_sampler.source_sizes,
                    source_batch_sampler.source_group_counts)
    else:
        loader = DataLoader(
            dataset, batch_size=t.batch_size, shuffle=True, num_workers=t.dataset_num_workers,
            collate_fn=collate_ewm_sparse_events if gpu_voxelization else None,
            pin_memory=gpu_voxelization,
            generator=loader_generator,
        )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=t.learning_rate,
        betas=tuple(t.adam_betas),
        weight_decay=t.weight_decay,
    )
    scheduler = (
        build_cosine_scheduler(optimizer, total_opt_steps=lr_schedule_steps, cfg=cfg, num_processes=accelerator.num_processes)
        if t.lr_scheduler == "cosine"
        else None
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)
    output = Path(resume_path) if resume_path else Path(t.output_path)
    contract = resume_contract(cfg, dataset, loader, accelerator, lr_schedule_steps)
    if resume_path:
        normalizer_path = output / "normalization.json"
        if not normalizer_path.is_file() or json.loads(normalizer_path.read_text()) != dataset.normalizer.metadata:
            raise ValueError("EWM resume normalization metadata does not match the saved run")
    elif accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "config.yaml")
        (output / "normalization.json").write_text(json.dumps(dataset.normalizer.metadata, indent=2))
    accelerator.wait_for_everyone()
    run = init_wandb(cfg) if accelerator.is_main_process else None
    step, epoch, skip_batches = 0, 0, 0
    resumed_epoch_generator_state = None
    if resume_path:
        state_dir, metadata = resume_state
        step, epoch, skip_batches, resumed_epoch_generator_state = load_resume_state(
            state_dir, metadata, accelerator, model, optimizer, scheduler, contract
        )
        if not 0 <= skip_batches <= len(loader):
            raise ValueError(f"EWM resume batch offset {skip_batches} is outside loader length {len(loader)}")
        if step > int(t.max_steps):
            raise ValueError(f"EWM resume step {step} exceeds max_steps={t.max_steps}")
        if accelerator.is_main_process:
            _repoint_latest_policy(output, step)
            if pure_noise_eval is not None and eval_due(step, t.max_steps, pure_noise_eval):
                submit_pure_noise_eval(output, step, pure_noise_eval)
        if step == int(t.max_steps):
            logger.info("[resume] step=%d already complete", step)
            if run is not None:
                run.finish()
            return output / "checkpoint.pt"
        loader_generator.set_state(resumed_epoch_generator_state)
        logger.info("[resume] loaded %s at optimizer step=%d epoch=%d batch=%d",
                    state_dir, step, epoch, skip_batches)
    model.train()
    optimizer.zero_grad()
    video_loss_key = "loss_rgb" if rgb_mode else "loss_event"
    metric_keys = (("loss", "loss_action") if action_only else
                   ("loss", video_loss_key, "loss_action", "latent_flow_mse", "latent_x0_mse"))
    diagnostic_keys = (("action_x0_mse", "action_x0_mae") if action_only else
                       ("rgb_x0_mse", "rgb_x0_mae", "rgb_x0_psnr") if rgb_mode else
                       ("voxel_x0_mse", "voxel_x0_active_mse", "voxel_x0_inactive_mse",
                        "event_precision", "event_recall", "event_f1",
                        "pred_nonzero_ratio", "target_nonzero_ratio"))
    diagnostic_accumulated = torch.zeros(len(diagnostic_keys), device=accelerator.device)
    accumulated = torch.zeros(len(metric_keys), device=accelerator.device)
    micro_steps = 0
    initial_step = step
    started = time.monotonic()
    profile_totals = {}
    profile_count = 0
    profile_data_clock = time.perf_counter()

    def profile_mark(timings, name):
        if timings is not None:
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize(accelerator.device)
            now = time.perf_counter()
            timings[name] = now - timings.pop("_clock")
            timings["_clock"] = now

    while step < t.max_steps:
        if source_batch_sampler is not None:
            source_batch_sampler.set_epoch(epoch)
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        epoch_generator_state = loader_generator.get_state()
        if skip_batches:
            from accelerate import skip_first_batches
            epoch_loader = skip_first_batches(loader, skip_batches)
            batches_consumed = skip_batches
        else:
            epoch_loader = loader
            batches_consumed = 0
        for batch in epoch_loader:
            batches_consumed += 1
            timings = None
            if profile_segments:
                now = time.perf_counter()
                timings = {"data_wait": now - profile_data_clock, "_clock": now}
            if gpu_voxelization:
                batch = materialize_sparse_event_voxels(
                    batch, num_cameras=num_event_cameras, count_clip=voxel_count_clip
                )
            profile_mark(timings, "materialize")
            with accelerator.accumulate(model):
                diagnose = (step + 1) % diagnostic_every == 0 or step + 1 == t.max_steps
                losses = model(batch, diagnostics=diagnose, profile_timing=profile_segments)
                if profile_segments:
                    timings.update(losses.pop("_profile"))
                    timings["_clock"] = time.perf_counter()
                if diagnose:
                    diagnostic_accumulated += torch.stack([losses[k].detach().float() for k in diagnostic_keys])
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError("nonfinite EWM training loss")
                accelerator.backward(losses["loss"])
                profile_mark(timings, "backward")
                accumulated += torch.stack([losses[k].detach().float() for k in metric_keys])
                micro_steps += 1
                grad_norm = torch.tensor(0.0, device=accelerator.device)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), t.max_grad_norm)
                profile_mark(timings, "clip_and_accumulate")
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                profile_mark(timings, "optimizer")
            if not accelerator.sync_gradients:
                profile_data_clock = time.perf_counter()
                continue
            step += 1
            metrics_values = accelerator.reduce(accumulated / micro_steps, reduction="mean").tolist()
            diagnostic_values = accelerator.reduce(diagnostic_accumulated / micro_steps, reduction="mean").tolist() if diagnose else None
            profile_mark(timings, "metric_reduce")
            diagnostic_accumulated.zero_()
            accumulated.zero_()
            micro_steps = 0
            should_log = step % log_every == 0 or step == t.max_steps
            should_diagnose = step % diagnostic_every == 0 or step == t.max_steps
            if accelerator.is_main_process and (should_log or should_diagnose):
                metrics = dict(
                    zip(("loss_total", *metric_keys[1:]), metrics_values),
                    grad_norm=float(grad_norm),
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
                if diagnostic_values is not None:
                    metrics.update(zip(diagnostic_keys, diagnostic_values))
                if not action_only:
                    metrics["loss_video"] = metrics[video_loss_key]
                write_debug_loss_row(
                    str(output),
                    labels=([("action", "loss_action")] if action_only else
                            [("rgb" if rgb_mode else "event", "loss_video"), ("action", "loss_action")]),
                    # The shared debug CSV has a legacy loss_video column;
                    # zero is only a schema placeholder, not a computed loss.
                    metrics=dict(metrics, loss_video=0.0) if action_only else metrics,
                    global_step=step,
                    opt_step=step,
                    epoch=epoch,
                    lr=optimizer.param_groups[0]["lr"],
                    steps_per_sec=(step - initial_step) / (time.monotonic() - started),
                )
                if should_diagnose:
                    event_log = output / ("action_metrics.csv" if action_only else
                                          "rgb_metrics.csv" if rgb_mode else "event_metrics.csv")
                    diagnostic_loss_keys = ("loss_action",) if action_only else metric_keys[3:]
                    write_header = not event_log.exists()
                    with event_log.open("a", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=["step", *diagnostic_loss_keys, *diagnostic_keys])
                        if write_header:
                            writer.writeheader()
                        writer.writerow(dict(step=step, **{key: metrics[key] for key in (*diagnostic_loss_keys, *diagnostic_keys)}))
                    if action_only:
                        logger.info("step=%d action=%.6f action_x0_mse=%.6f action_x0_mae=%.6f",
                                    step, metrics["loss_action"], metrics["action_x0_mse"], metrics["action_x0_mae"])
                    elif rgb_mode:
                        logger.info(
                            "step=%d latent_flow=%.6f action=%.6f rgb_x0_mse=%.6f rgb_x0_psnr=%.2f",
                            step, metrics["latent_flow_mse"], metrics["loss_action"],
                            metrics["rgb_x0_mse"], metrics["rgb_x0_psnr"])
                    else:
                        logger.info(
                            "step=%d latent_flow=%.6f action=%.6f voxel_x0=%.6f precision=%.4f recall=%.4f f1=%.4f",
                            step, metrics["latent_flow_mse"], metrics["loss_action"], metrics["voxel_x0_mse"],
                            metrics["event_precision"], metrics["event_recall"], metrics["event_f1"])
                elif action_only:
                    logger.info("step=%d action=%.6f grad_norm=%.6f",
                                step, metrics["loss_action"], metrics["grad_norm"])
                else:
                    logger.info(
                        "step=%d total=%.6f %s=%.6f action=%.6f grad_norm=%.6f",
                        step, metrics["loss_total"], "rgb" if rgb_mode else "event",
                        metrics[video_loss_key], metrics["loss_action"], metrics["grad_norm"],
                    )
                if run is not None:
                    wandb_metrics = (metrics if should_diagnose else {
                        key: metrics[key] for key in
                        (("loss_total", "loss_action", "grad_norm", "lr") if action_only else
                         ("loss_total", video_loss_key, "loss_action", "loss_video", "grad_norm", "lr"))
                    })
                    run.log({f"train/{key}": value for key, value in wandb_metrics.items()}, step=step)
            should_pure_noise_eval = (
                pure_noise_eval is not None and eval_due(step, t.max_steps, pure_noise_eval)
            )
            if save_checkpoints and (step == t.max_steps or
                                     (t.save_steps and step % t.save_steps == 0) or
                                     should_pure_noise_eval):
                # Serialize only EWM weights, excluding external frozen encoders.
                # Do not call Accelerator.get_state_dict() here: on a normal
                # single-GPU run it imports optional DeepSpeed solely while
                # checking wrappers, which fails when CUDA_HOME is absent.
                # DDP exposes the actual EWM module via .module; otherwise the
                # prepared object is already the model.
                policy_model = getattr(model, "module", model)
                state = policy_model.state_dict()
                if accelerator.is_main_process:
                    temporary = output / "checkpoint.tmp"
                    torch.save(
                        dict(
                            format=getattr(policy_model, "checkpoint_format",
                                           "ewm_action_only_v1" if action_only else "ewm_rgb_action_latent_v1" if rgb_mode else "ewm_wan_latent_v2"),
                            model=policy_model.policy_state_dict(state),
                            model_config=OmegaConf.to_container(cfg.model, resolve=True),
                            step=step,
                            horizon=cfg.dataloader.chunk_size,
                        ),
                        temporary,
                    )
                    temporary.replace(output / "checkpoint.pt")
                    # Preserve a policy snapshot from an interrupted earlier attempt.
                    _publish_policy_snapshot(output, step)
                accelerator.wait_for_everyone()
                if save_full_states:
                    save_resume_state(accelerator, output, step, epoch, batches_consumed,
                                      epoch_generator_state, optimizer, scheduler, contract)
                    if accelerator.is_main_process:
                        prune_resume_states(output, keep_resume_states)
                    accelerator.wait_for_everyone()
                if accelerator.is_main_process and should_pure_noise_eval:
                    submit_pure_noise_eval(output, step, pure_noise_eval)
                if accelerator.is_main_process and keep_policy_checkpoints is not None:
                    prune_policy_checkpoints(output, keep_policy_checkpoints)
                accelerator.wait_for_everyone()
            if profile_segments and step > profile_warmup_steps and not diagnose:
                for key, value in timings.items():
                    if key != "_clock":
                        profile_totals[key] = profile_totals.get(key, 0.0) + value
                profile_count += 1
                if profile_count >= profile_log_every:
                    names = sorted(profile_totals)
                    local = torch.tensor([profile_totals[name] / profile_count for name in names],
                                         device=accelerator.device)
                    means = accelerator.reduce(local, reduction="mean").tolist()
                    if accelerator.is_main_process:
                        values = dict(zip(names, means))
                        path = output / "segment_profile.csv"
                        write_header = not path.exists()
                        with path.open("a", newline="") as handle:
                            writer = csv.DictWriter(handle, fieldnames=["step", "samples", *names])
                            if write_header:
                                writer.writeheader()
                            writer.writerow({"step": step, "samples": profile_count, **values})
                        logger.info("profile step=%d samples=%d %s", step, profile_count,
                                    " ".join(f"{name}={values[name]:.4f}s" for name in names))
                    profile_totals.clear()
                    profile_count = 0
            profile_data_clock = time.perf_counter()
            if step >= t.max_steps:
                break
        epoch += 1
        skip_batches = 0
    if run is not None:
        run.finish()
    return output / "checkpoint.pt"


@hydra.main(version_base=None, config_path="../../configs", config_name="train_ewm")
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
