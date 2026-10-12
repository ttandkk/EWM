"""Independent EWM policy inference, with one executed action per observation.

An environment factory `package.module:factory` must return a single Gym-style
simulator with reset()/step(action). Its observation dictionary contains raw
obs_video uint8 [2,3,H,W], obs_voxel counts [event_cameras,2,H,W] for
one 30 Hz event interval, proprio [8], prompt (str), plus obs_video_time_s,
proprio_time_s, obs_event_start_s, obs_event_end_s. RGB and proprio must
match the end of the observed event interval, regardless of training history.
step receives a physical 7-D numpy action.
Use --dataset for offline sampling against the existing normalized EWMDataset.
"""

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np

from openwam.dataloader.ewm_normalizer import EWMNormalizer
from openwam.model.architectures.ewm import EWMModel


def load_policy(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") not in ("ewm_wan_latent_v2", "ewm_action_only_v1", "ewm_rgb_event_memory_latent_v1"):
        raise ValueError("Dense-voxel EWM checkpoints are incompatible with the framewise Wan latent architecture")
    if (payload["model_config"].get("training_mode", "joint") == "action_only") != (payload["format"] == "ewm_action_only_v1"):
        raise ValueError("action-only checkpoint format and training_mode must match")
    model_cls = EWMModel
    memory_format = payload["format"] == "ewm_rgb_event_memory_latent_v1"
    if memory_format != (payload["model_config"].get("type") == "ewm_rgb_event_memory"):
        raise ValueError("RGB/event memory checkpoint format and model type must match")
    if memory_format:
        from openwam.model.architectures.hierarchical_ewm import HierarchicalMemoryEWMModel
        model_cls = HierarchicalMemoryEWMModel
    if payload["model_config"].get("type", "ewm") == "ewm_sparse":
        from openwam.model.architectures.sparse_token_ewm import SparseTokenEWMModel
        model_cls = SparseTokenEWMModel
    model = model_cls.from_config(payload["model_config"])
    model.load_policy_state_dict(payload["model"])
    model.to(device).eval()
    normalizer = EWMNormalizer(checkpoint.parent / "normalization.json")
    return model, normalizer, int(payload["horizon"])


def validate_observation_timing(observation, *, fps: float = 30.0) -> None:
    """Require one causal event interval ending at the RGB/state observation."""
    names = ("obs_video_time_s", "proprio_time_s", "obs_event_start_s", "obs_event_end_s")
    missing = [name for name in names if name not in observation]
    if missing:
        raise ValueError(f"live EWM observation needs explicit timestamps: {missing}")
    video_t, state_t, start_t, end_t = (float(observation[name]) for name in names)
    if not all(map(math.isfinite, (video_t, state_t, start_t, end_t))):
        raise ValueError("live EWM observation timestamps must be finite")
    dt = 1.0 / fps
    tolerance = 0.001  # 1 ms, wider than the collector's sub-microsecond grid drift.
    if abs((end_t - start_t) - dt) > tolerance:
        raise ValueError("event voxel must cover one 30 Hz control interval")
    if abs(video_t - state_t) > tolerance:
        raise ValueError("RGB and proprioception timestamps differ")
    if abs(video_t - end_t) > tolerance:
        raise ValueError("RGB/proprio time must equal the observed event interval end")


def prepare_observation(observation, normalizer, device):
    validate_observation_timing(observation)
    rgb = torch.as_tensor(observation["obs_video"])
    voxel = torch.as_tensor(observation["obs_voxel"])
    state = torch.as_tensor(observation["proprio"])
    if rgb.ndim != 4 or rgb.shape[:2] != (2, 3) or rgb.dtype != torch.uint8:
        raise ValueError("environment RGB must be uint8 [2,3,H,W]")
    if voxel.ndim != 4 or voxel.shape[0] not in (1, 2) or voxel.shape[1:] != (2 * normalizer.voxel_bins, *rgb.shape[-2:]) or state.shape != (8,):
        raise ValueError("environment voxel/state shape mismatch")
    return dict(
        obs_video=rgb[None].to(device),
        obs_voxel=normalizer.normalize_voxel(voxel)[None].to(device),
        proprio=normalizer.normalize_proprio(state)[None].to(device),
        prompt=[str(observation["prompt"])],
    )


@torch.no_grad()
def evaluate(model, normalizer, env, *, horizon, num_steps=20, episodes=1, max_steps=600,
             output=None):
    if min(episodes, max_steps, horizon, num_steps) <= 0:
        raise ValueError("episodes, max_steps, horizon and num_steps must be positive")
    device = next(model.parameters()).device
    model.eval()
    records = []
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    for episode in range(episodes):
        reset = env.reset()
        observation = reset[0] if isinstance(reset, tuple) else reset
        reward_sum = 0.0
        history = None
        if getattr(model, "checkpoint_format", "") == "ewm_rgb_event_memory_latent_v1":
            from scripts.eval_ewm.ewm_memory_history import RGBEventHistoryBuffer
            history = RGBEventHistoryBuffer()
        for step in range(max_steps):
            batch = prepare_observation(observation, normalizer, device)
            if history is not None:
                inputs = history.observe(np.asarray(observation["obs_video"]), np.asarray(observation["obs_voxel"]), step)
                for key,value in inputs.items():
                    tensor = torch.from_numpy(value)
                    if key == "memory_event": tensor = normalizer.normalize_voxel(tensor)
                    batch[key] = tensor[None].to(device)
            result = model.sample_policy(batch, horizon=horizon, normalizer=normalizer, num_steps=num_steps)
            # Receding horizon: never execute the rest of the old chunk.
            action = result["action"][0, 0].detach().cpu().numpy()
            if output is not None:
                torch.save(
                    dict(
                        observation={k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
                        prediction={k: v.cpu() for k, v in result.items()},
                    ),
                    output / f"episode_{episode:04d}_step_{step:04d}.pt",
                )
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            if terminated or truncated:
                break
        records.append(
            dict(
                episode=episode,
                steps=step + 1,
                reward=reward_sum,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )
    if output is not None:
        (output / "metrics.json").write_text(json.dumps(records, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--env-factory", help="Single simulator factory: module:function")
    source.add_argument("--dataset", help="Offline EWMDataset root; no environment actions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", default="outputs/ewm_policy")
    parser.add_argument("--compile-core", action="store_true", help="Compile the fixed-shape EWM denoising core")
    args = parser.parse_args()
    model, normalizer, trained_horizon = load_policy(args.checkpoint, args.device)
    if args.compile_core:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile-core requires torch.compile support")
        model.predict_velocity = torch.compile(model.predict_velocity, dynamic=False)
    horizon = trained_horizon if args.horizon is None else args.horizon
    if horizon != trained_horizon:
        raise ValueError("EWM inference horizon must match the checkpoint training horizon")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(args.checkpoint).parent / "config.yaml")
    if args.device == "cpu":
        torch.backends.mkldnn.enabled = False
    if args.dataset:
        from openwam.dataloader.registry import build_dataset
        from scripts.eval_ewm.ewm_eval_time_audit import verify_exported_event_alignment

        cfg.dataloader.dataset_dir = str(Path(args.dataset).resolve())
        cfg.dataloader.gpu_voxelization = False
        cfg.dataloader.observation_row_offset = 0  # Same-row RGB/state/event-frame.
        cfg.dataloader.normalization_stats_path = str(Path(args.checkpoint).parent / "normalization.json")
        dataset = build_dataset(cfg.dataloader, split="train")
        if model.training_mode != "action_only":
            if not dataset.direct_event_frames:
                raise ValueError("evaluation requires an aligned direct event-frame dataset")
            verify_exported_event_alignment(dataset, [args.sample_index], horizon)
        sample = dataset[args.sample_index]
        batch = {k: sample[k][None].to(args.device) if isinstance(sample[k], torch.Tensor)
                 else [sample[k]] for k in
                 (("obs_video", "proprio", "prompt") if model.training_mode == "action_only" else
                  ("obs_video", "obs_voxel", "proprio", "prompt"))}
        if getattr(model, "checkpoint_format", "") == "ewm_rgb_event_memory_latent_v1":
            for key,value in sample.items():
                if key.startswith("memory_"):
                    batch[key] = value[None].to(args.device)
            batch["fps"] = torch.tensor([sample["fps"]],device=args.device)
        result = model.sample_policy(batch, horizon=horizon, normalizer=normalizer, num_steps=args.num_steps)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in result.items()}, output / "prediction.pt")
    else:
        module, factory = args.env_factory.split(":", 1)
        env = getattr(importlib.import_module(module), factory)()
        try:
            print(
                json.dumps(
                    evaluate(
                        model,
                        normalizer,
                        env,
                        horizon=horizon,
                        num_steps=args.num_steps,
                        episodes=args.episodes,
                        max_steps=args.max_steps,
                        output=args.output,
                    ),
                    indent=2,
                )
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
