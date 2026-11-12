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

from openwam.dataloader.ewm_normalizer import EWMNormalizer
from openwam.model.architectures.ewm import EWMModel
from openwam.model.architectures.ewm_checkpoint import normalize_policy_payload


def load_policy(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    payload = normalize_policy_payload(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = EWMModel.from_config(payload["model_config"])
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


def prepare_observation(observation, normalizer, device, *, training_mode="joint", event_camera_mode="wrist_only"):
    from scripts.eval_ewm.ewm_action_only_protocol import validate_timing
    if training_mode in ("action_only", "rgb_action"):
        validate_timing(observation)
    else:
        validate_observation_timing(observation)
    rgb = torch.as_tensor(observation["obs_video"])
    state = torch.as_tensor(observation["proprio"])
    if rgb.ndim != 4 or rgb.shape[:2] != (2, 3) or rgb.dtype != torch.uint8:
        raise ValueError("environment RGB must be uint8 [2,3,H,W]")
    if state.shape != (8,):
        raise ValueError("environment state shape mismatch")
    batch = dict(
        obs_video=rgb[None].to(device),
        proprio=normalizer.normalize_proprio(state)[None].to(device),
        prompt=[str(observation["prompt"])],
    )
    if training_mode not in ("action_only", "rgb_action"):
        voxel = torch.as_tensor(observation["obs_voxel"])
        if voxel.ndim != 4 or voxel.shape[0] not in (1, 2) or voxel.shape[1:] != (2 * normalizer.voxel_bins, *rgb.shape[-2:]):
            raise ValueError("environment voxel shape mismatch")
        batch['obs_voxel'] = normalizer.normalize_voxel(voxel)[None].to(device)
    if training_mode == "rgb_action":
        views = {"dual": slice(None), "wrist_only": slice(1, 2), "global_only": slice(0, 1)}
        batch['obs_rgb'] = batch['obs_video'][:, views[event_camera_mode]]
    return batch


def prepare_sample(sample, model, device):
    """Forward current observations (raw or cached), never future labels."""
    cached = 'current_dino_features' in sample
    keys = ['current_dino_features' if cached else 'obs_video', 'proprio', 'prompt']
    if model.training_mode == 'rgb_action':
        keys.append('obs_rgb')
    elif model.training_mode != 'action_only':
        keys.append('obs_voxel')
    batch = {key: sample[key][None].to(device) if isinstance(sample[key], torch.Tensor)
             else [sample[key]] for key in keys}
    return batch


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
        for step in range(max_steps):
            batch = prepare_observation(observation, normalizer, device,
                training_mode=model.training_mode, event_camera_mode=model.event_camera_mode)
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
        if model.training_mode not in ("action_only", "rgb_action"):
            if not dataset.direct_event_frames:
                raise ValueError("evaluation requires an aligned direct event-frame dataset")
            verify_exported_event_alignment(dataset, [args.sample_index], model.event_prediction_horizon(horizon))
        sample = dataset[args.sample_index]
        batch = prepare_sample(sample, model, args.device)
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
