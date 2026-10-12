#!/usr/bin/env python3
"""Count EWM policy and frozen-encoder parameters without loading full weights.

From OpenWAM/:
    ../envs/openwam/bin/python tests/count_ewm_parameters.py
    ../envs/openwam/bin/python tests/count_ewm_parameters.py model.dim=512 model.action_hidden_dim=512
    ../envs/openwam/bin/python tests/count_ewm_parameters.py --skip-frozen --json model.event_camera_mode=dual
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors import safe_open
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.model.architectures.ewm import EWMContextEncoder, EWMModel  # noqa: E402


class _DinoShape(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.properties = SimpleNamespace(z_dim=width)


class _TextShape(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.hidden_size = width


class _CodecShape(nn.Module):
    def __init__(self, impl: str):
        super().__init__()
        specs = {
            "leanvae16_joint": (16, 8, "joint"),
            "wan22_joint": (48, 16, "joint"),
            "wan22_frame": (48, 16, "frame"),
        }
        if impl not in specs:
            raise ValueError(f"Unsupported event codec: {impl}")
        self.latent_channels, self.spatial_compression, self.mode = specs[impl]


def _json_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _hf_tensor_count(directory: Path) -> int:
    """Count unique saved tensors, including sharded safetensors checkpoints."""
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        files = sorted(set(_json_config(index)["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    total = 0
    for name in files:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen weight file: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            total += sum(math.prod(handle.get_slice(key).get_shape()) for key in handle.keys())
    return total


def _tensor_count(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel()
    if isinstance(obj, dict):
        return sum(_tensor_count(value) for value in obj.values())
    return 0


def _vae_tensor_count(impl: str, location: Path) -> int:
    if impl == "leanvae16_joint":
        path = location / "LeanVAE-dim16.ckpt" if location.is_dir() else location
    else:
        if not location.is_dir():
            raise FileNotFoundError(f"Wan VAE directory does not exist: {location}")
        candidates = sorted({path for pattern in ("*VAE*.pth", "*VAE*.safetensors", "*vae*.pth", "*vae*.safetensors")
                             for path in location.glob(pattern)})
        if len(candidates) != 1:
            raise ValueError(f"Expected one Wan VAE checkpoint in {location}, found {len(candidates)}")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen VAE checkpoint: {path}")
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as handle:
            return sum(math.prod(handle.get_slice(key).get_shape()) for key in handle.keys())
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("state_dict", checkpoint)
    return _tensor_count(state)


def count_parameters(model_config: dict, *, skip_frozen: bool = False) -> dict:
    options = dict(model_config)
    if options.pop("type", "ewm") != "ewm":
        raise ValueError("This script counts only EWM models")
    dino_path = Path(options.pop("dino_path"))
    text_path = Path(options.pop("text_path"))
    vae_path = Path(options["event_vae_path"])
    options.pop("text_max_length", None)
    dino_width = int(_json_config(dino_path / "config.json")["hidden_size"])
    text_width = int(_json_config(text_path / "config.json")["d_model"])
    with torch.device("meta"):
        context = EWMContextEncoder(_DinoShape(dino_width), _TextShape(text_width), options["dim"])
        model = EWMModel(context_encoder=context, event_codec=_CodecShape(options["event_codec_impl"]), **options)

    groups = {"event_experts": 0, "action_experts": 0, "context_projection": 0, "other_policy": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("blocks.") and ".event." in name:
            group = "event_experts"
        elif name.startswith("blocks.") and ".action." in name:
            group = "action_experts"
        elif name.startswith("context_encoder."):
            group = "context_projection"
        else:
            group = "other_policy"
        groups[group] += parameter.numel()
    policy_total = sum(groups.values())
    result = {
        "config": {"event_dim": options["dim"], "action_hidden_dim": options["action_hidden_dim"],
                   "heads": options["heads"], "layers": options["layers"],
                   "event_camera_mode": options["event_camera_mode"], "event_codec_impl": options["event_codec_impl"]},
        "trainable": {**groups, "total": policy_total},
    }
    if not skip_frozen:
        frozen = {
            "dino": _hf_tensor_count(dino_path),
            "t5": _hf_tensor_count(text_path),
            "vae": _vae_tensor_count(options["event_codec_impl"], vae_path),
        }
        frozen["total"] = sum(frozen.values())
        result["frozen"] = frozen
        result["full_total"] = policy_total + frozen["total"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="train_ewm", help="Hydra config name under OpenWAM/configs")
    parser.add_argument("--skip-frozen", action="store_true", help="Count only trainable EWM parameters")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model.dim=512")
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name=args.config_name, overrides=args.overrides)
    result = count_parameters(OmegaConf.to_container(config.model, resolve=True), skip_frozen=args.skip_frozen)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("EWM configuration:", result["config"])
        for label in ("trainable", "frozen"):
            if label in result:
                print(f"{label}:")
                for name, count in result[label].items():
                    print(f"  {name:20s} {count:>13,d}")
        if "full_total" in result:
            print(f"full_total:          {result['full_total']:>13,d}")


if __name__ == "__main__":
    main()
