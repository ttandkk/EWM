"""Wan video LoRA; same adapter math and targets as the FastWAM baseline.

Base Linear weights retain their names for native full-state checkpoints.
"""
from __future__ import annotations

import math

from torch import nn
from torch.nn import functional as F

DEFAULT_TARGETS = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)


class LoRALinear(nn.Linear):
    """Keep original weight names; the zero-initialized update preserves outputs."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        # Reuse the loaded base parameters rather than allocating another large weight.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        self.reset_lora_parameters()
        self.train(base.training)

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return F.linear(x, self.weight, self.bias) + update * self.scaling


def normalize_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("video_lora must be a mapping")
    unknown = set(config) - {"rank", "alpha", "dropout", "target_modules"}
    if unknown:
        raise ValueError(f"Unknown video_lora settings: {sorted(unknown)}")
    rank = config.get("rank", 16)
    alpha = float(config.get("alpha", rank))
    dropout = float(config.get("dropout", 0.0))
    targets = config.get("target_modules", DEFAULT_TARGETS)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("video_lora.rank must be a positive integer")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("video_lora.alpha must be finite and positive")
    if not 0 <= dropout < 1:
        raise ValueError("video_lora.dropout must be in [0, 1)")
    if not isinstance(targets, (list, tuple)) or not targets or any(not isinstance(t, str) or not t for t in targets):
        raise ValueError("video_lora.target_modules must be a nonempty list of module suffixes")
    return {"rank": rank, "alpha": alpha, "dropout": dropout, "target_modules": list(dict.fromkeys(targets))}


def inject_video_lora(video: nn.Module, config: dict) -> dict:
    config = normalize_config(config)
    matches = []
    found = set()
    # Only transformer blocks; embeddings, VAE and action expert are untouched.
    for name, module in video.named_modules():
        if not name.startswith("blocks."):
            continue
        targets = [t for t in config["target_modules"] if name.endswith("." + t)]
        if targets:
            if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
                raise ValueError(f"video_lora target must be an unwrapped Linear: {name}")
            matches.append((name, module))
            found.update(targets)
    missing = set(config["target_modules"]) - found
    if missing:
        raise ValueError(f"video_lora targets not found: {sorted(missing)}")
    for name, base in matches:
        parent, _, leaf = name.rpartition(".")
        setattr(video.get_submodule(parent), leaf, LoRALinear(base, config["rank"], config["alpha"], config["dropout"]))
    mark_only_lora_trainable(video)
    return config


def mark_only_lora_trainable(video: nn.Module):
    # Do not wrap frozen modules in no_grad: gradients must cross the frozen
    # video layers to reach adapters and the coupled action expert.
    video.requires_grad_(False)
    for module in video.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)


def is_adapter_key(key: str) -> bool:
    return key.endswith((".lora_A", ".lora_B"))

