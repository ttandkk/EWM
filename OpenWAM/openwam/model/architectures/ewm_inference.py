"""Inference-only weight snapshots for the unchanged EWM velocity core."""

import torch
from torch import nn


class _VelocityWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.predict_velocity(*args, **kwargs)


def bf16_velocity_snapshot(model):
    """Detached Linear weights owned by the callable, excluded from policy state."""
    wrapper = _VelocityWrapper(model)
    weights = {}
    for name, module in model.named_modules():
        if name.startswith(('context_encoder', 'event_codec')):
            continue
        if isinstance(module, nn.Linear):
            for key, value in module.named_parameters(recurse=False):
                cached = value.detach().to(dtype=torch.bfloat16, copy=True)
                # Otherwise CUDA Graph stages these ordinary tensor inputs every step.
                torch._dynamo.mark_static_address(cached)
                weights[f'model.{name}.{key}'] = cached

    def call(*args, **kwargs):
        return torch.func.functional_call(wrapper, weights, args, kwargs, strict=False)

    return call
