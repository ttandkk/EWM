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


def configure_inference_compile(model, *, mode, cache_bf16_weights, cuda_graphs):
    from torch._inductor import config, list_mode_options
    options = dict(list_mode_options(mode))
    if not cuda_graphs:
        options['triton.cudagraphs'] = False
    enabled = bool(options.get('triton.cudagraphs', config.triton.cudagraphs))
    # torch.compile accepts either mode or options; resolve mode before overriding.
    model.compile_inference(mode=None, cache_bf16_weights=cache_bf16_weights, options=options)
    return enabled

def configure_action_compile(model, *, mode, cuda_graphs):
    from torch._inductor import config, list_mode_options
    options = dict(list_mode_options(mode))
    if not cuda_graphs:
        options['triton.cudagraphs'] = False
    enabled = bool(options.get('triton.cudagraphs', config.triton.cudagraphs))
    model.predict_velocity = torch.compile(model.predict_velocity, dynamic=False, options=options)
    return enabled
