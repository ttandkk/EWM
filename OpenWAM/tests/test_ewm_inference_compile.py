"""Sampling compilation must not replace training or retain request outputs."""
from types import SimpleNamespace

import pytest
import torch

from openwam.model.architectures.ewm import EWMModel
from openwam.model.architectures.sparse_token_ewm import SparseTokenEWMModel
from tests.test_sparse_token_ewm import model_options, small_batch


@pytest.mark.parametrize("cls", [EWMModel, SparseTokenEWMModel])
@pytest.mark.parametrize("training_mode", ["joint", "event_only"])
def test_compiled_sampling_preserves_training_and_handles_new_requests(cls, training_mode):
    model = cls(**model_options(training_mode=training_mode))
    original_velocity = model.predict_velocity
    keys = set(model.state_dict())
    options = dict(horizon=4, num_steps=3, decode_future=False,
                   normalizer=SimpleNamespace(denormalize_action=lambda x: x))
    batches = [small_batch(), small_batch()]
    batches[1]["obs_voxel"][..., :128, :] = 0
    expected = [model.sample_policy(b, generator=torch.Generator().manual_seed(43), **options)
                for b in batches]
    model.compile_inference(mode=None, backend="eager")
    assert model.predict_velocity == original_velocity
    outputs = [model.sample_policy(b, generator=torch.Generator().manual_seed(43), **options)
               for b in batches]
    for actual, reference in zip(outputs, expected):
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name])
    assert model.training
    assert set(model.state_dict()) == keys

    # Compilation only affects sampling, and a preceding sample leaves no stale state.
    loss = model(batches[1])["loss"]
    loss.backward()
    assert model.event_input.weight.grad is not None
    assert torch.isfinite(model.event_input.weight.grad).all()
    model.disable_inference_compile()
    assert model._inference_velocity is None
    actual = model.sample_policy(batches[1], generator=torch.Generator().manual_seed(43), **options)
    for name in expected[1]:
        torch.testing.assert_close(actual[name], expected[1][name])


def test_inference_compile_defaults_to_fullgraph_reduce_overhead(monkeypatch):
    model = EWMModel(**model_options())
    calls = []

    def compile_stub(function, **options):
        calls.append(options)
        return function

    monkeypatch.setattr(torch, "compile", compile_stub)
    model.compile_inference()
    assert calls == [dict(mode="reduce-overhead", dynamic=False, fullgraph=True)]


@pytest.mark.parametrize("cls", [EWMModel, SparseTokenEWMModel])
@pytest.mark.parametrize("starts_training", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_sampling_mode_switches_only_when_needed(cls, starts_training, fails, monkeypatch):
    model = cls(**model_options()).train(starts_training)
    calls = []
    train = model.train

    def record(mode=True):
        calls.append(mode)
        return train(mode)

    monkeypatch.setattr(model, "train", record)
    if fails:
        def broken(*args):
            raise RuntimeError("context failed")
        monkeypatch.setattr(model, "_context", broken)
        with pytest.raises(RuntimeError, match="context failed"):
            model.sample_policy(small_batch(), horizon=4)
    else:
        model.sample_policy(small_batch(), horizon=4, num_steps=1, decode_future=False,
                            normalizer=SimpleNamespace(denormalize_action=lambda x: x))
    assert model.training == starts_training
    assert calls == ([False, True] if starts_training else [])
    assert not model.event_codec.training


@pytest.mark.parametrize("cls", [EWMModel, SparseTokenEWMModel])
@pytest.mark.parametrize("invalidate", ["train", "load", "policy_load", "move", "disable"])
def test_cached_compile_keeps_parameters_and_invalidates_snapshot(cls, invalidate, monkeypatch):
    model = cls(**model_options()).eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    grads = {k: v.requires_grad for k, v in model.named_parameters()}
    calls, resets = [], []
    monkeypatch.setattr(torch, "compile", lambda f, **kw: calls.append(kw) or f)
    monkeypatch.setattr(torch._dynamo, "reset_code", lambda code: resets.append(code))
    model.compile_inference(cache_bf16_weights=True)
    assert calls[0] == dict(mode='reduce-overhead', dynamic=False, fullgraph=True)
    assert model._inference_bf16_weights
    model.eval()
    assert model._inference_velocity is not None
    if invalidate == 'train':
        model.train()
    elif invalidate == 'load':
        model.load_state_dict(before)
    elif invalidate == 'policy_load':
        model.load_policy_state_dict(model.policy_state_dict(before))
    elif invalidate == 'move':
        model.to('cpu')
    else:
        model.disable_inference_compile()
    assert model._inference_velocity is None
    assert not model._inference_bf16_weights
    assert len(resets) == 1
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert grads == {k: v.requires_grad for k, v in model.named_parameters()}


def test_cached_weights_require_eval_and_bf16_autocast(monkeypatch):
    model = EWMModel(**model_options())
    with pytest.raises(ValueError, match="requires model.eval"):
        model.compile_inference(cache_bf16_weights=True)
    model.eval()
    monkeypatch.setattr(torch, 'compile', lambda f, **kw: f)
    model.compile_inference(cache_bf16_weights=True)
    with pytest.raises(ValueError, match="requires BF16 autocast"):
        model.sample_policy(small_batch(), horizon=4)
    model.disable_inference_compile()


@pytest.mark.parametrize("cls", [EWMModel, SparseTokenEWMModel])
def test_training_after_cached_inference_uses_original_fp32_parameters(cls, monkeypatch):
    model = cls(**model_options()).eval()
    weight = model.event_input.weight
    monkeypatch.setattr(torch, 'compile', lambda f, **kw: f)
    model.compile_inference(cache_bf16_weights=True)
    model.train()
    assert model._inference_velocity is None
    assert model.event_input.weight is weight
    assert weight.dtype == torch.float32
    loss = model(small_batch())["loss"]
    loss.backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
