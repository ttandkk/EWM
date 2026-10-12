"""Unit tests for the lightweight EWM T5-small prompt encoder."""

from __future__ import annotations

import sys
import types

import torch
from torch import nn

from openwam.model.text_encoder.t5_small import T5SmallTextEncoder


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls()
        instance.path = path
        instance.kwargs = kwargs
        return instance

    def __call__(self, prompts, *, max_length, return_tensors, **kwargs):
        assert return_tensors == "pt"
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(prompts):
            length = min(len(prompt) + 1, max_length)
            ids[row, :length] = 7
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FakeT5(nn.Module):
    def __init__(self, d_model=512, model_type="t5"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = types.SimpleNamespace(d_model=d_model, model_type=model_type)
        self.calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls()
        instance.path = path
        instance.kwargs = kwargs
        return instance

    def forward(self, *, input_ids, attention_mask, return_dict):
        self.calls.append((input_ids, attention_mask, return_dict))
        hidden = torch.ones(
            input_ids.shape[0],
            input_ids.shape[1],
            self.config.d_model,
            device=input_ids.device,
        )
        return types.SimpleNamespace(last_hidden_state=hidden)


def _install_fake_transformers(monkeypatch):
    fake = types.SimpleNamespace(AutoTokenizer=_FakeTokenizer, T5EncoderModel=_FakeT5)
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_t5_small_text_encoder_returns_tokens_and_mask(monkeypatch):
    _install_fake_transformers(monkeypatch)
    encoder = T5SmallTextEncoder("/fake/t5-small", max_length=8, freeze=True)

    tokens, mask = encoder(["stack blocks", "open drawer"])

    assert tokens.shape == (2, 8, 512)
    assert tokens.dtype == torch.float32
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool
    assert mask[:, 0].all()
    assert not encoder.model.training
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())

    encoder.train(True)
    assert not encoder.model.training


def test_t5_small_text_encoder_rejects_wrong_checkpoint_geometry(monkeypatch):
    class _WrongT5(_FakeT5):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls(d_model=768)

    fake = types.SimpleNamespace(AutoTokenizer=_FakeTokenizer, T5EncoderModel=_WrongT5)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    try:
        T5SmallTextEncoder("/fake/not-small")
    except ValueError as exc:
        assert "d_model=512" in str(exc)
    else:
        raise AssertionError("expected geometry validation failure")
