"""Lightweight T5-small prompt encoder for EWM policies.

This is intentionally independent of Wan's UMT5-XXL text path. It returns
token-level context and a padding mask so EWM action/event experts can consume
it through cross-attention without depending on a video backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


class T5SmallTextEncoder(nn.Module):
    """Frozen-by-default HuggingFace T5-small prompt encoder.

    forward(prompts) returns (text_tokens, text_mask):

    - text_tokens: [B, max_length, 512];
    - text_mask: [B, max_length] boolean, where True marks real tokens.

    model_path accepts a local HuggingFace snapshot or google-t5/t5-small.
    The latter is downloaded by HuggingFace only when it is not already cached.
    """

    MODEL_ID = "google-t5/t5-small"
    HIDDEN_SIZE = 512

    def __init__(
        self,
        model_path: str | Path = MODEL_ID,
        *,
        max_length: int = 64,
        freeze: bool = True,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if int(max_length) <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")

        # Kept local so importing OpenWAM does not require transformers unless
        # this optional prompt encoder is actually constructed.
        from transformers import AutoTokenizer, T5EncoderModel

        load_kwargs = {"local_files_only": local_files_only}
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), **load_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(load_kwargs)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = T5EncoderModel.from_pretrained(str(model_path), **model_kwargs)
        if device is not None:
            self.model.to(device)

        config = self.model.config
        if getattr(config, "model_type", None) != "t5":
            raise ValueError(
                f"T5SmallTextEncoder requires a T5 checkpoint, got model_type={getattr(config, 'model_type', None)!r}"
            )
        if int(getattr(config, "d_model", -1)) != self.HIDDEN_SIZE:
            raise ValueError(
                f"T5SmallTextEncoder requires d_model={self.HIDDEN_SIZE}, got {getattr(config, 'd_model', None)}. "
                "Use google-t5/t5-small rather than a larger T5 checkpoint."
            )

        self.max_length = int(max_length)
        self.freeze = bool(freeze)
        if self.freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.eval()

    @property
    def hidden_size(self) -> int:
        return self.HIDDEN_SIZE

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def train(self, mode: bool = True) -> "T5SmallTextEncoder":
        # A frozen language encoder must keep dropout disabled even while the
        # enclosing policy is in train mode.
        super().train(False if self.freeze else mode)
        self.model.train(False if self.freeze else mode)
        return self

    def forward(self, prompts: str | Sequence[str]) -> tuple[Tensor, Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [str(prompt) for prompt in prompts]
        if not prompts:
            raise ValueError("T5SmallTextEncoder received an empty prompts list.")

        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.set_grad_enabled(not self.freeze):
            tokens = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        return tokens, attention_mask.to(dtype=torch.bool)

    encode = forward


__all__ = ["T5SmallTextEncoder"]
