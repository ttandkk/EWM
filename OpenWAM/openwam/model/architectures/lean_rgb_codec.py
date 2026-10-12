"""Frozen LeanVAE-16 codec for real RGB video in the EWM latent layout."""
from __future__ import annotations

import torch
from torch import Tensor

from .lean_event_codec import LeanEventCodec


class LeanRGBCodec(LeanEventCodec):
    """[B,T,camera,3,H,W] RGB uint8 or float [0,1] <-> LeanVAE latents."""

    @torch.no_grad()
    def encode(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 6 or rgb.shape[3] != 3:
            raise ValueError("RGB input must be [B,T,camera,3,H,W]")
        b, t, cameras, _, h, w = rgb.shape
        if min(b, t, cameras, h, w) <= 0 or h % (2 * self.spatial_compression) or w % (2 * self.spatial_compression):
            raise ValueError("RGB spatial dimensions must be positive multiples of twice the VAE stride")
        if (t - 1) % 4:
            raise ValueError("joint RGB encoding requires observed1 + future4n frames")
        if rgb.dtype != torch.uint8 and not rgb.is_floating_point():
            raise TypeError("RGB input must be uint8 [0,255] or floating [0,1]")
        anchor = next(self.vae.parameters())
        videos = rgb.permute(0, 2, 3, 1, 4, 5).reshape(b * cameras, 3, t, h, w)
        chunks = []
        for part in videos.split(self.batch_size):
            # _LeanAdapter applies the final 0.5 scale required by LeanVAE.
            scale = (2 / 255) if part.dtype == torch.uint8 else 2.0
            chunks.append(self.vae.batch_encode((part.float() * scale - 1).to(anchor)).float())
        z = torch.cat(chunks)
        return z.reshape(b, cameras, self.latent_channels, 1 + (t - 1) // 4,
                         h // self.spatial_compression, w // self.spatial_compression).permute(0, 3, 1, 2, 4, 5).contiguous()

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 6 or latent.shape[3] != self.latent_channels:
            raise ValueError("RGB latent must be [B,T,camera,latent_channels,h,w]")
        b, t, cameras, _, h, w = latent.shape
        anchor = next(self.vae.parameters())
        videos = latent.permute(0, 2, 3, 1, 4, 5).reshape(b * cameras, self.latent_channels, t, h, w)
        chunks = []
        for part in videos.split(self.batch_size):
            rgb = self.vae.decode(part.to(anchor), tiled=False).float()
            chunks.append(((rgb + 1) * .5).clamp(0, 1))
        return torch.cat(chunks).reshape(b, cameras, 3, 1 + 4 * (t - 1),
                                          h * self.spatial_compression, w * self.spatial_compression).permute(0, 3, 1, 2, 4, 5).contiguous()
