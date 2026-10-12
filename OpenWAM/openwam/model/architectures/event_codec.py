"""Frozen pseudo-RGB event codec layout shared by pretrained video VAEs."""
from __future__ import annotations
import torch
from torch import nn, Tensor

class FrozenEventCodec(nn.Module):
    def __init__(self, *, vae, latent_channels, spatial_compression, batch_size=16, mode="joint"):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("codec batch_size must be positive")
        if mode not in ("joint", "frame"):
            raise ValueError("VAE mode must be joint or frame")
        self.vae = vae.eval().requires_grad_(False)
        self.latent_channels = latent_channels
        self.spatial_compression = spatial_compression
        self.batch_size = int(batch_size)
        self.mode = mode

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    @staticmethod
    def to_rgb(voxel: Tensor) -> Tensor:
        """[N,2,H,W] normalized OFF/ON -> [N,3,1,H,W] in [-1,1]."""
        if voxel.ndim != 4 or voxel.shape[1] != 2:
            raise ValueError("one-bin events must have exactly two channels: OFF, ON")
        n, p = voxel.unbind(1)
        return (torch.stack((n, (n + p) * .5, p), 1) * 2 - 1).unsqueeze(2)

    @staticmethod
    def from_rgb(rgb: Tensor) -> Tensor:
        if rgb.ndim != 5 or rgb.shape[1:3] != (3, 1):
            raise ValueError("framewise decoder must return [N,3,1,H,W]")
        return ((rgb[:, (0, 2), 0].float() + 1) * .5).clamp(0, 1)

    @torch.no_grad()
    def encode(self, voxel: Tensor) -> Tensor:
        """[B,T,camera,2,H,W] -> latents; joint T must be 4n+1."""
        if voxel.ndim != 6 or voxel.shape[3] != 2:
            raise ValueError("event input must be [B,K,camera,2,H,W]")
        b, k, c, _, h, w = voxel.shape
        if min(b, k, c, h, w) <= 0 or h % (self.spatial_compression * 2) or w % (self.spatial_compression * 2):
            raise ValueError("event image dimensions must be positive multiples of twice the codec spatial stride for VAE + DiT patching")
        if self.mode == "joint":
            return self._encode_joint(voxel)
        anchor = next(self.vae.parameters())
        chunks = []
        for part in voxel.reshape(-1, 2, h, w).split(self.batch_size):
            # Normalize in float32 before converting to VAE compute precision.
            rgb = self.to_rgb(part.float()).to(device=anchor.device, dtype=anchor.dtype)
            z = self.vae.batch_encode(rgb)
            if z.shape[1:] != (self.latent_channels, 1, h // self.spatial_compression, w // self.spatial_compression):
                raise RuntimeError(f"unexpected framewise VAE latent shape: {z.shape}")
            chunks.append(z[:, :, 0].float())
        return torch.cat(chunks).reshape(b, k, c, self.latent_channels, h // self.spatial_compression, w // self.spatial_compression)

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode with the same temporal mode used for encoding."""
        if latent.ndim != 6 or latent.shape[3] != self.latent_channels:
            raise ValueError("event latent must be [B,K,camera,latent_channels,h,w]")
        b, k, c, _, h, w = latent.shape
        if self.mode == "joint":
            return self._decode_joint(latent)
        anchor = next(self.vae.parameters())
        chunks = []
        for part in latent.reshape(-1, self.latent_channels, h, w).split(self.batch_size):
            z = part.unsqueeze(2).to(device=anchor.device, dtype=anchor.dtype)
            chunks.append(self.from_rgb(self.vae.decode(z, tiled=False)))
        return torch.cat(chunks).reshape(b, k, c, 2, h * self.spatial_compression, w * self.spatial_compression)

    def _encode_joint(self, voxel):
        b, t, c, _, h, w = voxel.shape
        if (t - 1) % 4:
            raise ValueError("joint event encoding requires observed1 + future4n frames")
        anchor = next(self.vae.parameters())
        videos = voxel.permute(0, 2, 3, 1, 4, 5).reshape(b*c, 2, t, h, w)
        chunks = []
        for part in videos.split(self.batch_size):
            n, p = part.float().unbind(1)
            rgb = (torch.stack((n, (n+p)*.5, p), 1)*2-1).to(anchor)
            chunks.append(self.vae.batch_encode(rgb).float())
        z = torch.cat(chunks)
        return z.reshape(b,c,self.latent_channels,1+(t-1)//4,h//self.spatial_compression,w//self.spatial_compression).permute(0,3,1,2,4,5).contiguous()

    def _decode_joint(self, latent):
        b, t, c, _, h, w = latent.shape
        anchor = next(self.vae.parameters())
        videos = latent.permute(0,2,3,1,4,5).reshape(b*c,self.latent_channels,t,h,w)
        chunks = []
        for part in videos.split(self.batch_size):
            rgb = self.vae.decode(part.to(anchor), tiled=False).float()
            chunks.append(((rgb[:, (0,2)]+1)*.5).clamp(0,1))
        return torch.cat(chunks).reshape(b,c,2,1+4*(t-1),h*self.spatial_compression,w*self.spatial_compression).permute(0,3,1,2,4,5).contiguous()
