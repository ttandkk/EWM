"""Frozen LeanVAE-16 event codec, retaining the pretrained raw latent coordinates."""
from argparse import Namespace
from pathlib import Path
import torch
from torch import nn
from .event_codec import FrozenEventCodec

class _LeanAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.to(torch.bfloat16).eval().requires_grad_(False)
        self.model.set_tile_inference(False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Keep the frozen codec in its configured BF16 compute precision.
        self.model.to(torch.bfloat16)
        return self

    def batch_encode(self, rgb):
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            return self.model.encode((rgb.float() * .5).to(torch.bfloat16))

    def decode(self, z, *, tiled=False):
        if tiled:
            raise ValueError("Lean event codec does not use temporal tiling")
        with torch.autocast(device_type=z.device.type, enabled=False):
            rgb = self.model.decode(z.to(torch.bfloat16), is_image=z.shape[2] == 1)
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(2)
            return rgb.float() * 2

class LeanEventCodec(FrozenEventCodec):
    latent_channels = 16
    spatial_compression = 8

    def __init__(self, model_path=None, *, batch_size=16, vae=None):
        if vae is None:
            if not model_path:
                raise ValueError("event_vae_path must point to LeanVAE-dim16.ckpt or its directory")
            path = Path(model_path)
            if path.is_dir():
                path = path / "LeanVAE-dim16.ckpt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            args = Namespace(**checkpoint["args"])
            if args.latent_dim != 16:
                raise ValueError("leanvae16_joint requires the 16-channel LeanVAE checkpoint")
            from openwam.model.video_backbone.encoder.leanvae_src.autoencoder import LeanVAE
            vae = LeanVAE(args)
            vae.load_state_dict(checkpoint["state_dict"], strict=True)
        super().__init__(vae=_LeanAdapter(vae), latent_channels=16, spatial_compression=8,
                         batch_size=batch_size, mode="joint")
