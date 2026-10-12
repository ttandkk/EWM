"""Frozen Wan2.2 event codec; joint temporal compression or independent frames."""
from .event_codec import FrozenEventCodec

class WanEventCodec(FrozenEventCodec):
    latent_channels = 48
    spatial_compression = 16

    def __init__(self, model_path=None, *, batch_size=16, vae=None, mode="joint"):
        if vae is None:
            if not model_path:
                raise ValueError("event_vae_path must point to the pretrained Wan2.2 VAE directory")
            from openwam.model.video_backbone.encoder.wan22_vae import WanVideoVAEEncoder
            vae = WanVideoVAEEncoder.from_pretrained(str(model_path))
        if vae.properties.z_dim != 48 or vae.properties.spatial_compression != 16:
            raise ValueError("event codec requires Wan2.2 VAE (48 latent channels, spatial factor16)")
        super().__init__(vae=vae, latent_channels=48, spatial_compression=16, batch_size=batch_size, mode=mode)
