"""Tiny frozen encoders and Lean codec shared by ordinary EWM tests."""
import torch

from openwam.model.architectures.ewm import EWMContextEncoder
from openwam.model.architectures.lean_event_codec import LeanEventCodec
from tests.test_ewm_model import FakeLeanVAE, FakeDino, FakeText


def options(cameras=1,mode='joint',checkpoint=False):
    return dict(dim=16,action_hidden_dim=16,heads=2,layers=2,
        event_codec_impl='leanvae16_joint',event_temporal_compression=4,
        event_spatial_compression=32,event_patch_size=4,
        event_camera_mode='dual' if cameras==2 else 'wrist_only',
        event_codec=LeanEventCodec(vae=FakeLeanVAE(),batch_size=2),
        context_encoder=EWMContextEncoder(FakeDino(),FakeText(),16),
        training_mode=mode,gradient_checkpointing=checkpoint)



def full_grid_batch(cameras=1, horizon=4):
    return dict(
        obs_voxel=torch.rand(1, cameras, 2, 256, 256),
        future_voxel=torch.rand(1, horizon, cameras, 2, 256, 256),
        proprio=torch.rand(1, 8), action=torch.rand(1, horizon, 7),
        obs_video=torch.zeros(1, 2, 3, 256, 256, dtype=torch.uint8),
        prompt=["stack"],
    )

