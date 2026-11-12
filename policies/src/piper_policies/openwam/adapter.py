"""Piper image/state adapter around OpenWAM's native deployment engine."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from ..contract import action_reply, validate_request
from ..metadata import load_metadata
from ..resources import ResourceResolver
from .normalization import build_compatible_normalizer


class OpenWAMPolicy:
    def __init__(self, checkpoint_dir, *, device="cuda", resolver=None, **metadata_options):
        import torch
        from omegaconf import OmegaConf
        from openwam.deploy.engine import JointInferenceEngine
        from openwam.deploy.model_loader import load_from_checkpoint_dir

        cfg, checkpoint, stats, self.descriptor = load_metadata(checkpoint_dir, "openwam", **metadata_options)
        if stats != (Path(checkpoint_dir).resolve() / "normalization_stats.npy"):
            raise ValueError("OpenWAM native loader requires normalization_stats.npy in checkpoint-dir")
        cfg = (resolver or ResourceResolver()).config(cfg, "openwam")
        if (int(cfg.model.architecture.action_dim) != 7 or int(cfg.model.architecture.state_dim) != 8
                or int(cfg.dataloader.height) != 256 or int(cfg.dataloader.width) != 512
                or str(cfg.dataloader.action_mode) != "eef" or bool(cfg.dataloader.get("unify_action", False))):
            raise ValueError("checkpoint is not a Piper horizontal dual-camera delta-EEF policy")
        stride = int(cfg.dataloader.get("video_stride", 1))
        horizon = self.descriptor["action_horizon"]
        if stride <= 0 or horizon % stride or (horizon // stride) % 4:
            raise ValueError("training horizon/video stride is incompatible with Wan video geometry")
        self.video_frames = horizon // stride + 1
        cfg, self.model = load_from_checkpoint_dir(str(Path(checkpoint_dir).resolve()), device=device,
                                                   ckpt_name=str(checkpoint), config_override=cfg)
        if self.descriptor["normalization_adapter"]["mapping"]:
            self.model.attach_normalizer(build_compatible_normalizer(
                cfg, stats, self.descriptor["normalization_adapter"]))
        OmegaConf.update(cfg, "inference", {"denoise_steps": 20, "denoise_mode": "sync"}, merge=True, force_add=True)
        OmegaConf.update(cfg, "optimization.decode_video", False, force_add=True)
        self.engine = JointInferenceEngine(cfg, architecture=self.model)
        self.device = torch.device(device)

    def handle(self, request, arrays):
        import torch
        from PIL import Image

        if request.get("op") != "predict":
            raise ValueError("OpenWAM adapter supports hello/predict/shutdown only")
        obs = validate_request(request, arrays, self.descriptor)
        # Exact training layout: cam0 on the left, cam_wrist on the right.
        frame = np.concatenate([view.transpose(1, 2, 0) for view in obs.video], axis=1)
        conditions = dict(prompt=obs.prompt, first_frame_image=[Image.fromarray(frame)],
                          proprio=obs.proprio, num_frames=self.descriptor["action_horizon"] + 1,
                          video_num_frames=self.video_frames, height=256, width=512,
                          seed=obs.seed, denoise_steps=obs.sampling_steps, tiled=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.engine.generate(conditions)
        # The native architecture already unnormalizes actions. Only apply its
        # documented final binary command projection, as WAMPolicy does.
        actions = np.asarray(result["actions"], dtype=np.float32).copy()
        for dim in getattr(self.model, "binary_command_dims", ()):
            actions[..., dim] = np.where(actions[..., dim] > .5, 1., -1.)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return action_reply(actions, obs, self.descriptor, time.perf_counter() - started, prediction_recorded=False)
