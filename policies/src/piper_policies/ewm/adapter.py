"""Reuse EWM's checkpoint compatibility, normalization and latent sampler."""
from __future__ import annotations

import time

from ..contract import action_reply, validate_request
from ..metadata import load_metadata
from ..resources import ResourceResolver


class EWMPolicy:
    def __init__(self, checkpoint_dir, *, device="cuda", resolver=None, cache_context_kv=True,
                 inference_options=None, **metadata_options):
        import torch
        from openwam.dataloader.ewm_normalizer import EWMNormalizer
        from openwam.model.architectures.ewm import EWMModel
        from openwam.model.architectures.ewm_checkpoint import normalize_policy_payload
        from piper_bench.evaluation.prediction_recording import PredictionArchive

        cfg, checkpoint, normalization, self.descriptor = load_metadata(checkpoint_dir, "ewm", **metadata_options)
        payload = normalize_policy_payload(torch.load(checkpoint, map_location="cpu", weights_only=True))
        config = (resolver or ResourceResolver()).ewm_model_config(payload["model_config"])
        self.model = EWMModel.from_config(config)
        self.model.load_policy_state_dict(payload["model"])
        self.model.to(device).eval()
        self.device = torch.device(device)
        self.normalizer = EWMNormalizer(normalization)
        self.archive = PredictionArchive()
        self.cache_context_kv = cache_context_kv
        d = self.descriptor
        if "step" in payload and int(payload["step"]) != d["checkpoint_step"]:
            raise ValueError("checkpoint payload and filename step differ")
        if int(payload["horizon"]) != d["action_horizon"] or self.model.training_mode != d["mode"]:
            raise ValueError("policy payload and training config horizon/mode differ")
        if d["mode"] != "action_only":
            if self.model.event_prediction_horizon(d["action_horizon"]) != d["event_horizon"]:
                raise ValueError("policy payload and training config event horizon differ")
            for section in (cfg.model, cfg.dataloader):
                configured = section.get("event_horizon")
                if (d["action_horizon"] if configured is None else int(configured)) != d["event_horizon"]:
                    raise ValueError("training loader and model prediction horizons differ")
            if self.model.event_camera_mode != d["event_camera_mode"]:
                raise ValueError("policy payload and config camera mode differ")
        if d["mode"] == "joint":
            if (self.model.voxel_bins != d["voxel_bins"] or int(cfg.dataloader.voxel_bins) != d["voxel_bins"]
                    or float(cfg.dataloader.count_clip) != d["count_clip"]
                    or cfg.dataloader.event_camera_mode != d["event_camera_mode"]):
                raise ValueError("model, loader and normalization event contract differ")

        from .optimization import prepare_inference
        self.optimization = prepare_inference(self, **(inference_options or {}))

    def handle(self, request, arrays):
        import torch

        if request.get("op", "").startswith("visualization_"):
            if not self.descriptor["post_trial_visualization_supported"]:
                raise ValueError("this model has no future decoder")
            return self.archive.handle(request, self.model)
        if request.get("op") != "predict":
            raise ValueError("unknown policy operation")
        obs = validate_request(request, arrays, self.descriptor)
        batch = dict(obs_video=torch.as_tensor(obs.video)[None].to(self.device),
                     proprio=self.normalizer.normalize_proprio(torch.as_tensor(obs.proprio))[None].to(self.device),
                     prompt=[obs.prompt])
        if obs.voxel is not None:
            batch["obs_voxel"] = self.normalizer.normalize_voxel(torch.as_tensor(obs.voxel))[None].to(self.device)
        if self.descriptor["mode"] == "rgb_action":
            views = {"dual": slice(None), "wrist_only": slice(1, 2), "global_only": slice(0, 1)}
            batch["obs_rgb"] = batch["obs_video"][:, views[self.descriptor["event_camera_mode"]]]
        generator = torch.Generator(device=self.device).manual_seed(obs.seed)
        recording = request.get("record_prediction") is not None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            result = self.model.sample_policy(batch, horizon=self.descriptor["action_horizon"],
                normalizer=self.normalizer, num_steps=obs.sampling_steps, generator=generator,
                decode_future=False, cache_context_kv=self.cache_context_kv, return_observed_latent=recording)
        normalized_request = {**request, "step": self.descriptor["checkpoint_step"], "seed": obs.seed,
                              "sampling_steps": obs.sampling_steps, "timing": obs.timing}
        recorded = self.archive.record(normalized_request, result)
        actions = result["action"][0].detach().float().cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        return action_reply(actions, obs, self.descriptor, elapsed, prediction_recorded=recorded)
