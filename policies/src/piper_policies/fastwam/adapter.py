"""Use the native FastWAM sampler and saved processor normalization."""
from __future__ import annotations

import inspect
import os
import time

import numpy as np

from ..contract import action_reply, validate_request
from ..metadata import load_metadata
from ..resources import ResourceResolver

# Same template as native RobotVideoDataset; importing the whole training
# dataset would force video readers into an otherwise model-only service.
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class FastWAMPolicy:
    def __init__(self, checkpoint_dir, *, device="cuda", resolver=None, **metadata_options):
        import torch
        from hydra.utils import instantiate
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        cfg, checkpoint, normalization, self.descriptor = load_metadata(checkpoint_dir, "fastwam", **metadata_options)
        resolver = resolver or ResourceResolver()
        cfg = resolver.config(cfg, "fastwam")
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        if resolver.pretrained_root:
            os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(resolver.pretrained_root)
        data, processor = cfg.data.train, cfg.data.train.processor
        if (int(processor.action_output_dim) != 7 or int(processor.proprio_output_dim) != 8
                or list(data.video_size) != [256, 512] or data.concat_multi_camera != "horizontal"
                or [item.key for item in data.shape_meta.images] != ["cam0", "cam_wrist"]):
            raise ValueError("checkpoint is not a Piper horizontal dual-camera delta-EEF policy")
        if len(data.shape_meta.action) != 1 or len(data.shape_meta.state) != 1:
            raise ValueError("Piper expects one physical action and one state field")
        self.action_key = str(data.shape_meta.action[0].key)
        self.state_key = str(data.shape_meta.state[0].key)
        precision = str(cfg.get("mixed_precision", "bf16"))
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[precision]
        # Native online encoding supports arbitrary evaluation instructions and
        # uses the same template/tokenizer as cached training text.
        cfg.model.load_text_encoder = True
        self.model = instantiate(cfg.model, model_dtype=dtype, device=device)
        payload = self.model.load_checkpoint(str(checkpoint))
        if "mot" not in payload or "proprio_encoder" not in payload:
            raise ValueError("Piper inference requires trained action and proprio weights")
        # The native dense loader is permissive; reject missing trained keys.
        if payload.get("video_lora") is None:
            expected = set(self.model.mot.state_dict())
            provided = set(payload["mot"])
            if expected != provided:
                raise ValueError("dense FastWAM checkpoint has missing or unexpected model weights")
        if "step" in payload and int(payload["step"]) != self.descriptor["checkpoint_step"]:
            raise ValueError("checkpoint payload and filename step differ")
        del payload
        self.model.to(device).eval()
        self.processor = instantiate(processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(normalization)))
        self.device = torch.device(device)
        ratio = int(data.get("action_video_freq_ratio", 1))
        horizon = self.descriptor["action_horizon"]
        if ratio <= 0 or horizon % ratio or (horizon // ratio) % 4:
            raise ValueError("training action/video frequency ratio is incompatible with Wan")
        self.video_frames = horizon // ratio + 1

    def handle(self, request, arrays):
        import torch

        if request.get("op") != "predict":
            raise ValueError("FastWAM adapter supports hello/predict/shutdown only")
        obs = validate_request(request, arrays, self.descriptor)
        state = {"state": {self.state_key: torch.as_tensor(obs.proprio, dtype=torch.float32)[None]}}
        state = self.processor.action_state_transform(state)
        state = self.processor.normalizer.forward(state)["state"][self.state_key]
        frame = np.concatenate(list(obs.video), axis=2)
        image = torch.as_tensor(frame)[None].to(device=self.device, dtype=self.model.torch_dtype) * (2. / 255.) - 1.
        kwargs = dict(prompt=DEFAULT_PROMPT.format(task=obs.prompt), input_image=image,
                      action_horizon=self.descriptor["action_horizon"], proprio=state,
                      num_inference_steps=obs.sampling_steps, seed=obs.seed,
                      text_cfg_scale=1., negative_prompt="", rand_device="cpu", tiled=False)
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            kwargs["num_video_frames"] = self.video_frames
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            action = self.model.infer_action(**kwargs)["action"]
        if action.ndim == 2:
            action = action[None]
        normalizer = self.processor.normalizer.normalizers["action"][self.action_key]
        actions = normalizer.backward(action.detach().float().cpu()).numpy()[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return action_reply(actions, obs, self.descriptor, time.perf_counter() - started, prediction_recorded=False)
