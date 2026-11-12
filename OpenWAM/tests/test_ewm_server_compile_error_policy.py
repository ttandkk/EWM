"""CPU startup checks for compile-error reporting and opt-in strict fallback."""
import contextlib
import importlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


@pytest.mark.parametrize("kind", ["action_only", "rgb_action", "joint"])
@pytest.mark.parametrize("flags,error,active,exceeded", [
    ([], 0.1, True, True),
    (["--allow-compile-action-error"], 0.1, True, True),
    (["--no-allow-compile-action-error"], 0.1, False, True),
    (["--no-allow-compile-action-error"], 0.01, True, False),
])
def test_server_compile_error_policy_at_startup(tmp_path, monkeypatch, kind, flags, error, active, exceeded):
    suffix = {"action_only": "action_only_eval_server", "rgb_action": "rgb_action_eval_server",
              "joint": "eval_server"}[kind]
    module = importlib.import_module(f"scripts.eval_ewm.ewm_{suffix}")
    checkpoint = tmp_path / "checkpoint_step_80000.pt"
    checkpoint.touch()
    (tmp_path / "config.yaml").write_text("fixture")
    (tmp_path / "normalization.json").write_text("{}")
    ready = tmp_path / "ready.json"
    config = OmegaConf.create({"model": {"training_mode": kind, "voxel_bins": 1,
                                        "event_camera_mode": "dual"},
        "dataloader": {"type": {"action_only": "ewm_piper_action_only",
                               "rgb_action": "ewm_piper_rgb_action", "joint": "ewm_piper_multitask"}[kind],
                       "chunk_size": 24, "voxel_bins": 1, "count_clip": 15,
                       "event_camera_mode": "dual"}})

    class Model:
        training_mode = kind
        event_codec = None if kind == "action_only" else object()
        event_camera_mode = "dual"
        num_event_cameras = 2
        voxel_bins = 1
        _inference_bf16_weights = False
        compiled = False

        def predict_velocity(self):
            return 0.0

        def event_prediction_horizon(self, horizon):
            return horizon

        def disable_inference_compile(self):
            self.compiled = False

        def sample_policy(self, *args, **kwargs):
            value = error if self.compiled else self.predict_velocity()
            return {"action_normalized": torch.tensor([value]), "future_latent": torch.tensor([value])}

    model = Model()
    normalizer = SimpleNamespace(count_clip=15, voxel_bins=1)
    monkeypatch.setattr(module.OmegaConf, "load", lambda _: config)
    monkeypatch.setattr(module, "load_policy", lambda *a, **kw: (model, normalizer, 24))
    monkeypatch.setattr(sys, "argv", ["server", "--checkpoint-dir", str(tmp_path),
        "--ready-file", str(ready), "--compile-core", *flags])

    original_zeros = torch.zeros
    original_generator = torch.Generator
    def cpu_zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)
    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch, "Generator", lambda **kw: original_generator(device="cpu"))
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def configure_action(model, **kwargs):
        model.predict_velocity = lambda: error
        return False
    def configure_joint(model, **kwargs):
        model.compiled = True
        return False

    captured = {}
    if kind in ("joint", "rgb_action"):
        monkeypatch.setattr(module, "configure_inference_compile", configure_joint)
        monkeypatch.setattr(module, "serve_on_inference_thread",
                            lambda process, metadata, ready_file: captured.update(metadata))
    else:
        if kind == "action_only":
            monkeypatch.setattr(module, "configure_action_compile", configure_action)
        class FakeServer:
            socket = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 12345))
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def serve_forever(self):
                captured.update(json.loads(ready.read_text()))
        monkeypatch.setattr(module, "serve", lambda *a, **kw: FakeServer())

    module.main()
    stats = captured["startup_benchmark"]
    assert stats["iterations"] == 1
    assert captured["compile_core_active"] is active
    assert stats["compile_accepted"] is active
    assert stats["compile_error_exceeded"] is exceeded
    assert stats["compile_rejected_for_error"] is (exceeded and not active)
    assert stats["compiled_kv_cache_action_max_abs_error"] == pytest.approx(error)
    assert stats["max_compile_action_error"] == pytest.approx(0.02)
    assert stats["allow_compile_action_error"] is ("--no-allow-compile-action-error" not in flags)
