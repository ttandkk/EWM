"""Exercise RGB-action startup and reconnects through the real CPU transport."""
import contextlib
import json
import sys
import threading
import time
from types import SimpleNamespace

import torch
import numpy as np
from omegaconf import OmegaConf
from websockets.sync.client import connect

from scripts.eval_ewm import ewm_rgb_action_eval_server as server
from scripts.eval_ewm.check_ewm_policy_server import check_server
from scripts.eval_ewm.ewm_eval_wire import decode_message, encode_message


def test_rgb_compile_warmup_requests_and_reconnect_share_owner(tmp_path, monkeypatch):
    (tmp_path / "checkpoint_step_080000.pt").touch()
    (tmp_path / "normalization.json").write_text("{}")
    OmegaConf.save(OmegaConf.create({
        "model": {"training_mode": "rgb_action"},
        "dataloader": {"type": "ewm_piper_multitask_rgb_action", "chunk_size": 24},
    }), tmp_path / "config.yaml")
    ready = tmp_path / "ready.json"
    calls = []

    class Model(torch.nn.Module):
        training_mode = "rgb_action"
        event_codec = object()
        event_camera_mode = "dual"
        num_event_cameras = 2
        _inference_bf16_weights = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def _decode_future(self, observed, future):
            calls.append(("decode", threading.get_ident()))
            return observed[:, None] + future

        def compile_inference(self, **kwargs):
            assert kwargs["options"]["triton.cudagraphs"] is False
            assert kwargs["cache_bf16_weights"] is True
            self._inference_bf16_weights = True
            calls.append(("compile", threading.get_ident()))

        def sample_policy(self, batch, **kwargs):
            assert kwargs["decode_future"] is False
            assert set(batch) == {"obs_video", "obs_rgb", "proprio", "prompt"}
            assert batch["obs_rgb"] is batch["obs_video"]
            calls.append(("sample", threading.get_ident()))
            action = torch.zeros((1, 24, 7))
            result = {"action": action, "action_normalized": action}
            if kwargs.get("return_observed_latent"):
                result.update(observed_latent=torch.zeros((1, 2, 3, 256, 256)),
                              future_latent=torch.full((1, 2, 2, 3, 256, 256), .5))
            return result

    def load(*args, **kwargs):
        calls.append(("load", threading.get_ident()))
        return Model(), SimpleNamespace(normalize_proprio=lambda x: x), 24

    monkeypatch.setattr(server, "load_policy", load)
    original_zeros, original_generator = torch.zeros, torch.Generator
    def zeros(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_zeros(*args, **kwargs)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch, "Generator", lambda **kwargs: original_generator(device="cpu"))
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    original_prepare = server.prepare_request
    monkeypatch.setattr(server, "prepare_request", lambda request, arrays, norm, **kwargs:
        original_prepare(request, arrays, norm, horizon=kwargs["horizon"], device="cpu"))
    monkeypatch.setattr(sys, "argv", ["server", "--checkpoint-dir", str(tmp_path),
        "--ready-file", str(ready), "--compile-core", "--compile-mode", "default",
        "--no-cuda-graphs", "--cache-bf16-weights", "--warmup-iters", "2",
        "--benchmark-iters", "1"])
    errors = []
    def run():
        try:
            server.main()
        except BaseException as exc:
            errors.append(exc)
    owner = threading.Thread(target=run, daemon=True)
    owner.start()
    metadata = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if errors:
                raise errors[0]
            try:
                metadata = json.loads(ready.read_text())
                break
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(.01)
        assert metadata is not None, "RGB-action server readiness timeout"
        assert metadata["cache_bf16_weights"] is True
        assert metadata["cuda_graphs_enabled"] is False
        with connect(metadata["url"], open_timeout=5) as conn:
            conn.send(encode_message({"op": "invalid"}))
            reply, _ = decode_message(conn.recv(timeout=5))
            assert not reply["ok"]
        report = check_server(metadata, "catch", require_compiled=True, require_no_cudagraphs=True)
        assert report["passed"] and report["requests"] == 4 and report["connections"] == 2
        assert report["max_reconnect_action_error"] == 0
        assert metadata["post_trial_visualization_supported"]
        with connect(metadata["url"], open_timeout=5) as conn:
            def request(header, arrays=None):
                conn.send(encode_message(header, arrays))
                reply, values = decode_message(conn.recv(timeout=10))
                assert reply["ok"], reply
                return reply, values
            request(dict(op="visualization_begin", session="trial"))
            request(dict(op="predict", observation_contract=server.OBSERVATION_CONTRACT,
                observation_row_offset=0, raw_observation=True, decode_future_rgb=False,
                record_prediction=dict(session="trial", observation_step=0),
                seed=5, step=80000, sampling_steps=4, action_chunk_size=24, prompt="catch",
                timing=dict(obs_video_time_s=0., proprio_time_s=0.)),
                dict(obs_video=np.zeros((2, 3, 256, 256), np.uint8), proprio=np.zeros(8, np.float32)))
            assert not any(op == "decode" for op, _ in calls)
        # Latents survive a reconnect; all CUDA work still belongs to the owner.
        with connect(metadata["url"], open_timeout=5) as conn:
            reply, values = request(dict(op="visualization_decode", session="trial", observation_step=0))
            assert reply["prediction_kind"] == "rgb"
            assert values["future_rgb"].shape == (2, 2, 3, 256, 256)
            assert (values["future_rgb"] == 128).all()
            reply, _ = request(dict(op="visualization_end", session="trial"))
            assert reply["recorded_predictions"] == 1
    finally:
        if metadata and owner.is_alive():
            with connect(metadata["url"], open_timeout=5) as conn:
                conn.send(encode_message({"op": "shutdown"}))
                reply, _ = decode_message(conn.recv(timeout=5))
                assert reply["ok"]
        owner.join(timeout=5)
    assert not owner.is_alive() and not errors and not ready.exists()
    assert {tid for _, tid in calls} == {owner.ident}
    assert [op for op, _ in calls].count("compile") == 1
    assert [op for op, _ in calls].count("sample") == 12
    assert [op for op, _ in calls].count("decode") == 1
