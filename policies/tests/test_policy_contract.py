import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from piper_policies.contract import action_reply, validate_request
from piper_policies.metadata import hello_metadata, load_metadata
from piper_policies.resources import ResourceResolver


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "checkpoint_step_42.pt").write_bytes(b"test checkpoint content")
    (tmp_path / "config.yaml").write_text("model:\n  training_mode: joint\n  event_camera_mode: dual\n  voxel_bins: 1\ndataloader:\n  chunk_size: 4\n")
    stats = dict(version=1, voxel=dict(num_bins=1, count_clip=15), sources=dict(conveyor=dict(
        dataset_dir="/old/dataset", collection_sha256="a" * 64, input_signature="b" * 64)))
    (tmp_path / "normalization.json").write_text(json.dumps(stats))
    return tmp_path


@pytest.fixture
def descriptor(bundle):
    return load_metadata(bundle)[3]


def observation(descriptor):
    request = dict(op="predict", raw=True, seed=4, sampling_steps=2, action_chunk_size=3,
                   prompt="move the object", step=42, observation_step=3, observation_time_s=.1,
                   timing=dict(obs_video_time_s=.1, proprio_time_s=.1, obs_event_start_s=2/30, obs_event_end_s=.1))
    request["event_time_contract" if descriptor["mode"] == "joint" else "observation_contract"] = descriptor[
        "event_time_contract" if descriptor["mode"] == "joint" else "observation_contract"]
    arrays = dict(obs_video=np.zeros((2, 3, 256, 256), np.uint8), proprio=np.zeros(8, np.float32))
    if descriptor["mode"] == "joint":
        arrays["obs_voxel"] = np.zeros((2, 2, 256, 256), np.float32)
    return request, arrays


def test_content_identity_survives_relocation(bundle, tmp_path):
    import shutil
    descriptor = load_metadata(bundle)[3]
    moved = tmp_path / "moved" / "model"
    shutil.copytree(bundle, moved, ignore=shutil.ignore_patterns("moved"))
    assert load_metadata(moved)[3] == descriptor
    source = descriptor["training_data"]["conveyor"]
    assert "dataset_dir" not in source and "input_signature" not in source
    assert source["collection_sha256"] == "a" * 64
    assert source["legacy_input_signature"] == "b" * 64
    (moved / "normalization.json").write_text((moved / "normalization.json").read_text() + " ")
    assert load_metadata(moved)[3]["normalization_sha256"] != descriptor["normalization_sha256"]


def test_path_maps_use_boundaries_and_allowlisted_fields(tmp_path):
    resolver = ResourceResolver(project_root=tmp_path, pretrained_root=tmp_path / "weights",
                                path_maps=["/old=/moved", "/old/specific=/precise"])
    config = dict(dino_path="/old/specific/dino", text_path="/oldish/text", event_vae_path="/host/pretrained/vae", prompt="/old/prompt")
    relocated = resolver.ewm_model_config(config)
    assert relocated == dict(dino_path="/precise/dino", text_path="/oldish/text", event_vae_path=str(tmp_path / "weights/vae"), prompt="/old/prompt")
    assert config["dino_path"] == "/old/specific/dino"
    cfg = OmegaConf.create({"model": {"action_dit_pretrained_path": "relative.pt", "model_id": "vendor/model"}, "data": {"dataset_dir": "/old/data"}})
    moved = resolver.config(cfg, "fastwam")
    assert moved.model.action_dit_pretrained_path == str(tmp_path / "relative.pt")
    assert moved.model.model_id == "vendor/model" and moved.data.dataset_dir == "/old/data"


def test_roundtrip_action_plan(descriptor):
    from piper_bench.evaluation.wire import encode_message, decode_message
    request, arrays = observation(descriptor)
    header, decoded = decode_message(encode_message(request, arrays))
    obs = validate_request(header, decoded, descriptor)
    reply, actions = action_reply(np.ones((4, 7)), obs, descriptor, .023, prediction_recorded=False)
    reply, actions = decode_message(encode_message(reply, actions))
    assert actions["executed_actions"].shape == (3, 7)
    assert actions["executed_actions"].dtype == np.float32
    assert reply["observation_time_s"] == .1 and reply["observation_step"] == 3
    assert reply["control_hz"] == 30 and reply["inference_seconds"] == .023


@pytest.mark.parametrize("change", [
    {"step": 43}, {"raw": False}, {"raw_observation": False}, {"control_hz": 60}, {"observation_step": 4},
    {"observation_time_s": float("nan")}, {"observation_row_offset": 1},
    {"action_chunk_size": 5}, {"sampling_steps": 0}, {"seed": -1}, {"prompt": ""},
    {"event_time_contract": "old"}, {"decode_future_events": True},
    {"timing": {"obs_video_time_s": .1, "proprio_time_s": .1, "obs_event_start_s": .1, "obs_event_end_s": .1}},
])
def test_invalid_requests_fail_before_inference(descriptor, change):
    request, arrays = observation(descriptor)
    request.update(change)
    with pytest.raises(ValueError):
        validate_request(request, arrays, descriptor)


def test_array_contract_and_no_future_labels(descriptor):
    request, arrays = observation(descriptor)
    arrays["future_voxel"] = arrays["obs_voxel"]
    with pytest.raises(ValueError, match="exactly"):
        validate_request(request, arrays, descriptor)
    del arrays["future_voxel"]
    arrays["proprio"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_request(request, arrays, descriptor)


@pytest.mark.parametrize("mode", ["joint", "action_only", "rgb_action"])
def test_real_websocket_and_inference_thread(descriptor, tmp_path, mode):
    descriptor.update(mode=mode, observation_contract=("rgb_state_pre_action_rgb_latent_v1" if mode == "rgb_action" else "rgb_state_pre_action_v1"))
    from websockets.sync.client import connect
    from piper_bench.evaluation.wire import decode_message, encode_message
    from piper_policies.server import serve_policy
    owner_ids, errors = [], []

    class Policy:
        def __init__(self):
            self.descriptor = descriptor
        def handle(self, request, arrays):
            owner_ids.append(threading.get_ident())
            obs = validate_request(request, arrays, self.descriptor)
            return action_reply(np.zeros((4, 7)), obs, self.descriptor, .01)

    ready = tmp_path / "ready.json"
    def run():
        try:
            serve_policy(Policy(), ready_file=ready)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 10
    while not ready.exists() and not errors and time.monotonic() < deadline:
        time.sleep(.02)
    assert not errors
    metadata = json.loads(ready.read_text())
    with connect(metadata["url"], max_size=16 * 1024**2, compression=None) as socket:
        socket.send(encode_message({"op": "hello"}))
        hello, _ = decode_message(socket.recv())
        assert hello["policy_descriptor"] == descriptor
        assert hello["training_mode"] == mode
        request, arrays = observation(descriptor)
        socket.send(encode_message({**request, "seed": -1}, arrays))
        assert decode_message(socket.recv())[0]["ok"] is False
        socket.send(encode_message(request, arrays))
        reply, actions = decode_message(socket.recv())
        assert reply["ok"] and actions["executed_actions"].shape == (3, 7)
        socket.send(encode_message({"op": "shutdown"}))
        assert decode_message(socket.recv())[0]["ok"]
    thread.join(timeout=10)
    assert not thread.is_alive() and not errors
    assert set(owner_ids) == {thread.ident}
    assert not ready.exists()


def test_describe_does_not_import_models(bundle):
    import subprocess
    import sys
    code = "from piper_policies.cli import main; main(['describe','--checkpoint-dir',%r]); import sys; assert 'torch' not in sys.modules; assert 'sapien' not in sys.modules" % str(bundle)
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["checkpoint_step"] == 42


def test_explicit_checkpoint_step_selects_older_file_and_rejects_ambiguity(bundle):
    from piper_policies.metadata import select_checkpoint
    (bundle / "checkpoint_step_100.pt").write_bytes(b"newer")
    assert select_checkpoint(bundle, "ewm", checkpoint_step=42)[1] == 42
    assert select_checkpoint(bundle, "ewm")[1] == 100
    (bundle / "checkpoint_step_000042.pt").write_bytes(b"duplicate step")
    with pytest.raises(ValueError, match="ambiguous"):
        select_checkpoint(bundle, "ewm", checkpoint_step=42)
