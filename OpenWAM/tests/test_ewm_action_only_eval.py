"""Independent action-only evaluation contract and closed-loop workflow tests."""
from __future__ import annotations

import ast
import contextlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.eval_ewm.ewm_action_only_protocol import (
    OBSERVATION_CONTRACT, prepare_request, validate_timing,
)
from scripts.eval_ewm.ewm_eval_wire import MAX_MESSAGE, decode_message, encode_message
from scripts.eval_ewm.ewm_eval_summary import write_summary
from scripts.eval_ewm.ewm_eval_recovery import archive_interrupted_trial, load_completed_results
from scripts.eval_ewm.ewm_event_replan import horizon_statistics
from scripts.eval_ewm.ewm_rgb_ood import RGBOODConfig, TrialRGBOOD, check_resume_rgb_ood
from scripts.eval_ewm.ewm_scene_rng import restore_scene_rng, save_scene_rng, state_path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts/eval_ewm"
CLIENT = EVAL / "eval_ewm_action_only_piper_sim.py"


class Normalizer:
    def normalize_proprio(self, value):
        return value / 2


def observation():
    return {"obs_video": np.zeros((2, 3, 256, 256), dtype=np.uint8),
            "proprio": np.arange(8, dtype=np.float32)}


def predict_header():
    return {"op": "predict", "step": 1, "seed": 10000, "sampling_steps": 4,
            "action_chunk_size": 16, "prompt": "stack",
            "observation_contract": OBSERVATION_CONTRACT, "raw_observation": True,
            "observation_row_offset": 0,
            "timing": {"obs_video_time_s": 1., "proprio_time_s": 1.}}


def test_wire_raw_input_normalization_and_no_voxel():
    header, arrays = decode_message(encode_message(predict_header(), observation()))
    batch, seed, steps, chunk = prepare_request(header, arrays, Normalizer(), horizon=24, device="cpu")
    assert set(batch) == {"obs_video", "proprio", "prompt"}
    assert batch["obs_video"].dtype == torch.uint8
    assert batch["obs_video"].shape == (1, 2, 3, 256, 256)
    torch.testing.assert_close(batch["proprio"], torch.arange(8)[None].float() / 2)
    assert (seed, steps, chunk) == (10000, 4, 16)


@pytest.mark.parametrize("mutation", [
    {"observation_contract": "aligned_event_frame_pre_action_v1"},
    {"observation_row_offset": 1}, {"raw_observation": False},
    {"decode_future_events": True}, {"seed": -1}, {"seed": 2**63},
    {"sampling_steps": 0}, {"action_chunk_size": 0}, {"action_chunk_size": 25},
    {"prompt": ""}, {"timing": {}},
    {"timing": {"obs_video_time_s": 1., "proprio_time_s": 1.01}},
])
def test_invalid_request_is_rejected(mutation):
    with pytest.raises(ValueError):
        prepare_request(dict(predict_header(), **mutation), observation(), Normalizer(),
                        horizon=24, device="cpu")


@pytest.mark.parametrize("kind", ["event", "missing", "shape", "dtype", "nonfinite"])
def test_invalid_arrays_are_rejected(kind):
    arrays = observation()
    if kind == "event":
        arrays["obs_voxel"] = np.zeros((2, 2, 256, 256), dtype=np.float32)
    elif kind == "missing":
        del arrays["proprio"]
    elif kind == "shape":
        arrays["obs_video"] = arrays["obs_video"][:1]
    elif kind == "dtype":
        arrays["obs_video"] = arrays["obs_video"].astype(np.float32)
    else:
        arrays["proprio"][0] = np.nan
    with pytest.raises(ValueError):
        prepare_request(predict_header(), arrays, Normalizer(), horizon=24, device="cpu")


def test_timing_matches_original_one_ms_tolerance():
    validate_timing({"obs_video_time_s": 1., "proprio_time_s": 1.0009})
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            validate_timing({"obs_video_time_s": value, "proprio_time_s": value})


def client_definitions():
    """Load pure definitions without importing or starting Isaac in unit tests."""
    tree = ast.parse(CLIENT.read_text())
    definitions = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    definitions += [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    namespace = dict(Path=Path, np=np, torch=torch, time=time, math=math, json=json,
                     cv2=cv2, subprocess=subprocess, shutil=shutil, traceback=traceback, sys=sys,
                     PROJECT=ROOT.parent, CONTROL_HZ=30, CAMERAS=("cam0", "cam_wrist"),
                     RESET_WARMUP_STEPS=60, TASK="Isaac-Stack-Cube-Piper-IK-Rel-v0",
                     PROMPT="stack", OBSERVATION_CONTRACT=OBSERVATION_CONTRACT,
                     MAX_MESSAGE=MAX_MESSAGE, encode_message=encode_message, decode_message=decode_message,
                     _write_summary=write_summary, horizon_statistics=horizon_statistics,
                     RGBOODConfig=RGBOODConfig, TrialRGBOOD=TrialRGBOOD,
                     check_resume_rgb_ood=check_resume_rgb_ood,
                     restore_scene_rng=restore_scene_rng, save_scene_rng=save_scene_rng,
                     state_path=state_path, load_completed_results=load_completed_results,
                     archive_interrupted_trial=archive_interrupted_trial)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(CLIENT), "exec"), namespace)
    return namespace


def test_scene_geometry_cameras_materials_and_physics_match_original():
    original = ast.parse((EVAL / "eval_ewm_piper_sim.py").read_text())
    action = ast.parse(CLIENT.read_text())
    for name in ("camera", "set_views", "configure_env", "apply_patterns", "simulation_time"):
        left = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        right = next(n for n in action.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(left) == ast.dump(right), name
    assert "from dvs_gen" not in CLIENT.read_text()
    assert '"obs_voxel"' not in CLIENT.read_text()


def test_launcher_protocol_matches_and_rejects_event_flags():
    source = (EVAL / "run_ewm_action_only_piper_eval.sh").read_text()
    assert "--trials 50 --max-steps 500 --sampling-steps 4 --execution-horizon 16 --profile-timing" in source
    assert "--seed 10000 --scene-seed 20260930 --scene-seed-mode stream" in source
    assert "eval_piper_sim_action_only/job" in source
    assert "ewm_eval_watchdog.py" in source
    subprocess.run(["bash", "-n", str(EVAL / "run_ewm_action_only_piper_eval.sh")], check=True)
    for flag in ("--event-guided-replan", "--decode-predicted-events", "--event-replan-threshold"):
        result = subprocess.run(["bash", str(EVAL / "run_ewm_action_only_piper_eval.sh"), flag],
                                capture_output=True, text=True)
        assert result.returncode == 2
        assert "does not accept event" in result.stderr


class FakeEnv:
    device = "cpu"

    def __init__(self, *, termination_step=None):
        self.steps = 0
        self.commands = []
        self.closed = False
        self.termination_step = termination_step
        self.sim = SimpleNamespace(reset=lambda: None, get_physics_step_count=lambda: self.steps * 4,
                                   get_physics_dt=lambda: 1 / 120)
        self.scene = {name: SimpleNamespace(data=SimpleNamespace(output={
            "rgb": torch.zeros(1, 256, 256, 4, dtype=torch.uint8)}))
            for name in ("cam0", "cam_wrist")}
        self.scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.arange(8)[None].float()))
        for name in ("cube_1", "cube_2", "cube_3"):
            self.scene[name] = SimpleNamespace(data=SimpleNamespace(
                root_pos_w=torch.zeros(1, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]])))

    def reset(self, seed=None):
        self.steps = 0
        if seed is not None:
            torch.manual_seed(seed)
        for name in ("cube_1", "cube_2", "cube_3"):
            self.scene[name].data.root_pos_w = torch.rand(1, 3)

    def step(self, command):
        self.commands.append(command.clone())
        self.steps += 1
        terminated = self.termination_step is not None and self.steps >= 61 + self.termination_step
        return {}, torch.tensor([1.]), torch.tensor([terminated]), torch.tensor([False]), {}

    def close(self):
        self.closed = True


def test_capture_is_causal_and_only_reads_current_rgb_state():
    namespace = client_definitions()
    namespace["set_views"] = lambda env: None
    env = FakeEnv()
    obs, done, terminated, extra = namespace["capture_step"](
        env, torch.zeros(1, 7), profile_timing=True)
    assert set(obs) == {"obs_video", "proprio", "timing", "profile_seconds"}
    assert obs["timing"] == {"obs_video_time_s": 1 / 30, "proprio_time_s": 1 / 30}
    assert not done and not terminated and extra["reward"] == 1.
    assert "dvs_warp_process" not in obs["profile_seconds"]
    obs["obs_video"][0, 0, 0, 0] = 123
    assert env.scene["cam0"].data.output["rgb"][0, 0, 0, 0] == 0  # copied buffer


def test_rgb_video_shape_codec_and_frame_count(tmp_path):
    namespace = client_definitions()
    path = tmp_path / "trial_000_h264_failure.mp4"
    video = namespace["TrialVideo"](path, rgb_ood_label="RGB OOD")
    try:
        for step in range(3):
            video.write(observation(), step, 0)
    finally:
        video.close()
    reader = cv2.VideoCapture(str(path))
    try:
        assert reader.isOpened()
        assert reader.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        assert reader.get(cv2.CAP_PROP_FPS) == 30
        ok, frame = reader.read()
        assert ok and frame.shape == (256, 512, 3)
    finally:
        reader.release()


@pytest.mark.parametrize("success,termination,budget,expected_steps,reason", [
    (False, None, 19, 19, "step_budget"), (True, None, 25, 10, "success"),
    (False, 3, 25, 3, "terminated"),
])
def test_closed_loop_outputs_fixed_horizon_and_success_contract(
        tmp_path, monkeypatch, success, termination, budget, expected_steps, reason):
    namespace = client_definitions()
    env = FakeEnv(termination_step=termination)
    output = tmp_path / "eval"
    ready_file = tmp_path / "ready.json"
    ready_file.write_text(json.dumps({"url": "ws://fixture", "checkpoint_dir": "fixture"}))
    args = SimpleNamespace(output_dir=output, ready_file=ready_file, resume=False, trials=2,
                           max_steps=budget, sampling_steps=4, execution_horizon=16,
                           profile_timing=True, seed=10000, scene_seed_mode="stream",
                           scene_rng_reference_dir=None, checkpoint_step=None,
                           renderer_mode="pathtracing", pathtracing_spp=64,
                           rgb_ood_severity="moderate")
    fake_rep = SimpleNamespace(settings=SimpleNamespace(set_render_pathtraced=lambda **kw: None))
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(replicator=SimpleNamespace(core=fake_rep)))
    monkeypatch.setitem(sys.modules, "omni.replicator", SimpleNamespace(core=fake_rep))
    monkeypatch.setitem(sys.modules, "omni.replicator.core", fake_rep)
    namespace.update(args=args, scene_seed=20260930, rgb_ood_config=RGBOODConfig(seed=10000),
                     carb=SimpleNamespace(settings=SimpleNamespace(get_settings=lambda:
                          SimpleNamespace(set=lambda *a: None))),
                     gym=SimpleNamespace(make=lambda *a, **kw: SimpleNamespace(unwrapped=env)),
                     configure_env=lambda: (None, SimpleNamespace(params={},
                       func=lambda env: torch.tensor([success]))),
                     set_views=lambda env: None, apply_patterns=lambda env: None,
                     ws_connect=lambda *a, **kw: contextlib.nullcontext(None))
    requests = []

    def request(conn, header, arrays=None):
        if header["op"] == "hello":
            return {"training_mode": "action_only", "observation_contract": OBSERVATION_CONTRACT,
                    "inference_observation_row_offset": 0, "training_observation_row_offset": 0,
                    "horizon": 24, "checkpoints": [10000], "config_sha256": "cfg",
                    "normalization_sha256": "norm", "checkpoint_sha256": {"10000": "ckpt"}}, {}
        if header["op"] == "shutdown":
            return {"ok": True}, {}
        requests.append(header)
        batch, seed, steps, chunk = prepare_request(header, arrays, Normalizer(), horizon=24, device="cpu")
        assert set(batch) == {"obs_video", "proprio", "prompt"}
        return {"inference_seconds": .01}, {"executed_actions": np.zeros((chunk, 7), dtype=np.float32)}

    namespace["request"] = request
    namespace["main"]()
    complete = json.loads((output / "complete.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert complete["training_mode"] == "action_only"
    assert complete["success_rate"] == float(success)
    assert manifest["event_input"] is False and manifest["event_replan"]["enabled"] is False
    for result in complete["results"]:
        assert result["steps"] == expected_steps and result["success"] == success
        assert result["execution_chunks"][-1]["end_reason"] == reason
        assert max(c["executed_horizon"] for c in result["execution_chunks"]) <= 16
        assert Path(result["video"]).is_file()
        assert state_path(output, result["trial"]).is_file()
        rows = json.loads((output / f"trial_{result['trial']:03d}_steps.json").read_text())
        assert all("event_count" not in row for row in rows)
    assert "event_input: false" in (output / "summary.txt").read_text()
    assert env.closed
    assert requests[0]["seed"] == 10000
    assert any(r["seed"] == 10000 + budget for r in requests)
    if reason == "step_budget":
        assert [c["executed_horizon"] for c in complete["results"][0]["execution_chunks"]] == [16, 3]
    with pytest.raises(FileExistsError):
        namespace["main"]()
    if reason == "step_budget":
        # Replay the same scenes, interrupt trial 1, then use the shared resume path.
        args.output_dir = tmp_path / "paired_eval"
        args.scene_rng_reference_dir = output
        interrupted = False

        def interrupt_once(conn, header, arrays=None):
            nonlocal interrupted
            if header["op"] == "predict" and header["seed"] >= 10000 + budget and not interrupted:
                interrupted = True
                raise RuntimeError("test interruption")
            return request(conn, header, arrays)

        namespace["request"] = interrupt_once
        with pytest.raises(RuntimeError, match="test interruption"):
            namespace["main"]()
        assert len(load_completed_results(args.output_dir, 2)) == 1
        assert (args.output_dir / "failure.json").is_file()
        args.resume = True
        namespace["main"]()
        paired = json.loads((args.output_dir / "complete.json").read_text())
        assert len(paired["results"]) == 2
        for left, right in zip(complete["results"], paired["results"]):
            assert left["initial_cube_pose"] == right["initial_cube_pose"]
            assert left["initial_joint_state"] == right["initial_joint_state"]
        assert list(args.output_dir.glob("trial_001_*_interrupted_*"))


@pytest.mark.parametrize("benchmark_iters", [0, 1])
def test_server_handler_wire_actions_and_event_rejection(tmp_path, monkeypatch, benchmark_iters):
    from scripts.eval_ewm import ewm_action_only_eval_server as server

    cfg = OmegaConf.create({"model": {"training_mode": "action_only"},
                           "dataloader": {"type": "ewm_piper_action_only", "chunk_size": 24}})
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    (tmp_path / "normalization.json").write_text("{}")
    (tmp_path / "checkpoint_step_1.pt").write_bytes(b"fixture1")
    (tmp_path / "checkpoint_step_2.pt").write_bytes(b"fixture2")
    ready = tmp_path / "ready.json"
    seen = []

    class Model:
        training_mode = "action_only"
        event_codec = None
        predict_velocity = None

        def load_policy_state_dict(self, state):
            seen.append("switched")

        def sample_policy(self, batch, **kw):
            assert set(batch) == {"obs_video", "proprio", "prompt"}
            assert kw["horizon"] == 24 and kw["num_steps"] == 4
            assert isinstance(kw["cache_context_kv"], bool) and kw["decode_future"] is False
            action = torch.ones(1, 24, 7)
            return {"action": action, "action_normalized": action / 2}

    bad_arrays = dict(observation(), obs_voxel=np.zeros((2, 2, 256, 256), dtype=np.float32))
    messages = [encode_message({"op": "hello"}),
                encode_message(predict_header(), bad_arrays),
                encode_message(predict_header(), observation()),
                encode_message({"op": "clean_codec"}), encode_message({"op": "shutdown"})]
    replies = []
    connection = SimpleNamespace(recv=lambda **kw: messages.pop(0), send=lambda data: replies.append(decode_message(data)))

    class FakeServer:
        socket = SimpleNamespace(getsockname=lambda: ("127.0.0.1", 12345))

        def __init__(self, handler):
            self.handler = handler

        def serve_forever(self):
            assert json.loads(ready.read_text())["checkpoint_format"] == "ewm_action_only_v1"
            self.handler(connection)

        def shutdown(self):
            seen.append("shutdown")

    monkeypatch.setattr(server, "serve", lambda handler, *a, **kw: contextlib.nullcontext(FakeServer(handler)))
    monkeypatch.setattr(server, "load_policy", lambda *a, **kw: (Model(), Normalizer(), 24))
    monkeypatch.setattr(server, "prepare_request", lambda request, arrays, norm, **kw:
                        prepare_request(request, arrays, norm, horizon=24, device="cpu"))
    generator = torch.Generator
    monkeypatch.setattr(torch, "Generator", lambda **kw: generator())
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: zeros(*a, **dict(kw, device="cpu")))
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {
        "format": "ewm_action_only_v1", "model_config": {"training_mode": "action_only"},
        "step": 1, "horizon": 24, "model": {}})
    monkeypatch.setattr(sys, "argv", ["server", "--checkpoint-dir", str(tmp_path), "--ready-file", str(ready),
                                     "--benchmark-iters", str(benchmark_iters),
                                     "--warmup-iters", str(benchmark_iters)])
    server.main()
    assert replies[0][0]["input_keys"] == ["obs_video", "proprio", "prompt"]
    assert replies[0][0]["cache_context_kv"] is True
    assert replies[1][0]["ok"] is False and "no events" in replies[1][0]["error"]
    assert replies[2][0]["future_decoded"] is False
    assert set(replies[2][1]) == {"executed_actions"}
    np.testing.assert_array_equal(replies[2][1]["executed_actions"], np.ones((16, 7), dtype=np.float32))
    assert replies[3][0]["ok"] is False
    assert seen == ["switched", "shutdown"]
    assert not ready.exists()
    if benchmark_iters:
        assert replies[0][0]["startup_benchmark"]["action_max_abs_error"] == 0.
    # Fail before loading a GPU model if pointed at an existing joint training run.
    cfg.model.training_mode = "joint"
    OmegaConf.save(cfg, tmp_path / "config.yaml")
    monkeypatch.setattr(server, "load_policy", lambda *a, **kw: pytest.fail("unexpected model load"))
    with pytest.raises(ValueError, match="only action_only"):
        server.main()
