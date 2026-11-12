"""CPU regression for six-task sampling, policy loops and recovery; no Isaac."""
import copy
import json
import hashlib
import shutil
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Event-DataCollection/scripts"))

from piper_tasks import load_task
from piper_tasks.collection_protocol import TASKS, SPEEDS, VERSION, base_config, episode_spec, make_plan
from piper_tasks.evaluation_protocol import (
    digest, load_scene_manifest, scene_manifest, stable_steps, validate_collection,
)
from scripts.eval_ewm import eval_ewm_piper_tasks_sim as evaluator
from scripts.eval_ewm.ewm_eval_wire import encode_message, decode_message


def collection(task, version=VERSION):
    plan = make_plan(task, 50, 20261016, "train", version=version)
    module = load_task(task)
    return dict(version=plan["version"], task=task, task_id=module.TASK, plan=plan,
                base_config=base_config(task, plan).metadata(), camera=module.camera_metadata(),
                renderer="pathtracing", standard_imaging=True, pathtracing_spp=64,
                control_hz=30, physics_hz=120, event_hz=240, initial_hold_steps=1,
                success_version="existing_physical_predicates_v1")


def export(tmp_path, task):
    data = collection(task)
    path = tmp_path/task
    meta = path/"meta"
    meta.mkdir(parents=True)
    files = {"piper_collection_config.json": data, "piper_camera_calibration.json": data["camera"],
             "piper_validation.json": {"task": task, "passed": True},
             "piper_temporal_contract.json": dict(temporal_contract="aligned_event_frame_pre_action_v1",
                 task_selector=task, collection_protocol=data["version"], control_hz=30, event_hz=240)}
    for name, value in files.items():
        (meta/name).write_text(json.dumps(value))
    for name in (".collection_complete", ".export_complete"):
        (path/name).write_text("{}")
    return path


@pytest.mark.parametrize("task", TASKS)
def test_scenes_are_deterministic_and_disjoint_from_training(task):
    data = collection(task)
    scenes = scene_manifest(data)
    assert scenes == scene_manifest(data)
    assert len(scenes["episodes"]) == 50
    training = {episode_spec(data["plan"], i)["seed"] for i in range(50)}
    assert not training.intersection(s["seed"] for s in scenes["episodes"])
    assert all(s["split"] == "evaluation" and s["retry"] == 0 for s in scenes["episodes"])
    assert scenes["collection_sha256"] == digest(data)


@pytest.mark.parametrize("task", SPEEDS)
def test_speed_ood_is_strictly_outside_training_and_preserves_poses(task):
    data = collection(task)
    low, high = (data["plan"]["speed"][key] for key in ("low", "high"))
    conditions = {c:scene_manifest(data, condition=c) for c in ("id", "slow", "fast")}
    for i in range(50):
        id_, slow, fast = (conditions[c]["episodes"][i] for c in ("id", "slow", "fast"))
        assert low <= id_["speed"] <= high
        assert .85*low <= slow["speed"] < low
        assert high < fast["speed"] <= 1.15*high
        for key in ("object_poses", "object_yaws", "phase", "seed", "cabinet_pose"):
            assert id_[key] == slow[key] == fast[key]
        cfg = load_task(task).Config(**fast["config"])
        actual = cfg.peak_speed if task == "moving_carton" else abs(cfg.omega)
        assert actual == pytest.approx(fast["speed"])


def test_push_t_legacy_and_pose_v2_remain_separate():
    from piper_tasks.push_t.protocol import VERSION as WIDE
    for version, distance, angle in [(VERSION, .01, math.radians(10)), (WIDE, .03, math.pi/4)]:
        data = collection("push_t", version)
        scenes = scene_manifest(data)
        for spec in scenes["episodes"]:
            assert abs(spec["object_poses"][0][0]-.36) <= distance
            assert abs(spec["object_poses"][0][1]+.105) <= distance
            assert abs(spec["object_yaws"][0]) <= angle
            assert spec["config"]["position_jitter"] == distance
            assert spec["config"]["target_y"] == .105


@pytest.mark.parametrize("task", ("push_t", "drawer_cup", "square_nut"))
def test_static_tasks_reject_speed_ood(task):
    with pytest.raises(ValueError, match="dynamic"):
        scene_manifest(collection(task), condition="fast")


@pytest.mark.parametrize("task", TASKS)
def test_export_metadata_and_time_limits(tmp_path, task):
    path = export(tmp_path, task)
    assert validate_collection(task, path) == collection(task)
    args = evaluator.build_parser().parse_args(["--task", task, "--dataset-dir", str(path), "--plan-only"])
    evaluator.load_settings(args)
    assert args.max_steps == round(collection(task)["base_config"]["max_seconds"]*30)


@pytest.mark.parametrize("change", ["task", "camera", "clock", "geometry", "incomplete", "contract"])
def test_export_mismatches_fail_before_isaac(tmp_path, change):
    path = export(tmp_path, "square_nut")
    file = path/"meta/piper_collection_config.json"
    data = json.loads(file.read_text())
    if change == "task": data["task"] = "push_t"
    elif change == "camera": data["camera"]["calibration_id"] = "wrong"
    elif change == "clock": data["event_hz"] = 30
    elif change == "geometry": data["base_config"]["stable_seconds"] = .01
    elif change == "incomplete": (path/".export_complete").unlink()
    elif change == "contract":
        (path/"meta/piper_temporal_contract.json").write_text('{}')
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError): validate_collection("square_nut", path)


def test_manifest_tampering_is_rejected(tmp_path):
    data = scene_manifest(collection("rotating_tube"))
    file = tmp_path/"scenes.json"
    file.write_text(json.dumps(data))
    assert load_scene_manifest(file, data) == data
    bad = copy.deepcopy(data)
    bad["episodes"][0]["object_poses"][0][0] += .001
    file.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="manifest differs"): load_scene_manifest(file, data)


def test_dataset_relocation_content_and_shortened_budget_are_checked_on_cpu(tmp_path):
    dataset = export(tmp_path, "drawer_cup")
    checkpoint = tmp_path / "run"
    checkpoint.mkdir()
    # Deliberately invalid YAML proves that the benchmark never parses model config.
    (checkpoint / "config.yaml").write_text("[invalid training yaml")
    argv = ["--task", "drawer_cup", "--checkpoint-dir", str(checkpoint),
            "--dataset-dir", str(dataset), "--plan-only"]
    parser = evaluator.build_parser()
    assert evaluator.load_settings(parser.parse_args(argv))[0] == dataset
    moved = tmp_path / "relocated" / "drawer_cup"
    shutil.copytree(dataset, moved)
    assert evaluator.load_settings(parser.parse_args(argv + ["--dataset-dir", str(moved)]))[0] == moved
    expected = hashlib.sha256((dataset / "meta/piper_collection_config.json").read_bytes()).hexdigest()
    descriptor = {"training_data": {"drawer_cup": {"collection_sha256": expected}}}
    assert evaluator.verify_policy_data(descriptor, "drawer_cup", moved, "piper_eight_v1")["status"] == "verified"
    metadata = moved / "meta/piper_collection_config.json"
    value = json.loads(metadata.read_text())
    value["base_config"]["stable_seconds"] = .01
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata differs"):
        evaluator.verify_policy_data(descriptor, "drawer_cup", moved, "piper_eight_v1")
    with pytest.raises(ValueError, match="time limit"):
        evaluator.load_settings(parser.parse_args(argv + ["--max-steps", "4"]))
    assert evaluator.load_settings(parser.parse_args(argv + ["--max-steps", "4", "--interface-only"]))


def test_static_stability_counter_takes_precedence_over_sequence_tracker():
    assert stable_steps(SimpleNamespace(stable_steps=21, tracker=SimpleNamespace(steps=0))) == 21
    assert stable_steps(SimpleNamespace(tracker=SimpleNamespace(steps=30))) == 30


def test_rgb_transform_preserves_current_inputs_without_changing_events():
    scene = {name:SimpleNamespace(data=SimpleNamespace(output={
        "rgb": torch.zeros((1, 256, 256, 4), dtype=torch.uint8)})) for name in ("cam0", "cam_wrist")}
    scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((1, 8))))
    runtime = SimpleNamespace(env=SimpleNamespace(scene=scene), time=0.)
    dvs = SimpleNamespace(_conveyor_snapshot={}, recorder=SimpleNamespace(_events={}),
                          warp_and_process=lambda *a: None)
    obs = evaluator.shared.observation(runtime, dvs, ("cam0", "cam_wrist"), rgb_transform=lambda x:x+64)
    assert np.all(obs["obs_video"] == 64) and np.all(obs["obs_voxel"] == 0)
    assert not any(key.startswith("memory_") for key in obs)
    # CHW-contiguous sensor postprocessing must still support OpenCV labeling.
    assert evaluator.shared.preview(obs, 0).shape == (256, 512, 3)


def test_real_cpu_video_encoding_and_decode(tmp_path):
    obs = dict(obs_video=np.zeros((2, 3, 256, 256), np.uint8),
               obs_voxel=np.zeros((2, 2, 256, 256), np.float32),
               event_counts={"cam0": 0, "cam_wrist": 0})
    path = tmp_path/"panels.mp4"
    video = evaluator.shared.TrialVideo(path, ("cam0", "cam_wrist"))
    try:
        video.write(obs, 0, 0)
        video.write(obs, 1, 0)
    finally:
        video.close()
    subprocess.run([video.ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
    assert path.stat().st_size > 0


def test_plan_only_launcher_cannot_submit_gpu_job(tmp_path):
    path = export(tmp_path, "moving_carton")
    bin_dir = tmp_path/"bin"
    bin_dir.mkdir()
    marker = tmp_path/"sbatch_was_called"
    sbatch = bin_dir/"sbatch"
    sbatch.write_text('#!/usr/bin/env bash\ntouch "$SBATCH_TEST_MARKER"\nexit 99\n')
    sbatch.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ["PATH"], SBATCH_TEST_MARKER=str(marker))
    command = ["bash", str(ROOT/"OpenWAM/scripts/eval_ewm/run_ewm_piper_tasks_eval.sh"),
               "--task", "moving_carton", "--dataset-dir", str(path), "--plan-only", "--trials", "2"]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90, check=True)
    assert len(json.loads(completed.stdout)["episodes"]) == 2
    assert not marker.exists()


def test_task_capture_keeps_dynamic_hooks_and_event_clock(monkeypatch):
    import piper_tasks.collection_runtime as capture_api
    calls = []
    module = SimpleNamespace(set_views=lambda env:calls.append("views"))
    monkeypatch.setattr(evaluator, "load_task", lambda name:module)
    monkeypatch.setattr(evaluator.shared, "snapshot", lambda *a:calls.append("snapshot") or {"new": 1})
    monkeypatch.setattr(evaluator.shared, "observation", lambda *a, **kw:{"observed": True})

    def result(rt, name, action):
        calls.append("predicate")
        rt.time += 1/30
        return False, None, False

    monkeypatch.setattr(capture_api, "step_result", result)
    for task in TASKS:
        calls.clear()
        runtime = SimpleNamespace(_eval_task=task, time=0., before_step=lambda:calls.append("before"),
            env=SimpleNamespace(step=lambda action:calls.append("physics")),
            _eval_rgb=SimpleNamespace(apply_rgb=lambda rgb:rgb))
        warps = []
        dvs = SimpleNamespace(_conveyor_snapshot={"old": 1}, warp_and_process=lambda *a:warps.append(a))
        obs, success, reason = evaluator.capture_step(runtime, dvs, None, ("cam_wrist",))
        expected = (["before"] if task == "turntable_cups" else [])+["views", "physics", "predicate", "snapshot"]
        assert calls == expected
        assert warps[0] == ({"old": 1}, {"new": 1}, 8, 0., 1/240)
        assert runtime.time == 1/30 and runtime._eval_opened is False
        assert obs == {"observed": True} and success is False and reason is None


def test_resume_preserves_failure_and_archives_only_interrupted_trial(tmp_path):
    scenes = scene_manifest(collection("push_t"), trials=2)
    manifest = {"scene_manifest_sha256": digest(scenes), "model": "abc"}
    args = SimpleNamespace(resume=False, trials=2)
    assert evaluator.prepare_results(tmp_path, manifest, scenes, args) == []
    video = tmp_path/"trial_000_h264_failure.mp4"
    video.write_bytes(b"video")
    result = dict(trial=0, success=False, video=str(video))
    (tmp_path/"trial_000_result.json").write_text(json.dumps(result))
    (tmp_path/"trial_000_steps.json").write_text('[]')
    interrupted = tmp_path/"trial_001_h264_recording.mp4"
    interrupted.write_bytes(b"partial")
    args.resume = True
    assert evaluator.prepare_results(tmp_path, manifest, scenes, args) == [result]
    assert video.exists() and not interrupted.exists()
    assert (tmp_path/"trial_001_h264_recording_interrupted_01.mp4").exists()
    with pytest.raises(ValueError, match="resume"):
        evaluator.prepare_results(tmp_path, dict(manifest, model="changed"), scenes, args)


@pytest.mark.parametrize("controller", ["paused", "serial", "buffered", "continuous"])
@pytest.mark.parametrize("policy_mode", ["joint", "action_only", "rgb_action"])
def test_model_loop_records_failures_without_resampling_and_cleans_up(tmp_path, monkeypatch, controller, policy_mode):
    """Exercise real wire validation, scheduler, report and result files on CPU.

    Only Isaac physics and video encoding are replaced. One timeout must remain
    in the success denominator, and no Oracle or privileged observation is used.
    """
    import websockets.sync.client
    import piper_tasks.collection_runtime as capture_api
    import piper_tasks.event_preview as sensors
    data = collection("square_nut")
    module = load_task("square_nut")
    scenes = scene_manifest(data, trials=2)
    checkpoint = tmp_path/"checkpoint"
    checkpoint.mkdir()
    (checkpoint/"config.yaml").write_text("model: {}")
    (checkpoint/"checkpoint_step_10.pt").write_bytes(b"mock_policy")
    ready = tmp_path/"ready.json"
    ready.write_text('{"url":"ws://test"}')
    out = tmp_path/"eval"
    args = evaluator.build_parser().parse_args([
        "--task", "square_nut", "--checkpoint-dir", str(checkpoint), "--ready-file", str(ready),
        "--output-dir", str(out), "--trials", "2", "--max-steps", "8", "--interface-only",
        "--execution-horizon", "3", "--latency-controller", "serial" if controller == "paused" else controller,
        "--policy-mode", policy_mode, "--no-latency-aware" if controller == "paused" else "--latency-aware"])
    evaluator.shared.validate_args(args)
    env = SimpleNamespace(device="cpu", closed=False, sim=SimpleNamespace(), scene={})
    for camera in ("cam0", "cam_wrist"):
        env.scene[camera] = SimpleNamespace(data=SimpleNamespace(
            output={"rgb": torch.zeros((1, 256, 256, 4), dtype=torch.uint8)}))
    env.scene['robot'] = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 8)))
    env.reset = lambda **kw: None
    env.step = lambda action: None
    env.close = lambda: setattr(env, "closed", True)
    runtime = SimpleNamespace(env=env, cfg=module.Config(**data["base_config"]), time=0., stable_steps=0)
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(
        make=lambda *a, **kw: SimpleNamespace(unwrapped=env)))
    monkeypatch.setattr(evaluator, "register_tasks", lambda: None)
    fake_task = SimpleNamespace(TASK=module.TASK, PROMPT=module.PROMPT, Runtime=lambda *a: runtime,
                               make_env_cfg=lambda *a, **kw: None, camera_metadata=module.camera_metadata,
                               set_views=lambda env: None)
    monkeypatch.setattr(evaluator, "load_task", lambda name: fake_task)
    monkeypatch.setattr(evaluator.shared, "configure_renderer", lambda spp: None)
    dvs = SimpleNamespace(recorder=SimpleNamespace(_events={}), reset=lambda x: None,
                          warp_and_process=lambda *a: None)
    monkeypatch.setattr(sensors, "create_event_camera", lambda *a: dvs)
    monkeypatch.setattr(evaluator.shared, "snapshot", lambda *a: {})
    original_tensor = torch.tensor
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: original_tensor(
        *a, **(dict(kw, device="cpu") if kw.get("device") == "cuda:0" else kw)))
    initialized = []

    def initialize(rt, spec):
        initialized.append(spec)
        rt.time = 0.
        rt.trial = spec["slot"]
        for camera in ("cam0", "cam_wrist"):
            env.scene[camera].data.output['rgb'].fill_(rt.trial * 20)
        return {"object_poses": spec["object_poses"], "robot_joints": [0.]*8}

    def step_result(rt, name, action):
        rt.time += 1/30
        for camera in ("cam0", "cam_wrist"):
            env.scene[camera].data.output['rgb'].fill_(rt.trial * 20 + round(rt.time * 30))
        done = rt.time >= 7/30-1e-8
        return done and rt.trial == 1, "timeout" if done and rt.trial == 0 else None, True

    monkeypatch.setattr(capture_api, "initialize", initialize)
    monkeypatch.setattr(capture_api, "step_result", step_result)
    monkeypatch.setattr(capture_api, "terminal_state", lambda *a: {"stable_steps": 0})
    monkeypatch.setattr(capture_api, "terminal_valid", lambda *a: True)

    class Video:
        def __init__(self, path, *a, **kw): self.path, self.ffmpeg = path, "mock_ffmpeg"
        def write(self, *a, **kw): pass
        def close(self): self.path.write_bytes(b"mock_video")

    monkeypatch.setattr(evaluator.shared, "TrialVideo", Video)
    monkeypatch.setattr(evaluator.subprocess, "run", lambda *a, **kw: None)
    headers = []

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def send(self, message):
            self.header, arrays = decode_message(message)
            if self.header["op"] == "predict":
                headers.append(self.header)
                expected = ({"obs_video", "proprio"} if policy_mode != "joint"
                            else {"obs_video", "obs_voxel", "proprio"})
                assert set(arrays) == expected
                if policy_mode == "rgb_action":
                    assert self.header["decode_future_rgb"] is False
                assert self.header["prompt"] == module.PROMPT
        def recv(self, **kw):
            if self.header["op"] == "hello":
                hello = dict(ok=True, checkpoints=[10], horizon=24, config_sha256=evaluator.hashlib.sha256(
                    (checkpoint/"config.yaml").read_bytes()).hexdigest(), normalization_sha256="test",
                    checkpoint_sha256={"10": hashlib.sha256(b"mock_policy").hexdigest()},
                    event_time_contract=evaluator.shared.EVENT_TIME_CONTRACT,
                    inference_observation_row_offset=0, training_observation_row_offset=0,
                    event_camera_mode="dual", event_cameras=2, voxel_bins=1, count_clip=15,
                    deferred_visualization_supported=True)
                if policy_mode == "action_only":
                    hello.update(training_mode="action_only", checkpoint_format="ewm_action_only_v1",
                                 observation_contract=evaluator.shared.OBSERVATION_CONTRACT,
                                 input_keys=["obs_video", "proprio", "prompt"])
                elif policy_mode == "rgb_action":
                    hello.update(training_mode="rgb_action", checkpoint_format="ewm_rgb_action_latent_v1",
                                 observation_contract=evaluator.shared.RGB_ACTION_CONTRACT,
                                 input_keys=["obs_video", "obs_rgb", "proprio", "prompt"])
                return encode_message(hello)
            horizon = self.header["action_chunk_size"]
            return encode_message(dict(ok=True, inference_seconds=.001),
                                  {"executed_actions": np.zeros((horizon, 7), np.float32)})

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: Connection())
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/piper_collection_config.json").write_text(json.dumps(data))
    evaluator.run(args, dataset, data, runtime.cfg, scenes)
    report = json.loads((out/"complete.json").read_text())
    assert report["trials"] == 2 and report["successes"] == 1
    assert report["success_rate"] is None  # interface diagnostics never report a formal rate
    assert report["success_rate_denominator"] == "requested_trials"
    assert [r["success"] for r in report["results"]] == [False, True]
    assert initialized == scenes["episodes"] and len(headers) >= 2
    if controller == "continuous":
        for result in report['results']:
            timing = result['latency_timing']
            assert timing['controller'] == 'continuous'
            requests = timing['inference_requests']
            stats = result['continuous_plan_statistics']
            assert stats['submitted'] == len(requests)
            assert stats['received'] + stats['censored'] == stats['submitted']
            assert len(requests) >= 2
            for previous, following in zip(requests, requests[1:]):
                assert following['observation_step'] == max(previous['scheduled_ready_step'],
                                                            previous['observation_step']+1)
            rows = json.loads((out/f"trial_{result['trial']:03d}_steps.json").read_text())
            for row in rows:
                if row['action_source'] == 'policy':
                    assert row['chunk_index'] == row['step']-1-row['source_observation_step']
    assert env.closed
