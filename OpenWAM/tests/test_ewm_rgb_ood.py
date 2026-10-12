"""RGB OOD reproducibility, directional PSF, and event isolation checks."""
import random
import argparse
import ast
import io
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts.eval_ewm.ewm_rgb_ood import RGBOODConfig, TrialRGBOOD, check_resume_rgb_ood, motion_kernel
from scripts.eval_ewm.ewm_eval_wire import decode_message, encode_message

SOURCE = Path(__file__).resolve().parents[1] / "scripts/eval_ewm/eval_ewm_piper_sim.py"


def extract_definitions(*names, namespace):
    # Importing the full client would launch Isaac. Execute the actual function
    # bodies with a tiny simulator double instead, not a reimplementation.
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


def observation():
    y, x = np.indices((64, 64))
    rgb = np.stack([((x // 4 + y // 4) % 2) * 160 + 40] * 3).astype(np.uint8)
    rgb = np.stack([rgb, np.roll(rgb, 3, axis=1)])
    return {"obs_video": rgb, "obs_voxel": np.arange(128, dtype=np.float32).reshape(2, 2, 8, 4),
            "proprio": np.arange(8, dtype=np.float32),
            "timing": {"obs_event_start_s": 0.0, "obs_event_end_s": 1 / 30},
            "event_count": 123, "event_counts": {"cam0": 60, "cam_wrist": 63}}


def test_disabled_is_identity():
    obs = observation()
    assert TrialRGBOOD(RGBOODConfig(), 0).apply_observation(obs) is obs


@pytest.mark.parametrize("lighting,blur", [(True, False), (False, True), (True, True)])
def test_rgb_only_uint8_and_no_input_mutation(lighting, blur):
    obs = observation()
    rgb_before, event_before = obs["obs_video"].tobytes(), obs["obs_voxel"].tobytes()
    obs["obs_video"].setflags(write=False)
    obs["obs_voxel"].setflags(write=False)
    result = TrialRGBOOD(RGBOODConfig(lighting, blur), 3).apply_observation(obs)
    assert result["obs_video"].dtype == np.uint8
    assert result["obs_video"].flags.c_contiguous
    assert result["obs_video"].shape == obs["obs_video"].shape
    assert result["obs_video"].tobytes() != rgb_before
    for key in obs:
        if key != "obs_video":
            assert result[key] is obs[key]
    assert obs["obs_video"].tobytes() == rgb_before
    assert obs["obs_voxel"].tobytes() == event_before
    for camera in range(2):
        assert not np.array_equal(result["obs_video"][camera], obs["obs_video"][camera])


def test_deterministic_trial_and_repeated_frames():
    config = RGBOODConfig(True, True, seed=77)
    first, resumed = TrialRGBOOD(config, 5), TrialRGBOOD(config, 5)
    assert first.metadata() == resumed.metadata()
    frame = observation()["obs_video"]
    np.testing.assert_array_equal(first.apply_rgb(frame), resumed.apply_rgb(frame))
    np.testing.assert_array_equal(first.apply_rgb(frame), first.apply_rgb(frame))
    assert first.metadata() != TrialRGBOOD(config, 6).metadata()


def test_component_streams_do_not_depend_on_other_switch():
    both = TrialRGBOOD(RGBOODConfig(True, True, seed=42), 2)
    lighting = TrialRGBOOD(RGBOODConfig(True, False, seed=42), 2)
    blur = TrialRGBOOD(RGBOODConfig(False, True, seed=42), 2)
    for b, l, m in zip(both.parameters, lighting.parameters, blur.parameters):
        assert b["lighting"] == l["lighting"]
        assert b["motion_blur"] == m["motion_blur"]


@pytest.mark.parametrize("config", [RGBOODConfig(True, True),
    *[RGBOODConfig(mode=mode) for mode in
      ("clean", "blur-3", "blur-5", "exposure-0.8", "exposure-0.6", "exposure-0.4")]])
def test_no_global_rng_consumption(config):
    import torch

    random.seed(5)
    np.random.seed(5)
    torch_before = torch.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    transform = TrialRGBOOD(config, 1)
    transform.apply_rgb(observation()["obs_video"])
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0]
    np.testing.assert_array_equal(after[1], numpy_before[1])
    assert after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_motion_psf_is_normalized_and_directional():
    horizontal, vertical = motion_kernel(11, 0), motion_kernel(11, 90)
    assert horizontal.min() >= 0
    assert horizontal.sum() == pytest.approx(1.0)
    y, x = np.indices(horizontal.shape)
    assert float((horizontal * (x - 5)**2).sum()) > 5 * float((horizontal * (y - 5)**2).sum())
    assert float((vertical * (y - 5)**2).sum()) > 5 * float((vertical * (x - 5)**2).sum())


def test_blur_preserves_constant_and_channel_order():
    transform = TrialRGBOOD(RGBOODConfig(motion_blur=True), 1)
    rgb = np.zeros((2, 3, 64, 64), dtype=np.uint8)
    rgb[:, 0] = 170
    np.testing.assert_array_equal(transform.apply_rgb(rgb), rgb)
    rgb[:, 0] = 0
    rgb[:, 0, 30, 30] = 255
    blurred = transform.apply_rgb(rgb)
    assert np.count_nonzero(blurred[:, 0]) > 2
    assert np.count_nonzero(blurred[:, 1:]) == 0


@pytest.mark.parametrize("severity", ["mild", "moderate", "strong"])
def test_all_presets_finite_bounded_and_serializable(severity):
    import json
    transform = TrialRGBOOD(RGBOODConfig(True, True, severity, 7), 0)
    out = transform.apply_rgb(observation()["obs_video"])
    assert out.min() >= 0 and out.max() <= 255
    json.dumps(transform.metadata(), allow_nan=False)


def test_invalid_settings_and_resume_guard():
    with pytest.raises(ValueError):
        RGBOODConfig(severity="unknown")
    with pytest.raises(ValueError):
        RGBOODConfig(seed=-1)
    with pytest.raises(ValueError):
        motion_kernel(8, 0)
    clean = RGBOODConfig().metadata()
    ood = RGBOODConfig(lighting=True).metadata()
    check_resume_rgb_ood(None, clean)
    check_resume_rgb_ood(ood, ood)
    with pytest.raises(ValueError):
        check_resume_rgb_ood(None, ood)
    with pytest.raises(ValueError):
        check_resume_rgb_ood(clean, ood)
    with pytest.raises(ValueError):
        check_resume_rgb_ood(ood, {**ood, "seed": 99})
    with pytest.raises(ValueError):
        check_resume_rgb_ood(ood, {**ood, "severity": "strong"})
    with pytest.raises(ValueError):
        check_resume_rgb_ood(ood, {**ood, "motion_blur": True})


@pytest.mark.parametrize("flags,expected", [
    ([], (False, False)),
    (["--rgb-ood"], (True, True)),
    (["--rgb-lighting-ood"], (True, False)),
    (["--rgb-motion-blur-ood"], (False, True)),
    (["--rgb-ood", "--no-rgb-motion-blur-ood"], (True, False)),
    (["--rgb-ood", "--no-rgb-lighting-ood"], (False, True)),
    (["--no-rgb-ood"], (False, False)),
])
def test_actual_cli_switches(flags, expected):
    tree = ast.parse(SOURCE.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == "add_argument"
                   and node.value.args and isinstance(node.value.args[0], ast.Constant)
                   and str(node.value.args[0].value).startswith("--rgb-")]
    namespace = {"argparse": argparse, "parser": argparse.ArgumentParser(), "RGBOODConfig": RGBOODConfig}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    args = namespace["parser"].parse_args([*flags, "--rgb-ood-severity", "strong", "--rgb-ood-seed", "23"])
    namespace.update(args=args, rgb_ood_seed=args.rgb_ood_seed)
    config_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "rgb_ood_config"
                               for target in node.targets))
    exec(compile(ast.Module(body=[config_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    config = namespace["rgb_ood_config"]
    assert (config.lighting, config.motion_blur) == expected
    assert config.severity == "strong" and config.seed == 23


class TensorDouble:
    def __init__(self, array):
        self.array = np.asarray(array)

    def __getitem__(self, index):
        return TensorDouble(self.array[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def any(self):
        return self.array.any()

    def __float__(self):
        return float(self.array)


class SimDouble:
    def __init__(self):
        self.counter = 0
        rgb = np.tile(observation()["obs_video"], (1, 1, 4, 4)).transpose(0, 2, 3, 1)
        self.scene = {name: SimpleNamespace(data=SimpleNamespace(output={"rgb": TensorDouble(image[None].copy())}))
                      for name, image in zip(("cam0", "cam_wrist"), rgb)}
        self.scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_pos=TensorDouble(np.arange(8)[None])))
        self.sim = SimpleNamespace(get_physics_step_count=lambda: self.counter, get_physics_dt=lambda: 1 / 240)

    def step(self, action):
        self.counter += 8
        for name in ("cam0", "cam_wrist"):
            rgb = self.scene[name].data.output["rgb"].array
            rgb[:] = np.roll(rgb, 1, axis=2)
        return None, TensorDouble([0.5]), TensorDouble([False]), TensorDouble([False]), {"ok": True}


class DVSDouble:
    def __init__(self, env):
        self.env = env
        self.recorder = SimpleNamespace(_events={})
        self.snapshots = []

    def snapshot(self):
        snapshot = tuple(self.env.scene[name].data.output["rgb"].array.copy()
                         for name in ("cam0", "cam_wrist"))
        self.snapshots.append(snapshot)
        return snapshot

    def warp_and_process(self, previous, current, warp, start, delta):
        # Events depend on the camera source pixels, so a mutation would also
        # break cross-step comparisons. Include both OFF and ON events.
        events = self.recorder._events.setdefault(0, {})
        for index, name in enumerate(("cam0", "cam_wrist")):
            x = int(current[index][0, 0, 0, 0])
            event = (np.array([x, x], np.uint16), np.array([90, 91], np.uint16),
                     np.array([start, start + warp * delta / 2]), np.array([-1, 1], np.int8))
            events.setdefault(f"DVS/{name}", []).append(event)


@pytest.mark.parametrize("event_cameras", [("cam0", "cam_wrist"), ("cam_wrist",)])
@pytest.mark.parametrize("config", [RGBOODConfig(True, False), RGBOODConfig(False, True),
    RGBOODConfig(True, True), *[RGBOODConfig(mode=mode) for mode in
      ("blur-3", "blur-5", "exposure-0.8", "exposure-0.6", "exposure-0.4")]])
def test_actual_capture_keeps_evis_source_events_and_state_clean(event_cameras, config):
    torch_double = SimpleNamespace(Tensor=TensorDouble,
        logical_or=lambda a, b: TensorDouble(np.logical_or(a.array, b.array)))
    namespace = extract_definitions("capture_step", "simulation_time", "event_voxel",
        namespace={"np": np, "math": math, "time": time, "torch": torch_double,
                   "set_views": lambda env: None, "CAMERAS": ("cam0", "cam_wrist"), "CONTROL_HZ": 30, "WARP": 8})
    clean_env, ood_env = SimDouble(), SimDouble()
    clean_dvs, ood_dvs = DVSDouble(clean_env), DVSDouble(ood_env)
    transform = TrialRGBOOD(config, 2)
    for _ in range(3):
        clean, *clean_extra = namespace["capture_step"](clean_env, clean_dvs, None, event_cameras=event_cameras)
        ood, *ood_extra = namespace["capture_step"](ood_env, ood_dvs, None,
                                                    event_cameras=event_cameras, rgb_ood=transform)
        np.testing.assert_array_equal(ood["obs_voxel"], clean["obs_voxel"])
        np.testing.assert_array_equal(ood["proprio"], clean["proprio"])
        assert ood["timing"] == clean["timing"]
        assert ood["event_count"] == clean["event_count"] > 0
        assert ood["event_counts"] == clean["event_counts"]
        assert ood_extra == clean_extra
        np.testing.assert_array_equal(ood["obs_video"], transform.apply_rgb(clean["obs_video"]))
        assert not np.array_equal(ood["obs_video"], clean["obs_video"])
        for name in ("cam0", "cam_wrist"):
            np.testing.assert_array_equal(ood_env.scene[name].data.output["rgb"].array,
                                          clean_env.scene[name].data.output["rgb"].array)
    for clean, ood in zip(clean_dvs.snapshots, ood_dvs.snapshots):
        for clean_camera, ood_camera in zip(clean, ood):
            np.testing.assert_array_equal(clean_camera, ood_camera)


def test_initial_and_every_control_capture_receive_the_transform():
    calls = [node for node in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "capture_step"]
    assert len(calls) == 2
    assert all(any(keyword.arg == "rgb_ood" and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == "rgb_ood" for keyword in call.keywords) for call in calls)


def test_disabled_capture_bypasses_corruption_and_ood_timing_is_nested():
    torch_double = SimpleNamespace(Tensor=TensorDouble,
        logical_or=lambda a, b: TensorDouble(np.logical_or(a.array, b.array)))
    namespace = extract_definitions("capture_step", "simulation_time", "event_voxel",
        namespace={"np": np, "math": math, "time": time, "torch": torch_double,
                   "set_views": lambda env: None, "CAMERAS": ("cam0", "cam_wrist"), "CONTROL_HZ": 30, "WARP": 8})
    disabled = TrialRGBOOD(RGBOODConfig(), 0)
    disabled.apply_rgb = lambda rgb: pytest.fail("disabled OOD must not enter the corruption path")
    baseline_env, disabled_env, enabled_env = SimDouble(), SimDouble(), SimDouble()
    baseline, *_ = namespace["capture_step"](baseline_env, DVSDouble(baseline_env), None)
    clean, *_ = namespace["capture_step"](disabled_env, DVSDouble(disabled_env), None,
                                         rgb_ood=disabled, profile_timing=True)
    for key in ("obs_video", "obs_voxel", "proprio"):
        np.testing.assert_array_equal(clean[key], baseline[key])
    assert "rgb_ood" not in clean["profile_seconds"]
    ood, *_ = namespace["capture_step"](enabled_env, DVSDouble(enabled_env), None,
                                       rgb_ood=TrialRGBOOD(RGBOODConfig(True, True), 0), profile_timing=True)
    timings = ood["profile_seconds"]
    assert 0 < timings["rgb_ood"] <= timings["capture_total"]


def test_wire_contract_preserves_uint8_rgb_and_original_events():
    obs = observation()
    transformed = TrialRGBOOD(RGBOODConfig(True, True), 0).apply_observation(obs)
    header, decoded = decode_message(encode_message({"op": "predict", "raw_observation": True},
        {name: transformed[name] for name in ("obs_video", "obs_voxel", "proprio")}))
    assert header["raw_observation"] is True
    assert decoded["obs_video"].dtype == np.uint8
    np.testing.assert_array_equal(decoded["obs_video"], transformed["obs_video"])
    np.testing.assert_array_equal(decoded["obs_voxel"], obs["obs_voxel"])
    np.testing.assert_array_equal(decoded["proprio"], obs["proprio"])


def test_six_panel_video_changes_rgb_only():
    namespace = extract_definitions("TrialVideo", namespace={"np": np, "cv2": cv2, "Path": Path,
                                                              "CAMERAS": ("cam0", "cam_wrist")})
    obs = {"obs_video": np.tile(observation()["obs_video"], (1, 1, 4, 4)),
           "obs_voxel": np.zeros((2, 2, 256, 256), np.float32),
           "event_counts": {"cam0": 123, "cam_wrist": 456}}
    obs["obs_voxel"][0, 0, 100, 100] = 1
    prediction = np.ones((2, 2, 256, 256), np.uint8)
    frames = []
    for config in (RGBOODConfig(), RGBOODConfig(True, True),
                   *[RGBOODConfig(mode=mode) for mode in
                     ("blur-3", "blur-5", "exposure-0.8", "exposure-0.6", "exposure-0.4")]):
        video = namespace["TrialVideo"].__new__(namespace["TrialVideo"])
        video.event_cameras, video.show_prediction = ("cam0", "cam_wrist"), True
        video.rgb_ood_label = config.label()
        video.process = SimpleNamespace(stdin=io.BytesIO())
        video.write(TrialRGBOOD(config, 2).apply_observation(obs), 1, 0, predicted_counts=prediction)
        frames.append(np.frombuffer(video.process.stdin.getvalue(), np.uint8).reshape(768, 512, 3))
    for frame in frames[1:]:
        assert not np.array_equal(frames[0][:256], frame[:256])
        np.testing.assert_array_equal(frames[0][256:], frame[256:])
