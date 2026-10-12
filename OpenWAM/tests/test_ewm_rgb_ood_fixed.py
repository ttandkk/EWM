"""Fixed pure OOD controls, all three CLI entrypoints, and resume isolation."""
import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.eval_ewm.ewm_rgb_ood import FIXED_MODES, RGBOODConfig, TrialRGBOOD, check_resume_rgb_ood

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = [
    ROOT / "scripts/eval_ewm/eval_ewm_piper_sim.py",
    ROOT / "scripts/eval_ewm/eval_ewm_action_only_piper_sim.py",
    ROOT / "scripts/evel_piper_sim_rgb.py",
]


@pytest.mark.parametrize("mode", list(FIXED_MODES))
def test_fixed_parameters_and_exact_pixels(mode):
    frame = np.random.default_rng(2).integers(0, 256, (2, 3, 16, 16), dtype=np.uint8)
    before = frame.copy()
    frame.setflags(write=False)
    first = TrialRGBOOD(RGBOODConfig(mode=mode, seed=1), 0)
    second = TrialRGBOOD(RGBOODConfig(mode=mode, severity="strong", seed=999), 40)
    assert first.parameters == second.parameters
    assert first.parameters[0]["lighting"] == first.parameters[1]["lighting"]
    assert first.parameters[0]["motion_blur"] == first.parameters[1]["motion_blur"]
    np.testing.assert_array_equal(first.apply_rgb(frame), second.apply_rgb(frame))
    obs = {"obs_video": frame, "obs_voxel": object(), "proprio": object(), "timing": object()}
    result = first.apply_observation(obs)
    for key in ("obs_voxel", "proprio", "timing"):
        assert result[key] is obs[key]
    np.testing.assert_array_equal(frame, before)
    json.dumps(first.metadata(), allow_nan=False)
    assert first.metadata()["parameters_fixed_across_trials"] is True
    assert first.metadata()["severity"] == "fixed"
    if mode == "clean":
        assert result is obs
        assert first.config.label() == ""
        return
    assert first.config.label() == f"RGB OOD {mode}"
    assert result["obs_video"].dtype == np.uint8
    assert result["obs_video"].flags.c_contiguous
    # Independent double-precision reference for linear-light processing.
    linear = frame.astype(np.float64) / 255
    linear = np.where(linear <= 0.04045, linear / 12.92, ((linear + 0.055) / 1.055) ** 2.4)
    gain, length = FIXED_MODES[mode]
    if gain is not None:
        linear *= gain
        assert first.config.lighting and not first.config.motion_blur
        assert np.all(result["obs_video"] <= frame)
    else:
        assert first.config.motion_blur and not first.config.lighting
        kernel = np.full((1, length), 1 / length, dtype=np.float64)
        linear = np.stack([cv2.filter2D(im.transpose(1, 2, 0), -1, kernel,
                          borderType=cv2.BORDER_REFLECT_101).transpose(2, 0, 1) for im in linear])
    srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    np.testing.assert_array_equal(result["obs_video"], np.rint(srgb * 255).astype(np.uint8))


@pytest.mark.parametrize("length", [3, 5])
def test_blur_horizontal_only_and_constant_preserving(length):
    transform = TrialRGBOOD(RGBOODConfig(mode=f"blur-{length}"), 0)
    frame = np.zeros((2, 3, 32, 32), np.uint8)
    frame[:, 0] = 170
    np.testing.assert_array_equal(transform.apply_rgb(frame), frame)
    frame[:, 0] = 0
    frame[:, 0, 16, 16] = 255
    result = transform.apply_rgb(frame)
    assert np.count_nonzero(result[0, 0]) == length
    assert np.count_nonzero(result[:, 1:]) == 0
    assert np.count_nonzero(result[:, :, :16]) == 0
    assert np.count_nonzero(result[:, :, 17:]) == 0


def cli_namespace(source):
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.Expr)
                   and isinstance(node.value, ast.Call)
                   and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == "add_argument"
                   and node.value.args and isinstance(node.value.args[0], ast.Constant)
                   and str(node.value.args[0].value).startswith("--rgb-")]
    validation = next(node for node in tree.body if isinstance(node, ast.If)
                      and any(isinstance(child, ast.Attribute) and child.attr == "rgb_ood_mode"
                              for child in ast.walk(node.test)))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "rgb_ood_config"
                              for target in node.targets))
    namespace = {"parser": argparse.ArgumentParser(), "argparse": argparse, "RGBOODConfig": RGBOODConfig}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, [validation, assignment]


@pytest.mark.parametrize("source", CLIENTS, ids=lambda p: p.name)
@pytest.mark.parametrize("mode", ["legacy", *FIXED_MODES])
def test_actual_cli_accepts_mode_and_constructs_config(source, mode):
    namespace, nodes = cli_namespace(source)
    namespace["args"] = namespace["parser"].parse_args(["--rgb-ood-mode", mode])
    namespace["rgb_ood_seed"] = 10000
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    config = namespace["rgb_ood_config"]
    assert config.mode == mode
    assert (config.lighting, config.motion_blur) == (mode.startswith("exposure-"), mode.startswith("blur-"))


@pytest.mark.parametrize("source", CLIENTS, ids=lambda p: p.name)
@pytest.mark.parametrize("flag", ["--rgb-ood", "--rgb-lighting-ood", "--no-rgb-motion-blur-ood"])
def test_cli_rejects_mixing_modes_before_sim_start(source, flag):
    namespace, nodes = cli_namespace(source)
    namespace["args"] = namespace["parser"].parse_args(["--rgb-ood-mode", "blur-3", flag])
    namespace["rgb_ood_seed"] = 10000
    with pytest.raises(SystemExit) as error:
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    assert error.value.code == 2


def test_legacy_metadata_and_label_unchanged():
    assert RGBOODConfig().metadata() == {
        "enabled": False, "lighting": False, "motion_blur": False, "severity": "moderate",
        "seed": 10000, "implementation": "rgb_sensor_postprocess_v1",
        "scope": "obs_video_only", "parameters_fixed_per_trial": True,
    }
    assert RGBOODConfig(True, True).label() == "RGB OOD moderate: lighting+motion-blur"
    assert RGBOODConfig().label() == ""
    with pytest.raises(ValueError):
        RGBOODConfig(mode="unknown")
    with pytest.raises(ValueError):
        RGBOODConfig(lighting=True, mode="blur-3")
    with pytest.raises(ValueError):
        RGBOODConfig(motion_blur=True, mode="clean")


@pytest.mark.parametrize("mode", list(FIXED_MODES))
def test_resume_guard_and_summary_report_mode(mode, tmp_path):
    from scripts.eval_ewm.ewm_eval_summary import write_summary

    metadata = RGBOODConfig(mode=mode).metadata()
    check_resume_rgb_ood(metadata, metadata)
    if mode != "clean":
        with pytest.raises(ValueError):
            check_resume_rgb_ood(None, metadata)
    with pytest.raises(ValueError):
        check_resume_rgb_ood(RGBOODConfig().metadata(), metadata)
    other = "blur-5" if mode == "blur-3" else "blur-3"
    with pytest.raises(ValueError):
        check_resume_rgb_ood(RGBOODConfig(mode=other).metadata(), metadata)
    write_summary(tmp_path, max_steps=500, sampling_steps=4, requested_trials=50,
                  attempted_trials=0, results=[], task_name="stack", rgb_ood=metadata)
    text = (tmp_path / "summary.txt").read_text()
    assert f"rgb_ood_mode: {mode}" in text
    assert "rgb_ood_severity: fixed" in text
    assert "rgb_ood_parameters_fixed_across_trials: true" in text
    assert f"rgb_ood_exposure_gain: {metadata['exposure_gain']}" in text
    assert f"rgb_ood_blur_kernel_length_px: {metadata['blur_kernel_length_px']}" in text
