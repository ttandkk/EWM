"""CPU checks for the formal wrapper's resolved protocol and launch boundary."""
import importlib.util
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("piper_final_profile", ROOT / "scripts/eval_piper_final.py")
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)


@pytest.fixture
def options(tmp_path):
    checkpoint = tmp_path / "checkpoint with spaces"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("model: {training_mode: joint}\n")
    return ["--checkpoint-dir", str(checkpoint), "--output-dir", str(tmp_path / "new results")]


@pytest.mark.parametrize("mode,recording", [("action_only", False), ("joint", True), ("rgb_action", True)])
def test_formal_command_reaches_general_suite_parser(options, mode, recording):
    suite_spec = importlib.util.spec_from_file_location("final_protocol_suite", ROOT / "scripts/eval_piper_eight_dynamic.py")
    suite = importlib.util.module_from_spec(suite_spec)
    suite_spec.loader.exec_module(suite)
    args = formal.parser().parse_args(options + ["--conditions", "id", "speed_ood", "--preflight"])
    command = formal.build_command(args, json.loads(formal.PROFILE.read_text()), mode)
    resolved = suite.parser().parse_args(command[2:])
    assert resolved.tasks == ["conveyor", "moving_carton", "turntable_cups", "rotating_tube",
                              "tennis_rolling", "tennis_falling", "tennis_parabolic", "tennis_container"]
    assert resolved.conditions == ["id", "speed_ood"]
    assert (resolved.trials, resolved.seed, resolved.scene_seed) == (50, 10000, 20280002)
    assert (resolved.latency_controller, resolved.min_execution_steps, resolved.sampling_steps) == ("continuous", 8, 4)
    assert resolved.latency_aware and resolved.preflight
    assert resolved.execution_horizon == 10
    assert resolved.checkpoint_step == 80000
    assert resolved.record_predictions is recording
    assert not resolved.decode_predicted_events  # No inline decoding in equivalent latency.
    assert "--max-steps" not in command and not resolved.interface_only


def test_profile_overrides_stale_protocol_environment_but_preserves_resources(options, monkeypatch):
    monkeypatch.setenv("COMPILE_CORE", "false")
    monkeypatch.setenv("COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("COMPILE_CUDA_GRAPHS", "true")
    monkeypatch.setenv("CACHE_CONTEXT_KV", "false")
    monkeypatch.setenv("ALLOW_COMPILE_ACTION_ERROR", "false")
    monkeypatch.setenv("WARMUP_ITERS", "99")
    monkeypatch.setenv("EWM_EVAL_GRES", "gpu:pro6000:1")
    monkeypatch.setenv("EWM_EVAL_TIME", "3-00:00:00")
    monkeypatch.setenv("SLURM_JOB_ID", "test-existing-allocation")
    calls = []

    def execute(command, *, env, check):
        calls.append(command)
        assert check is False
        assert env["COMPILE_CORE"] == "true"
        assert env["COMPILE_MODE"] == "default"
        assert env["COMPILE_CUDA_GRAPHS"] == "false"
        assert env["CACHE_CONTEXT_KV"] == env["CACHE_BF16_WEIGHTS"] == "true"
        assert env["ALLOW_COMPILE_ACTION_ERROR"] == env["POLICY_SERVER_PROBE"] == "true"
        assert env["WARMUP_ITERS"] == env["BENCHMARK_ITERS"] == "3"
        assert env["EWM_EVAL_GRES"] == "gpu:pro6000:1"
        assert env["EWM_EVAL_TIME"] == "3-00:00:00"
        assert env["SLURM_JOB_ID"] == "test-existing-allocation"
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(formal.subprocess, "run", execute)
    assert formal.main(options) == 7  # Preserve failures instead of reporting a successful suite.
    assert len(calls) == 1


@pytest.mark.parametrize("mode,phase", [("action_only", "disabled"), ("joint", "post_trial"), ("rgb_action", "post_trial")])
def test_dry_run_reads_modality_without_weights_output_or_gpu(options, monkeypatch, capsys, mode, phase):
    checkpoint = Path(options[1])
    (checkpoint / "config.yaml").write_text(f"model: {{training_mode: {mode}}}\n")
    monkeypatch.setattr(formal.subprocess, "run", lambda *a, **kw: pytest.fail("dry run attempted execution"))
    assert formal.main(options + ["--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training_mode"] == mode
    assert report["prediction_decode_phase"] == phase
    assert report["prediction_decode_in_equivalent_latency"] is False
    assert not Path(options[3]).exists()
    assert not list(checkpoint.glob("*.pt"))


@pytest.mark.parametrize("override", [
    ["--min-execution-steps", "4"], ["--latency-controller", "buffered"],
    ["--no-latency-aware"], ["--no-record-predictions"], ["--decode-predicted-events"],
    ["--trials", "2"], ["--max-steps", "32"], ["--sampling-steps", "1"],
])
def test_fixed_protocol_cannot_be_silently_overridden(options, override):
    with pytest.raises(SystemExit) as error:
        formal.parser().parse_args(options + override)
    assert error.value.code == 2


def test_duplicate_conditions_and_unsupported_checkpoint_fail_before_launch(options, monkeypatch):
    monkeypatch.setattr(formal.subprocess, "run", lambda *a, **kw: pytest.fail("invalid request launched"))
    with pytest.raises(SystemExit):
        formal.main(options + ["--conditions", "id", "id"])
    (Path(options[1]) / "config.yaml").write_text("model: {training_mode: event_only}\n")
    with pytest.raises(SystemExit):
        formal.main(options)


def test_submission_validates_first_and_preserves_argument_boundaries(options, monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("EWM_EVAL_DIRECT", raising=False)
    monkeypatch.setenv("EWM_EVAL_GRES", "gpu:rtx5090:1")
    monkeypatch.setenv("EWM_EVAL_TIME", "12:00:00")
    root = tmp_path / "project root"
    root.mkdir()
    monkeypatch.setattr(formal, "ROOT", root)
    calls = []

    def execute(command, *, env, check):
        calls.append(command)
        assert env["COMPILE_MODE"] == "default"
        if len(calls) == 1:
            assert command[0] == "bash" and command[-1] == "--preflight"
        else:
            assert command[0] == "sbatch"
            assert "--gres=gpu:rtx5090:1" in command
            assert "--time=12:00:00" in command
            assert shlex.split(command[-1]) == calls[0][:-1]
            assert command[-2] == "--wrap"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(formal.subprocess, "run", execute)
    # These characters are passed as one literal pathname, never executed by the shell.
    options[3] = str(tmp_path / "result with spaces;$(literal)")
    assert formal.main(options) == 0
    assert len(calls) == 2


def test_preflight_failure_prevents_gpu_submission(options, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("EWM_EVAL_DIRECT", raising=False)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert command[0] == "bash" and command[-1] == "--preflight"
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(formal.subprocess, "run", execute)
    assert formal.main(options) == 3
    assert len(calls) == 1


@pytest.mark.parametrize("flag", ["--preflight", "--plan-only"])
def test_cpu_modes_never_submit_slurm_job(options, monkeypatch, flag):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("EWM_EVAL_DIRECT", raising=False)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert command[0] == "bash" and command[-1] == flag
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(formal.subprocess, "run", execute)
    assert formal.main(options + [flag]) == 0
    assert len(calls) == 1
