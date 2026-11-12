"""Portable native run layouts and the actual legacy launcher command boundary."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from piper_policies.metadata import file_sha256, implementation_identity, load_metadata, select_checkpoint
from piper_policies.resources import ResourceResolver


@pytest.fixture
def native_run(tmp_path):
    run = tmp_path / "native-run"
    weights = run / "checkpoints/weights"
    weights.mkdir(parents=True)
    for step in (30, 60):
        (weights / f"step_{step:06}.pt").write_bytes(f"checkpoint {step}".encode())
    (run / "config.yaml").write_text("data:\n  train:\n    num_frames: 25\n    action_video_freq_ratio: 2\n")
    (run / "dataset_stats.json").write_text('{"native": true}')
    return run


def test_native_checkpoint_selection_and_no_recursive_search(native_run):
    checkpoint, step = select_checkpoint(native_run, "fastwam")
    assert step == 60 and checkpoint == native_run / "checkpoints/weights/step_000060.pt"
    assert select_checkpoint(native_run, "fastwam", checkpoint_step=30)[1] == 30
    unrelated = native_run / "archive/weights"
    unrelated.mkdir(parents=True)
    (unrelated / "step_999999.pt").write_bytes(b"unrelated")
    assert select_checkpoint(native_run, "fastwam")[1] == 60
    (native_run / "step_000030.pt").write_bytes(b"ambiguous")
    with pytest.raises(ValueError, match="ambiguous"):
        select_checkpoint(native_run, "fastwam", checkpoint_step=30)
    assert select_checkpoint(native_run, "fastwam", checkpoint="checkpoints/weights/step_000030.pt")[1] == 30
    with pytest.raises(ValueError, match="unavailable or filename disagrees"):
        select_checkpoint(native_run, "fastwam", checkpoint="checkpoints/weights/step_000030.pt", checkpoint_step=60)


def test_native_stats_fallback_and_override_preserve_hashes(native_run):
    _, checkpoint, stats, descriptor = load_metadata(native_run, "fastwam")
    assert stats == native_run / "dataset_stats.json"
    assert descriptor["checkpoint_sha256"] == file_sha256(checkpoint)
    assert descriptor["normalization_sha256"] == file_sha256(stats)
    exported = native_run / "fastwam_stats.json"
    exported.write_text('{"exported": true}')
    assert load_metadata(native_run, "fastwam")[2] == exported
    override = native_run / "chosen-stats.json"
    override.write_text('{"chosen": true}')
    _, _, stats, selected = load_metadata(native_run, "fastwam", normalization=override)
    assert stats == override and selected["normalization_sha256"] == file_sha256(override)


def test_pretrained_relative_paths_and_explicit_map_priority(tmp_path):
    resolver = ResourceResolver(project_root=tmp_path / "project", pretrained_root=tmp_path / "weights",
                                path_maps=["/old/pretrained=/explicit"])
    for path in ("../pretrained/Wan-AI/weights.pt", "pretrained/Wan-AI/weights.pt", "/old-other/pretrained/Wan-AI/weights.pt"):
        assert resolver.resolve(path) == str(tmp_path / "weights/Wan-AI/weights.pt")
    assert resolver.resolve("/old/pretrained/Wan-AI/weights.pt") == "/explicit/Wan-AI/weights.pt"
    assert resolver.resolve("../other/weights.pt") == str(tmp_path / "other/weights.pt")
    assert ResourceResolver(project_root=tmp_path / "project").resolve("../pretrained/Wan-AI") == str(tmp_path / "pretrained/Wan-AI")
    cfg = OmegaConf.create({"model": {"video_backbone": {"model_path": "../pretrained/Wan-AI"}}})
    assert resolver.config(cfg, "openwam").model.video_backbone.model_path == str(tmp_path / "weights/Wan-AI")


@pytest.mark.parametrize("relative", [
    "model/architectures/ewm_inference.py", "model/action_backbone/components.py",
    "model/action_backbone/separate_action_dit.py", "model/architectures/event_codec.py",
    "model/video_backbone/cosmos_predict25/scheduler.py",
])
def test_direct_native_inference_source_changes_identity(tmp_path, monkeypatch, relative):
    from piper_policies import metadata
    root = tmp_path / "openwam"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_text("# native implementation one\n")
    monkeypatch.setattr(metadata.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(root)]))
    before, files = implementation_identity("ewm")
    key = "openwam/" + relative
    assert files[key] == file_sha256(source)
    source.write_text("# native implementation two\n")
    after, changed = implementation_identity("ewm")
    assert before != after and changed[key] != files[key]
    assert all(not Path(key).is_absolute() for key in changed)


@pytest.mark.parametrize("mismatched_ready", [False, True])
def test_legacy_suite_forwards_to_describe_and_serve(native_run, tmp_path, monkeypatch, mismatched_ready):
    from piper_bench.evaluation import suite
    from piper_policies import legacy_suite

    config = tmp_path / "chosen-config.yaml"
    config.write_text("data:\n  train:\n    num_frames: 17\n    action_video_freq_ratio: 2\n")
    stats = tmp_path / "chosen-stats.json"
    stats.write_text('{"chosen": true}')
    commands, state = [], {}
    real_run = legacy_suite.subprocess.run
    real_popen = legacy_suite.subprocess.Popen

    def describe(command, **kwargs):
        commands.append(command)
        result = real_run(command, **kwargs)
        state["descriptor"] = json.loads(result.stdout)
        return result

    class Process:
        pid = 123456

        def __init__(self, command, **kwargs):
            commands.append(command)
            descriptor = dict(state["descriptor"])
            if mismatched_ready:
                descriptor["checkpoint_sha256"] = "different"
            ready = Path(command[command.index("--ready-file") + 1])
            ready.write_text(json.dumps(dict(policy_descriptor=descriptor, url="ws://localhost:1234")))

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def run_suite(options, *, descriptor_provider, policy_context):
        descriptor = descriptor_provider("conveyor")
        assert descriptor["checkpoint_step"] == 30
        assert descriptor["action_horizon"] == 16
        assert descriptor["config_sha256"] == file_sha256(config)
        assert descriptor["normalization_sha256"] == file_sha256(stats)
        with policy_context("conveyor") as endpoint:
            assert endpoint == "ws://localhost:1234"
        return 0

    monkeypatch.setattr(legacy_suite.subprocess, "run", describe)
    monkeypatch.setattr(legacy_suite.subprocess, "Popen",
                        lambda command, **kwargs: real_popen(command, **kwargs) if command[4] == "describe" else Process(command, **kwargs))
    monkeypatch.setattr(legacy_suite.os, "killpg", lambda *args: None)
    monkeypatch.setattr(suite, "run", run_suite)
    arguments = ["--preflight", "--family", "fastwam", "--checkpoint-dir", str(native_run),
                 "--checkpoint", "checkpoints/weights/step_000030.pt", "--checkpoint-step", "30",
                 "--config", str(config), "--normalization", str(stats),
                 "--path-map", "/old=/new", "--path-map", "/specific=/precise",
                 "--output-dir", str(tmp_path / "results"), "--tasks", "conveyor"]
    if mismatched_ready:
        with pytest.raises(ValueError, match="differs from preflight descriptor"):
            legacy_suite.main(arguments)
    else:
        assert legacy_suite.main(arguments) == 0
    assert [command[4] for command in commands] == ["describe", "serve"]
    for command in commands:
        for flag in ("--checkpoint", "--checkpoint-step", "--config", "--normalization"):
            assert command[command.index(flag) + 1] == arguments[arguments.index(flag) + 1]
        assert [command[i + 1] for i, token in enumerate(command) if token == "--path-map"] == ["/old=/new", "/specific=/precise"]
