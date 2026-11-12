"""Regression tests for installation safety and honest diagnostic states."""
import hashlib
import io
import json
import os
import builtins
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from urllib.request import Request

import pytest

from piper_bench.assets.resolver import _TokenSafeRedirect, download, verify
from piper_bench.deployment.checks import classify_pip_check
from piper_bench.deployment.config import load_lock, runtime_environment, simulation_python
from piper_bench.deployment.doctor import _runtime_results, diagnose
from piper_bench.deployment.setup import install, install_plan, matches_patch, project_snapshot, stage_project


def manifest(payload=b"locked-content"):
    return {"schema_version": 1, "revision": "a" * 40, "repository": "test/assets", "repo_type": "dataset",
            "files": {"nested/asset.usd": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}}


class Opener:
    def __init__(self, payload):
        self.payload = payload

    def open(self, request, timeout):
        return io.BytesIO(self.payload)


def test_download_checks_content_and_never_overwrites(tmp_path):
    root = tmp_path / "assets"
    report = download(root, manifest(), opener=Opener(b"locked-content"))
    assert report["passed"] and report["verified_files"] == 1
    target = root / "nested/asset.usd"
    target.write_bytes(b"user-modified")
    with pytest.raises(FileExistsError):
        download(root, manifest(), opener=Opener(b"locked-content"))
    assert target.read_bytes() == b"user-modified"


def test_download_rejects_corruption_without_partial_target(tmp_path):
    with pytest.raises(ValueError, match="verification"):
        download(tmp_path, manifest(), opener=Opener(b"broken-content"))
    assert not (tmp_path / "nested/asset.usd").exists()
    assert not list(tmp_path.rglob(".asset-*"))


def test_assets_do_not_escape_via_symlink(tmp_path):
    root, outside = tmp_path / "assets", tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        download(root, manifest(), opener=Opener(b"locked-content"))
    assert not list(outside.iterdir())
    (outside / "asset.usd").write_bytes(b"locked-content")
    assert not verify(root, manifest())["passed"]


def test_redirect_does_not_leak_private_hub_token():
    request = Request("https://huggingface.co/source", headers={"Authorization": "Bearer test-secret"})
    redirect = _TokenSafeRedirect().redirect_request(request, None, 302, "Found", {}, "https://cdn.example/file")
    assert redirect.get_header("Authorization") is None


def test_only_exact_known_metadata_errors_are_accepted():
    line = "dvs-gen 0.1.0 has requirement numpy<2, but you have numpy 2.3.1."
    assert classify_pip_check(line, 1, "simulation")["passed"]
    assert not classify_pip_check(line.replace("2.3.1", "2.4.0"), 1, "simulation")["passed"]
    assert not classify_pip_check(line + "\nother 1.0 requires missing, which is not installed.", 1, "simulation")["passed"]
    assert not classify_pip_check(line, 1, "writer")["passed"]


def test_venv_interpreter_symlink_is_not_dereferenced(tmp_path, monkeypatch):
    target, binary = tmp_path / "system-python", tmp_path / "venv/bin/python"
    target.touch(); binary.parent.mkdir(parents=True); binary.symlink_to(target)
    monkeypatch.delenv("PIPER_SIM_PYTHON", raising=False)
    config = {"profiles": {"simulation": {"python": str(binary)}}}
    assert simulation_python(config) == binary
    assert simulation_python(config) != target


def test_runtime_settings_preserve_explicit_assets_and_apply_saved_license(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPER_ASSET_ROOT", str(tmp_path / "explicit-assets"))
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)
    cfg = {"prefix": str(tmp_path), "assets_dir": str(tmp_path / "configured-assets"),
           "profiles": {"simulation": {"installed": True}},
           "licenses": {"nvidia_isaac_sim": {"accepted": True, "version": "6.0.1.0"}}}
    env = runtime_environment(cfg)
    assert env["PIPER_ASSET_ROOT"] == str(tmp_path / "explicit-assets")
    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"
    monkeypatch.delenv("PIPER_ASSET_ROOT")
    assert "PIPER_ASSET_ROOT" not in runtime_environment({"prefix": str(tmp_path), "assets_dir": str(tmp_path / "absent")})


def test_backend_applies_settings_before_app_launcher_import(monkeypatch):
    from piper_bench.backend import isaac
    monkeypatch.setenv("PIPER_TEST_LAUNCH_READY", "no")
    monkeypatch.setattr(isaac, "runtime_environment", lambda: {"PIPER_TEST_LAUNCH_READY": "yes"})
    original = builtins.__import__
    sentinel = object()

    def importing(name, *args, **kwargs):
        if name == "isaaclab.app":
            assert os.environ["PIPER_TEST_LAUNCH_READY"] == "yes"
            return SimpleNamespace(AppLauncher=lambda args: SimpleNamespace(app=sentinel))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    assert isaac.launch(SimpleNamespace()) is sentinel


def test_setup_refuses_existing_unmanaged_data(tmp_path):
    (tmp_path / "user-data").write_text("preserve")
    plan = {"eula_required": False, "prefix": str(tmp_path)}
    with pytest.raises(ValueError, match="unmanaged"):
        install(plan)
    assert list(tmp_path.iterdir()) == [tmp_path / "user-data"]


def test_patch_resume_checks_new_files_and_rejects_extra_edits(tmp_path):
    target = tmp_path / "new_module.py"
    target.write_bytes(b"locked source\n")
    files = {target.name: {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "mode": "100644"}}
    assert matches_patch(tmp_path, files, "?? new_module.py\0")
    assert not matches_patch(tmp_path, files, "?? new_module.py\0 M unrelated.py\0")
    assert not matches_patch(tmp_path, files, "A  new_module.py\0")
    target.write_bytes(b"unrecognized change\n")
    assert not matches_patch(tmp_path, files, "?? new_module.py\0")


@pytest.mark.parametrize("resume", [False, True])
def test_setup_populates_clone_even_when_remote_head_matches_lock(tmp_path, monkeypatch, resume):
    from piper_bench.deployment import setup
    origin, prefix = tmp_path / "origin", tmp_path / "install"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", str(origin)], check=True)
    (origin / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    subprocess.run(["git", "-C", str(origin), "add", "pyproject.toml"], check=True)
    subprocess.run(["git", "-C", str(origin), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()
    target = prefix / "sources/fixture"
    clone = ["git", "clone", "--no-checkout", str(origin), str(target)]
    if resume:
        prefix.mkdir()
        (prefix / ".piper-managed.json").write_text(json.dumps(
            {"schema_version": 1, "managed_by": "piper-bench", "release": "fixture"}))
        subprocess.run(clone, check=True)
        assert setup.unpopulated_clone(target)
    binary = prefix / "envs/writer/bin/python"
    original_run = setup.subprocess.run
    def run(command, **kwargs):
        if command == [str(binary), "-m", "pip", "check"]:
            return SimpleNamespace(returncode=0, stdout="No broken requirements found.\n", stderr="")
        return original_run(command, **kwargs)
    monkeypatch.setattr(setup.subprocess, "run", run)
    plan = {"eula_required": False, "prefix": str(prefix), "lock_release": "fixture",
            "profile": "writer", "python": str(binary), "requirements_sha256": "fixture",
            "sources": [{"name": "fixture", "commit": commit, "url": str(origin)}], "first_party_snapshots": [],
            "commands": [clone, ["git", "-C", str(target), "checkout", "--detach", commit]]}
    if resume:
        plan["sources"][0]["url"] = str(tmp_path / "unexpected-origin")
        with pytest.raises(ValueError, match="origin differs"):
            install(plan)
        assert setup.unpopulated_clone(target)
        plan["sources"][0]["url"] = str(origin)
        extra = target / "user-file.txt"
        extra.write_text("keep me\n")
        with pytest.raises(ValueError, match="unrecognized changes"):
            install(plan)
        assert extra.read_text() == "keep me\n" and not (target / "pyproject.toml").exists()
        extra.unlink()
        # Git diff may create an empty index while inspecting this synthetic invalid clone.
        index = target / ".git/index"
        if index.exists():
            index.unlink()
    assert install(plan)["installed"]
    assert (target / "pyproject.toml").read_bytes() == (origin / "pyproject.toml").read_bytes()
    # A later user's modification must still prevent resuming, without overwriting it.
    (target / "pyproject.toml").write_text("user edit\n")
    with pytest.raises(ValueError, match="unrecognized changes"):
        install(plan)
    assert (target / "pyproject.toml").read_text() == "user edit\n"
    plan["sources"][0]["patch"] = "fixture.patch"
    files = {"pyproject.toml": {"sha256": hashlib.sha256(b"user edit\n").hexdigest(), "mode": "100644"}}
    monkeypatch.setattr(setup, "patch_files", lambda source: files)
    assert install(plan)["installed"]
    assert (target / "pyproject.toml").read_text() == "user edit\n"


def test_first_party_build_snapshot_preserves_source_and_modified_copies(tmp_path):
    root, prefix = tmp_path / "checkout", tmp_path / "isolated"
    source = root / "benchmark"
    (source / "src/pkg").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (source / "src/pkg/__init__.py").write_text("value = 1\n")
    snapshot = project_snapshot(root, prefix, "benchmark")
    stage_project(snapshot)
    assert sorted(p.name for p in source.iterdir()) == ["pyproject.toml", "src"]
    copied = Path(snapshot["target"]) / "src/pkg/__init__.py"
    copied.write_text("user edit\n")
    with pytest.raises(ValueError, match="overwrite"):
        stage_project(snapshot)
    assert copied.read_text() == "user edit\n"
    assert (source / "src/pkg/__init__.py").read_text() == "value = 1\n"


def test_all_profile_plans_are_pinned_and_isolated(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for profile in load_lock()["profiles"]:
        plan = install_plan(profile, tmp_path / "fresh", root)
        assert plan["python"].startswith(str(tmp_path / "fresh/envs"))
        assert not (tmp_path / "fresh").exists()
        assert all(len(source["commit"]) == 40 for source in plan["sources"])
        # setup.py-only sources must not enter setuptools' legacy develop path:
        # it starts a nested pip invocation that drops --no-build-isolation.
        for command in plan["commands"]:
            if command[1:4] == ["-m", "pip", "install"] and "-e" in command:
                assert "--use-pep517" in command and "--no-build-isolation" in command


def test_setup_py_only_editable_build_stays_offline(tmp_path):
    """Exercise the real pip backend without installing into its interpreter's env."""
    binary = os.environ.get("PIPER_INSTALLER_TEST_PYTHON", sys.executable)
    source, prefix = tmp_path / "source", tmp_path / "install"
    source.mkdir()
    name = "piper_installer_fixture_" + hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]
    (source / (name + ".py")).write_text("value = 731\n")
    (source / "setup.py").write_text(
        f"from setuptools import setup\nsetup(name={name!r}, version='0.0.0', py_modules=[{name!r}])\n")
    assert not (source / "pyproject.toml").exists()
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PIP_NO_INDEX="1", PIP_NO_CACHE_DIR="1", PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK="1", TMPDIR=str(tmp_path))
    for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_ROOT"):
        env.pop(key, None)
    command = [binary, "-I", "-B", "-m", "pip", "install", "--prefix", str(prefix),
               "--no-index", "--no-deps", "--no-build-isolation", "--use-pep517", "-e", str(source)]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    installed = next(prefix.glob("lib/python*/site-packages"))
    check = subprocess.run([binary, "-I", "-B", "-c",
        "import importlib,site,sys;site.addsitedir(sys.argv[1]);m=importlib.import_module(sys.argv[2]);"
        "assert m.value==731;print(m.__file__)", str(installed), name],
        env=env, capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stdout + check.stderr
    assert Path(check.stdout.strip()).parent == source
    # The interpreter's normal environment must not contain this test distribution.
    absent = subprocess.run([binary, "-I", "-B", "-c",
        "import importlib.util,sys;assert importlib.util.find_spec(sys.argv[1]) is None", name],
        env=env, capture_output=True, text=True, timeout=30)
    assert absent.returncode == 0, absent.stdout + absent.stderr


def test_writer_build_tools_are_installed_before_source_distributions(tmp_path):
    root = Path(__file__).resolve().parents[2]
    plan = install_plan("writer", tmp_path / "fresh", root)
    commands = plan["commands"]
    build_tools = next(i for i, cmd in enumerate(commands) if "cmake==4.1.3" in cmd)
    requirements = next(i for i, cmd in enumerate(commands) if "-r" in cmd)
    assert build_tools < requirements
    assert "--no-deps" in commands[build_tools]


@pytest.mark.parametrize("profile,deepspeed", [("policy-ewm", "0.18.9"), ("policy-fastwam", "0.18.7")])
def test_policy_build_has_complete_torch_dependencies_before_deepspeed(tmp_path, profile, deepspeed):
    from piper_bench.deployment.config import resource
    from piper_bench.deployment.setup import locked_requirements
    plan = install_plan(profile, tmp_path / "fresh", Path(__file__).resolve().parents[2])
    commands = plan["commands"]
    wheel_index = next(i for i, command in enumerate(commands) if "--only-binary=:all:" in command)
    import_index = next(i for i, command in enumerate(commands) if len(command) > 2
                        and command[1] == "-c" and "import torch" in command[2])
    source_index = next(i for i, command in enumerate(commands) if "--no-binary=:all:" in command)
    assert wheel_index < import_index < source_index
    wheels = {arg for arg in commands[wheel_index] if "==" in arg}
    sources = {arg for arg in commands[source_index] if "==" in arg}
    assert sources == {"antlr4-python3-runtime==4.9.3", "deepspeed==" + deepspeed}
    assert not wheels & sources
    assert wheels | sources == set(locked_requirements(resource(profile + ".requirements.txt")))
    assert {pin.split("==")[0] for pin in wheels} >= {
        "torch", "typing_extensions", "filelock", "sympy", "networkx", "Jinja2", "fsspec"}
    assert "--no-deps" in commands[wheel_index]
    assert "--no-deps" in commands[source_index] and "--no-build-isolation" in commands[source_index]


def test_installer_build_commands_use_target_environment_tools(tmp_path, monkeypatch):
    from piper_bench.deployment import setup
    prefix = tmp_path / "fresh"
    binary = prefix / "envs/writer/bin/python"
    output = tmp_path / "tool-used.txt"
    host_bin = tmp_path / "host/bin"
    host_bin.mkdir(parents=True)
    host_tool = host_bin / "cmake"
    host_tool.write_text("#!/bin/sh\nexit 99\n")
    host_tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(host_bin))
    original_run = setup.subprocess.run

    def run(command, **kwargs):
        if command == [str(binary), "-m", "pip", "check"]:
            return SimpleNamespace(returncode=0, stdout="No broken requirements found.\n", stderr="")
        # Model the tool installed by the preceding bootstrap command.
        binary.parent.mkdir(parents=True, exist_ok=True)
        tool = binary.parent / "cmake"
        tool.write_text('#!/bin/sh\nprintf target > "$1"\n')
        tool.chmod(0o755)
        return original_run(command, **kwargs)

    monkeypatch.setattr(setup.subprocess, "run", run)
    plan = {"eula_required": False, "prefix": str(prefix), "lock_release": "fixture",
            "profile": "writer", "python": str(binary), "requirements_sha256": "fixture",
            "sources": [], "first_party_snapshots": [], "commands": [["cmake", str(output)]]}
    assert install(plan)["installed"]
    assert output.read_text() == "target"


@pytest.mark.parametrize("pins", ["anyio==4.14.2\nanyio==4.15.1\n",
                                  "HuggingFace_Hub==1.32.0\nhuggingface-hub==1.32.0\n"])
def test_install_plan_rejects_duplicate_distributions_before_creating_prefix(tmp_path, monkeypatch, pins):
    from piper_bench.deployment import setup
    requirements = tmp_path / "conflicting.requirements.txt"
    requirements.write_text(pins)
    monkeypatch.setattr(setup, "resource", lambda _: requirements)
    lock = {"profiles": {"writer": {"requirements": requirements.name,
             "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest()}}}
    prefix = tmp_path / "uncreated"
    with pytest.raises(ValueError, match="Duplicate locked distribution"):
        install_plan("writer", prefix, tmp_path, lock=lock)
    assert not prefix.exists()


@pytest.mark.parametrize("pin", ["numpy>=2.2\n", "numpy==2.*\n"])
def test_lock_rejects_unpinned_versions(tmp_path, pin):
    from piper_bench.deployment.setup import locked_requirements
    requirements = tmp_path / "unpinned.requirements.txt"
    requirements.write_text(pin)
    with pytest.raises(ValueError, match="exact package version"):
        locked_requirements(requirements)


def test_metadata_only_does_not_claim_runtime_acceptance(tmp_path, monkeypatch):
    import piper_bench.deployment.doctor as doctor
    monkeypatch.setattr(doctor, "_probe", lambda *a: {"status": "passed"})
    monkeypatch.setattr(doctor, "verify", lambda *a: {"passed": True})
    cfg = {"prefix": str(tmp_path), "assets_dir": str(tmp_path / "assets")}
    report = diagnose(python="python3", output=tmp_path / "report.json", config=cfg)
    assert report["passed"] is True
    assert report["deployment_accepted"] is False
    assert all(report[name]["status"] == "not_run" for name in ("robot_rgb_wrist", "oracle", "events"))


def test_runtime_layers_distinguish_render_from_task_and_reset(tmp_path):
    for i in range(2):
        directory = tmp_path / f"oracle_{i:03d}"
        directory.mkdir()
        (directory / "deployment_audit.json").write_text(json.dumps({"passed": True, "checks": {"rgb": True},
                                                                    "events": {"passed": i == 0}}))
    (tmp_path / "complete.json").write_text(json.dumps({"passed": False, "results": [{}, {}]}))
    rgb, oracle, events = _runtime_results(tmp_path, True)
    assert rgb["status"] == "passed"
    assert oracle["status"] == events["status"] == "failed"
