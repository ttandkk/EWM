"""Install a pinned profile into an isolated, project-managed prefix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys

from .config import home, load_lock, resource, sha256, write_json
from .checks import classify_pip_check


def project_root(explicit=None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        root = next((p for p in Path(__file__).resolve().parents if (p / "OpenWAM").is_dir()), None)
    if root is None or not (root / "benchmark/pyproject.toml").is_file():
        raise ValueError("A source checkout is required for setup; pass --project-root /path/to/EWM")
    return root


def _run(command, **kwargs):
    print("[setup] " + " ".join(str(x) for x in command), flush=True)
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def patch_files(source):
    """Reuse migration/fastwam's exact resulting file state, including new files."""
    path = resource(source["patch_state"])
    if sha256(path) != source["patch_state_sha256"]:
        raise ValueError("Changed post-patch source manifest")
    state = json.loads(path.read_text())
    if (state["base_commit"] != source["commit"] or state["upstream_url"] != source["url"] or
            state["patch_sha256"] != source["patch_sha256"]):
        raise ValueError("Patch state provenance differs from source lock")
    for name, item in state["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError(f"Unsafe patch state path: {name}")
        if item is not None and (item["mode"] not in ("100644", "100755") or
                                 not re.fullmatch("[0-9a-f]{64}", item["sha256"])):
            raise ValueError(f"Invalid patch state: {name}")
    return state["files"]


def matches_patch(directory, files, status):
    paths = set()
    for record in filter(None, status.split("\0")):
        if record[:2] not in (" M", " D", "??"):
            return False
        paths.add(record[3:])
    if paths != set(files):
        return False
    for name, item in files.items():
        path = directory / name
        if item is None:
            if path.exists():
                return False
        elif (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or
              sha256(path) != item["sha256"] or
              ("100755" if path.stat().st_mode & stat.S_IXUSR else "100644") != item["mode"]):
            return False
    return True


def unpopulated_clone(directory):
    """Recognize an untouched --no-checkout clone, including an interrupted setup."""
    directory = Path(directory)
    return ((directory / ".git").is_dir() and not (directory / ".git/index").exists()
            and {path.name for path in directory.iterdir()} == {".git"})


def project_snapshot(root, prefix, name):
    """Describe only first-party build inputs, excluding datasets and outputs."""
    directory = root / name
    inputs = ["pyproject.toml", "src"] if name in ("benchmark", "policies") else ["pyproject.toml", "openwam", "configs"]
    inputs += [p.name for p in directory.iterdir() if p.is_file() and p.name.startswith(("LICENSE", "NOTICE", "README", "THIRD_PARTY"))]
    files = {}
    for relative in inputs:
        start = directory / relative
        if not start.exists():
            raise FileNotFoundError(f"Missing build input: {start}")
        for path in sorted(start.rglob("*") if start.is_dir() else [start]):
            relative_path = path.relative_to(directory)
            if any(part in ("__pycache__", "build", "dist", ".git") or part.endswith(".egg-info") for part in relative_path.parts):
                continue
            if path.is_file() and path.suffix != ".pyc":
                if path.is_symlink():
                    raise ValueError(f"Build input symlinks require an explicit source snapshot: {path}")
                files[relative_path.as_posix()] = sha256(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"project": name, "source": str(directory), "sha256": digest, "files": files,
            "target": str(prefix / "first-party" / (name + "-" + digest[:16]))}


def stage_project(snapshot):
    source, target = Path(snapshot["source"]), Path(snapshot["target"])
    if not target.resolve().is_relative_to(target.parents[1].resolve()):
        raise ValueError(f"Build snapshot escapes installation prefix: {target}")
    for relative, expected in snapshot["files"].items():
        original, staged = source / relative, target / relative
        if not staged.resolve().is_relative_to(target.resolve()):
            raise ValueError(f"Build input escapes snapshot directory: {staged}")
        if sha256(original) != expected:
            raise ValueError(f"Source changed during installation planning: {original}")
        if staged.exists():
            if not staged.is_file() or staged.is_symlink() or sha256(staged) != expected:
                raise ValueError(f"Refusing to overwrite modified build input: {staged}")
        else:
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, staged)


def locked_requirements(path):
    """Read one exact pin per normalized distribution before any installation."""
    entries, names = [], {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9._+!-]*)", line)
        if not match:
            raise ValueError(f"Expected an exact package version at {path}:{number}: {line}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in names:
            raise ValueError(f"Duplicate locked distribution {name}: {names[name]} and {line}")
        names[name] = line
        entries.append(line)
    return entries


def install_plan(profile, prefix, root, *, python=None, lock=None) -> dict:
    lock = lock or load_lock()
    if profile not in lock["profiles"]:
        raise ValueError(f"Unknown installation profile: {profile}")
    spec = lock["profiles"][profile]
    prefix, root = Path(prefix).expanduser().resolve(), Path(root).resolve()
    requirements = resource(spec["requirements"])
    if sha256(requirements) != spec["requirements_sha256"]:
        raise ValueError(f"Changed dependency lock: {requirements}")
    requirements_lines = locked_requirements(requirements)
    interpreter = python or f"python{spec['python']}"
    env = prefix / "envs" / profile
    binary = env / "bin/python"
    source_map = {s["name"]: s for s in lock["sources"]}
    sources = [source_map[name] for name in spec["sources"]]
    projects = [project_snapshot(root, prefix, name) for name in
                ([spec["local_project"]] if spec.get("local_project") else []) + ["benchmark"] +
                (["policies"] if profile.startswith("policy-") else [])]
    for source in sources:
        if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
            raise ValueError(f"Unpinned source revision: {source['name']}")
        if source.get("patch") and sha256(resource(source["patch"])) != source["patch_sha256"]:
            raise ValueError(f"Changed source patch: {source['name']}")
        if source.get("patch"):
            patch_files(source)
    indexes = [arg for index in lock["indexes"][1:] for arg in ("--extra-index-url", index)]
    # pip builds source distributions before installing the remaining wheels.
    # Build tools must therefore be available before egl_probe and similar builds.
    bootstrap = [s for s in requirements_lines if s.lower().startswith(
        ("torch==", "torchvision==", "nvidia-", "triton==", "cmake==", "ninja=="))]
    commands = [[interpreter, "-m", "venv", str(env)],
                [str(binary), "-m", "pip", "install", "pip==25.2", "setuptools==80.9.0", "wheel==0.45.1"]]
    if profile.startswith("policy-"):
        # DeepSpeed reads Torch during metadata/build generation. Installing only
        # Torch's wheel is insufficient: its import also needs the locked Python
        # dependencies. Install the audited wheels first, then the two sdists.
        source_pins = {"antlr4-python3-runtime==4.9.3",
                       "deepspeed==" + {"policy-ewm": "0.18.9", "policy-fastwam": "0.18.7"}[profile]}
        if not source_pins.issubset(requirements_lines):
            raise ValueError("Policy source-distribution versions changed; re-audit the build prerequisites")
        wheels = [pin for pin in requirements_lines if pin not in source_pins]
        torch_version = next(pin.split("==", 1)[1] for pin in wheels if pin.startswith("torch=="))
        commands.append([str(binary), "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
                         *indexes, *wheels])
        commands.append([str(binary), "-c", "import torch; assert torch.__version__ == " + repr(torch_version)
                         + "; print('Policy build Torch import:', torch.__version__)"])
        commands.append([str(binary), "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                         "--use-pep517", "--no-binary=:all:", *indexes, *sorted(source_pins)])
    else:
        if bootstrap:
            commands.append([str(binary), "-m", "pip", "install", "--no-deps", *indexes, *bootstrap])
        commands.append([str(binary), "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                         *indexes, "-r", str(requirements)])
    for source in sources:
        directory = prefix / "sources" / source["name"]
        commands.extend([["git", "clone", "--filter=blob:none", "--no-checkout", source["url"], str(directory)],
                         ["git", "-C", str(directory), "checkout", "--detach", source["commit"]]])
        if source.get("patch"):
            commands.append(["git", "-C", str(directory), "apply", str(resource(source["patch"]))])
        for relative in source["install"]:
            commands.append([str(binary), "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                             "--use-pep517", "-e", str(directory / relative)])
    for project in projects:
        command = [str(binary), "-m", "pip", "install", "--no-deps", "--no-build-isolation"]
        if project["project"] == "OpenWAM":
            command.extend(["--use-pep517", "-e"])
        commands.append(command + [project["target"]])
    for command in commands:
        if command[1:4] == ["-m", "pip", "install"]:
            command.insert(4, "--no-compile")
    return {"profile": profile, "prefix": str(prefix), "python_version": spec["python"],
            "python": str(binary), "requirements_sha256": spec["requirements_sha256"],
            "commands": commands, "sources": sources, "first_party_snapshots": projects, "lock_release": lock["release"],
            "eula_required": profile == "simulation", "system_requirements": ["git", "C/C++ compiler", "ffmpeg", "OpenGL/Vulkan system libraries"]}


def install(plan: dict, *, accept_eula=False) -> dict:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("This release is validated for Linux x86_64 only")
    if plan["eula_required"] and not accept_eula:
        raise ValueError("Isaac Sim requires NVIDIA license acceptance. Read the NVIDIA terms and rerun with --accept-eula.")
    prefix = Path(plan["prefix"])
    marker = prefix / ".piper-managed.json"
    if prefix.exists() and any(prefix.iterdir()) and not marker.is_file():
        raise ValueError(f"Refusing to modify an existing unmanaged prefix: {prefix}; choose a new --prefix")
    prefix.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write_json(marker, {"schema_version": 1, "managed_by": "piper-bench", "release": plan["lock_release"]})
    elif json.loads(marker.read_text()) != {"schema_version": 1, "managed_by": "piper-bench", "release": plan["lock_release"]}:
        raise ValueError("Prefix ownership/release differs; choose a new installation prefix")
    (prefix / "sources").mkdir(exist_ok=True)
    cfg_path = prefix / "deployment.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {
        "schema_version": 1, "prefix": str(prefix), "profiles": {}, "sources": {}, "assets_dir": str(prefix / "assets")}
    if plan["eula_required"] and accept_eula:
        cfg.setdefault("licenses", {})["nvidia_isaac_sim"] = {
            "accepted": True, "version": load_lock()["expected_simulation_versions"]["isaacsim"]}
    for snapshot in plan["first_party_snapshots"]:
        stage_project(snapshot)
    temporary = prefix / "tmp"
    temporary.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", GIT_LFS_SKIP_SMUDGE="1",
               PIP_CACHE_DIR=str(prefix / "cache/pip"), XDG_CACHE_HOME=str(prefix / "cache"),
               TMPDIR=str(temporary), PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1",
               PIP_CONFIG_FILE=os.devnull, PIP_INDEX_URL=load_lock()["indexes"][0])
    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_ROOT", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS",
                "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["PATH"] = os.pathsep.join((str(Path(plan["python"]).parent), env.get("PATH", os.defpath)))
    if accept_eula:
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    # No shell evaluation: paths containing spaces remain single arguments.
    for command in plan["commands"]:
        if command[1:3] == ["-m", "venv"] and Path(plan["python"]).exists():
            continue
        if command[:2] == ["git", "clone"]:
            target = Path(command[-1])
            if target.exists():
                source = next(s for s in plan["sources"] if target.name == s["name"])
                actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
                if actual != source["commit"]:
                    raise ValueError(f"Existing source revision differs: {target}; refusing to reset it")
                if unpopulated_clone(target):
                    remote = subprocess.check_output(
                        ["git", "-C", str(target), "remote", "get-url", "origin"], text=True).strip()
                    if remote != source["url"]:
                        raise ValueError(f"Unpopulated source origin differs: {target}; refusing to check it out")
                    recognized = True
                elif source.get("patch"):
                    dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain=v1", "-z", "--untracked-files=all"], text=True)
                    recognized = not dirty or matches_patch(target, patch_files(source), dirty)
                else:
                    dirty = subprocess.check_output(["git", "-C", str(target), "diff", "--name-only", "HEAD"], text=True)
                    recognized = not dirty
                if not recognized:
                    raise ValueError(f"Existing source has unrecognized changes: {target}; refusing to modify it")
                continue
        if command[0] == "git" and "checkout" in command:
            target = Path(command[2])
            actual = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], text=True, capture_output=True)
            if (actual.returncode == 0 and actual.stdout.strip() == command[-1]
                    and not unpopulated_clone(target)):
                continue
        if command[0] == "git" and "apply" in command:
            reverse = subprocess.run(command[:3] + ["apply", "--reverse", "--check", command[-1]], capture_output=True)
            if reverse.returncode == 0:
                continue
        try:
            _run(command, env=env)
        except subprocess.CalledProcessError as exc:
            write_json(prefix / f"setup-{plan['profile']}.json", {
                "installed": False, "runtime_verified": False, "profile": plan["profile"],
                "failed_command": command, "returncode": exc.returncode})
            raise
    check = subprocess.run([plan["python"], "-m", "pip", "check"], text=True, capture_output=True, env=env)
    metadata = classify_pip_check(check.stdout + check.stderr, check.returncode, plan["profile"])
    cfg["profiles"][plan["profile"]] = {"python": plan["python"], "requirements_sha256": plan["requirements_sha256"],
        "installed": metadata["passed"], "runtime_verified": False, "pip_check": metadata,
        "first_party": [{k: p[k] for k in ("project", "target", "sha256")} for p in plan["first_party_snapshots"]]}
    for source in plan["sources"]:
        cfg["sources"][source["name"]] = str(prefix / "sources" / source["name"])
    write_json(cfg_path, cfg)
    result = {"installed": metadata["passed"], "profile": plan["profile"], "config": str(cfg_path), "runtime_verified": False,
              "pip_check": cfg["profiles"][plan["profile"]]["pip_check"]}
    write_json(prefix / f"setup-{plan['profile']}.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("simulation", "policy-ewm", "policy-fastwam", "writer"), required=True)
    parser.add_argument("--prefix", type=Path, default=home())
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--python", help="Interpreter used to create this profile's isolated environment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    args = parser.parse_args(argv)
    plan = install_plan(args.profile, args.prefix, project_root(args.project_root), python=args.python)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    interpreter = shutil.which(plan["commands"][0][0])
    if not interpreter:
        parser.error(f"Install Python {plan['python_version']} or pass --python /path/to/python")
    version = subprocess.check_output([interpreter, "-c", "import sys; print('%d.%d'%sys.version_info[:2])"], text=True).strip()
    if version != plan["python_version"]:
        parser.error(f"Expected Python {plan['python_version']}, got {version}")
    result = install(plan, accept_eula=args.accept_eula)
    print(json.dumps(result, indent=2))
    return 0 if result["installed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
