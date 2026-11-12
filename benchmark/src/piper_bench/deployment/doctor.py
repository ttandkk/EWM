"""Layered diagnostics with real simulator evidence and explicit not-run states."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import unquote, urlparse

from piper_bench.assets.resolver import load_manifest, verify
from piper_bench.backend.video import find_ffmpeg
from .checks import classify_pip_check
from .config import load_config, load_lock, resource, runtime_environment, sha256, simulation_python, write_json


def _probe(python, env, lock):
    expected = dict(lock["expected_simulation_versions"])
    for line in resource(lock["profiles"]["simulation"]["requirements"]).read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            expected[name] = version
    script = '''import importlib.metadata as m,json,sys
names=json.loads(sys.argv[1]); versions={}; locations={}; missing=[]
for name in names:
 try:
  d=m.distribution(name); versions[name]=d.version
  locations[name]={'metadata':str(d.locate_file('')), 'direct_url':json.loads(d.read_text('direct_url.json') or '{}')}
 except m.PackageNotFoundError: missing.append(name)
print(json.dumps({'python':sys.version.split()[0], 'executable':sys.executable,'versions':versions,'locations':locations,'missing':missing}))'''
    result = subprocess.run([str(python), "-c", script, json.dumps(list(expected))],
                            text=True, capture_output=True, env=env, timeout=120)
    if result.returncode:
        return {"status": "failed", "error": result.stdout + result.stderr, "returncode": result.returncode}
    data = json.loads(result.stdout)
    data["version_mismatches"] = {name: {"expected": version, "actual": data["versions"].get(name)}
                                  for name, version in expected.items()
                                  if data["versions"].get(name) != version}
    check = subprocess.run([str(python), "-m", "pip", "check"], text=True, capture_output=True, env=env, timeout=120)
    data["pip_check"] = classify_pip_check(check.stdout + check.stderr, check.returncode, "simulation", lock)
    compat = Path(__file__).resolve().parents[1] / lock["compatibility"]["path"]
    data["compatibility_patch_matches"] = sha256(compat) == lock["compatibility"]["sha256"]
    recipe = lock["asset_derivation"]
    recipe_file = Path(__file__).resolve().parents[1] / recipe["path"]
    data["asset_derivation_recipe_matches"] = sha256(recipe_file) == recipe["sha256"]
    data["sources"] = {}
    for distribution, name in (("isaaclab", "runtime_isaaclab"), ("dvs_gen", "event_plugin"), ("agx_teleop", "agilex")):
        source = next(s for s in lock["sources"] if s["name"] == name)
        url = data["locations"].get(distribution, {}).get("direct_url", {}).get("url", "")
        entry = {"expected_commit": source["commit"], "direct_url": url, "passed": False}
        if urlparse(url).scheme == "file":
            path = unquote(urlparse(url).path)
            actual = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
            dirty = subprocess.run(["git", "-C", path, "diff", "--name-only", "HEAD"], capture_output=True, text=True)
            entry.update(commit=actual.stdout.strip(), tracked_changes=dirty.stdout.splitlines())
            entry["passed"] = actual.returncode == dirty.returncode == 0 and entry["commit"] == source["commit"] and not entry["tracked_changes"]
        data["sources"][name] = entry
    passed = (data["python"].startswith(lock["profiles"]["simulation"]["python"] + ".") and
              not data["version_mismatches"] and data["pip_check"]["passed"] and data["compatibility_patch_matches"] and
              data["asset_derivation_recipe_matches"] and
              all(s["passed"] for s in data["sources"].values()))
    data["status"] = ("passed_with_exceptions" if data["pip_check"]["known_exceptions"] else "passed") if passed else "failed"
    data["scope"] = "Package metadata and compatibility hash only; no renderer or EVIS execution in this layer."
    return data


def preview_command(python, output, *, task="tennis_rolling", events=False, trials=2, max_steps=600, audit=True):
    command = [str(python), "-B", "-m", f"piper_bench.tasks.{task}.preview", "--output-dir", str(output),
               "--mode", "oracle", "--trials", str(trials), "--max-steps", str(max_steps),
               "--headless", "--enable_cameras", "--device", "cpu"]
    if audit:
        command.append("--deployment-audit")
    if events:
        command.append("--events")
    return command


def _runtime_results(directory, requested_events):
    trials = sorted(directory.glob("oracle_*/deployment_audit.json"))
    results = [json.loads(path.read_text()) for path in trials]
    complete_path = directory / "complete.json"
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else {}
    render = {"status": "passed" if len(results) == 2 and all(r["passed"] for r in results) else "failed",
              "evidence": [str(path) for path in trials], "trials": len(results),
              "checks": [r["checks"] for r in results],
              "videos": [str(path.parent / "preview.mp4") for path in trials]}
    oracle = {"status": "passed" if complete.get("passed") is True and len(complete.get("results", [])) == 2 else "failed",
              "evidence": str(complete_path), "results": complete.get("results", [])}
    events = {"status": "not_run", "reason": "Run doctor --events to execute EVIS."}
    if requested_events:
        event_results = [r.get("events", {}) for r in results]
        events = {"status": "passed" if len(event_results) == 2 and all(r.get("passed") is True for r in event_results) else "failed",
                  "trials": event_results, "reset_scope": "Two episodes in one simulator process; both initial event frames checked empty.",
                  "videos": [str(path.parent / "preview_events.mp4") for path in trials]}
    return render, oracle, events


def diagnose(*, python=None, output=Path("deployment_report.json"), render=False, events=False,
             asset_root=None, timeout=1800, max_steps=600, config=None):
    cfg, lock = config or load_config(), load_lock()
    output = Path(output).expanduser().resolve()
    env = runtime_environment(cfg)
    from .process import package_environment
    root = Path(asset_root or env.get("PIPER_ASSET_ROOT", cfg["assets_dir"])).expanduser().resolve()
    env["PIPER_ASSET_ROOT"] = str(root)
    report = {"schema_version": 1, "time_utc": datetime.now(timezone.utc).isoformat(),
              "lock_release": lock["release"], "requested": {"render": render or events, "events": events},
              "simulation_lock_sha256": sha256(resource("simulation.lock.json")),
              "assets_lock_sha256": sha256(resource("assets.lock.json")),
              "benchmark_source": str(Path(__file__).resolve().parents[1]),
              "environment": {"status": "not_run"}, "assets": {},
              "robot_rgb_wrist": {"status": "not_run", "reason": "Run doctor --render to execute the simulator."},
              "oracle": {"status": "not_run", "reason": "Run doctor --render to execute the oracle."},
              "events": {"status": "not_run", "reason": "Run doctor --events to execute EVIS."}}
    try:
        python = Path(python).expanduser().absolute() if python else simulation_python(cfg)
        with package_environment(env) as child_env:
            report["environment"] = _probe(python, child_env, lock)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        report["environment"] = {"status": "failed", "error": str(exc)}
    assets = verify(root, load_manifest())
    report["assets"] = dict(assets, status="passed" if assets["passed"] else "failed")
    if render or events:
        smi = shutil.which("nvidia-smi")
        gpu = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             text=True, capture_output=True, timeout=30) if smi else None
        report["gpu"] = {"query_output": (gpu.stdout + gpu.stderr) if gpu else "nvidia-smi unavailable",
                         "query_returncode": gpu.returncode if gpu else None}
        reason = None
        if not gpu or gpu.returncode or not gpu.stdout.strip():
            reason = "No visible NVIDIA GPU; rendering and EVIS were not run."
        elif report["environment"]["status"] == "failed" or not assets["passed"]:
            reason = "Resolve environment/asset failures before running the simulator."
        else:
            try:
                env["PIPER_FFMPEG"] = find_ffmpeg()
            except (RuntimeError, FileNotFoundError) as exc:
                reason = str(exc)
        if reason:
            for layer in ("robot_rgb_wrist", "oracle", "events") if events else ("robot_rgb_wrist", "oracle"):
                report[layer] = {"status": "not_run", "reason": reason}
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            evidence = Path(tempfile.mkdtemp(prefix="deployment-smoke-", dir=output.parent))
            run_dir = evidence / "simulation"
            command = preview_command(python, run_dir, events=events, max_steps=max_steps)
            report["runtime"] = {"command": command, "log": str(evidence / "simulation.log"), "evidence": str(run_dir)}
            with (evidence / "simulation.log").open("w") as log:
                try:
                    with package_environment(env) as child_env:
                        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=child_env, timeout=timeout)
                    report["runtime"]["returncode"] = child.returncode
                except subprocess.TimeoutExpired:
                    report["runtime"]["timeout_seconds"] = timeout
            try:
                report["robot_rgb_wrist"], report["oracle"], report["events"] = _runtime_results(run_dir, events)
            except (OSError, ValueError, KeyError) as exc:
                for layer in ("robot_rgb_wrist", "oracle", "events") if events else ("robot_rgb_wrist", "oracle"):
                    report[layer] = {"status": "failed", "error": str(exc)}
            if report["runtime"].get("returncode") != 0:
                report["oracle"]["status"] = "failed"
    required = ["environment", "assets"] + (["robot_rgb_wrist", "oracle"] if render or events else []) + (["events"] if events else [])
    report["passed"] = all(report[layer]["status"].startswith("passed") for layer in required)
    report["deployment_accepted"] = all(report[layer]["status"].startswith("passed") for layer in
                                         ("environment", "assets", "robot_rgb_wrist", "oracle", "events"))
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("deployment_report.json"))
    parser.add_argument("--python", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-steps", type=int, default=600)
    args = parser.parse_args(argv)
    if args.timeout < 1 or args.max_steps < 1:
        parser.error("--timeout and --max-steps must be positive")
    report = diagnose(**vars(args))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
