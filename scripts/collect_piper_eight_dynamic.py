"""Serial eight-task collection with per-task 70% acceptance and export checks."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(os.environ.get("NEW_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
_frozen_package = Path(__file__).resolve().parent / "benchmark/src"
sys.path.insert(0, str(_frozen_package if (_frozen_package / "piper_bench").is_dir()
                       else ROOT / "benchmark/src"))
from piper_bench.collection.sampling import TASKS, VERSION as PROTOCOL, DEFAULT_PROTOCOLS
from piper_bench.collection.freeze import freeze_collection_source, frozen_pythonpath, hash_snapshot

TASK_PROTOCOLS = DEFAULT_PROTOCOLS.copy()
ACCEPTANCE_RATE = .7


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, help="Logs, frozen source and resume metadata")
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--acceptance-trials", type=int, default=20)
    p.add_argument("--max-attempts", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--pathtracing-spp", type=int, default=64)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Freeze source/assets and settings without a GPU")
    p.add_argument("--preflight-all", action="store_true",
                   help="Preflight all eight tasks, then collect only tasks that passed")
    p.add_argument("--dry-run", action="store_true", help="Print commands; no writes or GPU work")
    return p


def settings(args):
    value = dict(protocol=PROTOCOL, task_protocols=TASK_PROTOCOLS, tasks=list(TASKS), episodes=args.episodes, save_raw_events=True,
                acceptance_trials=args.acceptance_trials, acceptance_rate=ACCEPTANCE_RATE,
                max_attempts=args.max_attempts, seed=args.seed, pathtracing_spp=args.pathtracing_spp,
                dataset_root=str(args.dataset_root.resolve()))
    if args.preflight_all:
        value["preflight_all"] = True
    return value


def commands(args, task, resume=False):
    source = args.run_dir / "source" / "scripts"
    output = args.dataset_root / ("piper_" + task)
    collect = [
        os.environ.get("SIMPY", str(ROOT / "isaacsim601/env/bin/python")),
        str(source / "collect_piper_tasks_evis.py"), "--task", task,
        "--collection-protocol", TASK_PROTOCOLS[task], "--output", str(output),
        "--audit-dir", str(args.run_dir / "audit" / task),
        "--episodes", str(args.episodes), "--max-attempts", str(args.max_attempts),
        "--acceptance-trials", str(args.acceptance_trials),
        "--acceptance-rate", str(ACCEPTANCE_RATE), "--seed", str(args.seed),
        "--pathtracing-spp", str(args.pathtracing_spp),
        "--save-raw-events",
        "--writer-python", os.environ.get("LRPY", str(ROOT / "envs/lerobot/bin/python")),
        "--device", "cpu", "--headless", "--enable_cameras",
        "--kit_args=--/plugins/carb.tasking.plugin/threadCount=4",
    ]
    if resume:
        collect.append("--resume")
    validate = [collect[collect.index("--writer-python") + 1],
                str(source / "validate_piper_tasks_lerobot.py"), str(output),
                "--episodes", str(args.episodes)]
    return collect, validate


def verify_source(run):
    for name, expected in read(run / "source_sha256.json").items():
        if hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source snapshot changed: {name}")


def prepare(args):
    run = args.run_dir
    if args.resume:
        if read(run / "launch.json") != settings(args):
            raise ValueError("resume requires the original settings and dataset root")
        verify_source(run)
        return
    existing = [args.dataset_root / ("piper_" + t) for t in TASKS
                if (args.dataset_root / ("piper_" + t)).exists()]
    if existing:
        raise FileExistsError(f"datasets already exist; resume the original run or use a new root: {existing}")
    run.mkdir(parents=True, exist_ok=False)
    freeze_collection_source(ROOT, run / "source")
    for path in (Path(__file__), ROOT / "scripts/collect_piper_eight_dynamic.sh",
                 ROOT / "docs/proto.md", ROOT / "docs/piper_eight_task_collection_eval_protocol_v1.md"):
        shutil.copy2(path, run / "source" / path.name)
    write(run / "source_sha256.json", hash_snapshot(run / "source"))
    (run / "logs").mkdir()
    (run / "audit").mkdir()
    write(run / "launch.json", settings(args))


def acceptance(args, task):
    path = args.run_dir / "audit" / task / "acceptance.json"
    if not path.exists():
        return dict(passed=False, trials=0, successes=0)
    report = read(path)
    results = report["results"]
    successes = sum(bool(r["success"]) for r in results)
    passed = (len(results) == args.acceptance_trials
              and successes >= math.ceil(args.acceptance_trials * ACCEPTANCE_RATE)
              and report["passed"])
    return dict(passed=bool(passed), trials=len(results), successes=successes)


def execute(command, log, env):
    # Append on resume so the original failure and interrupted run remain inspectable.
    with log.open("a") as stream:
        stream.write("\n" + shlex.join(command) + "\n")
        stream.flush()
        return subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT).returncode


def preflight_all(args, env):
    """Finish every unbiased preflight before selecting eligible tasks."""
    run = args.run_dir
    previous = read(run / "preflight.json").get("tasks", {}) if (run / "preflight.json").exists() else {}
    reports = {}
    for task in TASKS:
        verify_source(run)
        gate = acceptance(args, task)
        code = 0
        if not gate["passed"] or not previous.get(task, {}).get("passed"):
            resume = (run / "audit" / task / "collection_config.json").exists()
            command, _ = commands(args, task, resume)
            print(f"PREFLIGHT {task}: {args.acceptance_trials} trials", flush=True)
            code = execute(command + ["--acceptance-only"], run / "logs" / f"{task}.preflight.log", env)
            gate = acceptance(args, task)
        reports[task] = dict(**gate, exit_code=code)
        reports[task]["passed"] = bool(gate["passed"] and code == 0)
        write(run / "preflight.json", dict(complete=False, passed=False, tasks=reports))
    passed = all(value["passed"] for value in reports.values())
    write(run / "preflight.json", dict(complete=True, passed=passed, tasks=reports))
    print(f"PREFLIGHT SUMMARY: {sum(value['passed'] for value in reports.values())}/{len(TASKS)} "
          "tasks eligible for collection", flush=True)
    return passed


def run_tasks(args):
    run = args.run_dir
    summary_path = run / "summary.json"
    summary = read(summary_path) if summary_path.exists() else {}
    env = source_environment(run)
    if args.preflight_all:
        preflight_all(args, env)
        preflights = read(run / "preflight.json")["tasks"]
    failed = False
    for task in TASKS:
        verify_source(run)
        output = args.dataset_root / ("piper_" + task)
        if args.preflight_all and not preflights[task]["passed"]:
            summary[task] = dict(status="skipped_preflight", dataset=str(output),
                                 acceptance=preflights[task])
            write(summary_path, summary)
            gate = preflights[task]
            print(f"SKIP {task}: preflight {gate['successes']}/{gate['trials']}, "
                  f"exit={gate['exit_code']}", flush=True)
            continue
        resume = (args.resume or args.preflight_all) and (run / "audit" / task / "collection_config.json").exists()
        collect, validate = commands(args, task, resume)
        complete = (output / ".export_complete").exists()
        print(f"BEGIN {task}: acceptance >= {math.ceil(args.acceptance_trials * ACCEPTANCE_RATE)}"
              f"/{args.acceptance_trials}, collect {args.episodes}", flush=True)
        summary[task] = dict(status="running", dataset=str(output))
        write(summary_path, summary)
        # A completed task is validated again, without recollection.
        code = 0 if complete else execute(collect, run / "logs" / f"{task}.collect.log", env)
        gate = acceptance(args, task)
        result = dict(dataset=str(output), acceptance=gate, collection_exit=code)
        if code or not gate["passed"]:
            result["status"] = ("acceptance_failed" if gate["trials"] == args.acceptance_trials
                                and not gate["passed"] else "collection_failed")
        else:
            code = execute(validate, run / "logs" / f"{task}.validate.log", env)
            result["validation_exit"] = code
            marker = output / ".export_complete"
            result["status"] = ("complete" if code == 0 and marker.exists()
                                and read(marker).get("passed") else "validation_failed")
        summary[task] = result
        write(summary_path, summary)
        failed |= result["status"] != "complete"
        print(f"END {task}: {result['status']} "
              f"(acceptance {gate['successes']}/{gate['trials']})", flush=True)
    return int(failed)


def source_environment(run):
    return dict(os.environ, PYTHONPATH=frozen_pythonpath(run / "source"), PYTHONDONTWRITEBYTECODE="1")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (args.episodes < 2 or args.episodes % 2 or args.acceptance_trials < 2
            or args.acceptance_trials % 2 or args.max_attempts < args.episodes
            or args.seed < 0 or args.pathtracing_spp < 1):
        p.error("episodes/acceptance-trials must be positive even counts (container pairs); "
                "max-attempts >= episodes, seed >= 0, SPP >= 1")
    if args.resume and args.run_dir is None:
        p.error("--resume requires --run-dir")
    if args.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.run_dir = ROOT / "tmp/piper_eight_collection_runs" / stamp
    args.run_dir = args.run_dir.resolve()
    args.dataset_root = args.dataset_root.resolve()
    print(f"Run: {args.run_dir}\nDatasets: {args.dataset_root}", flush=True)
    if args.dry_run:
        if args.preflight_all:
            for task in TASKS:
                print(shlex.join(commands(args, task, args.resume)[0] + ["--acceptance-only"]))
        for task in TASKS:
            for cmd in commands(args, task, args.resume or args.preflight_all):
                print(shlex.join(cmd))
        return 0
    if args.prepare_only:
        prepare(args)
        return 0
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("run inside an allocated GPU job; --dry-run requires no allocation")
    prepare(args)
    with (args.run_dir / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_tasks(args)


if __name__ == "__main__":
    raise SystemExit(main())
