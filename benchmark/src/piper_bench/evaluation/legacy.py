"""Thin compatibility for former single-task evaluator imports and CLI calls.

There is one rollout engine: runtime.py / runner.py. Checkpoint configuration
stays in the model service; historical controller settings require a profile.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from . import runtime, runner
from .constants import EVENT_TIME_CONTRACT
from .recovery import load_completed_results, archive_interrupted_trial
from piper_bench.collection.conveyor_speed import uniform_speed_spec

SPEED_OOD_SCALES = {"none": 1., "slow": 2 / 3, "fast": 4 / 3}


def build_parser(app_launcher=None, *, policy_mode="joint"):
    parser = runner.build_parser(app_launcher)
    for action in parser._actions:
        if action.dest == "task":
            action.required = False
    parser.set_defaults(task="conveyor", policy_mode=policy_mode)
    return parser


def _complete_args(args):
    defaults = vars(build_parser().parse_args([]))
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    # Old arbitrary belt-speed flags cannot silently become a different suite.
    if args.speed_ood != "none" or args.conveyor_speed is not None or args.conveyor_speed_range is not None:
        raise ValueError("Use a versioned scene manifest and --condition for conveyor speed changes")
    return args


def load_collection(args):
    settings = runner.load_settings(_complete_args(args))
    args._legacy_settings = settings
    dataset, collection, cfg, _ = settings
    return dataset, collection, cfg, collection["pathtracing_spp"]


def run(args, dataset_dir, collection, cfg, spp):
    settings = getattr(args, "_legacy_settings", None)
    if settings is None:
        settings = runner.load_settings(_complete_args(args))
    if collection != settings[1] or cfg != settings[2] or spp != collection["pathtracing_spp"]:
        raise ValueError("legacy run settings differ from the selected suite")
    return runner.run(args, *settings)


def capture_step(scene, dvs, action, event_cameras):
    from piper_bench.tasks.conveyor.core import set_views
    previous, previous_time = dvs._conveyor_snapshot, scene.time
    set_views(scene.env)
    scene.env.step(action)
    success, reason = scene.after_step()
    if scene.supply():
        runtime.refresh_spawn_views(scene.env)
    current = runtime.snapshot(dvs, scene.env)
    dvs.warp_and_process(previous, current, runtime.WARP, previous_time, 1 / (runtime.CONTROL_HZ * runtime.WARP))
    dvs._conveyor_snapshot = current
    return runtime.observation(scene, dvs, event_cameras), success, reason


def main(argv=None, *, policy_mode="joint"):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--task" not in argv and not any(item.startswith("--task=") for item in argv):
        argv[0:0] = ["--task", "conveyor"]
    if "--policy-mode" not in argv and not any(item.startswith("--policy-mode=") for item in argv):
        argv[0:0] = ["--policy-mode", policy_mode]
    return runner.main(argv)

def evaluation_speed_spec(args, collection):
    bounds = getattr(args, "conveyor_speed_range", None)
    if bounds is not None:
        return uniform_speed_spec(bounds)
    if args.conveyor_speed is None and args.speed_ood == "none":
        return collection.get("speed_sampling")
    return None

def speed_ood_metadata(args, collection, config):
    training_spec=collection.get("speed_sampling")
    eval_spec=evaluation_speed_spec(args, collection)
    training_speed=float(collection["scene_config"]["speed"])
    if training_spec is not None or eval_spec is not None:
        lo,hi=(eval_spec['min_m_s'],eval_spec['max_m_s']) if eval_spec else (config.speed,config.speed)
        in_range=(training_spec['min_m_s'] <= lo <= hi <= training_spec['max_m_s']) if training_spec else lo==hi==training_speed
        return dict(distribution="in_distribution" if in_range else "out_of_distribution",
                    speed_ood="uniform" if eval_spec else "custom",
                    training_speed_sampling=training_spec, evaluation_speed_sampling=eval_spec,
                    training_conveyor_speed_m_s=None if training_spec else training_speed,
                    evaluation_conveyor_speed_m_s=None if eval_spec else float(config.speed),
                    conveyor_speed_scale=None)
    training_speed = float(collection["scene_config"]["speed"])
    custom = args.conveyor_speed is not None
    shifted = not math.isclose(config.speed, training_speed, rel_tol=0, abs_tol=1e-12)
    return dict(distribution="out_of_distribution" if shifted else "in_distribution",
                speed_ood="custom" if custom else args.speed_ood,
                training_conveyor_speed_m_s=training_speed,
                evaluation_conveyor_speed_m_s=float(config.speed),
                conveyor_speed_scale=config.speed / training_speed if custom else SPEED_OOD_SCALES[args.speed_ood])

def prepare_results(output, manifest, args):
    path = output / "manifest.json"
    if args.resume:
        saved = json.loads(path.read_text())
        # Existing paused evaluations predate the optional latency switch.
        if "latency_aware" in manifest:
            saved.setdefault("latency_aware", False)
        saved.setdefault("speed_ood", "none")
        saved.setdefault("training_conveyor_speed_m_s", saved.get("scene_config", {}).get("speed"))
        saved.setdefault("evaluation_conveyor_speed_m_s", saved.get("scene_config", {}).get("speed"))
        saved.setdefault("conveyor_speed_scale", 1.)
        saved.setdefault("basket_geometry_tolerance_m", 0.)
        saved.setdefault("basket_success_criterion_version", "full_volume_v1")
        saved.setdefault("training_mode", "joint")
        saved.setdefault("event_input", True)
        saved.setdefault("policy_observation_contract", EVENT_TIME_CONTRACT)
        saved.setdefault("latency_controller", "serial" if saved.get("latency_aware") else None)
        saved.setdefault("latency_source", "client_round_trip_seconds" if saved.get("latency_aware") else None)
        immutable = set(manifest) - {"server_optimization"}
        mismatches = [key for key in sorted(immutable) if saved.get(key) != manifest[key]]
        if mismatches:
            raise ValueError(f"Resume settings differ from saved manifest: {mismatches}")
        results = load_completed_results(output, args.trials)
        archived = archive_interrupted_trial(output, len(results))
        print(f"[ewm-conveyor] resume completed={len(results)} archived={archived}", flush=True)
        (output / "failure.json").unlink(missing_ok=True)
        return results
    if path.exists() or list(output.glob("trial_*")):
        raise FileExistsError(f"Existing evaluation requires --resume: {output}")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return []


def conveyor_module():
    """Retain the old module identity for monkeypatching runtime APIs in callers."""
    for name in ("build_parser", "load_collection", "run", "capture_step", "main",
                 "evaluation_speed_spec", "speed_ood_metadata", "prepare_results", "SPEED_OOD_SCALES"):
        setattr(runtime, name, globals()[name])
    return runtime
