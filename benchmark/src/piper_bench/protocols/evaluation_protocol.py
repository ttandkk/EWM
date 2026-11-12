"""Deterministic, paired model-evaluation scenes; no Isaac or file mutations."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from piper_bench.tasks import load_task
from piper_bench.collection.protocol import TASKS
from piper_bench.collection.protocol import SPEEDS
from piper_bench.collection.protocol import base_config
from piper_bench.collection.protocol import episode_spec
from piper_bench.collection.protocol import make_plan
from piper_bench.collection.protocol import rng
from piper_bench.collection.protocol import protocol_camera

VERSION = "piper_six_model_eval_v1"
CONDITIONS = ("id", "slow", "fast", "lighting_ood", "blur_ood", "background_ood", "speed_ood")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_collection(task, dataset):
    """Refuse previews, another task, or a changed physical scene."""
    dataset = Path(dataset)
    meta = dataset / "meta"
    collection = json.loads((meta / "piper_collection_config.json").read_text())
    from piper_bench.collection.sampling import VERSIONS as DYNAMIC_VERSIONS
    from piper_bench.collection.sampling import physics_hz
    from piper_bench.protocols.dynamic_evaluation import numerical_equal
    dynamic = collection.get("version") in DYNAMIC_VERSIONS
    module = load_task(task)
    if collection.get("task") != task or collection.get("task_id") != module.TASK:
        raise ValueError("dataset task differs from --task")
    if not all((dataset / marker).is_file() for marker in (".collection_complete", ".export_complete")):
        raise ValueError("evaluation requires a completed, validated collector export")
    validation = json.loads((meta / "piper_validation.json").read_text())
    if validation.get("passed") is not True or validation.get("task") != task:
        raise ValueError("dataset validation did not pass for this task")
    plan = collection["plan"]
    expected = make_plan(task, plan["episodes"], plan["seed"], plan["split"], version=plan["version"])
    if plan != expected or collection["version"] != plan["version"]:
        raise ValueError("dataset sampling plan differs from its versioned protocol")
    if collection["base_config"] != base_config(task, plan).metadata():
        raise ValueError("current task geometry/thresholds differ from the training export")
    recorded_camera = protocol_camera(task, plan, calibration_id=collection["camera"]["calibration_id"])
    if not numerical_equal(collection["camera"], recorded_camera) or json.loads(
            (meta / "piper_camera_calibration.json").read_text()) != collection["camera"]:
        raise ValueError("camera calibration differs from the training export")
    if (collection.get("renderer") != "pathtracing" or collection.get("standard_imaging") is not True
            or collection.get("control_hz") != 30 or collection.get("physics_hz") != (physics_hz(task) if dynamic else 120)
            or collection.get("event_hz") != 240 or collection.get("initial_hold_steps") != (0 if dynamic and task.startswith("tennis_") else 1)
            or not isinstance(collection.get("pathtracing_spp"), int) or collection["pathtracing_spp"] < 1):
        raise ValueError("unsupported collector renderer or sensor/control clock")
    contract = json.loads((meta / "piper_temporal_contract.json").read_text())
    if (contract.get("temporal_contract") != "aligned_event_frame_pre_action_v1"
            or contract.get("task_selector") != task or contract.get("collection_protocol") != plan["version"]
            or contract.get("control_hz") != 30 or contract.get("event_hz") != 240):
        raise ValueError("dataset observation/action/event contract differs")
    return collection


def scene_manifest(collection, trials=50, seed=20280002, condition="id", *, background_version=None):
    """Keep poses/phases paired across conditions; change only the speed stream.

    Even the open OOD boundaries are strict, using the next representable float
    if necessary. Failed trials are never retried or replaced by easier scenes.
    """
    from piper_bench.collection.sampling import VERSIONS as DYNAMIC_VERSIONS
    if collection.get("version", collection["plan"]["version"]) in DYNAMIC_VERSIONS:
        from piper_bench.protocols.dynamic_evaluation import scene_manifest as dynamic_manifest
        return dynamic_manifest(collection, trials, seed, condition, background_version=background_version)
    task = collection["task"]
    if task not in TASKS or condition not in ("id", "slow", "fast") or trials < 1 or not 0 <= seed < 2**63:
        raise ValueError("invalid evaluation task, condition, trials or seed")
    if condition != "id" and task not in SPEEDS:
        raise ValueError("speed OOD applies only to the three dynamic tasks")
    plan = make_plan(task, trials, seed, "evaluation", version=collection["plan"]["version"])
    speed = copy.deepcopy(collection["plan"]["speed"])
    if speed is not None:
        low, high = speed["low"], speed["high"]
        if condition == "slow":
            low, high = .85*low, float(np.nextafter(low, -math.inf))
        elif condition == "fast":
            low, high = float(np.nextafter(high, math.inf)), 1.15*high
        draws = rng(seed, task, "evaluation", stream=4).random(trials)
        values = np.clip(low + (np.arange(trials)+draws)*(high-low)/trials, low, high)
        order = rng(seed, task, "evaluation", stream=5).permutation(trials)
        for slot, index in zip(plan["slots"], order):
            slot.update(speed=float(values[index]), stratum=int(index))
        speed.update(low=low, high=high)
        plan["speed"] = speed
    scenes = [episode_spec(plan, i) for i in range(trials)]
    return dict(version=VERSION, task=task, condition=condition, collection_sha256=digest(collection),
                plan=plan, episodes=scenes, speed_distribution=speed,
                success_version=collection["success_version"],
                scene_seed_rule="independent evaluation split; retry=0; no success filtering")


def load_scene_manifest(path, expected):
    saved = json.loads(Path(path).read_text())
    from piper_bench.protocols.dynamic_evaluation import numerical_equal
    if not numerical_equal(saved, expected):
        raise ValueError("scene manifest differs from the task, training export or evaluation settings")
    return saved


def stable_steps(runtime):
    # Static scenes (including drawer) count stability on the runtime itself.
    return int(getattr(runtime, "stable_steps", getattr(getattr(runtime, "tracker", None), "steps", 0)))
