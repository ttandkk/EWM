"""Reproducible six-task ID collection plans, independent of Isaac Sim.

Rotation bounds were reduced on 2026-10-31 for recollection. Existing exports
must not be mixed with the revised plans; resume validation detects the change.
"""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from piper_bench.tasks import load_task

VERSION = "piper_six_collection_v1"
TASKS = ("push_t", "drawer_cup", "square_nut", "moving_carton", "turntable_cups", "rotating_tube")
from piper_bench.collection.sampling import VERSION as DYNAMIC_VERSION
from piper_bench.collection.sampling import TASKS as DYNAMIC_TASKS
from piper_bench.collection.sampling import TENNIS_TASKS
from piper_bench.collection.sampling import NARROW_VERSION
from piper_bench.collection.sampling import CONTAINER_SPATIAL_VERSION
from piper_bench.collection.sampling import CONTAINER_WIDE_VERSION
from piper_bench.collection.sampling import CONTAINER_SPATIAL_VERSIONS
from piper_bench.collection.sampling import CARTON_LOWWALL_VERSION
from piper_bench.collection.sampling import LOW_GEOMETRY_VERSION
from piper_bench.collection.sampling import VERSIONS as DYNAMIC_VERSIONS
from piper_bench.collection.sampling import DEFAULT_PROTOCOLS
COLLECTION_TASKS = TASKS + tuple(t for t in DYNAMIC_TASKS if t not in TASKS)
SPLITS = {"train": 1, "validation": 2, "acceptance": 3, "evaluation": 4}
SPEEDS = {
    "moving_carton": ("peak_speed", .05, "m/s"),
    "turntable_cups": ("omega", .315, "rad/s"),
    "rotating_tube": ("omega", .21, "rad/s"),
}
# Explicit angular bounds shared by standard and narrow collection profiles.
ROTATION_SPEED_BOUNDS = {
    "turntable_cups": (.30, .33),
    "rotating_tube": (.20, .22),
}
OBJECT_COUNTS = {name: 2 if name in ("moving_carton", "turntable_cups") else 1 for name in TASKS}
OBJECT_COUNTS.update({name: 1 for name in TENNIS_TASKS})
OBJECT_COUNTS["conveyor"] = 2


def rng(seed, task, split, slot=0, retry=0, stream=0):
    """Named independent streams: position, orientation, phase, speed and order."""
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence(
        [seed, TASKS.index(task), SPLITS[split], slot, retry, stream]))


def make_plan(task, episodes=50, seed=20261016, split="train", *, version=None):
    if version is None:
        version = DEFAULT_PROTOCOLS.get(task)
    if version in DYNAMIC_VERSIONS or (version is None and task in COLLECTION_TASKS and task not in TASKS):
        from piper_bench.collection.sampling import make_plan as dynamic_plan
        return dynamic_plan(task, episodes, seed, split, version=version or DYNAMIC_VERSION)
    import numpy as np
    if task not in TASKS or split not in SPLITS or episodes < 1 or seed < 0:
        raise ValueError("invalid task, split, episode count or seed")
    if version not in (None, VERSION):
        from piper_bench.tasks.push_t.protocol import VERSION as PUSH_VERSION
        if task != "push_t" or version != PUSH_VERSION:
            raise ValueError("unsupported collection protocol version")
    slots = [dict(slot=i, speed=None, stratum=None) for i in range(episodes)]
    if task in SPEEDS:
        name, center, unit = SPEEDS[task]
        low, high = ROTATION_SPEED_BOUNDS.get(task, (.7 * center, 1.3 * center))
        values = low + (np.arange(episodes) + rng(seed, task, split, stream=4).random(episodes)) * (high-low)/episodes
        order = rng(seed, task, split, stream=5).permutation(episodes)
        for slot, index in zip(slots, order):
            slot.update(speed=float(values[index]), stratum=int(index))
        speed = dict(parameter=name, unit=unit, center=center, low=low, high=high,
                     sampling="shuffled_uniform_strata", retry="keep_slot_speed")
    else:
        speed = None
    plan = dict(version=VERSION, task=task, split=split, seed=seed, episodes=episodes,
                position_jitter_m=.01, yaw_jitter_rad=math.radians(10), speed=speed, slots=slots)
    if task == "push_t" and version != VERSION:
        from piper_bench.tasks.push_t.protocol import configure_plan
        configure_plan(plan)
    return plan


def base_config(task, plan=None):
    cfg = load_task(task).Config()
    if plan is not None and plan.get('version') == LOW_GEOMETRY_VERSION:
        if task not in ('moving_carton', 'turntable_cups'):
            raise ValueError('low-geometry task mismatch')
        return replace(cfg, **plan['config_overrides'])
    if plan is not None:
        # Versioned historical exports must still reconstruct their old sizes.
        if task == 'moving_carton':
            cfg = replace(cfg, basket_height=.035, stack_success_mode='legacy', controlled_placement=False)
        elif task == 'turntable_cups':
            cfg = replace(cfg, object_size=.06, cup_height=.06)
    if plan is not None and plan.get('version') == CARTON_LOWWALL_VERSION:
        if task != 'moving_carton':
            raise ValueError('low-wall protocol is moving-carton only')
        return replace(cfg, **plan['config_overrides'])
    if plan is not None and plan.get("version") in CONTAINER_SPATIAL_VERSIONS:
        if task != "tennis_container":
            raise ValueError("spatial collection is container-only")
        return replace(cfg, **plan["config_overrides"])
    if plan is not None and plan.get("version") == NARROW_VERSION:
        from piper_bench.collection.narrow import config_overrides
        return replace(cfg, **config_overrides(task))
    # The actual poses below use separate streams; these fields also document
    # the intended bounds in the per-episode scene configuration.
    if task in TENNIS_TASKS or task == "conveyor":
        return cfg
    if hasattr(cfg, "position_jitter"):
        wide_push = task == "push_t" and (plan is None or plan["version"] != VERSION)
        cfg = replace(cfg, position_jitter=.03 if wide_push else .01,
                      yaw_jitter=math.radians(45 if wide_push else 10))
    return cfg


def episode_spec(plan, slot, retry=0):
    if plan["version"] in DYNAMIC_VERSIONS:
        from piper_bench.collection.sampling import episode_spec as dynamic_spec
        return dynamic_spec(plan, slot, retry)
    import numpy as np
    if not 0 <= slot < plan["episodes"] or retry < 0:
        raise ValueError("invalid collection slot or retry")
    task, split, seed = plan["task"], plan["split"], plan["seed"]
    cfg = base_config(task, plan)
    if plan.get("config_overrides"):
        cfg = replace(cfg, **plan["config_overrides"])
    position = rng(seed, task, split, slot, retry, 1)
    orientation = rng(seed, task, split, slot, retry, 2)
    phase = float(rng(seed, task, split, slot, retry, 3).uniform(-math.pi, math.pi))
    if "fixed_phase" in plan:
        phase = plan["fixed_phase"]
    speed = plan["slots"][slot]["speed"]
    if task == "moving_carton":
        cfg = replace(cfg, peak_speed=speed)
        xy = np.array([[cfg.source_x, cfg.source_y-cfg.source_spacing/2],
                       [cfg.source_x, cfg.source_y+cfg.source_spacing/2]])
        z = cfg.object_size/2 + .001
    elif task == "turntable_cups":
        cfg = replace(cfg, speed=speed * cfg.orbit_radius)
        angles = np.array([-2.5, -.4])
        xy = cfg.center + cfg.orbit_radius * np.column_stack((np.cos(angles), np.sin(angles)))
        z = cfg.belt_top + cfg.cup_height/2 + .001
    elif task == "rotating_tube":
        cfg = replace(cfg, omega=speed)
        xy, z = np.array([[cfg.tube_x, cfg.tube_y]]), cfg.cap_radius + .001
    else:
        xy = np.array([[cfg.source_x, cfg.source_y]])
        z = {"push_t": lambda: cfg.height/2 + .001,
             "drawer_cup": lambda: cfg.cup_height/2 + .001,
             "square_nut": lambda: cfg.table_z + cfg.thickness/2 + .002}[task]()
    jitter = plan.get("position_jitter_m", .01)
    xy += position.uniform(-jitter, jitter, xy.shape)
    yaw_jitter = plan.get("yaw_jitter_rad", math.radians(10))
    yaw = orientation.uniform(-yaw_jitter, yaw_jitter, len(xy))
    if task == "push_t" and plan["version"] != VERSION:
        from piper_bench.tasks.push_t.protocol import sample_pose
        offset, angle = sample_pose(plan, slot, retry)
        xy = np.array([[cfg.source_x, cfg.source_y]]) + offset
        yaw = np.array([angle])
    if task == "square_nut":
        yaw += cfg.source_yaw
    if task == "rotating_tube":
        # Perturb the approved horizontal radial heading, independently of XY.
        yaw += math.atan2(cfg.tube_y, cfg.tube_x)
        from piper_bench.tasks.rotating_tube.oracle import yaw_pitch_quaternion
        quats = [yaw_pitch_quaternion(yaw[0], math.pi/2)]
    else:
        if task == "turntable_cups":
            yaw[:] = 0.  # Axisymmetric cups remain upright.
        quats = [[0., 0., math.sin(a/2), math.cos(a/2)] for a in yaw]
    poses = [list(p) + [z] + list(q) for p, q in zip(xy, quats)]
    cabinet_pose = None
    if task == "drawer_cup":
        offset = rng(seed, task, split, slot, retry, 6).uniform(-.01, .01, 2)
        cfg = replace(cfg, cabinet_x=cfg.cabinet_x+offset[0], cabinet_y=cfg.cabinet_y+offset[1])
        cabinet_pose = [cfg.cabinet_x, cfg.cabinet_y, cfg.drawer_z, 0., 0.,
                        math.sin(cfg.cabinet_yaw/2), math.cos(cfg.cabinet_yaw/2)]
    runtime_seed = int(rng(seed, task, split, slot, retry, 7).integers(0, 2**31-1))
    return dict(task=task, split=split, slot=slot, retry=retry, seed=runtime_seed,
                speed=speed, stratum=plan["slots"][slot]["stratum"],
                phase=phase if task in ("moving_carton", "rotating_tube") else None,
                object_poses=poses, object_yaws=yaw.tolist(), cabinet_pose=cabinet_pose,
                config=cfg.metadata())


def protocol_camera(task, plan, *, calibration_id=None):
    if task in TENNIS_TASKS:
        from piper_bench.tasks.tennis.camera import camera_metadata_for_config
        return camera_metadata_for_config(base_config(task, plan), calibration_id=calibration_id)
    value = load_task(task).camera_metadata()
    if calibration_id is not None and calibration_id != value['calibration_id']:
        raise ValueError('unknown camera calibration for this task')
    return value


def write_json(path, data):
    """Atomic metadata replacement; a crash leaves either complete version."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def source_hashes(scripts=None):
    """Fingerprint installed benchmark sources, independent of the install prefix.

    Old launchers may additionally supply a scripts directory. Keep its hashes
    for provenance, but always include the actual canonical implementation rather
    than recording only compatibility shims. Historical manifests remain intact;
    their different source layout is intentionally rejected on blind resume.
    """
    package = Path(__file__).resolve().parents[1]
    files = {f"piper_bench/{p.relative_to(package).as_posix()}": p
             for p in package.rglob("*.py")}
    if scripts is not None:
        scripts = Path(scripts).resolve()
        if not scripts.is_relative_to(package):
            legacy = list(scripts.glob("*.py")) + list((scripts / "piper_tasks").rglob("*.py"))
            files.update({p.relative_to(scripts).as_posix(): p for p in legacy})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(files.items())}


def check_attempts(plan, attempts, saved):
    """Refuse ambiguous resume; never silently resample a failed speed slot."""
    slot, retry = 0, 0
    for index, record in enumerate(attempts):
        if record["attempt"] != index or record["spec"] != episode_spec(plan, slot, retry):
            raise ValueError("attempt history differs from the saved sampling plan")
        if record.get("success"):
            if record.get("saved_episode") != slot:
                raise ValueError("successful attempt is missing its saved episode")
            slot, retry = slot+1, 0
        else:
            if "saved_episode" in record:
                raise ValueError("failed attempt must not be exported")
            retry += 1
    if slot != saved:
        raise ValueError("writer and attempt counts disagree; inspect pending_save.json before resuming")
    return retry
