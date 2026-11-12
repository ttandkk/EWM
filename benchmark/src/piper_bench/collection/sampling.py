"""Versioned eight-dynamic-task sampling, retaining the legacy six-task plan."""
from dataclasses import replace
import math

from piper_bench.tasks import load_task

VERSION = "piper_eight_dynamic_collection_v1"
NARROW_VERSION = "piper_eight_dynamic_narrow_v2"
CONTAINER_SPATIAL_VERSION = "piper_container_spatial_v3"
CONTAINER_WIDE_VERSION = "piper_container_spatial_v4"
CONTAINER_SPATIAL_VERSIONS = (CONTAINER_SPATIAL_VERSION, CONTAINER_WIDE_VERSION)
CARTON_LOWWALL_VERSION = "piper_carton_lowwall_v4"
LOW_GEOMETRY_VERSION = "piper_low_geometry_v5"
VERSIONS = (VERSION, NARROW_VERSION, *CONTAINER_SPATIAL_VERSIONS, CARTON_LOWWALL_VERSION, LOW_GEOMETRY_VERSION)
TASKS = ("conveyor", "moving_carton", "turntable_cups", "rotating_tube",
         "tennis_rolling", "tennis_falling", "tennis_parabolic", "tennis_container")
TENNIS_TASKS = TASKS[4:]
# One default table for batch collection, single-task plans and previews.
# Explicit historical versions keep their original distributions.
DEFAULT_PROTOCOLS = dict.fromkeys(TASKS, NARROW_VERSION)
DEFAULT_PROTOCOLS.update(conveyor=VERSION, moving_carton=LOW_GEOMETRY_VERSION,
                         turntable_cups=LOW_GEOMETRY_VERSION,
                         tennis_container=CONTAINER_SPATIAL_VERSION)
SPLITS = {"train": 1, "validation": 2, "acceptance": 3, "evaluation": 4}


def rng(task, seed, split, slot=0, retry=0, stream=0):
    import numpy as np
    return np.random.default_rng(np.random.SeedSequence(
        [813, seed, TASKS.index(task), SPLITS[split], slot, retry, stream]))


def make_plan(task, episodes, seed, split, *, version=None):
    if version is None:
        version = DEFAULT_PROTOCOLS.get(task, VERSION)
    if version == LOW_GEOMETRY_VERSION:
        from piper_bench.collection.low_geometry import make_plan as geometry_plan
        return geometry_plan(task, episodes, seed, split)
    if version == CARTON_LOWWALL_VERSION:
        from piper_bench.collection.carton_lowwall import make_plan as lowwall_plan
        return lowwall_plan(task, episodes, seed, split)
    if version in CONTAINER_SPATIAL_VERSIONS:
        from piper_bench.collection.container_spatial import make_plan as spatial_plan
        return spatial_plan(task, episodes, seed, split, version=version)
    if version == NARROW_VERSION:
        from piper_bench.collection.narrow import make_plan as narrow_plan
        return narrow_plan(task, episodes, seed, split)
    if version != VERSION:
        raise ValueError("unsupported dynamic protocol")
    import numpy as np
    if task not in TASKS or split not in SPLITS or episodes < 1 or seed < 0:
        raise ValueError("invalid dynamic collection request")
    if task == "tennis_container" and episodes % 2:
        raise ValueError("container collection requires an even episode count to retain both pair members")
    from piper_bench.collection.protocol import make_plan as legacy_plan
    from piper_bench.collection.protocol import VERSION as LEGACY
    if task in TASKS[1:4]:
        plan = legacy_plan(task, episodes, seed, split, version=LEGACY)
        plan["version"] = VERSION
        return plan
    cfg = load_task(task).Config()
    bounds = {
        "conveyor": ("belt_speed_m_s", .04, .06, "m/s"),
        "tennis_rolling": ("launch_planar_speed_m_s", .30, .46, "m/s"),
        "tennis_falling": ("initial_downward_speed_m_s", 0., .6, "m/s"),
        "tennis_parabolic": ("launch_horizontal_speed_m_s", 1.565, 1.635, "m/s"),
        # Horizontal velocity is jointly determined by source, endpoint and
        # flight duration. Stratify duration, never invent a uniform speed law.
        "tennis_container": ("flight_duration_s", .65, .70, "s"),
    }
    name, low, high, unit = bounds[task]
    paired = task == "tennis_container"
    count = episodes//2 if paired else episodes
    draws = rng(task, seed, split, stream=4)
    values = low+(np.arange(count)+draws.random(count))*(high-low)/count
    order = draws.permutation(count)
    delays = rng(task, seed, split, stream=8).integers(0, 61, count)
    slots = []
    for i in range(episodes):
        group = i//2 if paired else i
        slots.append(dict(slot=i, speed=float(values[order[group]]) if not paired else None,
                          stratum=int(order[group]),
                          flight_duration_s=float(values[order[group]]) if paired else None,
                          launch_time_s=int(delays[group])/30 if task in TENNIS_TASKS else None,
                          pair_slot=i//2 if paired else None, pair_member=i%2 if paired else None))
    return dict(version=VERSION, task=task, split=split, seed=seed, episodes=episodes,
                speed=None if paired else dict(parameter=name, low=low, high=high, unit=unit,
                    sampling="shuffled_uniform_strata", retry="keep_slot_speed_and_launch_time"),
                flight_duration=dict(parameter=name, low=low, high=high, unit=unit,
                    sampling="shuffled_uniform_strata_per_pair") if paired else None,
                launch_time=dict(low=0., high=2., hz=30, sampling="uniform_inclusive_ticks",
                    retry="keep_slot_time") if task in TENNIS_TASKS else None,
                retry="same_pair_trajectory" if paired else "resample_pose_keep_speed_and_time",
                slots=slots)


def episode_spec(plan, slot, retry):
    if plan["version"] == LOW_GEOMETRY_VERSION:
        from piper_bench.collection.low_geometry import episode_spec as geometry_spec
        return geometry_spec(plan, slot, retry)
    if plan["version"] == CARTON_LOWWALL_VERSION:
        from piper_bench.collection.carton_lowwall import episode_spec as lowwall_spec
        return lowwall_spec(plan, slot, retry)
    if plan["version"] in CONTAINER_SPATIAL_VERSIONS:
        from piper_bench.collection.container_spatial import episode_spec as spatial_spec
        return spatial_spec(plan, slot, retry)
    if plan["version"] == NARROW_VERSION:
        from piper_bench.collection.narrow import episode_spec as narrow_spec
        return narrow_spec(plan, slot, retry)
    import numpy as np
    if not 0 <= slot < plan["episodes"] or retry < 0:
        raise ValueError("invalid dynamic slot or retry")
    task, split, seed = plan["task"], plan["split"], plan["seed"]
    from piper_bench.collection.protocol import episode_spec as legacy_spec
    from piper_bench.collection.protocol import VERSION as LEGACY
    if task in TASKS[1:4]:
        return legacy_spec(dict(plan, version=LEGACY), slot, retry)
    cfg = load_task(task).Config()
    if plan.get("config_overrides"):
        cfg = replace(cfg, **plan["config_overrides"])
    item = plan["slots"][slot]
    paired = plan.get("paired", task == "tennis_container")
    draw = rng(task, seed, split, slot//2 if paired else slot, 0 if paired else retry, 7)
    runtime_seed = int(draw.integers(0, 2**30-1))
    if paired:
        runtime_seed = 2*runtime_seed+slot%2
    speed = item["speed"]
    spec = dict(task=task, split=split, slot=slot, retry=retry, seed=runtime_seed,
                speed=speed, stratum=item["stratum"], phase=None, cabinet_pose=None)
    if task == "conveyor":
        cfg = replace(cfg, speed=speed)
        x, y, yaw = cfg.supply_positions(runtime_seed)
        poses = [[float(a), float(b), cfg.belt_top+cfg.object_size/2+.001,
                  0., 0., math.sin(c/2), math.cos(c/2)] for a, b, c in zip(x, y, yaw)]
        return dict(spec, object_poses=poses, object_yaws=yaw.tolist(), config=cfg.metadata())
    delay = item["launch_time_s"]
    cfg = replace(cfg, launch_delay_min=delay, launch_delay_max=delay)
    if task == "tennis_rolling":
        cfg = replace(cfg, rolling_speed=speed, rolling_speed_jitter=0.)
    elif task == "tennis_falling":
        cfg = replace(cfg, downward_speed_min=speed, downward_speed_max=speed)
    elif task == "tennis_parabolic":
        cfg = replace(cfg, velocity_x=-speed, velocity_jitter=0.)
    else:
        cfg = replace(cfg, flight_time_min=item["flight_duration_s"],
                      flight_time_max=item["flight_duration_s"])
    from piper_bench.tasks.tennis.motion import launch_state
    position, velocity = launch_state(cfg, runtime_seed)
    # Parking follows the scene type, not the independent launch-pairing policy.
    park = [-3. if task == 'tennis_container' else -.7, 0., -.5]
    pose = (position.tolist() if delay == 0 else park)+[0., 0., 0., 1.]
    return dict(spec, object_poses=[pose], object_yaws=[0.], config=cfg.metadata(),
                launch_position_m=position.tolist(), launch_velocity=velocity.tolist(),
                launch_time_s=delay, pair_slot=item["pair_slot"], pair_member=item["pair_member"],
                launch_horizontal_speed_m_s=float(np.linalg.norm(velocity[:2])),
                flight_duration_s=item["flight_duration_s"])


def physics_hz(task):
    return 240 if task == "tennis_container" else 120
