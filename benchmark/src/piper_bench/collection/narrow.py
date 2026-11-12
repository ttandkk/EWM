"""Narrow ID profile; rotation speeds share the revised recollection bounds."""
from dataclasses import replace
import math

from piper_bench.collection.sampling import VERSION
from piper_bench.collection.sampling import NARROW_VERSION
from piper_bench.collection.sampling import TASKS
from piper_bench.collection.sampling import TENNIS_TASKS
from piper_bench.collection.sampling import rng
from piper_bench.collection.protocol import ROTATION_SPEED_BOUNDS

BOUNDS = {
    'moving_carton': (.045, .055),
    **ROTATION_SPEED_BOUNDS,
    'tennis_rolling': (.36, .40),
    'tennis_falling': (.25, .35),
    'tennis_parabolic': (1.58, 1.62),
    'tennis_container': (.67, .68),
}


def config_overrides(task):
    if task == 'conveyor':
        return {}
    if task == 'rotating_tube':
        return {}  # Pose jitter is in the protocol, not TubeConfig.
    if task in TASKS[1:3]:
        return dict(position_jitter=.003, yaw_jitter=math.radians(3))
    value = dict(launch_delay_min=.5, launch_delay_max=.5)
    if task == 'tennis_rolling':
        value.update(release_x_jitter=0., release_y_jitter=0., rolling_yaw_jitter_deg=5.,
                     rolling_speed=.38, rolling_speed_jitter=.02)
    elif task == 'tennis_falling':
        value.update(position_jitter=.002, downward_speed_min=.25, downward_speed_max=.35)
    elif task == 'tennis_parabolic':
        from piper_bench.tasks.tennis.config import LobConfig
        cfg = LobConfig()
        flight = .675
        value.update(source_x=1.56, position_jitter=0., launch_yaw_jitter_deg=3.,
                     velocity_jitter=.02,
                     velocity_z=(cfg.catch_z-cfg.source_z+.5*cfg.gravity*flight**2)/flight)
    elif task == 'tennis_container':
        value.update(paired_launch=False, release_x_jitter=0., release_z_jitter=0.,
                     receiving_x_min=.40, receiving_x_max=.44, receiving_y_limit=.04,
                     minimum_receiver_displacement=.03, flight_time_min=.67, flight_time_max=.68)
    else:
        raise ValueError('unknown narrow task')
    return value


def make_plan(task, episodes, seed, split):
    import numpy as np
    from piper_bench.collection.sampling import make_plan as original_plan
    # v1 requires even container counts; v2 has independent, unpaired episodes.
    plan = original_plan(task, episodes*2 if task == 'tennis_container' else episodes,
                         seed, split, version=VERSION)
    plan.update(version=NARROW_VERSION, episodes=episodes)
    if task == 'conveyor':
        return plan  # No change to its random streams, bounds or scene states.
    plan['config_overrides'] = config_overrides(task)
    low, high = BOUNDS[task]
    duration = task == 'tennis_container'
    parameter = plan['flight_duration'] if duration else plan['speed']
    parameter.update(low=low, high=high, sampling='shuffled_uniform_strata',
                     retry='keep_slot_speed_and_launch_time')
    if 'center' in parameter:
        parameter['center'] = (low+high)/2
    draw = rng(task, seed, split, stream=4)
    values = low+(np.arange(episodes)+draw.random(episodes))*(high-low)/episodes
    order = draw.permutation(episodes)
    plan['slots'] = [dict(slot=i, speed=None if duration else float(values[j]), stratum=int(j),
                          flight_duration_s=float(values[j]) if duration else None,
                          launch_time_s=.5 if task in TENNIS_TASKS else None,
                          pair_slot=None, pair_member=None) for i,j in enumerate(order)]
    if task in TASKS[1:4]:
        plan.update(position_jitter_m=.003, yaw_jitter_rad=math.radians(3))
        if task in ('moving_carton','rotating_tube'):
            plan['fixed_phase'] = 0.
    else:
        plan.update(paired=False, launch_time=dict(low=.5, high=.5, hz=30, sampling='fixed_tick',
                                                 retry='keep_slot_time'),
                    retry='resample_pose_keep_speed_and_time')
    return plan


def episode_spec(plan, slot, retry=0):
    from piper_bench.collection.sampling import episode_spec as original_spec
    if plan['version'] != NARROW_VERSION:
        raise ValueError('expected narrow ID plan')
    return original_spec(dict(plan, version=VERSION), slot, retry)
