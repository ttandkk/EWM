"""Half-height carton walls and shallow bowls, retaining narrow ID sampling."""
from piper_bench.collection.sampling import VERSION
from piper_bench.collection.sampling import LOW_GEOMETRY_VERSION

TASKS = ('moving_carton', 'turntable_cups')


def make_plan(task, episodes, seed, split):
    if task not in TASKS:
        raise ValueError('low-geometry protocol requires moving_carton or turntable_cups')
    from piper_bench.collection.narrow import make_plan as narrow_plan
    plan = narrow_plan(task, episodes, seed, split)
    plan['version'] = LOW_GEOMETRY_VERSION
    if task == 'moving_carton':
        plan['config_overrides'].update(basket_height=.0125,
            stack_success_mode='lowwall_v4', controlled_placement=True)
    else:
        plan['config_overrides'].update(object_size=.03, cup_height=.03)
    return plan


def episode_spec(plan, slot, retry=0):
    from piper_bench.collection.sampling import episode_spec as original_spec
    if plan['version'] != LOW_GEOMETRY_VERSION:
        raise ValueError('expected low-geometry plan')
    return original_spec(dict(plan, version=VERSION), slot, retry)
