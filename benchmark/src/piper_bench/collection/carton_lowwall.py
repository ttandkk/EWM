"""Opt-in low-wall controlled-placement dataset; archived v1/v2 stay unchanged."""
from piper_bench.collection.narrow import make_plan as narrow_plan
from piper_bench.collection.sampling import VERSION
from piper_bench.collection.sampling import CARTON_LOWWALL_VERSION


def make_plan(task, episodes, seed, split):
    if task != 'moving_carton':
        raise ValueError('low-wall protocol is moving-carton only')
    plan = narrow_plan(task, episodes, seed, split)
    plan['version'] = CARTON_LOWWALL_VERSION
    # Retain the complete narrow-v2 random streams and ranges. Reduce release
    # variability through controlled placement, not easier initial conditions.
    plan['config_overrides'].update(basket_height=.025,
        stack_success_mode='lowwall_v4', controlled_placement=True)
    return plan


def episode_spec(plan, slot, retry=0):
    from piper_bench.collection.sampling import episode_spec as original_spec
    if plan['version'] != CARTON_LOWWALL_VERSION:
        raise ValueError('expected low-wall plan')
    return original_spec(dict(plan, version=VERSION), slot, retry)
