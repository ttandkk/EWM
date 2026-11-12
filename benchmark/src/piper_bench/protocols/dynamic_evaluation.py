"""Eight-task ID/OOD manifests, paired initial states and data-derived budgets."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from piper_bench.collection.sampling import TASKS
from piper_bench.collection.sampling import VERSION as COLLECTION_V1
from piper_bench.collection.sampling import NARROW_VERSION
from piper_bench.collection.sampling import CARTON_LOWWALL_VERSION
from piper_bench.collection.sampling import LOW_GEOMETRY_VERSION
from piper_bench.collection.sampling import make_plan
from piper_bench.collection.sampling import episode_spec
from piper_bench.collection.sampling import rng
from piper_bench.protocols.ood_protocol import make_plan as imaging_plan, BACKGROUND_VERSION

VERSION = 'piper_eight_dynamic_eval_v1'
DEFAULT_CONDITIONS = ('id', 'lighting_ood', 'blur_ood', 'background_ood', 'speed_ood')
CONDITIONS = (*DEFAULT_CONDITIONS, 'slow', 'fast')
SPEED_VERSION = 'piper_id_boundary_speed_ood_v3'
SPEED_MULTIPLIERS = {'slow': (.8, .9), 'fast': (1.1, 1.2)}
# Fixed evaluation defaults approved on 2026-10-30; never derive limits from demonstrations.
DEFAULT_MAX_STEPS = dict(conveyor=350, moving_carton=650, turntable_cups=600,
                         rotating_tube=350, tennis_rolling=100, tennis_falling=100,
                         tennis_parabolic=100, tennis_container=100)
# Frozen provenance: maxima of the 50 successful ID demonstrations on 2026-10-21.
COLLECTED_MAX_STEPS = dict(zip(TASKS, (300, 643, 586, 292, 139, 220, 221, 102)))


def next_fifty(maximum):
    if isinstance(maximum, bool) or int(maximum) != maximum or maximum < 1:
        raise ValueError('positive integer trajectory length required')
    return (int(maximum)//50+1)*50


def supported_conditions(task):
    if task not in TASKS:
        raise ValueError('unknown dynamic task')
    return CONDITIONS


def step_budget(dataset, *, legacy_max_steps=None):
    root = Path(dataset)
    info = json.loads((root/'meta/info.json').read_text())
    records = json.loads((root/'meta/piper_task_attempts.json').read_text())
    saved = [r for r in records if r['success']]
    if (len(saved) != info['total_episodes'] or not saved
            or sorted(r['saved_episode'] for r in saved) != list(range(len(saved)))
            or sum(r['frames'] for r in saved) != info['total_frames']):
        raise ValueError('collection episode lengths do not match exported dataset')
    maximum = max(r['frames'] for r in saved)
    collection = json.loads((root/'meta/piper_collection_config.json').read_text())
    task = collection['task']
    limit = DEFAULT_MAX_STEPS[task]
    # Only the new low-wall carton receives the requested 600 -> 650 change.
    if task == 'moving_carton' and collection['version'] not in (CARTON_LOWWALL_VERSION, LOW_GEOMETRY_VERSION):
        limit = 550 if collection['version'] == NARROW_VERSION else 650
    rule = 'fixed_task_defaults_v1'
    if legacy_max_steps is not None:
        # Existing manifests pin their historical limit; do not recalculate it.
        if type(legacy_max_steps) is not int or legacy_max_steps <= 0 or legacy_max_steps % 50:
            raise ValueError('invalid fixed legacy evaluation budget')
        limit, rule = legacy_max_steps, 'strictly_next_multiple_of_50'
    return dict(collected_max_steps=maximum, max_steps=limit,
                rule=rule, episodes=len(saved),
                includes_launch_wait=True, shared_across_conditions=True)


def numerical_equal(a, b, atol=1e-14):
    """Allow only cross-library floating arithmetic roundoff, never structure drift."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(numerical_equal(a[k], b[k], atol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(numerical_equal(x, y, atol) for x,y in zip(a,b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= atol
    return type(a) is type(b) and a == b


def speed_bounds(task, condition, plan):
    if condition not in ('slow', 'fast') or condition not in supported_conditions(task):
        raise ValueError(f'{task} has no {condition} speed OOD')
    duration = task == 'tennis_container'
    parameter = copy.deepcopy(plan['flight_duration'] if duration else plan['speed'])
    low, high = parameter['low'], parameter['high']
    factor_low, factor_high = SPEED_MULTIPLIERS[condition]
    # Expand outward from one ID boundary, never scale an individual ID draw.
    # Longer flight duration is the slow condition for container catching.
    use_lower = condition == ('fast' if duration else 'slow')
    reference = low if use_lower else high
    bounds = ((reference/factor_high, reference/factor_low) if duration else
              (reference*factor_low, reference*factor_high))
    if not (bounds[0] < bounds[1] and (bounds[1] < low or bounds[0] > high)):
        raise ValueError(f'{task} {condition}: cannot construct a non-overlapping speed OOD '
                         f'outside ID [{low}, {high}]')
    parameter.update(low=float(bounds[0]), high=float(bounds[1]),
                     version=SPEED_VERSION, id_low=low, id_high=high,
                     reference='ID_lower_bound' if use_lower else 'ID_upper_bound',
                     reference_value=float(reference), multiplier_low=factor_low, multiplier_high=factor_high,
                     sampling='shuffled_uniform_parameter_strata',
                     retry='keep_ID_slot_and_parameter', id_endpoints_excluded=True,
                     absolute_range_may_overlap_id=False)
    parameter.pop('center', None)
    if duration:
        parameter['interpretation'] = ('slow: T in [ID_max/0.9, ID_max/0.8]; '
                                       'fast: T in [ID_min/1.2, ID_min/1.1]; '
                                       'launch velocity is derived with gravity unchanged')
    return parameter


def scene_manifest(collection, trials=50, seed=20280002, condition='id', *, background_version=None):
    from piper_bench.protocols.evaluation_protocol import digest
    task = collection['task']
    if task not in TASKS or condition not in supported_conditions(task) or not 1 <= trials <= 50 or not 0 <= seed < 2**63:
        raise ValueError('invalid eight-task evaluation condition, seed or trial count (1..50)')
    if condition == 'speed_ood' and trials % 2:
        raise ValueError('speed_ood requires an even trial count for equal slow/fast groups')
    collection_version = collection.get('version', collection.get('plan', {}).get('version', COLLECTION_V1))
    version = LOW_GEOMETRY_VERSION if task in ('moving_carton', 'turntable_cups') else collection_version
    plan = make_plan(task, trials, seed, 'evaluation', version=version)
    speed = None
    if condition in ('slow', 'fast', 'speed_ood'):
        duration = task == 'tennis_container'
        key = 'flight_duration_s' if duration else 'speed'
        bounds_key = 'flight_duration' if duration else 'speed'
        mixed = condition == 'speed_ood'
        names = ('slow', 'fast') if mixed else (condition,)
        distributions = {name: speed_bounds(task, name, plan) for name in names}
        draw = rng(task, seed, 'evaluation', stream=37)
        for index, name in enumerate(names):
            bounds = distributions[name]
            slots = plan['slots'][index*(trials//2):(index+1)*(trials//2)] if mixed else plan['slots']
            # Keep all ID scene slots. Mixed OOD stratifies each half separately;
            # standalone historical container conditions retain paired times.
            groups = [slot['pair_slot'] if not mixed and slot.get('pair_slot') is not None else slot['slot']
                      for slot in slots]
            unique = list(dict.fromkeys(groups))
            units = (np.arange(len(unique))+draw.random(len(unique)))/len(unique)
            order = draw.permutation(len(unique))
            units = units[order]
            values = dict(zip(unique, bounds['low'] + units*(bounds['high']-bounds['low'])))
            strata = dict(zip(unique, order))
            for slot, group in zip(slots, groups):
                value, reference = float(values[group]), bounds['reference_value']
                factor = reference/value if duration else value/reference
                slot.update(speed_reference_boundary=reference, speed_multiplier=factor, stratum=int(strata[group]))
                slot[key] = value
                if mixed:
                    slot['speed_condition'] = name
        speed = (dict(version='piper_speed_ood_balanced_v1', parameter=distributions['slow']['parameter'],
                      unit=distributions['slow']['unit'], groups=distributions, trials_per_group=trials//2,
                      group_order=list(names)) if mixed else distributions[condition])
        plan[bounds_key] = speed
    imaging = imaging_plan(task, condition if condition not in ('slow','fast','speed_ood') else 'id', seed,
                           background_version=background_version or BACKGROUND_VERSION)
    specs = [episode_spec(plan, i, 0) for i in range(trials)]
    for i, spec in enumerate(specs):
        spec['imaging'] = dict(imaging['episodes'][i], condition=imaging['condition'])
        if speed is not None:
            spec.update(speed_reference_boundary=plan['slots'][i]['speed_reference_boundary'],
                        speed_multiplier=plan['slots'][i]['speed_multiplier'])
            if condition == 'speed_ood':
                spec['speed_condition'] = plan['slots'][i]['speed_condition']
    success_version = collection['success_version']
    success_overrides = {}
    if task == 'tennis_rolling':
        from piper_bench.tasks.tennis_rolling import SUCCESS_VERSION
        success_version = SUCCESS_VERSION
    if plan['version'] == LOW_GEOMETRY_VERSION:
        if task == 'moving_carton':
            from piper_bench.tasks.moving_carton.lowwall import VERSION as success_version
        else:
            from piper_bench.tasks.turntable_cups.scene import BOWL_SUCCESS_VERSION as success_version
    if task == 'moving_carton' and plan['version'] not in (CARTON_LOWWALL_VERSION, LOW_GEOMETRY_VERSION):
        from piper_bench.tasks.moving_carton.scene import STACK_SUCCESS_V3
        from piper_bench.tasks.moving_carton.scene import STACK_SUCCESS_V3_OVERRIDES
        success_version = STACK_SUCCESS_V3
        success_overrides = dict(STACK_SUCCESS_V3_OVERRIDES)
        for spec in specs:
            spec['config'].update(success_overrides)
    return dict(version=VERSION, task=task, condition=condition, collection_sha256=digest(collection),
                plan=plan, episodes=specs, speed_distribution=speed,
                **(dict(geometry_version=LOW_GEOMETRY_VERSION, collection_geometry_version=collection_version)
                   if plan['version'] == LOW_GEOMETRY_VERSION else {}),
                imaging=imaging, success_version=success_version,
                **(dict(collection_success_version=collection['success_version'])
                   if success_version != collection['success_version'] else {}),
                **(dict(success_config_overrides=success_overrides) if success_overrides else {}),
                scene_seed_rule='paired evaluation seed/slot; retry=0; no success filtering',
                formal_trial_count=50)


def condition_breakdown(results, *, interface_only=False):
    """Report speed groups, light strata and exposure levels without filtering failures."""
    groups = {}
    for result in results:
        imaging = result['spec'].get('imaging', {})
        if result['spec'].get('speed_condition') in ('slow', 'fast'):
            key = result['spec']['speed_condition']
        elif imaging.get('condition') == 'lighting_ood':
            key = imaging['group']
        elif imaging.get('condition') == 'blur_ood':
            key = f"exposure_{imaging['intervals']}_of_240_s"
        else:
            continue
        group = groups.setdefault(key, dict(trials=0, successes=0))
        group['trials'] += 1
        group['successes'] += int(result['success'])
    for group in groups.values():
        group['success_rate'] = None if interface_only else group['successes']/group['trials']
    return groups
