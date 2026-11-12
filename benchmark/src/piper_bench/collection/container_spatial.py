"""Container spatial grids: default v3 (+/-10 cm), explicit wider v4 (+/-12 cm)."""
import copy
import numpy as np
from piper_bench.collection.sampling import VERSION
from piper_bench.collection.sampling import CONTAINER_SPATIAL_VERSION
from piper_bench.collection.sampling import CONTAINER_WIDE_VERSION
from piper_bench.collection.sampling import CONTAINER_SPATIAL_VERSIONS
from piper_bench.collection.sampling import rng
from piper_bench.collection.narrow import make_plan as narrow_plan


def make_plan(task, episodes, seed, split, *, version=CONTAINER_SPATIAL_VERSION):
    if task != 'tennis_container' or version not in CONTAINER_SPATIAL_VERSIONS:
        raise ValueError('container spatial protocol requires tennis_container and a supported version')
    y_limit = .10 if version == CONTAINER_SPATIAL_VERSION else .12
    plan = narrow_plan(task, episodes, seed, split)
    plan['version'] = version
    plan['config_overrides'].update(receiving_x_min=.40, receiving_x_max=.50,
                                    receiving_y_limit=y_limit)
    plan['receiving_grid'] = dict(shape=[5,5], x=[.40,.50], y=[-y_limit,y_limit],
        sampling='shuffled_balanced_cells_uniform_within_cell', retry='keep_cell_flight_time_and_launch_time')
    draw = rng(task, seed, split, stream=19)
    # Every complete cycle visits all cells once; 50 demonstrations = two/cell.
    cells = np.concatenate([draw.permutation(25) for _ in range((episodes+24)//25)])[:episodes]
    xs, ys = np.linspace(.40,.50,6), np.linspace(-y_limit,y_limit,6)
    for item, cell in zip(plan['slots'], cells):
        ix, iy = divmod(int(cell),5)
        item.update(receiving_cell_id=int(cell), receiving_cell=[float(xs[ix]),float(xs[ix+1]),float(ys[iy]),float(ys[iy+1])])
    plan['retry'] = 'resample_within_same_spatial_cell_keep_speed_and_time'
    return plan


def episode_spec(plan, slot, retry=0):
    from piper_bench.collection.sampling import episode_spec as original_spec
    if plan['version'] not in CONTAINER_SPATIAL_VERSIONS or not 0 <= slot < plan['episodes'] or retry < 0:
        raise ValueError('invalid container spatial slot/retry')
    adapted=copy.deepcopy(plan)
    adapted['version']=VERSION
    adapted['config_overrides']['receiving_cell']=plan['slots'][slot]['receiving_cell']
    result=original_spec(adapted,slot,retry)
    result['receiving_cell_id']=plan['slots'][slot]['receiving_cell_id']
    return result
