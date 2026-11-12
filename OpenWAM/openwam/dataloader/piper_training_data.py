"""Training-only export validation and pooled statistics for eight dynamic tasks."""
import hashlib
import json
from pathlib import Path

import numpy as np

DYNAMIC_TASKS = ('conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube',
                 'tennis_rolling', 'tennis_falling', 'tennis_parabolic', 'tennis_container')
PROTOCOL = 'piper_eight_dynamic_collection_v1'
PROTOCOLS = (PROTOCOL, 'piper_eight_dynamic_narrow_v2', 'piper_container_spatial_v3')


def source_signatures(root, tasks=DYNAMIC_TASKS):
    root = Path(root)
    if (Path(root)/'datasets/.narrow_recollection_in_progress.json').exists():
        raise ValueError('seven-task replacement incomplete; training inputs are not ready')
    result = {}
    for task in tasks:
        directory = root/'datasets'/f'piper_{task}'
        for path in sorted((directory/'data').rglob('*.parquet')):
            stat = path.stat()
            result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
        for name in ('info.json', 'ewm_normalization_stats.json'):
            path = directory/'meta'/name
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def validate_exports(root, tasks=DYNAMIC_TASKS, expected_episodes=50):
    """Read the exported contract, not the simulator's current scene defaults."""
    import pyarrow.parquet as pq
    if (Path(root)/'datasets/.narrow_recollection_in_progress.json').exists():
        raise ValueError('seven-task replacement incomplete; training inputs are not ready')
    result = {}
    for task in tasks:
        directory = Path(root)/'datasets'/f'piper_{task}'
        read = lambda name: json.loads((directory/'meta'/name).read_text())
        info, collection, contract, validation = map(read, ('info.json', 'piper_collection_config.json',
            'piper_temporal_contract.json', 'piper_validation.json'))
        if not all((directory/name).is_file() for name in ('.collection_complete', '.export_complete')):
            raise ValueError(f'{task}: incomplete formal export')
        protocol_valid = (collection['version'] in PROTOCOLS or
                          (task == 'tennis_container' and collection['version'] == 'piper_container_spatial_v4') or
                          (task == 'moving_carton' and collection['version'] == 'piper_carton_lowwall_v4') or
                          (task in ('moving_carton', 'turntable_cups') and collection['version'] == 'piper_low_geometry_v5'))
        if (not protocol_valid or collection['task'] != task
                or collection['plan']['split'] != 'train' or not collection['standard_imaging']
                or collection['control_hz'] != 30 or collection['event_hz'] != 240
                or collection['physics_hz'] != (240 if task == 'tennis_container' else 120)):
            raise ValueError(f'{task}: expected current standard-ID training protocol')
        if (info['fps'] != 30 or info['total_episodes'] != expected_episodes
                or info['splits'].get('train') != f'0:{expected_episodes}'
                or not validation['passed'] or validation['task'] != task
                or validation['episodes'] != expected_episodes or validation['frames'] != info['total_frames']):
            raise ValueError(f'{task}: inconsistent validated training counts/split')
        if (contract['temporal_contract'] != 'aligned_event_frame_pre_action_v1'
                or contract['task_selector'] != task or contract['collection_protocol'] != collection['version']
                or contract['control_hz'] != 30 or contract['event_hz'] != 240):
            raise ValueError(f'{task}: incompatible observation/action/event clock')
        if read('piper_camera_calibration.json') != collection['camera']:
            raise ValueError(f'{task}: inconsistent camera calibration')
        if not (directory/'events/cam0').is_dir() or not (directory/'events/cam_wrist').is_dir():
            raise ValueError(f'{task}: missing original event archives')
        episodes = [r for p in sorted((directory/'meta/episodes').rglob('*.parquet'))
                    for r in pq.read_table(p).to_pylist()]
        episodes.sort(key=lambda r:r['episode_index'])
        if ([r['episode_index'] for r in episodes] != list(range(expected_episodes))
                or sum(r['length'] for r in episodes) != info['total_frames']
                or any(r['length'] <= 24 for r in episodes)):
            raise ValueError(f'{task}: incomplete episodes or no K24 training window')
        files = list((directory/'data').rglob('*.parquet'))
        if sum(pq.ParquetFile(p).metadata.num_rows for p in files) != info['total_frames']:
            raise ValueError(f'{task}: Parquet row counts disagree with metadata')
        inputs = {}
        for folder in ('meta', 'videos', 'events'):
            for p in sorted((directory/folder).rglob('*')):
                if p.is_file():
                    s = p.stat()
                    inputs[p.relative_to(directory).as_posix()] = [s.st_size,s.st_mtime_ns]
        result[task] = dict(episodes=len(episodes),frames=info['total_frames'],
            windows=sum(r['length']-24 for r in episodes),dataset_dir=str(directory.resolve()),
            input_signature=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest(),
            collection_sha256=hashlib.sha256((directory/'meta/piper_collection_config.json').read_bytes()).hexdigest())
    return result


def prepare_normalization(root, output, tasks=DYNAMIC_TASKS):
    """Pool actual train rows, never average quantiles or reuse an old run's stats."""
    import pyarrow.parquet as pq
    root, output = Path(root), Path(output)
    signatures = source_signatures(root, tasks)
    if output.exists():
        payload = json.loads(output.read_text())
        if (payload.get('source_split') != 'train' or payload.get('sources') != signatures
                or payload.get('tasks') != list(tasks)):
            raise ValueError('stale pooled normalization; choose a new cache directory')
        if payload.get('action', {}).get('type') != 'minmax':
            raise ValueError('legacy quantile action bounds clip expert commands; regenerate pooled normalization')
        return payload
    states, actions, reference = [], [], None
    for task in tasks:
        directory = root/'datasets'/f'piper_{task}'
        local = json.loads((directory/'meta/ewm_normalization_stats.json').read_text())
        if reference is None:
            reference = local
        if local['voxel'] != reference['voxel'] or local['rgb'] != reference['rgb']:
            raise ValueError('incompatible RGB/event normalization across tasks')
        info = json.loads((directory/'meta/info.json').read_text())
        if info.get('splits',{}).get('train') != f"0:{info['total_episodes']}":
            raise ValueError('pooled normalization requires training-only exports')
        count = 0
        for p in sorted((directory/'data').rglob('*.parquet')):
            table = pq.read_table(p,columns=['observation.state','action'])
            state = np.asarray(table['observation.state'].to_pylist(),dtype=np.float64)
            action = np.asarray(table['action'].to_pylist(),dtype=np.float64)
            if state.shape != (len(table),8) or action.shape != (len(table),7):
                raise ValueError(f'{task}: incompatible state/action shapes')
            states.append(state); actions.append(action); count += len(table)
        if count != info['total_frames']:
            raise ValueError(f'{task}: normalization input row count mismatch')
    state, action = np.concatenate(states), np.concatenate(actions)
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('nonfinite training state/action')
    q = np.quantile(action,[.01,.5,.99],axis=0)
    payload = dict(version=1,source_split='train',source_stats='eight_dynamic_pooled_raw_train_rows',
        tasks=list(tasks),sources=signatures,frames=len(state),rgb=reference['rgb'],voxel=reference['voxel'],
        proprio=dict(type='z-score',mean=state.mean(0).tolist(),std=state.std(0).tolist()),
        action=dict(type='minmax',min=action.min(0).tolist(),max=action.max(0).tolist(),
                    bounds_source='pooled_train_extrema',
                    q01=q[0].tolist(),q50=q[1].tolist(),q99=q[2].tolist(),degenerate_threshold=1e-6))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(output)
    return payload
