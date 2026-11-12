"""Container-only replacement: delete old data/cache first, collect, then rebuild.

Frozen launchers reuse the existing 70% expert gate and full eight-task cache
verification. Other seven datasets and their per-task caches are preserved.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(os.environ.get('NEW_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
TASK = 'tennis_container'
PROTOCOL = 'piper_container_spatial_v4'


def helper(run, preparing=False):
    path = ROOT/'scripts/recollect_piper_narrow.py' if preparing else run/'replacement_helper.py'
    spec = importlib.util.spec_from_file_location('replacement_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def protected_fingerprints(h):
    tasks = ('conveyor', *[t for t in h.TASKS if t != TASK])
    paths = [ROOT/'datasets'/f'piper_{t}' for t in tasks]
    for name in h.CACHE_NAMES:
        cache = ROOT/'tmp/training_data_cache'/name
        for t in tasks:
            paths.extend([cache/f'piper_{t}', cache/'current_dino_fixed32'/f'piper_{t}'])
    return {str(p): h.fingerprint(p) for p in paths if p.exists()}


def prepare(run):
    h = helper(run, True)
    preserved = protected_fingerprints(h)
    h.TASKS = (TASK,)
    h.PROTOCOL = PROTOCOL
    h.prepare(run)
    shutil.copy2(ROOT/'scripts/recollect_piper_narrow.py', run/'replacement_helper.py')
    shutil.copy2(__file__, run/'launcher.py')
    h.write(run/'protected_fingerprints.json', preserved)
    plan = json.loads((run/'replacement_plan.json').read_text())
    plan.update(delete_old_before_collection=True, replace_only_after_validation=True,
        cache_roots=None, deletion_allowlist=[str(p) for p in deletion_targets()],
        protected_tasks=['conveyor', 'moving_carton', 'turntable_cups', 'rotating_tube',
                         'tennis_rolling', 'tennis_falling', 'tennis_parabolic'])
    h.write(run/'replacement_plan.json', plan)


def deletion_targets():
    paths = [ROOT/'datasets'/f'piper_{TASK}']
    for name in ('piper_dynamic_v1', 'piper_dynamic_rgb_v1'):
        cache = ROOT/'tmp/training_data_cache'/name
        paths += [cache/f'piper_{TASK}', cache/'current_dino_fixed32'/f'piper_{TASK}']
        paths += [cache/n for n in ('complete.json','data_manifest.json','shared_normalization.json',
            'cpu_validation.json','memory120_complete.json','progress.json','preflight','action_preflight')]
    return paths


def preserved(run, h):
    for path, expected in json.loads((run/'protected_fingerprints.json').read_text()).items():
        if h.fingerprint(Path(path)) != expected:
            raise ValueError(f'protected data/cache changed: {path}')


def clean(run):
    h = helper(run)
    if (run/'cache_cleanup_complete.json').exists():
        return
    preserved(run, h)
    pending = ROOT/'datasets/.narrow_recollection_in_progress.json'
    if pending.exists() and json.loads(pending.read_text())['run'] != str(run):
        raise ValueError('another replacement is incomplete')
    # Validate the complete allowlist before deleting anything.
    paths = deletion_targets()
    for p in paths:
        if p.is_symlink() or p.resolve() != p:
            raise ValueError(f'refuse symlink deletion: {p}')
    h.write(pending, dict(run=str(run), tasks=[TASK], protocol=PROTOCOL))
    archive = run/'retired_metadata'
    archive.mkdir(exist_ok=True)
    oldmeta = ROOT/'datasets'/f'piper_{TASK}'/'meta'
    if oldmeta.exists():
        shutil.copytree(oldmeta, archive/'dataset_meta', dirs_exist_ok=True)
    deleted = []
    for p in paths:
        if not p.exists():
            continue
        size = p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        if p.is_file():
            shutil.copy2(p, archive/(p.parent.name+'_'+p.name))
            p.unlink()
        else:
            shutil.rmtree(p)
        deleted.append(dict(path=str(p), bytes=size))
    preserved(run, h)
    h.write(run/'cache_cleanup_complete.json', dict(deleted=deleted, bytes=sum(x['bytes'] for x in deleted)))


def collect(run):
    h = helper(run)
    if not (run/'cache_cleanup_complete.json').exists():
        raise ValueError('delete old container data/cache before collection')
    preserved(run, h)
    h.TASKS = (TASK,)
    h.PROTOCOL = PROTOCOL
    m = h.load_launcher(run)
    m.verify_source(run)
    args = h.launcher_args(m, run)
    args.resume = True
    target = ROOT/'datasets'/f'piper_{TASK}'
    pending = ROOT/'datasets/.narrow_recollection_in_progress.json'
    if pending.exists() and json.loads(pending.read_text())['run'] != str(run):
        raise ValueError('another replacement is incomplete')
    if not target.exists():
        if m.run_tasks(args):
            raise RuntimeError('container acceptance/collection/export validation failed')
        (run/'staging'/f'piper_{TASK}').rename(target)
    # Also validate an interrupted publication before releasing the cache dependency.
    cfg = json.loads((target/'meta/piper_collection_config.json').read_text())
    valid = json.loads((target/'meta/piper_validation.json').read_text())
    if cfg['version'] != PROTOCOL or not valid['passed'] or valid['episodes'] != 50 or not valid['raw_events_validated']:
        raise ValueError('invalid container publication')
    preserved(run, h)
    pending.unlink(missing_ok=True)
    h.write(run/'collection_complete.json', dict(passed=True, tasks=[TASK], episodes=50, other_seven_preserved=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--stage', choices=('prepare','clean','collect','cache'), required=True)
    a = p.parse_args()
    run = a.run_dir.resolve()
    if a.stage == 'prepare':
        prepare(run)
        return
    if a.stage in ('collect','cache') and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('GPU stages require a Slurm allocation')
    with (ROOT/'datasets/.narrow_recollection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'clean':
            clean(run)
        elif a.stage == 'collect':
            collect(run)
        else:
            # Keep the helper's original seven-task list for the full eight-task
            # readiness marker, while only rebuilding invalid/missing caches.
            helper(run).cache(run)


if __name__ == '__main__':
    main()
