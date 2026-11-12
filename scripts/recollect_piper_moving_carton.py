"""Collect 50 carton demonstrations, then replace data and all derived inputs.

The old export remains available until raw-event validation passes. During cache
replacement the training readiness marker is absent. Ordinary failures restore
the old data/caches; a hard node failure leaves the journal and blocks training.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASK = 'moving_carton'
CACHE = ROOT / 'tmp/training_data_cache/piper_dynamic_v1'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def inventory(path):
    return {str(p.relative_to(path)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(path.rglob('*')) if p.is_file()}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / 'benchmark/src'))
    from piper_bench.collection.freeze import freeze_collection_source
    freeze_collection_source(ROOT, run / 'source')
    shutil.copy2(__file__, run / 'launcher.py')
    training = list((ROOT / 'OpenWAM/openwam').rglob('*.py'))
    training += list((ROOT / 'OpenWAM/scripts/train_ewm').glob('*.py'))
    training += list((ROOT / 'OpenWAM/configs').rglob('*.yaml'))
    write(run / 'launch.json', dict(
        task=TASK, episodes=50, acceptance_trials=20, acceptance_rate=.7,
        seed=20261021, spp=64, raw_hz=240, memory_hz=120, future_hz=30,
        memory_bins=36, future_steps=24, dino_batch_size=32,
        dataset=str(ROOT / 'datasets/piper_moving_carton'), cache=str(CACHE),
        previous_dataset=inventory(ROOT / 'datasets/piper_moving_carton'),
        source_sha256={str(p.relative_to(run)): digest(p)
                       for p in sorted((run / 'source').rglob('*')) if p.is_file()},
        training_sha256={str(p.relative_to(ROOT)): digest(p) for p in training},
        launcher_sha256=digest(run / 'launcher.py')))
    print('PREPARED', run, flush=True)


def checked(command, env=None):
    print('RUN', json.dumps([str(p) for p in command]), flush=True)
    subprocess.run([str(p) for p in command], cwd=ROOT, env=env, check=True)


def verify_training_source(launch):
    for path, expected in launch['training_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'training source changed since submission: {path}')


@contextmanager
def replacement(entries, backup, journal):
    """Move only enumerated targets; restore the ready marker last on failure."""
    backup.mkdir(parents=True, exist_ok=False)
    moved = []
    try:
        for key, target in entries:
            if target.is_symlink():
                raise ValueError(f'refusing symlink replacement: {target}')
            existed = target.exists()
            if existed:
                target.rename(backup / key)
            moved.append((key, target, existed))
            write(journal, dict(state='replacing', entries=[
                dict(key=k, target=str(t), existed=e) for k, t, e in moved]))
        yield
    except BaseException:
        # Once rollback begins, a second TERM must not interrupt restoration.
        previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            for key, target, existed in reversed(moved):
                if target.exists() or target.is_symlink():
                    # Retain expensive newly collected data for diagnosis/reuse.
                    target.rename(backup / ('failed_' + key))
                if existed:
                    (backup / key).rename(target)
            write(journal, dict(state='rolled_back'))
        finally:
            signal.signal(signal.SIGTERM, previous)
        raise
    else:
        write(journal, dict(state='committed', old_export=str(backup / 'dataset')))


def current_dino():
    sys.path.insert(0, str(ROOT / 'OpenWAM'))
    import torch
    from scripts.train_ewm.prepare_piper_current_dino import prepare_task
    from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    encoder = DinoV3VideoEncoder.from_pretrained(str(ROOT / 'pretrained/dinov3-vitl16'))
    encoder = encoder.to('cuda').eval().requires_grad_(False)
    prepare_task(ROOT, CACHE / 'current_dino_fixed32', TASK, encoder, torch.device('cuda'), 32)


def run_job(run):
    launch = json.loads((run / 'launch.json').read_text())
    for base, key in ((run, 'source_sha256'), (ROOT, 'training_sha256')):
        for path, expected in launch[key].items():
            if digest(base / path) != expected:
                raise ValueError(f'source changed since submission: {path}')
    if digest(run / 'launcher.py') != launch['launcher_sha256']:
        raise ValueError('launcher changed since submission')
    target = ROOT / 'datasets/piper_moving_carton'
    stage = ROOT / 'datasets' / ('.moving_carton_' + run.name)
    if inventory(target) != launch['previous_dataset']:
        raise ValueError('canonical data changed since submission')
    source = run / 'source/scripts'
    package_parent = run / 'source/benchmark/src'
    env = None
    if (package_parent / 'piper_bench').is_dir():
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(package_parent), str(source))),
                   PYTHONDONTWRITEBYTECODE='1')
    checked([os.environ['SIMPY'], source / 'collect_piper_tasks_evis.py',
             '--task', TASK, '--collection-protocol', 'piper_eight_dynamic_collection_v1',
             '--output', stage, '--audit-dir', run / 'audit', '--episodes', '50',
             '--max-attempts', '300', '--acceptance-trials', '20', '--acceptance-rate', '.7',
             '--seed', '20261021', '--pathtracing-spp', '64', '--save-raw-events',
             '--writer-python', os.environ['LRPY'], '--device', 'cpu', '--headless',
             '--enable_cameras', '--kit_args=--/plugins/carb.tasking.plugin/threadCount=4'], env=env)
    checked([os.environ['LRPY'], source / 'validate_piper_tasks_lerobot.py', stage, '--episodes', '50'], env=env)
    report = json.loads((stage / 'meta/piper_validation.json').read_text())
    if not (report['passed'] and report['episodes'] == 50 and report['raw_events_validated']):
        raise ValueError('formal raw-event export validation failed')
    verify_training_source(launch)
    # Locks used by existing preparation entry points. Training also checks the
    # ready markers and source fingerprints before opening the cached tensors.
    with (CACHE / '.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if inventory(target) != launch['previous_dataset']:
            raise ValueError('canonical data changed during collection')
        entries = [('complete.json', CACHE / 'complete.json')]
        entries += [(name, CACHE / name) for name in (
            'memory120_complete.json', 'cpu_validation.json', 'data_manifest.json',
            'shared_normalization.json', 'progress.json', 'preflight')]
        entries += [('dataset', target), ('task_cache', CACHE / 'piper_moving_carton'),
                    ('alignment', CACHE / 'event_action_alignment_v1/piper_moving_carton'),
                    ('current_dino', CACHE / 'current_dino_fixed32/piper_moving_carton')]
        backup = run / 'previous'
        with replacement(entries, backup, run / 'replacement.json'):
            stage.rename(target)
            # Release the preparation lock only for our existing CPU builder,
            # which takes that same lock. No ready marker exists at this point.
            fcntl.flock(lock, fcntl.LOCK_UN)
            train = ROOT / 'OpenWAM/scripts/train_ewm'
            checked([os.environ['OWPY'], train / 'prepare_piper_dynamic_data.py',
                     '--stage', 'cpu', '--cache-root', CACHE])
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checked([os.environ['OWPY'], run / 'launcher.py', '--current-dino'])
            checked([os.environ['OWPY'], train / 'prepare_piper_eight.py', '--root', ROOT,
                     '--cache-root', CACHE, '--output', CACHE / 'preflight', '--compact-events',
                     '--current-dino-cache-root', CACHE / 'current_dino_fixed32'])
            final = json.loads((CACHE / 'preflight/training_manifest.json').read_text())
            if not final['passed'] or final['episodes'] != 400:
                raise ValueError('eight-task cached loader preflight failed')
            verify_training_source(launch)
            write(CACHE / 'complete.json', dict(passed=True, training_started=False,
                tasks=list(final['tasks']), episodes=400, frames=final['frames'],
                preflight=str(CACHE / 'preflight/training_manifest.json'), replacement_run=str(run)))
        # Only obsolete, reproducible carton caches are removed after commit.
        # Preserve the small old source dataset and metadata for traceability.
        for key in ('task_cache', 'alignment', 'current_dino'):
            if (backup / key).is_dir():
                shutil.rmtree(backup / key)
        write(run / 'complete.json', dict(passed=True, dataset=str(target), cache=str(CACHE),
             episodes=50, raw_event_hz=240, future_hz=30,
             replaced_dino=['dino_current16'], old_dataset=str(backup / 'dataset')))
        print('CARTON_DATA_AND_CACHES_REPLACED', target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--current-dino', action='store_true')
    args = p.parse_args()
    # Frozen launcher lives below tmp; use the explicit deployment environment.
    global ROOT, CACHE
    ROOT = Path(os.environ.get('NEW_PROJECT_ROOT', ROOT)).resolve()
    CACHE = ROOT / 'tmp/training_data_cache/piper_dynamic_v1'
    if args.current_dino:
        current_dino()
        return
    if args.run_dir is None:
        p.error('--run-dir is required')
    run = args.run_dir.resolve()
    if args.prepare:
        prepare(run)
    else:
        def interrupted(signum, frame):
            raise InterruptedError(f'scheduler signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        with (run / '.job.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_job(run)


if __name__ == '__main__':
    main()
