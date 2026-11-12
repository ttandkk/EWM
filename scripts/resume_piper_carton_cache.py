"""Finish the retained 220178 caches and promote the validated carton export."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal


def load_original(root, run):
    spec = importlib.util.spec_from_file_location('carton_collection', run / 'launcher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    module.CACHE = root / 'tmp/training_data_cache/piper_dynamic_v1'
    return module


def check_inputs(root, run, original):
    read = lambda p: json.loads(p.read_text())
    launch = read(run / 'launch.json')
    original.verify_training_source(launch)
    if original.digest(run / 'launcher.py') != launch['launcher_sha256']:
        raise ValueError('original frozen launcher changed')
    if read(run / 'replacement.json')['state'] != 'rolled_back':
        raise ValueError('expected the completed rollback of job 220178')
    if original.inventory(root / 'datasets/piper_moving_carton') != launch['previous_dataset']:
        raise ValueError('canonical data changed since original submission')
    previous = run / 'previous'
    validation = read(previous / 'failed_dataset/meta/piper_validation.json')
    if not (validation['passed'] and validation['raw_events_validated'] and validation['episodes'] == 50):
        raise ValueError('retained export did not pass full validation')
    for name in ('cpu_validation.json', 'memory120_complete.json'):
        if not read(previous / ('failed_' + name))['passed']:
            raise ValueError(f'retained cache validation failed: {name}')
    history = read(previous / 'failed_task_cache/dino_history8/manifest.json')
    if len(history['episodes']) != 50:
        raise ValueError('incomplete retained DINO history')
    current = previous / 'failed_current_dino/dino_current16'
    receipts = sorted(current.glob('episode_*.json'))
    for receipt in receipts:
        saved = read(receipt)
        ep = str(int(receipt.stem.split('_')[-1]))
        if not saved['verified'] or saved['entry']['source'] != history['episodes'][ep]['source']:
            raise ValueError(f'inconsistent completed DINO receipt: {receipt}')
    for path in previous.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'retained input must not be a symlink: {path}')
    return dict(episodes=50, frames=validation['frames'], current_dino_completed=len(receipts),
                current_dino_remaining=50-len(receipts), history_dino_completed=50)


def resume(root, run, original):
    cache = original.CACHE
    report = check_inputs(root, run, original)
    print('RETAINED_INPUTS_VERIFIED', json.dumps(report), flush=True)
    previous = run / 'previous'
    entries = [('complete.json', cache / 'complete.json')]
    entries += [(name, cache / name) for name in (
        'memory120_complete.json', 'cpu_validation.json', 'data_manifest.json',
        'shared_normalization.json', 'progress.json', 'preflight')]
    entries += [('dataset', root / 'datasets/piper_moving_carton'),
                ('task_cache', cache / 'piper_moving_carton'),
                ('alignment', cache / 'event_action_alignment_v1/piper_moving_carton'),
                ('current_dino', cache / 'current_dino_fixed32/piper_moving_carton')]
    backup = run / ('resume_previous_' + os.environ['SLURM_JOB_ID'])
    with (run / '.job.lock').open('a') as job_lock, (cache / '.prepare.lock').open('a') as lock:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_inputs(root, run, original)
        with original.replacement(entries, backup, run / 'resume_replacement.json'):
            for key, target in entries:
                retained = previous / ('failed_' + key)
                if retained.exists():
                    retained.rename(target)
            # Recheck raw-derived 120 Hz bins, causal future supervision, pooled
            # stats and real windows using the existing CPU preparation checks.
            train = root / 'OpenWAM/scripts/train_ewm'
            fcntl.flock(lock, fcntl.LOCK_UN)
            original.checked([os.environ['OWPY'], train / 'prepare_piper_dynamic_data.py',
                              '--stage', 'cpu', '--cache-root', cache])
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Existing fixed-32 receipts skip the 48 verified episodes. The
            # unfinished partial file has no receipt and is overwritten safely.
            original.checked([os.environ['OWPY'], run / 'launcher.py', '--current-dino'])
            original.checked([os.environ['OWPY'], train / 'prepare_piper_eight.py',
                              '--root', root, '--cache-root', cache, '--output', cache / 'preflight',
                              '--compact-events', '--current-dino-cache-root', cache / 'current_dino_fixed32'])
            final = json.loads((cache / 'preflight/training_manifest.json').read_text())
            if not final['passed'] or final['episodes'] != 400:
                raise ValueError('eight-task training loader preflight failed')
            original.verify_training_source(json.loads((run / 'launch.json').read_text()))
            original.write(cache / 'complete.json', dict(passed=True, training_started=False,
                tasks=list(final['tasks']), episodes=400, frames=final['frames'],
                preflight=str(cache / 'preflight/training_manifest.json'), replacement_run=str(run)))
        # This is the old reproducible cache, never the retained new export.
        for key in ('task_cache', 'alignment', 'current_dino'):
            if (backup / key).is_dir():
                shutil.rmtree(backup / key)
        original.write(run / 'complete.json', dict(passed=True, dataset=str(root / 'datasets/piper_moving_carton'),
            cache=str(cache), episodes=50, raw_event_hz=240, memory_hz=120, future_hz=30,
            replaced_dino=['dino_history8', 'dino_current16'], old_dataset=str(backup / 'dataset'),
            resume_job=os.environ['SLURM_JOB_ID']))
        print('CARTON_RESUME_AND_REPLACEMENT_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    root = Path(os.environ['NEW_PROJECT_ROOT']).resolve()
    run = args.run_dir.resolve()
    original = load_original(root, run)
    if args.check_only:
        print(json.dumps(check_inputs(root, run, original), indent=2))
        return
    def interrupted(signum, frame):
        raise InterruptedError(f'scheduler signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    resume(root, run, original)


if __name__ == '__main__':
    main()
