"""Recollect seven narrow-ID tasks, publish validated exports, invalidate old caches.

Conveyor source data is never modified. Use --prepare before sbatch so source and
commands are reviewable; the queued job runs the frozen snapshot. No training.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(os.environ.get('NEW_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
TASKS = ('moving_carton','turntable_cups','rotating_tube','tennis_rolling',
         'tennis_falling','tennis_parabolic','tennis_container')
PROTOCOL = 'piper_eight_dynamic_narrow_v2'
CACHE_NAMES = ('piper_dynamic_v1','piper_dynamic_rgb_v1')


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def load_launcher(run, preparing=False):
    path=ROOT/'scripts/collect_piper_eight_dynamic.py' if preparing else run/'source/collect_piper_eight_dynamic.py'
    spec=importlib.util.spec_from_file_location('serial_collect',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=ROOT;module.TASKS=TASKS;module.PROTOCOL=PROTOCOL
    return module


def launcher_args(module,run):
    return module.parser().parse_args(['--run-dir',str(run),'--dataset-root',str(run/'staging'),
        '--episodes','50','--acceptance-trials','20','--max-attempts','300','--seed','20261025'])


def fingerprint(path):
    return {str(p.relative_to(path)):[p.stat().st_size,p.stat().st_mtime_ns]
            for p in sorted(path.rglob('*')) if p.is_file()}


def prepare(run):
    module=load_launcher(run,True);args=launcher_args(module,run)
    module.prepare(args)
    shutil.copy2(__file__,run/'launcher.py')
    train=ROOT/'OpenWAM/scripts/train_ewm'
    files=list((ROOT/'OpenWAM/openwam').rglob('*.py'))+list(train.glob('*.py'))
    for source in files:
        target=run/'cache_source'/source.relative_to(ROOT/'OpenWAM')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    # prepare_piper_eight resolves Hydra presets relative to the source root.
    shutil.copytree(ROOT/'OpenWAM/configs',run/'cache_source/configs')
    write(run/'cache_source_sha256.json',{str(p.relative_to(run/'cache_source')):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (run/'cache_source').rglob('*') if p.is_file()})
    write(run/'replacement_plan.json',dict(protocol=PROTOCOL,tasks=list(TASKS),episodes_each=50,
        acceptance='at least 14/20 before each task',conveyor_before=fingerprint(ROOT/'datasets/piper_conveyor'),
        replace_only_after_validation=True,cache_roots=[str(ROOT/'tmp/training_data_cache'/n) for n in CACHE_NAMES],
        raw_event_hz=240,event_memory_hz=120,future_event_hz=30,current_dino_batch_size=32))
    print('PREPARED',run,flush=True)


def remove_directory(path, allowed_parent):
    if path.parent.resolve()!=allowed_parent.resolve() or path.is_symlink():
        raise ValueError(f'unsafe deletion target: {path}')
    if path.exists():shutil.rmtree(path)


def invalidate_cache(run):
    if (run/'cache_cleanup_complete.json').exists():return
    deleted=[]
    parent=ROOT/'tmp/training_data_cache'
    for name in CACHE_NAMES:
        target=parent/name
        # Fixed, task-specific allowlist; checkpoints and source data excluded.
        remove_directory(target,parent);deleted.append(str(target))
    write(run/'cache_cleanup_complete.json',dict(deleted=deleted))


def collect(run):
    module=load_launcher(run);module.verify_source(run)
    args=launcher_args(module,run);plan=json.loads((run/'replacement_plan.json').read_text())
    conveyor=ROOT/'datasets/piper_conveyor'
    if fingerprint(conveyor)!=plan['conveyor_before']:raise ValueError('conveyor changed since submission')
    pending=ROOT/'datasets/.narrow_recollection_in_progress.json'
    with (ROOT/'datasets/.narrow_recollection.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if pending.exists() and json.loads(pending.read_text())['run']!=str(run):
            raise ValueError('another replacement is incomplete')
        write(pending,dict(run=str(run),tasks=list(TASKS)))
        summary_path=run/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        env=(module.source_environment(run) if hasattr(module, 'source_environment')
             else dict(os.environ,PYTHONPATH=str(run/'source/scripts')))
        failed=False
        for task in TASKS:
            target=ROOT/'datasets'/f'piper_{task}'
            if summary.get(task,{}).get('status')=='published':
                if not (target/'.export_complete').exists():raise ValueError(f'{task}: published data missing')
                remove_directory(run/'retired'/f'piper_{task}',run/'retired')
                continue
            print('BEGIN',task,'acceptance >=14/20; collect 50 successes',flush=True)
            stage=run/'staging'/f'piper_{task}'
            backup=run/'retired'/f'piper_{task}'
            # Recover a process interruption between the two directory renames.
            if backup.exists() and not target.exists():backup.rename(target)
            if backup.exists():
                cfg=json.loads((target/'meta/piper_collection_config.json').read_text())
                if cfg['version']!=PROTOCOL or not (target/'.export_complete').exists():
                    raise ValueError('ambiguous interrupted publication')
                remove_directory(backup,run/'retired')
                summary[task]=dict(status='published',dataset=str(target));write(summary_path,summary);continue
            command,validate=module.commands(args,task,resume=(run/'audit'/task/'collection_config.json').exists())
            code=0 if (stage/'.export_complete').exists() else module.execute(command,run/'logs'/f'{task}.collect.log',env)
            gate=module.acceptance(args,task)
            if code or not gate['passed']:
                summary[task]=dict(status='collection_failed',code=code,acceptance=gate);failed=True;write(summary_path,summary);continue
            code=module.execute(validate,run/'logs'/f'{task}.validate.log',env)
            report=json.loads((stage/'meta/piper_validation.json').read_text())
            if code or not (report['passed'] and report['episodes']==50 and report['raw_events_validated']):
                raise ValueError(f'{task}: export/raw-event validation failed')
            invalidate_cache(run)
            backup.parent.mkdir(exist_ok=True)
            if target.is_symlink() or stage.is_symlink():raise ValueError('refuse symlink dataset swap')
            if target.exists():target.rename(backup)
            try:stage.rename(target)
            except BaseException:
                if backup.exists():backup.rename(target)
                raise
            # Old data are removed only after the validated replacement is in place.
            summary[task]=dict(status='published',dataset=str(target),frames=report['frames'],acceptance=gate)
            write(summary_path,summary)
            remove_directory(backup,run/'retired')
            print('PUBLISHED',task,report['frames'],flush=True)
        if failed:raise RuntimeError('some tasks failed; cache dependency must remain blocked')
        if fingerprint(conveyor)!=plan['conveyor_before']:raise ValueError('conveyor was modified')
        pending.unlink()
        write(run/'collection_complete.json',dict(passed=True,tasks=list(TASKS),episodes=350,conveyor_preserved=True))


def cache(run):
    if not json.loads((run/'collection_complete.json').read_text())['passed']:
        raise ValueError('collection not complete')
    if (ROOT/'datasets/.narrow_recollection_in_progress.json').exists():raise ValueError('replacement incomplete')
    for name,expected in json.loads((run/'cache_source_sha256.json').read_text()).items():
        if hashlib.sha256((run/'cache_source'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen cache source changed')
    source=ROOT/'OpenWAM';train=source/'scripts/train_ewm'
    env=dict(os.environ,PYTHONPATH=str(source))
    cache_root=ROOT/'tmp/training_data_cache/piper_dynamic_v1'
    def checked(script,*args):
        for name,expected in json.loads((run/'cache_source_sha256.json').read_text()).items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('cache source changed since submission; re-prepare before proceeding')
        subprocess.run([os.environ['OWPY'],str(train/script),*map(str,args)],env=env,cwd=ROOT,check=True)
    checked('prepare_piper_dynamic_data.py','--root',ROOT,'--cache-root',cache_root,'--stage','all','--batch-size','32','--defer-ready-marker')
    # Publish the readiness marker only after full-grid current DINO and both loaders pass.
    (cache_root/'complete.json').unlink(missing_ok=True)
    checked('prepare_piper_current_dino.py','--root',ROOT,'--cache-root',cache_root/'current_dino_fixed32','--batch-size','32')
    checked('prepare_piper_eight.py','--root',ROOT,'--cache-root',cache_root,'--output',cache_root/'preflight',
            '--compact-events','--current-dino-cache-root',cache_root/'current_dino_fixed32')
    checked('prepare_piper_action_eight.py','--root',ROOT,'--output',cache_root/'action_preflight')
    write(cache_root/'complete.json',dict(passed=True,episodes=400,tasks=['conveyor',*TASKS],training_started=False,
        collection_run=str(run),raw_event_hz=240,memory_hz=120,future_hz=30,current_dino_dtype='bf16',encoding_batch_size=32))
    write(run/'cache_complete.json',dict(passed=True,cache=str(cache_root)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','collect','cache'],required=True)
    a=p.parse_args();run=a.run_dir.resolve()
    if a.stage=='prepare':prepare(run)
    else:
        if not os.environ.get('SLURM_JOB_ID'):raise ValueError('GPU stage requires Slurm allocation')
        (collect if a.stage=='collect' else cache)(run)


if __name__=='__main__':main()
