"""Prepare current EWM inputs only: pooled stats, lossless events and DINO (no privileged geometry).

No optimizer, policy checkpoint, W&B run or training submission is started.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT/'OpenWAM'))
from openwam.dataloader.piper_training_data import DYNAMIC_TASKS, prepare_normalization, source_signatures, validate_exports


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def verify_cpu(root, cache, stats):
    import torch
    from openwam.dataloader.ewm_dataset import EWMDataset
    report = {}
    for task in DYNAMIC_TASKS:
        d = EWMDataset(root/'datasets'/f'piper_{task}',chunk_size=24,
            voxel_bins=1,event_camera_mode='dual',gpu_voxelization=False,normalization_stats_path=stats,
            event_frame_cache_dir=cache/f'piper_{task}/event_frames')
        checks = []
        # Every episode's first and final valid window: all 400 trajectories.
        for i,row in d.episodes.iterrows():
            for index in (int(d._cumulative[i]),int(d._cumulative[i+1])-1):
                sample = d[index]
                assert sample['obs_video'].shape==(2,3,256,256)
                assert sample['obs_voxel'].shape==(2,2,256,256)
                assert sample['action'].shape==(24,7)
                assert sample['future_voxel'].shape==(24,2,2,256,256)
                for key,value in sample.items():
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise ValueError(f'{task}/{index}: nonfinite {key}')
                anchor = d.first_anchor + (index - int(d._cumulative[i])) * d.window_stride
                if anchor == 0:
                    assert not sample['obs_voxel'].any()
                assert sample['prompt'].strip()
                checks.append(dict(episode=int(row['episode_index']),anchor=anchor))
        report[task]=dict(episodes=len(d.episodes),windows=len(d),checked_windows=checks,
                          prompts=list(d.prompts.values()),rgb_backend='raw current RGB, before GPU DINO cache')
        print('CPU_WINDOWS_PASSED',task,len(checks),flush=True)
    write_json(cache/'cpu_validation.json',dict(passed=True,tasks=report,
        checked_windows=sum(len(v['checked_windows']) for v in report.values()),
        event_window='(t_(k-1),t_k]',action_window='k:k+24',future_event_window='k+1:k+25',
        format='ewm_current_inputs_v1', raw_events_modified=False))


def run(args):
    root=args.root.resolve();cache=(args.cache_root or root/'tmp/training_data_cache/piper_dynamic_v1').resolve()
    if cache==root/'datasets' or (root/'datasets') in cache.parents:
        raise ValueError('derived caches must be outside source datasets')
    cache.mkdir(parents=True,exist_ok=True)
    with (cache/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        exports=validate_exports(root)
        source=dict(version='piper_eight_dynamic_training_inputs_v1',tasks=exports,sources=source_signatures(root))
        manifest=cache/'data_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())!=source:
            raise ValueError('source data changed; choose a new cache-root')
        write_json(manifest,source)
        stats=cache/'shared_normalization.json'
        prepare_normalization(root,stats)
        print('DATA_AUDIT_PASSED',sum(x['episodes'] for x in exports.values()),sum(x['frames'] for x in exports.values()),flush=True)
        if args.stage=='audit':return
        scripts=root/'OpenWAM/scripts/train_ewm'
        if args.stage in ('cpu','all'):
            from scripts.train_ewm.build_ewm_event_frame_cache import build
            for task in DYNAMIC_TASKS:
                dataset=root/'datasets'/f'piper_{task}'
                target=cache/f'piper_{task}/event_frames'
                if not (target/'manifest.json').exists():
                    build(dataset,target)
                write_json(cache/'progress.json',dict(stage='cpu',task=task))
            cpu_report=cache/'cpu_validation.json'
            if (not cpu_report.exists() or not json.loads(cpu_report.read_text()).get('passed')
                    or json.loads(cpu_report.read_text()).get('format')!='ewm_current_inputs_v1'):
                verify_cpu(root,cache,stats)
        if args.stage in ('dino','all'):
            if not json.loads((cache/'cpu_validation.json').read_text())['passed']:
                raise ValueError('CPU preparation must pass before DINO caching')
            current_cache = cache / 'current_dino_fixed32'
            subprocess.run([sys.executable, str(scripts/'prepare_piper_current_dino.py'),
                '--root', str(root), '--cache-root', str(current_cache),
                '--batch-size', str(args.batch_size)], check=True)
            subprocess.run([sys.executable,str(scripts/'prepare_piper_eight.py'),'--root',str(root),
                '--output',str(cache/'preflight'),'--cache-root',str(cache),
                '--current-dino-cache-root', str(current_cache)],check=True)
        if source['sources']!=source_signatures(root) or source['tasks']!=validate_exports(root):
            raise ValueError('source changed during preparation')
        if args.stage in ('dino','all') and not getattr(args, 'defer_ready_marker', False):
            write_json(cache/'complete.json',dict(passed=True,training_started=False,
                tasks=list(DYNAMIC_TASKS),episodes=sum(x['episodes'] for x in exports.values()),
                frames=sum(x['frames'] for x in exports.values()),
                preflight=str(cache/'preflight/training_manifest.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=PROJECT)
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--stage',choices=('audit','cpu','dino','all'),default='cpu')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--defer-ready-marker',action='store_true',help='Caller publishes readiness after additional current-DINO checks')
    args=p.parse_args()
    if args.batch_size<1:p.error('positive batch size required')
    run(args)


if __name__=='__main__':main()
