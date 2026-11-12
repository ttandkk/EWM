"""Validate cached eight-task inputs and persist the formal training manifest."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stats-source', type=Path, help='defaults to current dynamic cache statistics')
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--compact-events', action='store_true')
    parser.add_argument('--current-dino-cache-root', type=Path)
    parser.add_argument('--event-temporal-downsample', type=int, choices=(1, 2, 3),
                        help='defaults to the selected training preset')
    parser.add_argument('--config-name', default='train_ewm_eight',
                        choices=('train_ewm_eight', 'train_ewm_eight_rgb_action'))
    args = parser.parse_args()
    if args.config_name == 'train_ewm_eight_rgb_action' and args.compact_events:
        parser.error('RGB-action does not load compact events')
    sys.path.insert(0, str(args.root / 'OpenWAM'))
    import torch
    from accelerate.data_loader import BatchSamplerShard
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from openwam.dataloader.registry import build_dataset
    from openwam.model.architectures.temporal_sampling import TemporalSampling

    from openwam.dataloader.piper_training_data import DYNAMIC_TASKS, validate_exports
    args.root = args.root.resolve()
    exports = validate_exports(args.root)
    cache = (args.cache_root or args.root/'tmp/training_data_cache/piper_dynamic_v1').resolve()
    source = args.stats_source or cache/'shared_normalization.json'
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    stats = args.output / 'shared_normalization.json'
    data = source.read_bytes()
    if json.loads(data).get('action', {}).get('type') != 'minmax':
        raise ValueError('new eight-task training requires full-range minmax action statistics')
    if stats.exists() and stats.read_bytes() != data:
        raise ValueError('refusing to replace different pooled statistics')
    stats.write_bytes(data)
    os.environ.update(NEW_PROJECT_ROOT=str(args.root), EWM_EIGHT_STATS=str(stats),
                      EWM_EIGHT_OUTPUT=str(args.output.resolve()), EWM_EIGHT_RUN_NAME='eight-task-preflight', EWM_EIGHT_CACHE=str(cache))
    with initialize_config_dir(config_dir=str(args.root / 'OpenWAM/configs'), version_base=None):
        cfg = compose(config_name=args.config_name,
                      overrides=([f'model.event_temporal_downsample={args.event_temporal_downsample}']
                                 if args.event_temporal_downsample is not None else []))
    OmegaConf.resolve(cfg)
    horizon = int(cfg.dataloader.chunk_size)
    event_horizon = cfg.model.get('event_horizon')
    event_horizon = horizon if event_horizon is None else int(event_horizon)
    loader_events = cfg.dataloader.get('event_horizon')
    assert (horizon if loader_events is None else loader_events) == event_horizon
    temporal = TemporalSampling(stride=int(cfg.model.event_temporal_downsample),
                                compression=int(cfg.model.event_temporal_compression))
    future_slices = temporal.future_slices(event_horizon)
    cfg.dataloader.compact_events = args.compact_events
    cfg.dataloader.current_dino_cache_root = (str(args.current_dino_cache_root.resolve())
                                             if args.current_dino_cache_root else None)
    dataset = build_dataset(cfg.dataloader)
    assert dataset.tasks == DYNAMIC_TASKS
    tasks = {}
    for task_id, (task, child) in enumerate(zip(dataset.tasks, dataset.datasets)):
        for window in (0, min(100, len(child)-1), len(child) - 1):
            sample = dataset[task_id, window]
            assert sample['task_name'] == task and sample['prompt'].strip()
            assert sample['action'].shape == (horizon, 7)
            if cfg.model.get('training_mode', 'joint') == 'rgb_action':
                assert sample['future_rgb'].shape == (event_horizon, 2, 3, 256, 256)
                assert sample['obs_rgb'].shape == (2, 3, 256, 256)
                assert sample['future_rgb'].dtype == torch.uint8
            else:
                assert sample['future_voxel'].shape == (event_horizon, 2, 2, 256, 256)
            if args.compact_events:
                keys = ('obs_voxel', 'future_voxel')
                assert all(sample[k].dtype == torch.uint8 for k in keys)
            if args.current_dino_cache_root:
                assert 'obs_video' not in sample
                assert sample['current_dino_features'].shape == (2,1024,16,16)
                assert sample['current_dino_features'].dtype == torch.bfloat16
            for key, value in sample.items():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all(), (task, window, key)
        if child.pad_tail:
            terminal = child[len(child)-1]
            assert terminal['action_valid_mask'].sum() == 1
            assert not terminal['future_valid_mask'].any()
            assert len(child) == int(child.episodes['length'].sum())
        info = json.loads((child.dataset_dir / 'meta/info.json').read_text())
        tasks[task] = dict(episodes=info['total_episodes'], frames=info['total_frames'],
                           windows=len(child), dataset_dir=str(child.dataset_dir),
                           prompts=list(child.prompts.values()),
                           event_cache=str(child.event_frame_cache_dir),
                           current_dino_cache=(str(child.current_dino_cache.directory)
                                               if child.current_dino_cache else None))
        # Do not carry diagnostic-opened maps into worker spawn during later runs.
        print('TASK_PREFLIGHT_PASSED', task, tasks[task]['windows'], flush=True)
    sampler = dataset.make_batch_sampler(32, num_processes=2, seed=42)
    from itertools import islice
    ranks = [list(islice(BatchSamplerShard(sampler, 2, r, split_batches=False), 100)) for r in (0, 1)]
    for left, right in zip(*ranks):
        assert not set(left) & set(right)
        for batch in (left, right):
            assert Counter(t for t, _ in batch) == dict.fromkeys(range(8), 4)
    report = dict(passed=True, tasks=tasks, source_exports=exports, episodes=sum(t['episodes'] for t in tasks.values()),
                  frames=sum(t['frames'] for t in tasks.values()), windows=len(dataset),
                  sampling='uniform tasks and windows; disjoint ranks within each step',
                  pad_tail=bool(cfg.dataloader.pad_tail), chunk_size=horizon,
                  event_horizon=event_horizon, future_latent_slices=future_slices,
                  temporal_masks='action_valid_mask: recorded actions; future_valid_mask: available k+1 frames; only complete latent groups',
                  batch_per_gpu=32, global_batch=64, world_size=2,
                  workers_per_rank=4, prefetch_factor=2, persistent_workers=True,
                  optimizer_steps=80000, save_steps=20000, trainable_initialization='from_scratch',
                  frozen_encoders=['DINOv3', 'T5', 'LeanVAE'], config_name=args.config_name,
                  training_mode=cfg.model.get('training_mode', 'joint'), compact_events=args.compact_events,
                  current_dino_cache_root=cfg.dataloader.current_dino_cache_root,
                  visual_context_tokens=512,
                  event_temporal_downsample=cfg.model.event_temporal_downsample,
                  normalization_sha256=hashlib.sha256(data).hexdigest(),
                  validation='24 real windows finite; 100 balanced disjoint rank batches')
    (args.output / 'training_manifest.json').write_text(json.dumps(report, indent=2))
    OmegaConf.save(cfg, args.output / 'preflight_config.yaml')
    print('EIGHT_TASK_PREFLIGHT_PASSED', json.dumps({k: v for k, v in report.items() if k != 'tasks'}), flush=True)


if __name__ == '__main__':
    main()
