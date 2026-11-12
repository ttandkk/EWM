"""Validate cached RGB inputs before submitting/starting action-only training."""
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
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    sys.path.insert(0, str(root / 'OpenWAM'))
    import torch
    import numpy as np
    import pyarrow.parquet as pq
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from openwam.dataloader.registry import build_dataset
    from openwam.dataloader.ewm_action_only import PiperActionOnlyDataset
    from openwam.dataloader.ewm_dataset import EWMDataset

    output.mkdir(parents=True, exist_ok=True)
    source = root / 'tmp/training_data_cache/piper_dynamic_v1/shared_normalization.json'
    stats = output / 'shared_normalization.json'
    data = source.read_bytes()
    if stats.exists() and stats.read_bytes() != data:
        raise ValueError('refusing to replace different normalization')
    stats.write_bytes(data)
    os.environ.update(NEW_PROJECT_ROOT=str(root), EWM_ACTION_OUTPUT=str(output), EWM_ACTION_STATS=str(stats))
    with initialize_config_dir(version_base=None, config_dir=str(root / 'OpenWAM/configs')):
        cfg = compose(config_name='train_ewm_eight_action_only')
    OmegaConf.resolve(cfg)
    horizon = int(cfg.dataloader.chunk_size)
    # Current RGB uses the independent per-frame cache; never read events.
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected RGB decoding or event I/O in action-only preflight')
    PiperActionOnlyDataset._rgb = forbidden
    EWMDataset._direct_event_frames = forbidden
    dataset = build_dataset(cfg.dataloader)
    report = {}
    max_roundtrip_error = 0.0
    checked_actions = 0
    checked_terminals = 0
    for task_id, (task, child) in enumerate(zip(dataset.tasks, dataset.datasets)):
        for index in (0, min(100, len(child) - 1), len(child) - 1):
            sample = dataset[task_id, index]
            expected = {'current_dino_features', 'proprio', 'action', 'prompt', 'action_valid_mask', 'task_name'}
            assert set(sample) == expected
            assert sample['current_dino_features'].shape == (2, 1024, 16, 16)
            assert sample['current_dino_features'].dtype == torch.bfloat16
            assert sample['action'].shape == (horizon, 7) and sample['prompt'].strip()
            for value in sample.values():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all()
            assert not sample['action'][~sample['action_valid_mask']].any()
        assert sample['action_valid_mask'].sum() == 1
        assert len(child) == int(child.episodes['length'].sum())
        task_error = 0.0
        clipped = 0
        for path in sorted((child.dataset_dir / 'data').rglob('*.parquet')):
            raw = torch.from_numpy(np.asarray(pq.read_table(path, columns=['action'])['action'].to_pylist(), dtype=np.float32))
            outside = (raw < torch.as_tensor(child.normalizer.action_low)) | (raw > torch.as_tensor(child.normalizer.action_high))
            clipped += int(outside.any(-1).sum())
            restored = child.normalizer.denormalize_action(child.normalizer.normalize_action(raw))
            task_error = max(task_error, float((restored - raw).abs().max()))
            checked_actions += len(raw)
        assert clipped == 0 and task_error < 1e-6, (task, clipped, task_error)
        max_roundtrip_error = max(max_roundtrip_error, task_error)
        for endpoint in child._cumulative[1:]:
            terminal = child[int(endpoint) - 1]
            assert terminal['action_valid_mask'].tolist() == [True] + [False] * (horizon - 1)
            assert not terminal['action'][1:].any()
            checked_terminals += 1
        report[task] = dict(episodes=len(child.episodes), windows=len(child), prompts=list(child.prompts.values()),
                            clipped_action_rows=clipped, max_action_roundtrip_error=task_error)
        print('VALIDATED', task, report[task], flush=True)
    sampler = iter(dataset.make_batch_sampler(32, num_processes=2, seed=42))
    for _ in range(100):
        rank0, rank1 = next(sampler), next(sampler)
        assert not set(rank0) & set(rank1)
        assert Counter(t for t, _ in rank0 + rank1) == {i: 8 for i in range(8)}
    OmegaConf.save(cfg, output / 'preflight_config.yaml')
    (output / 'training_manifest.json').write_text(json.dumps(dict(
        passed=True, mode='action_only', event_prediction=False,
        tasks=report, windows=len(dataset), normalization_sha256=hashlib.sha256(data).hexdigest(),
        pad_tail=True, chunk_size=horizon, real_samples_checked=24, balanced_ddp_batches_checked=100,
        checked_action_rows=checked_actions, checked_episode_terminals=checked_terminals,
        max_action_roundtrip_error=max_roundtrip_error,
        visual_context_tokens=512,
        current_features='existing per-frame BF16 DINO at current rows; no event reads'), indent=2))
    print('ACTION_ONLY_PREFLIGHT_PASSED', len(dataset), flush=True)


if __name__ == '__main__':
    main()
