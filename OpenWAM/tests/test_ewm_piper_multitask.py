"""Balanced rank sampling, epoch replay and policy resume integration."""
from collections import Counter

import pytest
import torch
from accelerate import Accelerator
from accelerate.data_loader import BatchSamplerShard

from openwam.dataloader.ewm_piper_multitask import BalancedPiperBatchSampler
from scripts.train_ewm import train
from tests.test_ewm_resume import TinyDataset, TinyEWM, config


@pytest.mark.parametrize('world', [1, 2, 4])
def test_balanced_disjoint_shards_and_epoch_replay(world):
    sampler = BalancedPiperBatchSampler([67 + i for i in range(8)], 32, num_processes=world)
    full = list(sampler)
    shards = [list(BatchSamplerShard(sampler, world, r, split_batches=False)) for r in range(world)]
    assert len(full) == len(sampler) == len(shards[0]) * world
    for step in range(len(shards[0])):
        group = [shards[r][step] for r in range(world)]
        assert group == full[step * world:(step + 1) * world]
        for batch in group:
            assert Counter(t for t, _ in batch) == dict.fromkeys(range(8), 4)
        flat = [index for batch in group for index in batch]
        assert len(set(flat)) == 32 * world
    assert list(sampler) == full
    sampler.set_epoch(2)
    changed = list(sampler)
    assert changed != full
    replay = BalancedPiperBatchSampler(sampler.lengths, 32, num_processes=world)
    replay.set_epoch(2)
    assert list(replay) == changed


@pytest.mark.parametrize('lengths,batch,world', [([], 32, 2), ([100] * 8, 31, 2),
                                                ([2] * 8, 32, 2), ([100] * 8, 32, 0)])
def test_reject_invalid_batches(lengths, batch, world):
    with pytest.raises(ValueError):
        BalancedPiperBatchSampler(lengths, batch, num_processes=world)


class TinyTasks(TinyDataset):
    tasks = ('a', 'b')
    lengths = (9, 9)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        task, window = index
        return {'x': torch.tensor([float(task * 10 + window + 1)])}

    def make_batch_sampler(self, batch_size, *, num_processes, seed):
        return BalancedPiperBatchSampler(self.lengths, batch_size, num_processes=num_processes, seed=seed)




def test_multitask_resume_across_epochs(tmp_path):
    def execute(output, steps, weight, resume=None):
        cfg = config(output, steps, resume)
        cfg.training.lr_schedule_steps = 12
        return train(cfg, dataset=TinyTasks(), model=TinyEWM(weight),
                     accelerator=Accelerator(cpu=True, mixed_precision='no'))

    full, split = tmp_path / 'full', tmp_path / 'split'
    execute(full, 12, .7)
    execute(split, 10, .7)
    execute(split, 12, -7., split)
    a = torch.load(full / 'checkpoint_step_000012.pt', weights_only=True)
    b = torch.load(split / 'checkpoint_step_000012.pt', weights_only=True)
    assert a['format'] == 'ewm_wan_latent_v2'
    assert torch.equal(a['model']['weight'], b['model']['weight'])


def test_policy_artifact_uses_immutable_snapshot(monkeypatch, tmp_path):
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module('scripts.train_ewm.train_ewm')
    import wandb
    calls = []

    class Artifact:
        def __init__(self, name, **kw):
            assert name == 'ewm-policy-unit'
            assert kw == {'type': 'model', 'metadata': {'step': 20000}}

        def add_file(self, path, name):
            calls.append((path, name))

    monkeypatch.setattr(wandb, 'Artifact', Artifact)
    for name in ('config.yaml', 'normalization.json', 'training_manifest.json'):
        (tmp_path / name).write_text('{}')
    def log_artifact(artifact, aliases):
        assert aliases == ['step-020000', 'latest']
    module._log_policy_artifact(SimpleNamespace(id='unit', log_artifact=log_artifact), tmp_path, 20000)
    assert calls[0] == (str(tmp_path / 'checkpoint_step_020000.pt'), 'checkpoint.pt')
    assert len(calls) == 4
