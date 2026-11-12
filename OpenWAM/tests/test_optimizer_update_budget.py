"""Exercise the real trainer loop across accumulation and short epoch tails."""
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch import nn

import openwam.train.openwam_trainer as trainer_module
from openwam.train.openwam_trainer import OpenWAMTrainer


@pytest.mark.parametrize('epoch_batches', [3, 4])
@pytest.mark.parametrize('accumulation', [1, 2])
def test_exact_optimizer_budget_with_partial_epoch(epoch_batches, accumulation, monkeypatch, tmp_path):
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({'training':dict(num_epochs=None, max_steps=None,
        max_optimizer_steps=3, batch_size=1, gradient_accumulation_steps=accumulation,
        max_grad_norm=None, save_steps=0, keep_last_k_ckpts=1), 'project':{}})
    trainer.accelerator = accelerator
    trainer.architecture = nn.Linear(1, 1)
    trainer.dataset = [torch.ones(1) for _ in range(epoch_batches)]
    trainer._run_seed = None
    trainer._rank = 0
    trainer.build_optimizer = lambda: torch.optim.SGD(trainer.architecture.parameters(), lr=0.01)
    trainer.build_dataloader = lambda batch: torch.utils.data.DataLoader(trainer.dataset, batch_size=1)
    horizons = []
    trainer.build_lr_scheduler = lambda optimizer, total, debug=False: horizons.append(total)
    trainer.setup_output_dir = lambda debug, resume: (str(tmp_path), None)
    def prepare(opt, data, sched):
        trainer.architecture, opt, data = accelerator.prepare(trainer.architecture, opt, data)
        return opt, data, sched
    trainer.prepare_accelerate = prepare
    def compute(batch):
        loss = trainer.architecture(batch).square().mean()
        return dict(total=loss, video=loss.detach(), action=loss.detach())
    trainer.compute_loss = compute
    logs = []
    trainer.log_step = lambda **kw: logs.append((kw['global_step'], kw['opt_step']))
    trainer.finish_training = lambda *a: None
    monkeypatch.setattr(trainer_module, 'init_wandb', lambda cfg: None)
    trainer.train()
    assert horizons == [3]
    assert logs[-1][1] == 3
    # 3-batch epochs flush at batch 3: updates at 2,3,5. Full epochs: 2,4,6.
    assert logs[-1][0] == (3 if accumulation == 1 else (5 if epoch_batches == 3 else 6))
    accelerator.free_memory()


def test_resume_preserves_saved_optimizer_count(monkeypatch):
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({'training':{'max_optimizer_steps':30000}})
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    monkeypatch.setattr(trainer_module, 'load_full_state', lambda *a: {'global_step':3,'opt_step':2})
    assert trainer.resume_if_configured('/unused', [0,1,2], 2) == (3,2,1,0)


@pytest.mark.parametrize('budget', [0, -1, True, 2.5])
def test_reject_invalid_budget_before_loading_data(budget):
    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({'training':dict(num_epochs=None, max_steps=None,
        max_optimizer_steps=budget,batch_size=1,gradient_accumulation_steps=2)})
    with pytest.raises(ValueError, match='positive integer'):
        trainer.train()
