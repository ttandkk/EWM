"""Regression tests for EWM optimizer-step resume and checkpoint discovery."""

from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from openwam.train.utils.ewm_resume import find_latest_resume_state, prune_policy_checkpoints
from scripts.train_ewm import train


class TinyDataset(Dataset):
    gpu_voxelization = False
    voxel_bins = 1
    event_camera_mode = "dual"
    normalizer = SimpleNamespace(count_clip=15, metadata={"count_clip": 15, "voxel_bins": 1})

    def __len__(self):
        return 15

    def __getitem__(self, index):
        return {"x": torch.tensor([float(index + 1)])}


class TinyEWM(torch.nn.Module):
    num_event_cameras = 2
    voxel_bins = 1
    event_camera_mode = "dual"
    training_mode = "joint"

    def __init__(self, weight=0.7):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))

    def policy_state_dict(self, state=None):
        return self.state_dict() if state is None else state

    def load_policy_state_dict(self, state):
        return self.load_state_dict(state)

    def forward(self, batch, diagnostics=False, profile_timing=False):
        noise = torch.rand_like(batch["x"])
        if diagnostics:
            torch.rand(())  # Decoder diagnostics also consume RNG in production.
        loss = (self.weight * batch["x"] + noise).square().mean()
        zero = loss.detach() * 0
        return {
            "loss": loss,
            "loss_event": loss,
            "loss_action": zero,
            "latent_flow_mse": loss.detach(),
            "latent_x0_mse": zero,
            "voxel_x0_mse": zero,
            "voxel_x0_active_mse": zero,
            "voxel_x0_inactive_mse": zero,
            "event_precision": zero,
            "event_recall": zero,
            "event_f1": zero,
            "pred_nonzero_ratio": zero,
            "target_nonzero_ratio": zero,
        }


def config(output, max_steps, resume=None, grad_accum=1):
    return OmegaConf.create({
        "model": {"type": "ewm", "dim": 1},
        "dataloader": {"name": "tiny", "chunk_size": 1},
        "project": {"seed": 123, "wandb": {"project": None}},
        "training": {
            "output_path": str(output),
            "resume_ckpt_path": str(resume) if resume else None,
            "save_full_states_for_resume": True,
            "keep_last_k_resume_states": 2,
            "save_checkpoints": True,
            "save_steps": 3,
            "max_steps": max_steps,
            "batch_size": 2,
            "gradient_accumulation_steps": grad_accum,
            "dataset_num_workers": 0,
            "learning_rate": 0.01,
            "adam_betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler": "cosine",
            "lr_schedule_steps": 6,
            "warmup_ratio": 0.0,
            "lr_min_ratio": 0.01,
            "mixed_precision": "no",
            "compile_core": False,
            "log_every": 1,
            "diagnostic_every": 3,
        },
    })


def run(cfg, weight):
    return train(cfg, dataset=TinyDataset(), model=TinyEWM(weight),
                 accelerator=Accelerator(cpu=True, mixed_precision="no",
                                         gradient_accumulation_steps=cfg.training.gradient_accumulation_steps))


@pytest.mark.parametrize("grad_accum,expected_batches,stale_policy", [
    (1, 3, False), (2, 6, False), (1, 3, True),
])
def test_resume_matches_uninterrupted_training(tmp_path, grad_accum, expected_batches, stale_policy):
    full = tmp_path / "full"
    split = tmp_path / "split"
    run(config(full, 6, grad_accum=grad_accum), 0.7)
    run(config(split, 3, grad_accum=grad_accum), 0.7)
    state_dir, metadata = find_latest_resume_state(split, 1)
    assert state_dir.name == "resume_state_step_000003"
    assert metadata["batches_consumed"] == expected_batches
    if stale_policy:
        (split / "checkpoint_step_000006.pt").hardlink_to(split / "checkpoint_step_000003.pt")
    run(config(split, 6, resume=split, grad_accum=grad_accum), -7.0)
    if stale_policy:
        assert len(list((split / "incomplete_policy_checkpoints").glob("checkpoint_step_000006_*.pt"))) == 1

    uninterrupted = torch.load(full / "checkpoint_step_000006.pt", weights_only=True)
    resumed = torch.load(split / "checkpoint_step_000006.pt", weights_only=True)
    assert torch.equal(uninterrupted["model"]["weight"], resumed["model"]["weight"])
    assert [p.name for p in sorted(split.glob("resume_state_step_*"))] == [
        "resume_state_step_000003", "resume_state_step_000006"
    ]


def test_policy_only_checkpoint_cannot_resume(tmp_path):
    (tmp_path / "checkpoint_step_000003.pt").touch()
    with pytest.raises(FileNotFoundError, match="policy-only"):
        find_latest_resume_state(tmp_path, 1)


def test_prune_policy_checkpoints_keeps_latest_alias(tmp_path):
    for step in range(1, 7):
        (tmp_path / f"checkpoint_step_{step:06d}.pt").write_text(str(step))
    (tmp_path / "checkpoint.pt").hardlink_to(tmp_path / "checkpoint_step_000006.pt")
    prune_policy_checkpoints(tmp_path, 4)
    assert [path.name for path in sorted(tmp_path.glob("checkpoint_step_*.pt"))] == [
        f"checkpoint_step_{step:06d}.pt" for step in range(3, 7)
    ]
    assert (tmp_path / "checkpoint.pt").read_text() == "6"


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_event_only_source_batches_resume_across_epochs(tmp_path, grad_accum):
    from openwam.dataloader.ewm_event_only import EWMMixedEventOnlyDataset

    class MixedDataset(EWMMixedEventOnlyDataset):
        event_camera_mode = "dual"
        normalizer = TinyDataset.normalizer

        def __init__(self):
            self.source_ranges = {"reassemble": [(0, 9)], "dvla": [(9, 15)]}

        __len__ = TinyDataset.__len__

        def __getitem__(self, index):
            return dict(TinyDataset.__getitem__(self, index),
                        source="reassemble" if index < 9 else "dvla")

    class EventModel(TinyEWM):
        training_mode = "event_only"

        def forward(self, batch, **kwargs):
            assert len(set(batch["source"])) == 1
            return super().forward(batch, **kwargs)

    def execute(output, steps, weight, resume=None):
        cfg = config(output, steps, resume=resume, grad_accum=grad_accum)
        cfg.training.lr_schedule_steps = 12
        cfg.dataloader.homogeneous_source_batches = True
        return train(cfg, dataset=MixedDataset(), model=EventModel(weight),
                     accelerator=Accelerator(cpu=True, mixed_precision="no",
                                             gradient_accumulation_steps=grad_accum))

    full, split = tmp_path / "full", tmp_path / "split"
    execute(full, 10, .7)
    # With accumulation2, checkpoint after crossing into epoch1.
    execute(split, 3 if grad_accum == 1 else 6, .7)
    if grad_accum == 2:
        _, metadata = find_latest_resume_state(split, 1)
        assert metadata["epoch"] > 0
    execute(split, 10, -7.0, resume=split)
    a = torch.load(full / "checkpoint_step_000010.pt", weights_only=True)
    b = torch.load(split / "checkpoint_step_000010.pt", weights_only=True)
    assert torch.equal(a["model"]["weight"], b["model"]["weight"])
