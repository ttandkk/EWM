"""Integration tests for EWM loading, normalization, and temporal alignment.

Run against the exported LeRobot dataset:

    EWM_DATASET_DIR=/path/to/lerobot_dataset \
      pytest -s tests/test_ewm_dataset.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from openwam.dataloader.ewm_dataset import EWMDataset
from openwam.dataloader.ewm_normalizer import (
    EWMNormalizer,
    NEGATIVE_BINS,
    POSITIVE_BINS,
)
from scripts.eval_ewm.ewm_eval_time_audit import verify_exported_event_alignment


DEFAULT_DATASET_DIR = Path(
    "/projects/_hdd/hdd/myproject/CVPR2027/"
    "tmp/isaacsim601_stack/collection_eventframe_v1"
)


def _dataset_dir() -> Path:
    dataset_dir = Path(os.environ.get("EWM_DATASET_DIR", DEFAULT_DATASET_DIR))
    if not dataset_dir.is_dir():
        pytest.skip(
            "EWM LeRobot dataset is unavailable. Set EWM_DATASET_DIR to its root."
        )
    return dataset_dir


def _describe(value: object) -> str:
    if isinstance(value, torch.Tensor):
        message = f"shape={tuple(value.shape)}, dtype={value.dtype}"
        if value.is_floating_point():
            message += f", range=[{value.min().item():.4g}, {value.max().item():.4g}]"
        return message
    return f"type={type(value).__name__}, value={value!r}"


@pytest.mark.data
def test_ewm_normalizer_contract() -> None:
    """Check shared train/deploy/visualization normalization behavior."""
    normalizer = EWMNormalizer.from_dataset_dir(_dataset_dir(), expected_count_clip=15)

    assert NEGATIVE_BINS == slice(0, 4)
    assert POSITIVE_BINS == slice(4, 8)
    assert normalizer.degenerate_action_mask.tolist() == [False, False, False, True, True, True, False]

    raw_voxel = torch.tensor([0.0, 3.0, 15.0, 40.0])
    voxel = normalizer.normalize_voxel(raw_voxel)
    torch.testing.assert_close(voxel, torch.tensor([0.0, 0.2, 1.0, 1.0]))
    torch.testing.assert_close(
        normalizer.denormalize_voxel(voxel), torch.tensor([0.0, 3.0, 15.0, 15.0])
    )
    assert torch.equal(
        normalizer.denormalize_voxel(voxel, round_counts=True),
        torch.tensor([0.0, 3.0, 15.0, 15.0]),
    )

    state_mean = torch.tensor(normalizer.state_mean)
    torch.testing.assert_close(normalizer.normalize_proprio(state_mean), torch.zeros(8))

    q01 = torch.tensor(normalizer.action_q01)
    q50 = torch.tensor(normalizer.action_q50)
    q99 = torch.tensor(normalizer.action_q99)
    regular = ~torch.tensor(normalizer.degenerate_action_mask)
    in_range = (q01 + q99) * 0.5
    restored = normalizer.denormalize_action(normalizer.normalize_action(in_range))
    torch.testing.assert_close(restored[regular], in_range[regular], atol=1e-6, rtol=0)
    torch.testing.assert_close(restored[~regular], q50[~regular], atol=1e-6, rtol=0)

    above_q99 = q99 + (q99 - q01).abs()
    clipped_roundtrip = normalizer.denormalize_action(
        normalizer.normalize_action(above_q99)
    )
    expected = torch.where(torch.tensor(normalizer.degenerate_action_mask), q50, q99)
    torch.testing.assert_close(clipped_roundtrip, expected, atol=1e-6, rtol=0)


@pytest.mark.data
def test_ewm_dataset_loads_normalized_payload_and_time_alignment() -> None:
    """Read a real sample and guard the full EWM trainer contract."""
    dataset_dir = _dataset_dir()
    dataset = EWMDataset(dataset_dir, context_frames=1, chunk_size=8, voxel_bins=1,
                         event_camera_mode="wrist_only", observation_row_offset=0)
    sample = dataset[0]

    print(f"dataset_dir={dataset_dir}")
    print(f"num_samples={len(dataset)}")
    print("normalized sample payload:")
    for name, value in sample.items():
        print(f"  {name}: {_describe(value)}")

    assert set(sample) == {
        "obs_video",
        "obs_voxel",
        "proprio",
        "prompt",
        "action",
        "future_voxel",
    }
    assert sample["obs_video"].shape == (2, 3, 256, 256)
    assert sample["obs_video"].dtype == torch.uint8  # DINOv3 owns RGB preprocessing.

    assert sample["obs_voxel"].shape == (1, 2, 256, 256)
    assert sample["obs_voxel"].dtype == torch.float32
    assert 0.0 <= sample["obs_voxel"].min() <= sample["obs_voxel"].max() <= 1.0

    assert sample["proprio"].shape == (8,)
    assert sample["proprio"].dtype == torch.float32
    assert torch.isfinite(sample["proprio"]).all()

    assert isinstance(sample["prompt"], str) and sample["prompt"]

    assert sample["action"].shape == (8, 7)
    assert sample["action"].dtype == torch.float32
    assert -1.0 <= sample["action"].min() <= sample["action"].max() <= 1.0
    assert torch.equal(sample["action"][:, 3:6], torch.zeros_like(sample["action"][:, 3:6]))

    assert sample["future_voxel"].shape == (8, 1, 2, 256, 256)
    assert sample["future_voxel"].dtype == torch.float32
    assert 0.0 <= sample["future_voxel"].min() <= sample["future_voxel"].max() <= 1.0

    # For sample index 0, k=0: V_k, then V_{k+1} ... V_{k+K}.
    alignment = verify_exported_event_alignment(dataset, [0], dataset.chunk_size)
    assert alignment[0]["observation_row"] == 0
    assert alignment[0]["verified_frames"] == 9
