"""Protect mixed-source polarity and occupancy preprocessing contracts."""
import numpy as np
import pytest
import torch
from openwam.dataloader.ewm_event_only import _event_occupancy
from openwam.model.architectures.event_codec import FrozenEventCodec


def test_source_polarities_produce_identical_off_on_occupancy():
    reassemble = np.zeros((1, 2, 384, 512), dtype=np.uint8)
    reassemble[:, 0, :, :256] = 15  # OFF on the left.
    reassemble[:, 1, :, 256:] = 1   # ON on the right.
    dvla = reassemble[:, [1, 0]]   # Disk blue=ON, red=OFF.
    expected = _event_occupancy(reassemble, source="reassemble")
    actual = _event_occupancy(dvla, source="dvla")
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert actual.shape == (1, 1, 2, 256, 256)
    assert actual.min() == 0 and actual.max() == 1
    assert actual[..., :32, :].count_nonzero() == 0
    assert actual[..., 224:, :].count_nonzero() == 0
    assert actual[0, 0, 0, 32:224, :128].eq(1).all()
    assert actual[0, 0, 1, 32:224, 128:].eq(1).all()
    # Counts lose magnitude before area resize, retaining fractional coverage.
    sparse = np.zeros((1, 2, 384, 512), dtype=np.uint8)
    sparse[:, :, ::2, ::2] = 15
    occupancy = _event_occupancy(sparse, source="reassemble")
    torch.testing.assert_close(occupancy[..., 32:224, :], torch.full((1, 1, 2, 192, 256), .25))
    # Shared conversion maps floating [OFF, midpoint, ON] to Lean [-.5,.5].
    lean_rgb = FrozenEventCodec.to_rgb(actual[:, 0]) * .5
    assert lean_rgb.min() == -.5 and lean_rgb.max() == .5
    torch.testing.assert_close(lean_rgb[:, 1], (lean_rgb[:, 0] + lean_rgb[:, 2]) * .5)


def test_unknown_event_source_is_rejected():
    with pytest.raises(ValueError, match="unknown event source"):
        _event_occupancy(np.zeros((1, 2, 192, 256), dtype=np.uint8), source="unknown")
