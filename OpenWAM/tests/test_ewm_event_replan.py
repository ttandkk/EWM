import numpy as np
import pytest

from scripts.eval_ewm.ewm_event_replan import EventReplanGate, event_error, horizon_statistics


def events():
    return np.zeros((2, 2, 256, 256), dtype=np.float32)


def test_empty_sparse_and_one_sided():
    a, b = events(), events()
    assert event_error(a, b)[0] == 0
    a[0, 0, 0, 0] = 1
    assert event_error(a, b)[0] == 0
    a[:, 0, 0:16:4, 0] = 1
    assert event_error(a, b)[0] == 1


def test_tolerance_polarity_and_camera_aggregation():
    a = events()
    a[:, 0, 40:56:4, 40] = 1
    assert event_error(a, a)[0] == 0
    assert event_error(a, np.roll(a, 4, axis=-1))[0] == 0
    b = a.copy()
    b[0] = b[0, ::-1]
    score, cameras = event_error(a, b)
    assert cameras == [1, 0]
    assert score == 0.5
    with pytest.raises(ValueError):
        event_error(a, b[:1])


def test_gate_minimum_patience_reset_and_boundary():
    gate = EventReplanGate(0.6, 2, 4)
    assert not gate.update(0.8, 1)
    assert not gate.update(0.8, 2)
    assert not gate.update(0.6, 3)
    assert not gate.update(0.8, 4)
    assert gate.update(0.8, 5)
    assert not EventReplanGate().update(0.8, 1)


def test_statistics_weight_chunks_and_exclude_terminal_tail():
    chunks = [{"executed_horizon": 4, "end_reason": "event_mismatch"},
              {"executed_horizon": 16, "end_reason": "max_horizon"},
              {"executed_horizon": 1, "end_reason": "success"}]
    stats = horizon_statistics(chunks)
    assert stats["all_chunks"] == {"count": 3, "mean": 7, "min": 1, "max": 16}
    assert stats["replan_chunks"]["mean"] == 10
    assert horizon_statistics([])["all_chunks"]["mean"] is None
