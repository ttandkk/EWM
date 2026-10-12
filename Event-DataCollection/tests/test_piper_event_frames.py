"""Temporal-boundary checks for directly exported Piper event frames."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from piper_event_frames import events_to_frame


class EventFrameTests(unittest.TestCase):
    def test_event_at_boundary_belongs_to_earlier_interval(self):
        chunks = [(
            np.array([1, 1, 1], dtype=np.uint16),
            np.array([2, 2, 2], dtype=np.uint16),
            np.array([0.0, 0.5, 1.0]),
            np.array([-1, 1, -1], dtype=np.int8),
        )]
        first = events_to_frame(chunks, 0.0, 0.5, 4, 4)
        second = events_to_frame(chunks, 0.5, 1.0, 4, 4)
        self.assertEqual(first[2, 1].tolist(), [0, 1, 0])
        self.assertEqual(second[2, 1].tolist(), [1, 0, 0])

    def test_polarities_are_clipped_independently(self):
        count = 20
        chunks = [(
            np.ones(2 * count, dtype=np.uint16),
            np.ones(2 * count, dtype=np.uint16),
            np.full(2 * count, 0.5),
            np.array([-1] * count + [1] * count, dtype=np.int8),
        )]
        frame = events_to_frame(chunks, 0.0, 1.0, 4, 4)
        self.assertEqual(frame[1, 1].tolist(), [15, 15, 0])


if __name__ == "__main__":
    unittest.main()
