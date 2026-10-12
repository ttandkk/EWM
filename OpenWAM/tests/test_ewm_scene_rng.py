"""Checks scene RNG state survives a simulated client restart."""
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_ewm.ewm_scene_rng import restore_scene_rng, save_scene_rng, state_path


class SceneRngTests(unittest.TestCase):
    def test_restore_continues_all_reset_generators(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            random.seed(20260930)
            np.random.seed(20260930)
            torch.manual_seed(20260930)
            save_scene_rng(output, 4)
            expected = (random.random(), np.random.random(), torch.rand(1).item())
            random.random()
            np.random.random()
            torch.rand(1)
            restore_scene_rng(output, 4)
            actual = (random.random(), np.random.random(), torch.rand(1).item())
            self.assertEqual(actual, expected)
            self.assertTrue(state_path(output, 4).is_file())


if __name__ == "__main__":
    unittest.main()
