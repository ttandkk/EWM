"""Tests for compact simulator-evaluation summaries."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_ewm.ewm_eval_summary import write_summary


class EvalSummaryTests(unittest.TestCase):
    def test_one_average_and_blank_line_between_trials(self):
        results = [
            {"trial": 0, "steps": 10, "sim_seed": 123, "success": True,
             "video": "/tmp/trial_000_success.mp4",
             "inference_timing": {"calls": 2, "total_seconds": 0.2}},
            {"trial": 1, "steps": 20, "sim_seed": 124, "success": False,
             "video": "/tmp/trial_001_failure.mp4",
             "inference_timing": {"calls": 3, "total_seconds": 0.6}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_summary(output, max_steps=20, sampling_steps=4, requested_trials=2,
                          attempted_trials=2, results=results, task_name="stack")
            content = (output / "summary.txt").read_text()
        self.assertIn("trial_000: success, steps=10, seed=123, video=trial_000_success.mp4\n\ntrial_001:", content)
        self.assertIn("\naverage_inference_ms_per_call: 160.00\n", content)
        self.assertEqual(content.count("average_inference_ms_per_call:"), 1)
        self.assertNotIn("trial_000_inference", content)

    def test_rgb_ood_settings_are_visible(self):
        from scripts.eval_ewm.ewm_rgb_ood import RGBOODConfig

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_summary(output, max_steps=20, sampling_steps=4, requested_trials=1,
                          attempted_trials=0, results=[], task_name="stack",
                          rgb_ood=RGBOODConfig(True, True, "strong", 77).metadata())
            content = (output / "summary.txt").read_text()
        for line in ("rgb_ood_enabled: true", "rgb_lighting_ood: true",
                     "rgb_motion_blur_ood: true", "rgb_ood_severity: strong",
                     "rgb_ood_seed: 77", "rgb_ood_scope: obs_video_only"):
            self.assertIn(line, content)


if __name__ == "__main__":
    unittest.main()
