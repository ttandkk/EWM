"""Tests for interruption-safe simulator evaluation continuation."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_ewm.ewm_eval_recovery import archive_interrupted_trial, load_completed_results


class EvalRecoveryTests(unittest.TestCase):
    def test_contiguous_prefix_and_preserved_partial_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for index in range(2):
                video = output / f"trial_{index:03d}_h264_failure.mp4"
                video.write_bytes(b"video")
                (output / f"trial_{index:03d}_steps.json").write_text("[]")
                (output / f"trial_{index:03d}_result.json").write_text(
                    json.dumps({"trial": index, "video": str(video), "success": False}))
            self.assertEqual(len(load_completed_results(output, 3)), 2)
            partial = output / "trial_002_h264_recording.mp4"
            partial.write_bytes(b"partial")
            (output / "trial_002_steps.json").write_text("[1]")
            events = output / "trial_002_events"
            events.mkdir()
            (events / "cam0_raw.npz").write_bytes(b"partial raw events")
            archived = archive_interrupted_trial(output, 2)
            self.assertEqual(len(archived), 3)
            self.assertFalse(events.exists())
            self.assertEqual((output / "trial_002_events_interrupted_01/cam0_raw.npz").read_bytes(), b"partial raw events")
            self.assertTrue(all(path.exists() for path in archived))
            self.assertFalse(partial.exists())
            self.assertEqual(len(load_completed_results(output, 3)), 2)

    def test_missing_event_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            video = output / "trial_000_h264_success.mp4"
            video.write_bytes(b"video")
            (output / "trial_000_steps.json").write_text("[]")
            (output / "trial_000_result.json").write_text(json.dumps(dict(
                trial=0, video=str(video), event_archive=dict(directory=str(output / "trial_000_events")))))
            with self.assertRaisesRegex(ValueError, "incomplete event archive"):
                load_completed_results(output, 1)

    def test_missing_completed_video_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "trial_000_steps.json").write_text("[]")
            (output / "trial_000_result.json").write_text(
                json.dumps({"trial": 0, "video": str(output / "missing.mp4")}))
            with self.assertRaises(ValueError):
                load_completed_results(output, 2)


if __name__ == "__main__":
    unittest.main()
