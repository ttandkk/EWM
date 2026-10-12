"""CPU-only checks for decoded prediction display and temporal row selection."""
from __future__ import annotations

import ast
import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_ewm.ewm_event_visualization import predicted_event_counts


class EventVisualizationTests(unittest.TestCase):
    def test_threshold_inverse_quantization_and_spatial_filter(self):
        frame = np.zeros((1, 2, 2, 8, 8), dtype=np.float32)
        frame[0, 0, 0, 2, 2] = 0.0667  # one OFF count
        frame[0, 0, 0, 2, 3] = 0.14    # two OFF counts
        frame[0, 0, 0, 6, 6] = 0.8     # isolated decoder artifact
        frame[0, 1, 1, 3, 3] = 0.02    # below occupancy threshold
        frame[0, 1, 1, 4, 4] = 2.0     # clamp to count_clip
        counts = predicted_event_counts(frame, count_clip=15)
        self.assertEqual(counts.dtype, np.uint8)
        self.assertEqual(counts[0, 0, 0, 2, 2], 1)
        self.assertEqual(counts[0, 0, 0, 2, 3], 2)
        self.assertEqual(counts[0, 0, 0, 6, 6], 0)
        self.assertEqual(counts[0, 1, 1, 3, 3], 0)
        self.assertEqual(counts[0, 1, 1, 4, 4], 0)
        self.assertEqual(predicted_event_counts(frame, count_clip=15, min_neighbors=1)[0, 1, 1, 4, 4], 15)

    def test_optional_prediction_row_is_aligned_to_supplied_frame(self):
        source = ROOT / "scripts/eval_ewm/eval_ewm_piper_sim.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "TrialVideo")
        namespace = {"Path": Path, "np": np, "cv2": cv2,
                     "CAMERAS": ("cam0", "cam_wrist")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(source), "exec"), namespace)
        video = namespace["TrialVideo"].__new__(namespace["TrialVideo"])
        video.event_cameras = ("cam0", "cam_wrist")
        video.show_prediction = True
        video.process = type("Process", (), {"stdin": io.BytesIO()})()
        observation = {
            "obs_video": np.zeros((2, 3, 256, 256), dtype=np.uint8),
            "obs_voxel": np.zeros((2, 2, 256, 256), dtype=np.float32),
            "event_counts": {"cam0": 0, "cam_wrist": 0},
        }
        predicted = np.zeros((2, 2, 256, 256), dtype=np.uint8)
        predicted[0, 0, 100, 100] = 1
        predicted[1, 1, 100, 100] = 1
        video.write(observation, 1, 0, predicted_counts=predicted)
        pixels = np.frombuffer(video.process.stdin.getvalue(), dtype=np.uint8).reshape(768, 512, 3)
        self.assertEqual(tuple(pixels[256 + 100, 100]), (255, 255, 255))
        self.assertEqual(tuple(pixels[512 + 100, 100]), (255, 0, 0))
        self.assertEqual(tuple(pixels[512 + 100, 256 + 100]), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
