"""Portable RGB/event video recording without a model-environment dependency."""
from pathlib import Path
import subprocess
import numpy as np
from .constants import CAMERAS, CONTROL_HZ


from piper_bench.backend.video import find_ffmpeg


def preview(obs, step):
    import cv2
    image = np.ascontiguousarray(np.hstack(obs["obs_video"].transpose(0, 2, 3, 1)))
    for index, name in enumerate(CAMERAS):
        cv2.putText(image, f"{name} step={step}", (index * 256 + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class TrialVideo:
    """Stream RGB, observed events, and optionally predicted events to H.264."""

    def __init__(self, path: Path, event_cameras: tuple[str, ...], *, show_prediction: bool = False, scene_label: str = ""):
        self.scene_label = scene_label
        self.event_cameras = event_cameras
        self.show_prediction = show_prediction
        ffmpeg = find_ffmpeg()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.ffmpeg = ffmpeg
        self.process = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-video_size", "512x768" if show_prediction else "512x512",
                "-framerate", str(CONTROL_HZ),
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")

    @staticmethod
    def _label(image: np.ndarray, text: str) -> np.ndarray:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
        cv2.putText(image, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def write(self, observation: dict, step: int, success_steps: int,
              *, predicted_counts: np.ndarray | None = None, replan_label: str = "") -> None:
        import cv2
        rgb = observation["obs_video"].transpose(0, 2, 3, 1)
        rgb_row = np.hstack((
            self._label(rgb[0], f"cam0 step={step}"),
            self._label(rgb[1], f"wrist success={success_steps}"),
        ))
        event_images = {}
        displayed = observation.get("display_event_frames")
        observed_items = displayed.items() if displayed is not None else zip(self.event_cameras, observation["obs_voxel"], strict=True)
        for name, counts in observed_items:
            image = np.full((256, 256, 3), 255, dtype=np.uint8)
            image[counts[0] > 0] = (255, 0, 0)  # OFF: blue in BGR.
            image[counts[1] > 0] = (0, 0, 255)  # ON: red in BGR.
            cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
            cv2.putText(image, f"{name} events={observation['event_counts'][name]}",
                        (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
            event_images[name] = image
        event_row = np.hstack(tuple(
            event_images[name] if name in event_images else np.full((256, 256, 3), 255, dtype=np.uint8)
            for name in CAMERAS
        ))
        rows = [rgb_row, event_row]
        if self.show_prediction:
            if predicted_counts is not None and predicted_counts.shape != (len(self.event_cameras), 2, 256, 256):
                raise ValueError("predicted counts must be [camera,2,256,256]")
            predicted_images = {}
            if predicted_counts is not None:
                for name, counts in zip(self.event_cameras, predicted_counts, strict=True):
                    image = np.full((256, 256, 3), 255, dtype=np.uint8)
                    image[counts[0] > 0] = (255, 0, 0)  # OFF: blue in BGR.
                    image[counts[1] > 0] = (0, 0, 255)  # ON: red in BGR.
                    cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
                    count = int((counts > 0).sum())
                    cv2.putText(image, f"pred {name} active={count}",
                                (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    predicted_images[name] = image
            predicted_row = np.hstack(tuple(
                predicted_images[name] if name in predicted_images else self._blank_prediction(name, step)
                for name in CAMERAS
            ))
            rows.append(predicted_row)
        frame = np.vstack(rows)
        replan_label = " | ".join(label for label in (self.scene_label, replan_label) if label)
        if replan_label:
            cv2.rectangle(frame, (0, frame.shape[0] - 24), (512, frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, replan_label, (7, frame.shape[0] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    @staticmethod
    def _blank_prediction(name: str, step: int) -> np.ndarray:
        import cv2
        image = np.full((256, 256, 3), 255, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (256, 28), (0, 0, 0), -1)
        label = f"pred {name} pending" if step == 0 else f"pred {name} unavailable"
        cv2.putText(image, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with exit status {self.process.returncode}")
