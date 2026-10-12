"""Deterministic RGB-only OOD corruptions applied after EVIS event generation."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Photometric exposure is in linear RGB; spatial shading and white balance
# approximate sensor-side lighting variation without changing scene rendering.
PRESETS = {
    "mild": {"dark": (0.65, 0.8), "bright": (1.2, 1.4),
             "color": 0.08, "shade": 0.1, "kernels": (5, 7)},
    "moderate": {"dark": (0.4, 0.6), "bright": (1.65, 2.0),
                 "color": 0.15, "shade": 0.25, "kernels": (9, 11)},
    "strong": {"dark": (0.2, 0.35), "bright": (2.5, 3.0),
               "color": 0.25, "shade": 0.4, "kernels": (15, 19)},
}

# These sensor-only controls have no trial/camera randomness or extra effects.
FIXED_MODES = {
    "clean": (None, None),
    "blur-3": (None, 3),
    "blur-5": (None, 5),
    "exposure-0.8": (0.8, None),
    "exposure-0.6": (0.6, None),
    "exposure-0.4": (0.4, None),
}


@dataclass(frozen=True)
class RGBOODConfig:
    lighting: bool = False
    motion_blur: bool = False
    severity: str = "moderate"
    seed: int = 10000
    mode: str = "legacy"

    def __post_init__(self):
        if self.mode != "legacy":
            if self.mode not in FIXED_MODES:
                raise ValueError("unknown RGB OOD mode")
            if self.lighting or self.motion_blur:
                raise ValueError("fixed RGB OOD modes cannot combine with legacy component switches")
            gain, length = FIXED_MODES[self.mode]
            object.__setattr__(self, "lighting", gain is not None)
            object.__setattr__(self, "motion_blur", length is not None)
        if self.severity not in PRESETS:
            raise ValueError("unknown RGB OOD severity")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("RGB OOD seed must be in [0,2**63)")

    @property
    def enabled(self):
        return self.lighting or self.motion_blur

    def label(self):
        if not self.enabled:
            return ""
        if self.mode != "legacy":
            return f"RGB OOD {self.mode}"
        components = "+".join(name for name, enabled in
                              (("lighting", self.lighting), ("motion-blur", self.motion_blur))
                              if enabled)
        return f"RGB OOD {self.severity}: {components}"

    def metadata(self):
        if self.mode != "legacy":
            gain, length = FIXED_MODES[self.mode]
            return {"enabled": self.enabled, "lighting": self.lighting,
                    "motion_blur": self.motion_blur, "severity": "fixed",
                    "seed": self.seed, "implementation": "rgb_sensor_postprocess_v2",
                    "scope": "obs_video_only", "parameters_fixed_per_trial": True,
                    "parameters_fixed_across_trials": True, "mode": self.mode,
                    "exposure_gain": gain, "blur_kernel_length_px": length,
                    "blur_angle_degrees": 0.0 if length is not None else None}
        # Preserve the exact legacy schema for existing watchdog/resume manifests.
        return {"enabled": self.enabled, "lighting": self.lighting,
                "motion_blur": self.motion_blur, "severity": self.severity,
                "seed": self.seed, "implementation": "rgb_sensor_postprocess_v1",
                "scope": "obs_video_only", "parameters_fixed_per_trial": True}


def check_resume_rgb_ood(saved, current):
    """Permit old clean manifests, but never silently resume with new OOD."""
    if saved is None and not current["enabled"]:
        return
    if saved != current:
        raise ValueError("resume RGB OOD configuration differs")


def motion_kernel(length, angle_degrees):
    """Normalized anti-aliased straight-line PSF, not isotropic Gaussian blur."""
    if length < 3 or length % 2 != 1:
        raise ValueError("motion kernel length must be odd and >=3")
    if not np.isfinite(angle_degrees):
        raise ValueError("motion kernel angle must be finite")
    scale = 8
    mask = np.zeros((length * scale, length * scale), np.float32)
    center = (length * scale - 1) / 2
    radius = (length - 1) * scale / 2
    angle = np.deg2rad(angle_degrees)
    dx, dy = radius * np.cos(angle), radius * np.sin(angle)
    p0 = (round(center - dx), round(center - dy))
    p1 = (round(center + dx), round(center + dy))
    cv2.line(mask, p0, p1, 1.0, thickness=scale)
    kernel = cv2.resize(mask, (length, length), interpolation=cv2.INTER_AREA)
    return kernel / kernel.sum()


class TrialRGBOOD:
    """Legacy uses local RNGs; pure modes have identical, nonrandom parameters.

    Output RGB is CHW uint8, matching the existing policy wire contract. All
    non-RGB observations keep their original identity and values.
    """
    def __init__(self, config, trial, cameras=("cam0", "cam_wrist")):
        if trial < 0 or not cameras:
            raise ValueError("trial must be nonnegative and cameras nonempty")
        self.config, self.trial, self.cameras = config, trial, tuple(cameras)
        self.parameters, self.kernels = [], []
        if config.mode != "legacy":
            gain, length = FIXED_MODES[config.mode]
            for name in self.cameras:
                light = ({"exposure_mode": "dark", "exposure_gain": gain,
                          "channel_gains_rgb": [1.0, 1.0, 1.0],
                          "shade_x": 0.0, "shade_y": 0.0} if gain is not None else None)
                blur = ({"kernel_length_px": length, "angle_degrees": 0.0,
                         "boundary": "reflect101", "method": "horizontal_box_psf"}
                        if length is not None else None)
                kernel = (np.full((1, length), 1 / length, dtype=np.float32)
                          if length is not None else None)
                self.parameters.append({"camera": name, "lighting": light, "motion_blur": blur})
                self.kernels.append(kernel)
            self._shading = {}
            return
        preset = PRESETS[config.severity]
        for camera in range(len(self.cameras)):
            # Independent component streams preserve lighting draws when blur
            # is toggled, and reproduce a resumed trial without global RNG state.
            light_rng = np.random.default_rng(np.random.SeedSequence([config.seed, trial, camera, 7101]))
            blur_rng = np.random.default_rng(np.random.SeedSequence([config.seed, trial, camera, 7102]))
            light = None
            if config.lighting:
                mode = "dark" if light_rng.integers(2) == 0 else "bright"
                warmth = float(light_rng.uniform(-preset["color"], preset["color"]))
                light = {"exposure_mode": mode, "exposure_gain": float(light_rng.uniform(*preset[mode])),
                         "channel_gains_rgb": [1 + warmth, 1.0, 1 - warmth],
                         "shade_x": float(light_rng.uniform(-preset["shade"], preset["shade"])),
                         "shade_y": float(light_rng.uniform(-preset["shade"], preset["shade"]))}
            blur = None
            kernel = None
            if config.motion_blur:
                length = int(blur_rng.choice(preset["kernels"]))
                angle = float(blur_rng.uniform(0, 180))
                blur = {"kernel_length_px": length, "angle_degrees": angle,
                        "boundary": "reflect101", "method": "directional_line_psf"}
                kernel = motion_kernel(length, angle)
            self.parameters.append({"camera": self.cameras[camera], "lighting": light, "motion_blur": blur})
            self.kernels.append(kernel)
        self._shading = {}

    def metadata(self):
        return {**self.config.metadata(), "trial": self.trial, "cameras": self.parameters}

    def apply_rgb(self, rgb):
        if not self.config.enabled:
            return rgb
        if (rgb.ndim != 4 or rgb.shape[:2] != (len(self.cameras), 3)
                or rgb.dtype != np.uint8):
            raise ValueError("RGB OOD requires [camera,3,H,W] uint8")
        output = []
        height, width = rgb.shape[2:]
        for index, (image, parameters, kernel) in enumerate(zip(rgb, self.parameters, self.kernels)):
            image = image.transpose(1, 2, 0).astype(np.float32) / 255.0
            # Work in linear light for exposure and shutter PSF integration.
            image = np.where(image <= 0.04045, image / 12.92,
                             ((image + 0.055) / 1.055) ** 2.4)
            light = parameters["lighting"]
            if light is not None:
                key = (index, height, width)
                if key not in self._shading:
                    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
                    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
                    self._shading[key] = (1 + light["shade_x"] * x + light["shade_y"] * y)[..., None]
                image *= light["exposure_gain"] * self._shading[key]
                image *= np.asarray(light["channel_gains_rgb"], np.float32)
            if kernel is not None:
                image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
            image = np.clip(image, 0, 1)
            image = np.where(image <= 0.0031308, image * 12.92,
                             1.055 * image ** (1 / 2.4) - 0.055)
            output.append(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8).transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(output))

    def apply_observation(self, observation):
        if not self.config.enabled:
            return observation
        return {**observation, "obs_video": self.apply_rgb(observation["obs_video"])}
