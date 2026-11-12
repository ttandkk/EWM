"""Map tabletop motion to the current wrist camera's image axes."""
from __future__ import annotations

import numpy as np


def wrist_planar_delta(forward_world: np.ndarray, screen_up: float, screen_right: float, *, camera_rotation=None) -> np.ndarray:
    """Return XY motion whose image projection is up/right in the wrist view.

    Explicit camera_rotation uses the rigid OpenGL frame. Without it, retain
    the legacy world-up mapping for existing callers. The legacy camera uses
    Isaac Lab's Z-up look-at helper.  Its
    image right axis is ``forward × world_up`` and its image up axis is
    ``(-forward) × right``.  Only their tabletop projections are used: Q/E
    separately controls world height.
    """
    forward = np.asarray(forward_world, dtype=np.float64)
    if forward.shape != (3,) or not np.isfinite(forward).all():
        raise ValueError("forward_world must be one finite 3-vector")
    length = np.linalg.norm(forward)
    if length < 1e-6:
        raise ValueError("camera forward vector is undefined")
    forward = forward / length
    if camera_rotation is None:
        right = np.cross(forward, [0.0, 0.0, 1.0])
        if np.linalg.norm(right) < 5e-3:
            right = np.cross([1.0, 0.0, 0.0], -forward)
        right /= np.linalg.norm(right)
        up = np.cross(-forward, right)
    else:
        rotation = np.asarray(camera_rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError('camera_rotation must be one finite 3x3 rotation')
        right, up = rotation[:, 0], rotation[:, 1]
    right_xy = right[:2] / max(np.linalg.norm(right[:2]), 1e-6)
    up_xy = up[:2]
    if np.linalg.norm(up_xy) < 1e-6:
        up_xy = forward[:2]
    up_xy /= max(np.linalg.norm(up_xy), 1e-6)
    delta = float(screen_up) * up_xy + float(screen_right) * right_xy
    bound = np.hypot(screen_up, screen_right)
    norm = np.linalg.norm(delta)
    if norm > bound > 0:
        delta *= bound / norm
    return delta.astype(np.float32)
