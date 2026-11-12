"""Physical catch predicates, independent of simulator imports and oracle state."""
from collections import deque
import math
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw


def gripper_metrics(ball, velocity, tcp, tool_quat, tcp_velocity, aperture, closed, cfg):
    # TCP is 4.2 mm beyond the fingertip centre along the tool's Z axis.
    rotation = rotation_xyzw(tool_quat)
    depth = getattr(cfg, 'grasp_depth', 0.)
    local = rotation.T@(np.asarray(ball[:3])-tcp)+np.array([0., 0., .0042+(0. if cfg.mode == 'lob' else depth)])
    relative_speed = float(np.linalg.norm(np.asarray(velocity[:3])-tcp_velocity))
    if cfg.mode in ('roll', 'lob'):
        # Lob grasp_depth is an expert aiming offset, not a physical boundary.
        # Measure the actual ball centre in the fixed fingertip frame.
        # Piper USD collision bounds in the fingertip frame: finger X is
        # +/-28 mm and the palm front is Z=-68.8 mm (TCP Z minus 4.2 mm).
        # A sphere may sit deeper between the fingers than the old 22 mm
        # reference sphere. Keep its centre between the fingers and its rear
        # surface clear of the palm, with 1 mm contact/mesh tolerance.
        # Preserve the old 22 mm lateral and forward-tip limits.
        in_grasp_region = (
            abs(local[0]) <= .028
            and abs(local[1]) <= min(aperture/2, .022)
            and -.0688+cfg.radius-.001 <= local[2] <= .022
        )
    else:
        in_grasp_region = np.linalg.norm(local) < .022
    held = (closed and .04 < aperture < 2*cfg.radius+.015 and in_grasp_region)
    # Velocity is diagnostic for the tennis gripper tasks.
    # Keep the existing velocity predicates for the other tasks.
    if cfg.mode not in ('lob', 'roll', 'fall'):
        held = held and relative_speed < .08 and np.linalg.norm(velocity[:3]) < .12
    minimum_height = cfg.radius+cfg.lift_height*.8 if cfg.mode == 'roll' else cfg.catch_z+.04
    # Airborne lob catches need no extra lift. Runtime rejects table contact
    # at physics substeps; CatchTracker requires continuous relative stability.
    valid = held if cfg.mode == 'lob' else held and ball[2] > minimum_height
    return dict(valid=bool(valid), held=bool(held), local_ball_m=local.tolist(),
                relative_speed_m_s=relative_speed, ball_height_m=float(ball[2]), aperture_m=float(aperture))


def container_metrics(ball, velocity, container, container_velocity, tcp, aperture, closed, cfg):
    rotation = rotation_xyzw(container[3:])
    local = rotation.T@(np.asarray(ball[:3])-container[:3])
    # Root is at the outside bottom. Entire sphere must clear floor and walls;
    # rim perches and a ball merely passing through the opening are rejected.
    inside = (np.max(np.abs(local[:2]))+cfg.radius <= cfg.container_inner/2+.0005
              and local[2]-cfg.radius >= cfg.wall-.001
              and local[2]+cfg.radius < cfg.wall+cfg.container_height-.001)
    floor_error = float(local[2]-cfg.radius-cfg.wall)
    point_velocity = container_velocity[:3]+np.cross(container_velocity[3:], np.asarray(ball[:3])-container[:3])
    relative_speed = float(np.linalg.norm(velocity[:3]-point_velocity))
    grip_center = container[:3]+rotation@np.asarray(cfg.grip_offset)
    # Check the actual lowest bottom corner, rather than an arbitrary root height.
    half = cfg.container_inner/2+cfg.wall
    clearance = float(container[2]-half*(abs(rotation[2, 0])+abs(rotation[2, 1])))
    held = (closed and cfg.handle_width-.01 < aperture < cfg.handle_width+.015 and np.linalg.norm(tcp-grip_center) < .035
            and clearance > cfg.container_min_clearance and rotation[2, 2] > .985)
    # Container success is retention, not a level-grasp or low-velocity test.
    # Keep those diagnostics, but allow tilt, spin and motion inside the box.
    valid = inside and clearance > 0.
    return dict(valid=bool(valid), inside=bool(inside), container_held=bool(held),
                floor_error_m=floor_error, relative_speed_m_s=relative_speed,
                container_bottom_z_m=float(container[2]), container_clearance_m=clearance, local_ball_m=local.tolist())


class CatchTracker:
    def __init__(self, cfg):
        self.cfg, self.steps = cfg, 0
        # N samples span N-1 intervals: 16 observations at 30 Hz cover 0.5 s.
        self.window_samples = math.ceil(cfg.stable_seconds*30)+1
        self.positions = deque(maxlen=self.window_samples)
        self.position_diameter_m = 0.

    def _diameter(self):
        points = np.asarray(self.positions)
        return float(np.linalg.norm(points[:, None]-points[None, :], axis=-1).max())

    def update(self, metrics, launched, table_hit):
        valid = launched and not table_hit and metrics['valid']
        point = np.asarray(metrics.get('local_ball_m', []), dtype=float)
        valid = valid and point.shape == (3,) and np.all(np.isfinite(point))
        if not valid:
            self.positions.clear()
            self.steps, self.position_diameter_m = 0, 0.
            return False
        self.positions.append(point.copy())
        self.position_diameter_m = self._diameter()
        if self.cfg.mode in ('container', 'roll'):
            # Continuous containment or lifted grasp, irrespective of ball motion.
            self.steps = len(self.positions)
            return self.steps >= self.window_samples
        # Retain the longest stable suffix; reject slow drift as well as jitter.
        while self.position_diameter_m > self.cfg.stable_position_diameter_m:
            self.positions.popleft()
            self.position_diameter_m = self._diameter()
        self.steps = len(self.positions)
        return self.steps >= self.window_samples
