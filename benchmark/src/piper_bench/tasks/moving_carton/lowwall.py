"""Low-wall stack v4: geometric support plus a recorded relative-pose window.

This is a geometric support proxy, not a measurement of contact forces. The
low rim lies below the seated blue cube's top, avoiding rim-supported stacks.
"""
from itertools import product
import numpy as np
from piper_bench.tasks.conveyor.core import rotation_xyzw
from piper_bench.tasks.conveyor.core import CONTROL_HZ

VERSION = 'moving_carton_stack_v4_lowwall_support_posewindow'
CORNERS = np.array(list(product((-.5, .5), repeat=3)))


def cube_corners(pose, size):
    return CORNERS * size @ rotation_xyzw(pose[3:7]).T + np.asarray(pose[:3])


def support_metrics(poses, basket_pose, tcp, active, cfg):
    from piper_bench.tasks.moving_carton.scene import orange_overhang
    from piper_bench.tasks.moving_carton.scene import stack_geometry
    p = np.asarray(poses)
    geometry = stack_geometry(p, basket_pose, cfg)
    blue_rotation = rotation_xyzw(p[0, 3:7])
    axis = int(np.argmax(np.abs(blue_rotation[2])))
    normal = blue_rotation[:, axis] * np.sign(blue_rotation[2, axis])
    top = p[0, :3] + normal * cfg.object_size/2
    # Intersect the vertical line through the orange COM with the blue top.
    surface_z = top[2] - np.dot(normal[:2], p[1, :2]-top[:2])/normal[2]
    point = np.r_[p[1, :2], surface_z]
    face_coordinates = blue_rotation.T @ (point-p[0, :3])
    tangents = [i for i in range(3) if i != axis]
    support_margin = float(cfg.object_size/2 - np.max(np.abs(face_coordinates[tangents])))
    blue_corners = cube_corners(p[0], cfg.object_size)
    orange_corners = cube_corners(p[1], cfg.object_size)
    floor = basket_pose[2] + cfg.wall_thickness
    lower_gap = float(blue_corners[:, 2].min()-floor)
    # Flatness and centre-height constraints prevent a tipped corner alone
    # from satisfying this broad-phase surface-distance proxy.
    contact_gap = float(orange_corners[:, 2].min()-blue_corners[:, 2].max())
    orange_boundary = orange_overhang(p[1], basket_pose, cfg)
    blue_boundary = orange_overhang(p[0], basket_pose, cfg)
    retreat = np.linalg.norm(p[:, :3]-np.asarray(tcp), axis=1)
    checks = dict(active=bool(all(active)),
        lower_inside=bool(np.all(blue_boundary <= .001)),
        lower_seated=abs(lower_gap) <= .003,
        support=bool(support_margin >= .002),
        contact=bool(abs(contact_gap) <= .003),
        height=bool(abs(geometry['height_gap_m']-cfg.object_size) <= .004),
        flat=geometry['flat'],
        orange_boundary=bool(np.all(orange_boundary <= .002+1e-9)),
        retreat=bool(np.all(retreat >= cfg.retreat_distance)))
    return dict(valid=all(checks.values()), checks=checks,
        failed_conditions=[k for k,v in checks.items() if not v],
        **{k:v for k,v in geometry.items() if k != 'valid'},
        support_margin_m=support_margin, contact_gap_m=contact_gap,
        lower_surface_gap_m=lower_gap, orange_overhang_m=float(orange_boundary.max()),
        orange_overhang_by_axis_m=orange_boundary.tolist(), tcp_distance_m=retreat.tolist(),
        thresholds=dict(support_margin_m=.002, contact_gap_m=.003,
            lower_surface_gap_m=.003, orange_overhang_m=.002, tilt_deg=10.,
            corner_span_m=.003, stable_seconds=cfg.stable_seconds,
            retreat_m=cfg.retreat_distance))


def relative_corners(poses, basket_pose, cfg):
    return ((np.stack([cube_corners(p, cfg.object_size) for p in poses])
             - np.asarray(basket_pose[:3])) @ rotation_xyzw(basket_pose[3:7])).tolist()


def stable_window(window, cfg):
    """31 endpoints span one full second at 30 Hz; detect drift and oscillation."""
    required = round(cfg.stable_seconds*CONTROL_HZ)+1
    if len(window) != required:
        return False
    values = np.asarray(window, dtype=float)
    if values.shape != (required, 2, 8, 3) or not np.isfinite(values).all():
        return False
    # Bounding-box diagonal is a conservative bound on each corner's motion.
    return bool(np.max(np.linalg.norm(np.ptp(values, axis=0), axis=-1)) <= .003)
