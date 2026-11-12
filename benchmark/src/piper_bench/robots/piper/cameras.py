"""Shared rigid Piper wrist installation, derived from RoboTwin's Piper URDF.

This is a simulation calibration, not a measured real-robot hand-eye calibration.
All translations are metres; camera rotations use OpenGL (-Z forward, +Y up).
"""
import math
import numpy as np

CALIBRATION_ID = "piper_stack_front_global_tight_wrist_centered_v5"
WRIST_CALIBRATION_ID = "piper_centered_bracket_square75_v2"
CAM0_EYE_WORLD = (1.25, 0.0, 0.62)
CAM0_TARGET_WORLD = (0.24, 0.0, 0.16)
CAM0_FOCAL_LENGTH_CM = 30.0
CAM0_HORIZONTAL_APERTURE_CM = 20.955

REFERENCE_URDF_SHA256 = "fa27856fc91f085877c4d022790372fe8d38f27000e7bc0c8bd091cc986c02c2"
REFERENCE_URL = "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/resolve/main/embodiments.zip"
# RoboTwin's link6 is the current robot's gripper_base (identity fixed joint).
# The current task defines TCP at gripper_base + (0, 0, .14).
TCP_FROM_LINK6_M = (0., 0., .14)
REFERENCE_JOINTS = (
    ((0., 0., -.077), (0., 0., 1.5708)),
    ((.032, .055, .115), (0., -1.1116926535897932, -1.5707963267948966)),
)


def _transform(xyz, rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rz@ry@rx, xyz
    return result


def reference_transform():
    result = np.eye(4)
    for xyz, rpy in REFERENCE_JOINTS:
        result = result@_transform(xyz, rpy)
    result[:3, 3] -= TCP_FROM_LINK6_M
    # SAPIEN: +X forward, +Y left, +Z up. OpenGL: +X right, +Y up, -Z forward.
    result[:3, :3] = result[:3, :3]@np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
    return result


_REFERENCE = reference_transform()
# Retain a 55 mm lateral bracket and center on the finger symmetry plane.
# The reference +32 mm Y offset occludes thin grasped objects on our gripper.
WRIST_EYE_LOCAL = (-.055, 0., -.102)
# Fixed optical axis aimed 40 mm beyond TCP, never a world-space moving target.
WRIST_TARGET_LOCAL = (0., 0., .04)
WRIST_HORIZONTAL_APERTURE_CM = 20.955
# Retain square 256px streams. 75 degrees is a simulation framing preset,
# not a claim that a physical D435 has this field of view.
WRIST_FOV_DEG = 75.
WRIST_FOCAL_LENGTH_CM = WRIST_HORIZONTAL_APERTURE_CM/(2*math.tan(math.radians(WRIST_FOV_DEG/2)))


def wrist_mount_rotation(eye=None, target=None):
    eye = np.asarray(WRIST_EYE_LOCAL if eye is None else eye, dtype=float)
    target = np.asarray(WRIST_TARGET_LOCAL if target is None else target, dtype=float)
    if eye.shape != (3,) or target.shape != (3,) or not np.isfinite([eye, target]).all():
        raise ValueError('camera eye and target must be finite 3-vectors')
    back = eye-target
    if np.linalg.norm(back) < 1e-6:
        raise ValueError('camera eye and target must differ')
    back /= np.linalg.norm(back)
    right = np.cross(_REFERENCE[:3, 1], back)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError('custom optical axis is parallel to the fixed mount up axis')
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(back, right), back))


def wrist_transform():
    result = np.eye(4)
    result[:3, :3] = wrist_mount_rotation()
    result[:3, 3] = WRIST_EYE_LOCAL
    return result


def wrist_metadata(width=256, height=256):
    f = width*WRIST_FOCAL_LENGTH_CM/WRIST_HORIZONTAL_APERTURE_CM
    return dict(wrist_calibration_id=WRIST_CALIBRATION_ID, wrist_mount_mode='rigid_tcp',
        wrist_eye_local_m=list(WRIST_EYE_LOCAL), wrist_target_local_m=list(WRIST_TARGET_LOCAL),
        wrist_mount_rotation_local_opengl=wrist_mount_rotation().tolist(),
        wrist_tcp_to_camera_opengl=wrist_transform().tolist(),
        wrist_focal_length_cm=WRIST_FOCAL_LENGTH_CM,
        wrist_horizontal_aperture_cm=WRIST_HORIZONTAL_APERTURE_CM,
        wrist_intrinsic_matrix=[[f, 0., width/2], [0., f, height/2], [0., 0., 1.]],
        wrist_reference_urdf_sha256=REFERENCE_URDF_SHA256,
        wrist_reference_source=REFERENCE_URL, wrist_calibration_scope='simulation_adapted_bracket_not_measured_hand_eye')


def set_wrist_view(env):
    """Compatibility hook: the link-parented camera now follows the hand itself.

    Configure the rigid local pose once before scene creation so that mounting
    is independent of per-control-step world-pose updates.
    """
    pass


def configure_wrist_camera(cfg, *, eye=None, target=None):
    """Parent the sensor to the physical hand, so it follows physics substeps."""
    import torch
    from isaaclab.utils.math import quat_from_matrix
    from piper_bench.sensors.wrist import RigidPiperWristCamera
    cfg.class_type = RigidPiperWristCamera
    cfg.prim_path = '{ENV_REGEX_NS}/Robot/gripper_base/cam_wrist'
    cfg.update_latest_camera_pose = True
    cfg.offset.pos = tuple(np.asarray(WRIST_EYE_LOCAL if eye is None else eye)+TCP_FROM_LINK6_M)
    cfg.offset.rot = tuple(quat_from_matrix(torch.tensor(wrist_mount_rotation(eye, target), dtype=torch.float32)).tolist())
    cfg.offset.convention = 'opengl'
    cfg.spawn.focal_length = WRIST_FOCAL_LENGTH_CM
    cfg.spawn.horizontal_aperture = WRIST_HORIZONTAL_APERTURE_CM
    cfg.spawn.vertical_aperture = None
    return cfg
