"""Fixed task-specific wrist optics; no flight-state or world-target tracking."""
import math

import numpy as np
from piper_bench.robots.piper.cameras import wrist_metadata as shared_metadata
from piper_bench.robots.piper.cameras import wrist_mount_rotation


def wrist_metadata(mode):
    value = shared_metadata()
    if mode == 'lob':
        eye, target, fov = [-.055, 0., -.102], [.01, 0., .02], 75.
        calibration_id = 'piper_tennis_lob_fixed_flight75_v1'
    elif mode == 'container':
        eye, target, fov = [-.16, 0., -.16], [-.144, 0., .04], 110.
        calibration_id = 'piper_tennis_container_fixed_flight110_v1'
    else:
        return value
    rotation = wrist_mount_rotation(eye, target)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, eye
    aperture = value['wrist_horizontal_aperture_cm']
    focal = aperture / (2 * math.tan(math.radians(fov / 2)))
    pixel_focal = 256 * focal / aperture
    value.update(wrist_calibration_id=calibration_id, wrist_eye_local_m=eye,
                 wrist_target_local_m=target, wrist_focal_length_cm=focal,
                 wrist_mount_rotation_local_opengl=rotation.tolist(),
                 wrist_tcp_to_camera_opengl=transform.tolist(),
                 wrist_intrinsic_matrix=[[pixel_focal, 0., 128.], [0., pixel_focal, 128.], [0., 0., 1.]],
                 wrist_calibration_scope='task_specific_simulation_bracket_not_measured_hand_eye')
    return value


def configure_wrist_camera(camera_cfg, mode):
    from piper_bench.robots.piper.cameras import configure_wrist_camera as configure_shared
    calibration = wrist_metadata(mode)
    configure_shared(camera_cfg, eye=calibration['wrist_eye_local_m'],
                     target=calibration['wrist_target_local_m'])
    camera_cfg.spawn.focal_length = calibration['wrist_focal_length_cm']
