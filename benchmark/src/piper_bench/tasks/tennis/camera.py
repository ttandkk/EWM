"""Task-specific receiving views and fixed wrist flight calibrations."""
from piper_bench.tasks.common import camera_metadata as base_metadata
from piper_bench.tasks.common import set_views as base_views


def camera_metadata(mode=None, *, long_flight=False, spatial_receiving=False):
    value = base_metadata()
    value.update(calibration_id='piper_tennis_flight_front_v1_wrist_v1',
                 cam0_eye_world_m=[.95, .35, .75], cam0_target_world_m=[.35, 0., .25],
                 cam0_focal_length_cm=20.)
    if mode == 'roll':
        value = base_metadata()
        value['calibration_id'] = 'piper_tennis_rolling_frontal_v3_wrist_v1'
        value['cam0_focal_length_cm'] = 18.
    if mode == 'fall':
        # Face the robot across the table; cover the vertical release and grasp.
        value.update(calibration_id='piper_tennis_falling_tight_front_v4_wrist_v1',
                     cam0_eye_world_m=[1.18, 0., .75], cam0_target_world_m=[.35, 0., .45],
                     cam0_focal_length_cm=20.5)
    if mode == 'lob':
        # Close oblique view of interception; the rigid wrist covers the flight.
        value.update(calibration_id='piper_tennis_lob_receiving_oblique_v11_wrist_flight_v1',
                     cam0_eye_world_m=[1.15, .65, .70], cam0_target_world_m=[.40, 0., .50],
                     cam0_focal_length_cm=30.)
        if long_flight:
            value.update(calibration_id='piper_tennis_lob_long_flight_receiving_oblique_v12_wrist_flight_v1',
                         cam0_eye_world_m=[1.15, .65, .70],
                         cam0_target_world_m=[.40, 0., .50], cam0_focal_length_cm=30.)
    if mode == 'container':
        # Mild downward view exposes the opening without a high overhead angle.
        value.update(calibration_id='piper_tennis_container_receiving_oblique_v19_wrist_flight_v1',
                     cam0_eye_world_m=[1.15, .70, .65],
                     cam0_target_world_m=[.42, 0., .28], cam0_focal_length_cm=26.)
        if spatial_receiving:
            value.update(calibration_id='piper_tennis_container_spatial_receiving_oblique_v20_wrist_flight_v1',
                         cam0_focal_length_cm=32.)
    from piper_bench.tasks.tennis.wrist import wrist_metadata
    value.update(wrist_metadata(mode))
    return value


def camera_metadata_for_config(config, *, calibration_id=None):
    """Select the live view, or reconstruct a known historical export for validation."""
    long_flight = config.mode == 'lob' and config.source_x > 1.5
    value = camera_metadata(config.mode, long_flight=long_flight,
                            spatial_receiving=config.mode=='container' and not config.paired_launch)
    if calibration_id is None or calibration_id == value['calibration_id']:
        return value
    # Historical data keep their original observations. Only validators request
    # these exact calibrations; new scene construction always uses the live view.
    if config.mode == 'fall':
        legacy = {
            'piper_tennis_falling_side_v2_wrist_v1': ([.40,1.10,.65],[.35,0.,.32],20.),
            'piper_tennis_falling_front_v3_wrist_v1': ([1.30,0.,.75],[.35,0.,.40],20.),
        }
    elif config.mode == 'container':
        legacy = {
            'piper_tennis_container_long_flight_side_v14_wrist_v1': ([.68,1.12,.75],[.68,0.,.45],12.),
            'piper_tennis_container_low_oblique_v16_wrist_v1': ([2.0,.85,.90],[.75,0.,.50],16.),
            ('piper_tennis_container_tight_oblique_v17_wrist_v1' if config.paired_launch else
             'piper_tennis_container_spatial_tight_oblique_v18_wrist_v1'):
                ([1.90,1.0,.90],[.95,0.,.60],18.5 if config.paired_launch else 20.),
        }
    elif long_flight:
        legacy = {
            'piper_tennis_container_long_flight_side_v14_wrist_v1': ([.68,1.12,.75],[.68,0.,.45],12.),
            'piper_tennis_lob_long_flight_low_oblique_v8_wrist_v1': ([1.90,1.0,.75],[.75,0.,.50],17.),
            'piper_tennis_lob_long_flight_tight_oblique_v10_wrist_v1': ([1.90,1.0,.75],[.95,0.,.45],21.5),
        }
    elif config.mode == 'lob':
        legacy = {
            'piper_tennis_lob_side_v4_wrist_v1': ([.60,1.10,.75],[.60,0.,.35],14.),
            'piper_tennis_lob_low_oblique_v7_wrist_v1': ([1.60,.90,.75],[.60,0.,.45],19.),
            'piper_tennis_lob_tight_oblique_v9_wrist_v1': ([1.60,.60,.75],[.80,0.,.35],23.5),
        }
    else:
        raise ValueError('unknown camera calibration for this tennis configuration')
    if calibration_id not in legacy:
        raise ValueError('unknown camera calibration for this tennis configuration')
    from piper_bench.robots.piper.cameras import wrist_metadata as shared_wrist_metadata
    value.update(shared_wrist_metadata())
    eye,target,focal = legacy[calibration_id]
    value.update(calibration_id=calibration_id, cam0_eye_world_m=eye,
                 cam0_target_world_m=target, cam0_focal_length_cm=focal)
    return value


def set_views(env):
    import torch
    base_views(env)
    runtime = getattr(env, '_tennis_runtime', None)
    cal = camera_metadata_for_config(runtime.cfg) if runtime is not None else camera_metadata()
    def tensor(value):
        return torch.tensor([value], dtype=torch.float32, device=env.device)
    env.scene['cam0'].set_world_poses_from_view(
        tensor(cal['cam0_eye_world_m']), tensor(cal['cam0_target_world_m']))
    if runtime is not None and runtime.cfg.mode in ('fall','lob','container'):
        from piper_bench.tasks.tennis.background import fit_optional_room
        fit_optional_room(env,cal)


def wrist_ball_projection(ball, tcp, tool_quat, radius, *, calibration=None):
    """Frustum diagnostic using the fixed mount; this does not test occlusion."""
    import numpy as np
    from piper_bench.tasks.conveyor.core import rotation_xyzw
    from piper_bench.robots.piper.cameras import wrist_metadata
    cal = wrist_metadata() if calibration is None else calibration
    tool = rotation_xyzw(tool_quat)
    eye = np.asarray(tcp)+tool@np.asarray(cal['wrist_eye_local_m'])
    mount = np.asarray(cal['wrist_mount_rotation_local_opengl'])
    local = (tool@mount).T@(np.asarray(ball)-eye)
    depth = float(-local[2])
    half = np.arctan(cal['wrist_horizontal_aperture_cm']/(2*cal['wrist_focal_length_cm']))
    # Signed distance to the nearest of the four side planes, in metres.
    margin = float(depth*np.sin(half)-np.max(np.abs(local[:2]))*np.cos(half))
    uv = None
    if depth > 0:
        focal = 128/np.tan(half)
        uv = [float(128+focal*local[0]/depth), float(128-focal*local[1]/depth)]
    return dict(center_px=uv, depth_m=depth,
                center_in_frame=bool(depth > 0 and margin >= 0),
                whole_ball_in_frame=bool(depth > radius and margin >= radius),
                sphere_margin_m=margin-radius, occlusion_tested=False)
