"""Fit the optional standard room to the tennis main camera's upper rays."""
import math

import numpy as np


def room_wall_height(calibration, *, floor_z=-1.05, minimum=2.8):
    """Cover the top frustum edge with 15 cm of physical wall margin.

    The standard room occupies world X=[-4.6,2.8], Y=[-2,2]. Extremal ray
    intersections occur at image corners or where a ray crosses a room corner.
    The ceiling remains open, and this calculation does not alter exposure.
    """
    eye = np.asarray(calibration['cam0_eye_world_m'], dtype=float)
    forward = np.asarray(calibration['cam0_target_world_m'], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half = calibration['cam0_horizontal_aperture_cm'] / (2 * calibration['cam0_focal_length_cm'])
    center, side = forward + half * up, half * right
    horizontal = [-1., 1.]
    for x in (-4.6, 2.8):
        for y in (-2., 2.):
            dx, dy = x-eye[0], y-eye[1]
            denominator = dy*side[0] - dx*side[1]
            if abs(denominator) > 1e-12:
                u = (dx*center[1] - dy*center[0]) / denominator
                if -1 < u < 1:
                    horizontal.append(u)
    rays = center + np.asarray(horizontal)[:, None] * side
    distances = []
    for axis, bounds in ((0, (-4.6, 2.8)), (1, (-2., 2.))):
        if not bounds[0] < eye[axis] < bounds[1]:
            raise ValueError('tennis camera must be inside the standard room')
        edge = np.where(rays[:, axis] > 0, bounds[1]-eye[axis], bounds[0]-eye[axis])
        distances.append(np.divide(edge, rays[:, axis], out=np.full(len(rays), np.inf),
                                   where=abs(rays[:, axis]) > 1e-12))
    tops = eye[2] + np.minimum(*distances) * rays[:, 2]
    return max(minimum, math.ceil((float(tops.max())-floor_z+.15)*10)/10)


def fit_optional_room(env, calibration):
    """Keep ID and material-only OOD room walls at the same camera-fitted height."""
    if getattr(env, '_tennis_room_fitted', False):
        return
    env._tennis_room_fitted = True
    if getattr(env.cfg.scene, 'room_background', None) is None:
        return
    import omni.usd
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from piper_bench.scenes.background import BACKGROUND_ID
    from piper_bench.scenes.background_ood import BACKGROUND_ID as OOD_ID
    from piper_bench.scenes.background_ood import wall_mesh

    stage = omni.usd.get_context().get_stage()
    room = stage.GetPrimAtPath('/World/envs/env_0/RoomBackground')
    if not room or room.GetAttribute('piper:background_id').Get() not in (BACKGROUND_ID, OOD_ID):
        return
    floor_z = float(env.cfg.scene.room_background.init_state.pos[2])
    height = room_wall_height(calibration, floor_z=floor_z)
    for name in ('BackWall', 'FrontWall', 'LeftWall', 'RightWall'):
        wall = room.GetChild(name)
        position = list(wall.GetAttribute('xformOp:translate').Get())
        scale = list(wall.GetAttribute('xformOp:scale').Get())
        position[2], scale[2] = height/2, height
        wall.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*position))
        wall.GetAttribute('xformOp:scale').Set(Gf.Vec3d(*scale))
    if room.GetAttribute('piper:background_id').Get() == OOD_ID:
        # The plaster overlay must follow the walls, without stretching its texture.
        points, counts, indices, colors = wall_mesh(height)
        mesh = UsdGeom.Mesh.Get(stage, str(room.GetPath())+'/PlasterSurface')
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(counts)
        mesh.GetFaceVertexIndicesAttr().Set(indices)
        mesh.GetDisplayColorPrimvar().Set(colors)
    room.CreateAttribute('piper:wall_height_m', Sdf.ValueTypeNames.Float).Set(height)
    room.CreateAttribute('piper:framing_calibration_id', Sdf.ValueTypeNames.String).Set(
        calibration['calibration_id'])
