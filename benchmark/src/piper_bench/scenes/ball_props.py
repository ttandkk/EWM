"""Portable ball skins; appearance changes never replace the analytic collider."""
from pathlib import Path
from piper_bench.scenes.tabletop_props import catalog

from piper_bench.assets.resolver import asset_path

ASSET_DIR = asset_path('piper_ball_props', must_exist=False)
VERSION = 'piper_sports_ball_skins_v1'
ASSET_RADIUS = .03
MODE_SKINS = {'roll': 'tennis', 'fall': 'soccer_ball', 'lob': 'mini_basketball', 'container': 'baseball'}
TASK_SKINS = {'tennis_rolling': 'tennis', 'tennis_falling': 'soccer_ball',
              'tennis_parabolic': 'mini_basketball', 'tennis_container': 'baseball'}


def appearance_id(kind):
    if kind == 'tennis':
        from piper_bench.tasks.tennis.appearance import BALL_APPEARANCE
        return BALL_APPEARANCE
    if kind not in catalog(ASSET_DIR)['props']:
        raise ValueError(f'Unknown ball skin: {kind}')
    return VERSION + ':' + kind


def attach_ball_skin(prim, radius, kind):
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    stage, base = prim.GetStage(), str(prim.GetPath())
    spec = catalog(ASSET_DIR)['props'][kind]
    original = stage.GetPrimAtPath(base + '/geometry/mesh')
    if not original:
        raise ValueError(f'Missing original ball collider: {base}')
    path = base + '/BallAppearance'
    if stage.GetPrimAtPath(path):
        raise ValueError(f'Ball appearance already exists: {path}')
    visual = UsdGeom.Xform.Define(stage, path)
    visual.GetPrim().GetReferences().AddReference(str(ASSET_DIR / spec['usd']))
    visual.AddTranslateOp().Set(Gf.Vec3d(0., 0., -radius))
    visual.AddScaleOp().Set(Gf.Vec3f(radius / ASSET_RADIUS))
    UsdGeom.Imageable(original).MakeInvisible()
    prim.CreateAttribute('piper:ball_appearance', Sdf.ValueTypeNames.String).Set(appearance_id(kind))
    return visual.GetPrim()
