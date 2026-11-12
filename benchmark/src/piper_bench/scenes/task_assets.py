"""Object appearances and seeded, stationary tabletop dressing for eight tasks.

Original task objects retain their physics. Referenced USD props supply only
visuals. Background transforms change exclusively at reset.
"""
from dataclasses import asdict
import math
import numpy as np

from piper_bench.scenes.tabletop_props import ASSET_DIR
from piper_bench.scenes.tabletop_props import PropPlacement
from piper_bench.scenes.tabletop_props import catalog
from piper_bench.scenes.tabletop_props import configure_tabletop_props
from piper_bench.scenes.ball_props import ASSET_DIR as BALL_ASSET_DIR
from piper_bench.scenes.ball_props import TASK_SKINS

VERSION = 'piper_first_two_tabletop_assets_v4'
BACKGROUND_VERSION = 'piper_bowl_tube_tabletop_assets_v2'
BALL_BACKGROUND_VERSION = 'piper_ball_tabletop_assets_v1'
ROLLING_BACKGROUND_VERSION = 'piper_ball_tabletop_assets_v2'
FALLING_BACKGROUND_VERSION = 'piper_ball_tabletop_assets_v2'
TABLE_Z = .0002
POSITION_JITTER_M = .01
YAW_JITTER_DEG = 12.
PREFIX = 'background_prop'
CUBE_ASSETS = {'conveyor': ('rubik_cube', 'dice'),
               'moving_carton': ('gift_box', 'cardboard_box')}
# Spread over the exposed tabletop, outside the central manipulation area.
# Heights/scales are fixed; only XY and yaw vary between episodes.
LAYOUTS = {
    'conveyor': (
        PropPlacement('milk_box', (.45, .012), -15., .45),
        PropPlacement('tissue_box', (.04, -.235), 20., .40),
        PropPlacement('cardboard_box', (-.04, -.16), 12., .40),
    ),
    'moving_carton': (
        PropPlacement('juice_box', (.24, -.26), -15., .50),
        PropPlacement('tissue_box', (.445, -.024), 20., .40),
        PropPlacement('lego_block', (-.04, -.16), -20., .50),
    ),
    # User-marked foreground/side pockets break up the rear row. All complete
    # bounds remain visible and clear the moving disk/rack across reset jitter.
    'turntable_cups': (
        PropPlacement('ceramic_box', (.02, .30), 15., .65),
        PropPlacement('wooden_block', (.374, .198), -12., .60),
        PropPlacement('tissue_box', (.19, -.26), 8., .60),
    ),
    'rotating_tube': (
        PropPlacement('ceramic_box', (.02, .30), 15., .65),
        PropPlacement('wooden_block', (.02, -.17), -12., .60),
        PropPlacement('tissue_box', (.303, -.173), 8., .60),
    ),
    'tennis_rolling': (
        PropPlacement('soccer_ball', (.40, -.305), 15.),
        PropPlacement('mini_basketball', (.02, -.185), -12.),
        PropPlacement('baseball', (.02, .30), 8.),
    ),
    'tennis_falling': (
        PropPlacement('baseball', (.32, -.32), 15.),
        PropPlacement('globe_ball', (.02, -.185), -12.),
        PropPlacement('billiard_ball', (.16, .31), 8.),
    ),
    # The elevated oblique cameras see the rear-left tabletop. Keep the
    # forward prop beside the central flight/receiving corridor, not underneath it.
    'tennis_parabolic': (
        PropPlacement('soccer_ball', (-.03, -.185), 15.),
        PropPlacement('globe_ball', (.04, -.34), -12.),
        PropPlacement('billiard_ball', (.25, -.36), 8.),
    ),
    # Spread around the held container's projected silhouette. The forward
    # ball sits beyond the receiving X range; the right ball is behind it.
    'tennis_container': (
        PropPlacement('mini_basketball', (.17, .135), 15.),
        PropPlacement('soccer_ball', (.43, -.395), -12.),
        PropPlacement('globe_ball', (.58, -.145), 8.),
    ),
}
# Preserve surviving props' seeded poses after removing the two rear-right boxes.
_LAYOUT_RNG_SLOTS = {'conveyor': (0, 2, 4), 'moving_carton': (0, 2, 3)}


def sample_layout(task, seed):
    """Isolated RNG stream: never perturb object poses, speeds or oracle sampling."""
    if task not in LAYOUTS:
        raise ValueError(f'No tabletop dressing for {task}')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 87031, list(LAYOUTS).index(task)]))
    result = []
    previous_slot = -1
    slots = _LAYOUT_RNG_SLOTS.get(task, range(len(LAYOUTS[task])))
    for slot, p in zip(slots, LAYOUTS[task]):
        rng.random(3 * (slot - previous_slot - 1))
        previous_slot = slot
        xy = tuple(float(v) for v in np.asarray(p.xy) + rng.uniform(-POSITION_JITTER_M, POSITION_JITTER_M, 2))
        result.append(PropPlacement(p.kind, xy, p.yaw_degrees + float(rng.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG)), p.scale))
    return result


def appearance_metadata(task, seed):
    version = VERSION if task in CUBE_ASSETS else BALL_BACKGROUND_VERSION if task in TASK_SKINS else BACKGROUND_VERSION
    if task == 'tennis_rolling':
        version = ROLLING_BACKGROUND_VERSION
    elif task == 'tennis_falling':
        version = FALLING_BACKGROUND_VERSION
    data = dict(version=version, seed=int(seed),
                background_props=[asdict(p) for p in sample_layout(task, seed)],
                table_z_m=TABLE_Z, position_jitter_m=POSITION_JITTER_M,
                yaw_jitter_degrees=YAW_JITTER_DEG, randomization='episode_reset_only',
                background_collision=False)
    if task in CUBE_ASSETS:
        data.update(cube_assets=list(CUBE_ASSETS[task]),
                    cube_visual_variants=['flat_ribbon' if kind == 'gift_box' else 'default'
                                          for kind in CUBE_ASSETS[task]],
                    cube_physics='original_cuboid_unchanged')
    elif task in TASK_SKINS:
        data.update(ball_skin=TASK_SKINS[task], ball_physics='original_sphere_unchanged')
    else:
        data['task_objects'] = 'unchanged'
    return data


def attach_cube_visual(stage, root_path, kind, size):
    """Hide the stock cube's renderable geometry, retaining every physics API."""
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    if kind not in {kind for values in CUBE_ASSETS.values() for kind in values}:
        raise ValueError(f'Not a task cube asset: {kind}')
    spec = catalog()['props'][kind]
    dims = spec['dimensions_m']
    stock = stage.GetPrimAtPath(root_path + '/geometry/mesh')
    if not stock:
        raise ValueError(f'Missing original cube geometry: {root_path}')
    path = root_path + '/Appearance'
    if stage.GetPrimAtPath(path):
        raise ValueError(f'Appearance already exists: {path}')
    visual = UsdGeom.Xform.Define(stage, path)
    visual.GetPrim().GetReferences().AddReference(str(ASSET_DIR / spec['usd']))
    if kind == 'gift_box':
        # A bottom stacking object needs a flat top. Keep the wrapped ribbons
        # and lid; omit the raised bow only on this instance, not the asset.
        for part in ('BowLoop0', 'BowLoop1', 'BowLoop2', 'BowLoop3', 'BowKnot'):
            stage.GetPrimAtPath(path + '/Geometry/' + part).SetActive(False)
        dims = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default']).ComputeUntransformedBound(
            visual.GetPrim()).ComputeAlignedRange().GetSize()
    # Fit small seams/lid details to the original collider along each axis.
    if not all(math.isclose(d, dims[0], rel_tol=.01, abs_tol=1e-7) for d in dims):
        raise ValueError('Task appearances must be cubic apart from small surface details')
    visual.AddTranslateOp().Set(Gf.Vec3d(0., 0., -size / 2))
    visual.AddScaleOp().Set(Gf.Vec3f(*(size / d for d in dims)))
    UsdGeom.Imageable(stock).MakeInvisible()
    return visual.GetPrim()


def spawn_task_cube(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    prim = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    attach_cube_visual(sim.get_current_stage(), str(prim.GetPath()), spawn_cfg.prop_kind, spawn_cfg.size[0])
    return prim


def configure_task_assets(scene, task, seed):
    """Add props in task adapters; replace cube visuals only for the first two."""
    for i, kind in enumerate(CUBE_ASSETS.get(task, ()), 1):
        cube = getattr(scene, f'cube_{i}')
        if not all(math.isclose(d, cube.spawn.size[0]) for d in cube.spawn.size):
            raise ValueError('Only cubic task objects support this appearance')
        cube.spawn.func = spawn_task_cube
        cube.spawn.prop_kind = kind
    asset_dir = BALL_ASSET_DIR if task in TASK_SKINS else ASSET_DIR
    configure_tabletop_props(scene, sample_layout(task, seed), table_z=TABLE_Z, prefix=PREFIX, asset_dir=asset_dir)


def apply_layout(stage, env_paths, task, seed):
    """Author a reset layout on an existing USD stage; no simulator dependency."""
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    layout = sample_layout(task, seed)
    metadata = appearance_metadata(task, seed)
    for env_path in env_paths:
        root = stage.GetPrimAtPath(str(env_path))
        root.CreateAttribute('piper:appearance_version', Sdf.ValueTypeNames.String, custom=True).Set(metadata['version'])
        for i, p in enumerate(layout):
            prop = stage.GetPrimAtPath(f'{env_path}/{PREFIX}_{i:02d}')
            if not prop:
                raise RuntimeError(f'Missing configured tabletop prop: {prop.GetPath()}')
            transform = UsdGeom.Xformable(prop)
            # Spawner transforms may use orientation instead of RotateZ. Reset
            # the order and reuse only the named ops authored by this module.
            transform.ClearXformOpOrder()
            def op(kind):
                name = 'xformOp:' + kind + ':tabletop'
                attr = prop.GetAttribute(name)
                if attr:
                    value = UsdGeom.XformOp(attr)
                    transform.AddXformOp(value.GetOpType(), value.GetPrecision(), 'tabletop')
                    return value
                if kind == 'translate':return transform.AddTranslateOp(opSuffix='tabletop')
                if kind == 'rotateZ':return transform.AddRotateZOp(opSuffix='tabletop')
                return transform.AddScaleOp(opSuffix='tabletop')
            op('translate').Set(Gf.Vec3d(*p.xy, TABLE_Z))
            op('rotateZ').Set(p.yaw_degrees)
            op('scale').Set(Gf.Vec3f(p.scale))
    return metadata


def reset_task_assets(env, task, seed):
    """Update visual-only poses once per episode; return the exact serialized layout."""
    from isaaclab.sim.utils.stage import get_current_stage
    return apply_layout(get_current_stage(), env.scene.env_prim_paths, task, seed)
