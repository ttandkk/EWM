"""Opt-in, visual-only tabletop props. Origins sit on the tabletop (metres, Z-up)."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from piper_bench.assets.resolver import asset_path

ASSET_DIR = asset_path('piper_tabletop_props', must_exist=False)


def catalog(asset_dir=ASSET_DIR):
    """Return asset dimensions, materials and local USD filenames without Isaac."""
    return json.loads((Path(asset_dir) / 'catalog.json').read_text())


@dataclass(frozen=True)
class PropPlacement:
    kind: str
    xy: tuple[float, float]
    yaw_degrees: float = 0.
    scale: float = 1.

    def validate(self, assets):
        if self.kind not in assets['props']:
            raise ValueError(f'Unknown tabletop prop: {self.kind}')
        if len(self.xy) != 2 or not all(math.isfinite(v) for v in self.xy):
            raise ValueError('xy must contain two finite coordinates in metres')
        if not math.isfinite(self.yaw_degrees):
            raise ValueError('yaw_degrees must be finite')
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('scale must be positive and finite')


def _placements(placements, table_z, asset_dir):
    if not math.isfinite(table_z):
        raise ValueError('table_z must be finite')
    assets = catalog(asset_dir)
    result = []
    for placement in placements:
        placement.validate(assets)
        path = Path(asset_dir) / assets['props'][placement.kind]['usd']
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append((placement, path.resolve()))
    return result


def configure_tabletop_props(scene, placements, *, table_z, prefix='background_prop', asset_dir=ASSET_DIR):
    """Add named AssetBaseCfg entries BEFORE creating an IsaacLab environment.

    XY and table_z use the scene/environment frame. Every prop is static visual
    geometry: no collision, mass, rigid body or shadow/light overrides. Callers
    choose positions outside the task workspace; no random placement is implied.
    Returns scene entry names for logging. Existing entries are never replaced.
    """
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', prefix):
        raise ValueError('prefix must be a USD/Python identifier')
    entries = _placements(placements, table_z, asset_dir)
    names = [f'{prefix}_{i:02d}' for i in range(len(entries))]
    for name in names:
        if hasattr(scene, name):
            raise ValueError(f'Scene already contains {name}')
    from isaaclab.assets import AssetBaseCfg
    from piper_bench.backend.isaac_compat import quat_from_wxyz
    import isaaclab.sim as sim
    for name, (p, path) in zip(names, entries):
        half_yaw = math.radians(p.yaw_degrees) / 2
        setattr(scene, name, AssetBaseCfg(
            prim_path='{ENV_REGEX_NS}/' + name,
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(*p.xy, table_z), rot=quat_from_wxyz((math.cos(half_yaw), 0., 0., math.sin(half_yaw)))),
            spawn=sim.UsdFileCfg(usd_path=str(path), scale=(p.scale,) * 3)))
    return names


def place_tabletop_props(stage, placements, *, table_z, parent='/World/BackgroundProps', asset_dir=ASSET_DIR):
    """USD-only equivalent, useful in scene editors and standalone previews."""
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    entries = _placements(placements, table_z, asset_dir)
    path = Sdf.Path(parent)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError('parent must be an absolute, non-root prim path')
    if stage.GetPrimAtPath(path):
        raise ValueError(f'Prim already exists: {parent}')
    UsdGeom.Xform.Define(stage, path)
    for i, (p, asset_path) in enumerate(entries):
        prop = UsdGeom.Xform.Define(stage, path.AppendChild(f'prop_{i:02d}_{p.kind}'))
        prop.GetPrim().GetReferences().AddReference(str(asset_path))
        # Clear referenced transform order rather than inheriting asset transforms.
        prop.ClearXformOpOrder()
        prop.AddTranslateOp().Set(Gf.Vec3d(*p.xy, table_z))
        prop.AddRotateZOp().Set(p.yaw_degrees)
        prop.AddScaleOp().Set(Gf.Vec3f(p.scale))
    return stage.GetPrimAtPath(path)
