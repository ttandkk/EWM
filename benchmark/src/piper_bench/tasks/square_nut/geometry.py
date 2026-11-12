"""Separate box colliders preserve the square opening; no filled convex hull."""


def nut_boxes(cfg):
    """Local centres and sizes of four nonoverlapping walls and a grip tab."""
    wall = (cfg.outer_width - cfg.hole_width) / 2
    offset = (cfg.outer_width + cfg.hole_width) / 4
    return [
        ((sign * offset, 0., 0.), (wall, cfg.outer_width, cfg.thickness))
        for sign in (-1, 1)
    ] + [
        ((0., sign * offset, 0.), (cfg.hole_width, wall, cfg.thickness))
        for sign in (-1, 1)
    ] + [((cfg.handle_x, 0., 0.), (cfg.handle_length, cfg.handle_width, cfg.thickness))]


def spawn_nut(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import PhysxSchema
    from piper_bench.tasks.square_nut.config import SquareNutConfig
    cfg = SquareNutConfig(**spawn_cfg.nut_config)
    prim = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage, base = prim.GetStage(), str(prim.GetPath())
    old = stage.GetPrimAtPath(base + '/geometry/mesh')
    material, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial()
    physical, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial('physics')
    stage.RemovePrim(base + '/geometry/mesh')
    for i, (center, size) in enumerate(nut_boxes(cfg)):
        box = UsdGeom.Cube.Define(stage, base + f'/geometry/part_{i}')
        box.CreateSizeAttr(1.)
        box.AddTranslateOp().Set(Gf.Vec3d(*center))
        box.AddScaleOp().Set(Gf.Vec3d(*size))
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(physical, materialPurpose='physics')
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        collision = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
        collision.CreateContactOffsetAttr(.0003)
        collision.CreateRestOffsetAttr(0.)
    return prim
