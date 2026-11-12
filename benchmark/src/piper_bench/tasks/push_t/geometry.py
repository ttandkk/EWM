"""Two nonoverlapping boxes form one rigid T, with a visual-only goal silhouette."""
import numpy as np


def rectangles(cfg):
    return [((0., (cfg.length-cfg.bar)/2), (cfg.width, cfg.bar)),
            ((0., -cfg.bar/2), (cfg.stem, cfg.length-cfg.bar))]


def polygons(cfg, xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    unit = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    return [(unit*np.array(size)+center)@rot.T+xy for center, size in rectangles(cfg)]


def intersection_area(subject, clip):
    """Sutherland-Hodgman clipping of the convex rectangle pairs (CCW)."""
    points = list(subject)
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        incoming, points = points, []
        if not incoming:
            return 0.
        prev = incoming[-1]
        dprev = cross(b-a, prev-a)
        for point in incoming:
            d = cross(b-a, point-a)
            if (d >= 0) != (dprev >= 0):
                points.append(prev+(point-prev)*dprev/(dprev-d))
            if d >= 0:
                points.append(point)
            prev, dprev = point, d
    if len(points) < 3:
        return 0.
    p = np.array(points)
    return abs(float(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1))))/2


def coverage(cfg, xy, yaw):
    actual = polygons(cfg, xy, yaw)
    goal = polygons(cfg, [cfg.target_x, cfg.target_y], cfg.target_yaw)
    area = cfg.width*cfg.bar+cfg.stem*(cfg.length-cfg.bar)
    return min(1., sum(intersection_area(a, b) for a in actual for b in goal)/area)


def spawn_t(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import PhysxSchema
    from piper_bench.tasks.push_t.config import PushTConfig
    cfg = PushTConfig(**spawn_cfg.t_config)
    prim = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage, base = prim.GetStage(), str(prim.GetPath())
    old = stage.GetPrimAtPath(base+'/geometry/mesh')
    material, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial()
    physical, _ = UsdShade.MaterialBindingAPI(old).ComputeBoundMaterial('physics')
    stage.RemovePrim(base+'/geometry/mesh')
    goal = getattr(spawn_cfg, 'goal_only', False)
    for i, (center, size) in enumerate(rectangles(cfg)):
        box = UsdGeom.Cube.Define(stage, base+f'/geometry/part_{i}')
        box.CreateSizeAttr(1.)
        box.AddTranslateOp().Set(Gf.Vec3d(*center, 0.))
        box.AddScaleOp().Set(Gf.Vec3d(*size, .0003 if goal else cfg.height))
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material)
        if not goal:
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            api = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
            api.CreateContactOffsetAttr(.0005)
            api.CreateRestOffsetAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(physical, materialPurpose='physics')
    return prim
