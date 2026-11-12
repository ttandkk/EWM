"""Procedural red plastic lattice basket. Dimensions are in metres."""
from __future__ import annotations
import numpy as np


def rounded_rectangle(hx, hy, radius, z, segments=8):
    points = []
    for cx, cy, start in ((hx-radius, hy-radius, 0), (-hx+radius, hy-radius, 90),
                          (-hx+radius, -hy+radius, 180), (hx-radius, -hy+radius, 270)):
        for angle in np.deg2rad(np.linspace(start, start+90, segments+1)):
            points.append(np.array([cx+radius*np.cos(angle), cy+radius*np.sin(angle), z]))
    return points


def basket_mesh(cfg):
    """Closed rod meshes, slightly tapered walls, rounded lip, open lattice floor."""
    vertices, faces = [], []
    def rod(a, b, radius, sides=8):
        a, b = np.asarray(a), np.asarray(b)
        direction = (b-a)/np.linalg.norm(b-a)
        normal = np.cross(direction, [0., 0., 1.])
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(direction, [0., 1., 0.])
        normal /= np.linalg.norm(normal)
        tangent = np.cross(direction, normal)
        offset = len(vertices)
        for end in (a, b):
            for angle in np.linspace(0, 2*np.pi, sides, endpoint=False):
                vertices.append((end+radius*(np.cos(angle)*normal+np.sin(angle)*tangent)).tolist())
        faces.append(list(range(offset+sides-1, offset-1, -1)))
        faces.append(list(range(offset+sides, offset+2*sides)))
        for i in range(sides):
            j = (i+1) % sides
            faces.append([offset+i, offset+j, offset+sides+j, offset+sides+i])
    hx, hy = cfg.basket_length/2, cfg.basket_width/2
    t, h, taper = cfg.wall_thickness, cfg.basket_height, cfg.basket_taper
    lower = rounded_rectangle(hx+t/2, hy+t/2, 0.006, t)
    upper = rounded_rectangle(hx+t/2+taper, hy+t/2+taper, 0.008, t+h)
    # Perforated walls: vertical ribs plus five slender horizontal bands.
    for a, b in zip(lower, upper):
        rod(a, b, 0.0008)
    for k in range(6):
        fraction = k/5
        ring = [a+(b-a)*fraction for a, b in zip(lower, upper)]
        for a, b in zip(ring, ring[1:]+ring[:1]):
            rod(a, b, 0.0024 if k == 5 else 0.00085)
    # Add evenly spaced ribs along the straight parts of all four sides.
    for sign in (-1, 1):
        for x in np.arange(-hx+0.01, hx-0.005, 0.005):
            rod([x, sign*(hy+t/2), t], [x*(hx+taper)/hx, sign*(hy+t/2+taper), t+h], 0.00075)
        for y in np.arange(-hy+0.01, hy-0.005, 0.005):
            rod([sign*(hx+t/2), y, t], [sign*(hx+t/2+taper), y*(hy+taper)/hy, t+h], 0.00075)
    for x in np.arange(-hx+0.003, hx, 0.005):
        rod([x, -hy, t], [x, hy, t], 0.00085)
    for y in np.arange(-hy+0.003, hy, 0.005):
        rod([-hx, y, t], [hx, y, t], 0.00085)
    return vertices, faces


def spawn_basket(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import PhysxSchema
    from piper_bench.tasks.conveyor.core import ConveyorConfig
    cfg = ConveyorConfig(**spawn_cfg.basket_config)
    stage = sim.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0, 0, 0))))
    if orientation is not None:
        x, y, z, w = orientation
        root.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    points, faces = basket_mesh(cfg)
    mesh = UsdGeom.Mesh.Define(stage, prim_path+'/red_lattice')
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mat_path = prim_path+'/red_plastic'
    sim.PreviewSurfaceCfg(diffuse_color=(0.68, 0.012, 0.016), roughness=0.34, metallic=0.0).func(
        mat_path, sim.PreviewSurfaceCfg(diffuse_color=(0.68, 0.012, 0.016), roughness=0.34, metallic=0.0))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)))
    # Holes are much smaller than the 4 cm cubes. Thin solid collision proxies
    # preserve the sloping interior without expensive collision on every lattice rib.
    hx, hy, t, h, taper = cfg.basket_length/2, cfg.basket_width/2, cfg.wall_thickness, cfg.basket_height, cfg.basket_taper
    def collider(name, points):
        m = UsdGeom.Mesh.Define(stage, prim_path+'/collision_'+name)
        m.CreatePointsAttr(points)
        m.CreateFaceVertexCountsAttr([4]*6)
        m.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
        m.CreateSubdivisionSchemeAttr('none')
        m.CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
        api = PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim())
        api.CreateContactOffsetAttr(0.001)
        api.CreateRestOffsetAttr(0.0)
    def prism(low, high):
        return [[x,y,z] for z, x0,x1,y0,y1 in (low,high) for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1))]
    collider('bottom', prism((0,-hx-t,hx+t,-hy-t,hy+t),(t,-hx-t,hx+t,-hy-t,hy+t)))
    for s, name in ((-1,'left'),(1,'right')):
        a, b = sorted((s*hx, s*(hx+t)))
        collider(name, prism((t,a,b,-hy-t,hy+t),(t+h,a+s*taper,b+s*taper,-hy-t-taper,hy+t+taper)))
    for s, name in ((-1,'front'),(1,'back')):
        a, b = sorted((s*hy,s*(hy+t)))
        collider(name, prism((t,-hx,hx,a,b),(t+h,-hx-taper,hx+taper,a+s*taper,b+s*taper)))
    return root.GetPrim()
