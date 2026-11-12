"""Open tapered cups: separate convex wall wedges, never a filled convex hull."""
import math
import numpy as np

WALL_SEGMENTS = 32
VISUAL_SEGMENTS = 128


def cup_rings(cfg, segments):
    angle = np.arange(segments)*2*math.pi/segments
    xy = np.stack((np.cos(angle), np.sin(angle)), axis=1)
    h, t = cfg.cup_height, cfg.cup_wall
    # Root frame is centered halfway up the cup; the bottom is at -h/2.
    levels = [(cfg.cup_bottom_radius, -h/2), (cfg.cup_top_radius, h/2),
              (cfg.cup_top_radius-t, h/2),
              (cfg.cup_bottom_radius+cfg.taper*cfg.cup_floor-t, -h/2+cfg.cup_floor)]
    return np.array([np.column_stack((r*xy, np.full(segments,z))) for r,z in levels])


def cup_wall_wedges(cfg):
    rings = cup_rings(cfg, WALL_SEGMENTS)
    return [rings[:, [i,(i+1)%WALL_SEGMENTS]].reshape(8,3) for i in range(WALL_SEGMENTS)]


def spawn_cup(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import UsdShade
    from pxr import PhysxSchema
    from piper_bench.tasks.turntable_cups.config import CupConfig
    cfg = CupConfig(**spawn_cfg.cup_config)
    # Reuse the standard rigid-body/mass/material authoring, replacing its solid box.
    prim = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage, base = prim.GetStage(), str(prim.GetPath())
    original = stage.GetPrimAtPath(base+'/geometry/mesh')
    material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial()
    physics_material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial('physics')
    stage.RemovePrim(base+'/geometry/mesh')
    geometry = stage.GetPrimAtPath(base+'/geometry')
    UsdShade.MaterialBindingAPI.Apply(geometry).Bind(material)

    def mesh(name, points, faces):
        m = UsdGeom.Mesh.Define(stage,base+'/geometry/'+name)
        m.CreatePointsAttr(np.asarray(points).tolist())
        m.CreateFaceVertexCountsAttr([len(f) for f in faces])
        m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        m.CreateSubdivisionSchemeAttr('none')
        m.CreateDoubleSidedAttr(True)
        return m

    def collider(m):
        m.CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
        api=PhysxSchema.PhysxCollisionAPI.Apply(m.GetPrim())
        api.CreateContactOffsetAttr(.0005)
        api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(physics_material, materialPurpose='physics')

    # Eight vertices per wedge form a thin closed solid; the central cavity stays empty.
    wedge_faces = [[0,1,3,2],[2,3,5,4],[4,5,7,6],[6,7,1,0],[0,2,4,6],[1,7,5,3]]
    for i, points in enumerate(cup_wall_wedges(cfg)):
        m = mesh(f'wall_{i:02d}',points,wedge_faces)
        collider(m)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')
    bottom = UsdGeom.Cylinder.Define(stage,base+'/geometry/bottom_collision')
    bottom.CreateRadiusAttr(cfg.cup_bottom_radius)
    bottom.CreateHeightAttr(cfg.cup_floor)
    bottom.CreateAxisAttr('Z')
    bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,-cfg.cup_height/2+cfg.cup_floor/2))
    collider(bottom)
    n=VISUAL_SEGMENTS
    faces=[]
    for ring in range(3):
        for i in range(n):
            j=(i+1)%n
            faces.append([ring*n+i,ring*n+j,(ring+1)*n+j,(ring+1)*n+i])
    faces += [list(range(n-1,-1,-1)),list(range(3*n,4*n))]
    mesh('open_cup',cup_rings(cfg,n).reshape(-1,3),faces)
    return prim


def spawn_brown_basket(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    from piper_bench.tasks.conveyor.basket import spawn_basket
    from pxr import UsdShade
    from pxr import Gf
    prim=spawn_basket(prim_path,spawn_cfg,translation,orientation,**kwargs)
    shader=UsdShade.Shader(prim.GetStage().GetPrimAtPath(str(prim.GetPath())+'/red_plastic/Shader'))
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(.37,.24,.17))
    shader.GetInput('roughness').Set(.6)
    return prim


def add_stone_surface(stage, parent_path, radius):
    """Fine circular mesh with deterministic cream stone veins; rotates with disk marks."""
    from pxr import UsdGeom
    from pxr import UsdShade
    from pxr import Sdf
    n, rings=256,64
    angles=np.arange(n)*2*np.pi/n
    points=[[0.,0.,.00015]]
    for r in np.linspace(radius/rings,radius,rings):
        points.extend(np.column_stack((r*np.cos(angles),r*np.sin(angles),np.full(n,.00015))).tolist())
    faces=[[0,1+i,1+(i+1)%n] for i in range(n)]
    for r in range(rings-1):
        a,b=1+r*n,1+(r+1)*n
        faces.extend([[a+i,b+i,b+(i+1)%n,a+(i+1)%n] for i in range(n)])
    pts=np.asarray(points);x,y=pts[:,0],pts[:,1]
    phase=55*x+18*y+1.7*np.sin(29*y)+.35*np.sin(91*x+67*y)
    veins=np.exp(-np.square(np.sin(phase)/.10))
    fine=.5+.5*np.sin(143*x+37*y+np.sin(87*y))
    colors=np.array([.38,.34,.27])[None,:]-veins[:,None]*np.array([.055,.05,.045])-.016*fine[:,None]
    mesh=UsdGeom.Mesh.Define(stage,parent_path+'/StoneSurface')
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDisplayColorPrimvar('vertex').Set(colors.tolist())
    mat=UsdShade.Material.Define(stage,parent_path+'/StoneMaterial')
    shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
    reader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Colors')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor')
    reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(),'result')
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
