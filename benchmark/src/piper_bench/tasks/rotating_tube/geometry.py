"""Small perforated carousel and a capped, round-bottom test tube."""
import math
import numpy as np


def plate_cells(cfg, samples=96):
    """Tile an annulus with star-shaped sectors, each containing one open hole.

    Rays from a hole center stop at the sector's inner/outer circular boundary
    or radial sides. The resulting rings leave real openings in the collider.
    """
    half = math.pi/cfg.holes
    centers = np.array([cfg.hole_orbit, 0.])
    beta = np.arange(samples)*2*math.pi/samples
    directions = np.column_stack((np.cos(beta), np.sin(beta)))
    outer = []
    for d in directions:
        dot = float(centers@d)
        candidates = [-dot+math.sqrt(dot*dot+cfg.rack_radius**2-cfg.hole_orbit**2)]
        discriminant = dot*dot+cfg.plate_inner_radius**2-cfg.hole_orbit**2
        if discriminant >= 0:
            candidates += [t for t in (-dot-math.sqrt(discriminant), -dot+math.sqrt(discriminant)) if t > 1e-8]
        for a in (-half, half):
            n = np.array([-math.sin(a), math.cos(a)])
            denominator = n@d
            if abs(denominator)>1e-10:
                t = -float(n@centers)/float(denominator)
                if t>0:
                    point = centers+t*d
                    if point@np.array([math.cos(a),math.sin(a)])>0:
                        candidates.append(t)
        outer.append(centers+min(candidates)*d)
    inner = centers+cfg.hole_radius*directions
    cells = []
    for i in range(cfg.holes):
        a = i*2*math.pi/cfg.holes
        rotation = np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        cells.append((inner@rotation.T,np.asarray(outer)@rotation.T))
    return cells


def extruded_ring(inner, outer, low, high):
    n=len(inner)
    points=np.vstack([np.column_stack((ring,np.full(n,z)))
                      for z,ring in ((low,inner),(low,outer),(high,inner),(high,outer))])
    faces=[]
    for i in range(n):
        j=(i+1)%n
        # Triangles on the top and bottom, quads around the inner/outer walls.
        faces += [[2*n+i,3*n+i,3*n+j],[2*n+i,3*n+j,2*n+j],
                  [i,n+j,n+i],[i,j,n+j], [i,2*n+i,2*n+j,j],
                  [n+i,n+j,3*n+j,3*n+i]]
    return points,faces


def _materials(prim, color):
    from pxr import UsdShade
    from pxr import Sdf
    from pxr import Gf
    stage=prim.GetStage();base=str(prim.GetPath())
    mesh=stage.GetPrimAtPath(base+'/geometry/mesh')
    physics,_=UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial('physics')
    stage.RemovePrim(base+'/geometry/mesh')
    mat=UsdShade.Material.Define(stage,base+'/finish')
    shader=UsdShade.Shader.Define(stage,base+'/finish/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.38)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    return stage,base,mat,physics


def _collider(shape, material, physics, *, mesh=False):
    from pxr import UsdPhysics
    from pxr import PhysxSchema
    from pxr import UsdShade
    prim=shape.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    if mesh:
        # Concave triangle meshes are supported by the kinematic rack. Never
        # convex-hull the complete plate, which would close every hole.
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    contact=PhysxSchema.PhysxCollisionAPI.Apply(prim)
    contact.CreateContactOffsetAttr(.0003);contact.CreateRestOffsetAttr(0.)
    binding=UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(material)
    if physics and physics.GetPrim().IsValid():binding.Bind(physics,materialPurpose='physics')


def cylinder_mesh(radius,height,z,segments=128):
    """Smooth circular visual, independent of USD's low-resolution cylinder display."""
    angle=np.arange(segments)*2*math.pi/segments
    radial=np.column_stack((np.cos(angle),np.sin(angle),np.zeros(segments)))
    points=np.vstack((radial*radius+[0,0,z-height/2],radial*radius+[0,0,z+height/2],
                      [[0,0,z-height/2],[0,0,z+height/2]]))
    faces=[];normals=[]
    for i in range(segments):
        j=(i+1)%segments
        faces += [[i,j,segments+j,segments+i],[2*segments,j,i],[2*segments+1,segments+i,segments+j]]
        normals += [radial[i],radial[j],radial[j],radial[i],*[np.array([0,0,-1.])]*3,*[np.array([0,0,1.])]*3]
    return points,faces,np.asarray(normals)


def _cylinder(stage,path,radius,height,z,material,physics):
    from pxr import UsdGeom
    from pxr import Gf
    shape=UsdGeom.Cylinder.Define(stage,path)
    shape.CreateRadiusAttr(radius);shape.CreateHeightAttr(height);shape.CreateAxisAttr('Z')
    shape.AddTranslateOp().Set(Gf.Vec3d(0,0,z))
    _collider(shape,material,physics)
    shape.CreatePurposeAttr('guide')
    # Keep the analytic collider and replace only its faceted render surface.
    visual=UsdGeom.Mesh.Define(stage,path+'_visual')
    points,faces,normals=cylinder_mesh(radius,height,z)
    visual.CreatePointsAttr(points.tolist());visual.CreateFaceVertexCountsAttr([len(f) for f in faces])
    visual.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
    visual.CreateNormalsAttr(normals.tolist());visual.SetNormalsInterpolation('faceVarying')
    visual.CreateSubdivisionSchemeAttr('none')
    from pxr import UsdShade
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)
    return shape


def spawn_rack(path,spawn_cfg,translation=None,orientation=None,**kwargs):
    import isaaclab.sim as sim
    from pxr import UsdGeom
    from piper_bench.tasks.rotating_tube.config import TubeConfig
    cfg=TubeConfig(**spawn_cfg.tube_config)
    prim=sim.spawn_cuboid(path,spawn_cfg,translation,orientation,**kwargs)
    stage,base,mat,physics=_materials(prim,(.48,.30,.18))
    _cylinder(stage,base+'/geometry/bottom_tray',cfg.rack_radius,cfg.base_thickness,cfg.base_thickness/2,mat,physics)
    _cylinder(stage,base+'/geometry/center_post',.014,cfg.plate_top-cfg.base_thickness,
              (cfg.plate_top+cfg.base_thickness)/2,mat,physics)
    _cylinder(stage,base+'/geometry/plate_center',cfg.plate_inner_radius,
              cfg.plate_thickness,cfg.plate_top-cfg.plate_thickness/2,mat,physics)
    for i,(inner,outer) in enumerate(plate_cells(cfg)):
        points,faces=extruded_ring(inner,outer,cfg.plate_top-cfg.plate_thickness,cfg.plate_top)
        shape=UsdGeom.Mesh.Define(stage,base+f'/geometry/perforated_sector_{i}')
        shape.CreatePointsAttr(points.tolist());shape.CreateFaceVertexCountsAttr([len(f) for f in faces])
        shape.CreateFaceVertexIndicesAttr([v for f in faces for v in f])
        shape.CreateSubdivisionSchemeAttr('none');shape.CreateDoubleSidedAttr(True)
        _collider(shape,mat,physics,mesh=True)
    return prim


def spawn_tube(path,spawn_cfg,translation=None,orientation=None,**kwargs):
    import isaaclab.sim as sim
    from pxr import UsdGeom
    from pxr import UsdShade
    from pxr import Sdf
    from pxr import Gf
    from piper_bench.tasks.rotating_tube.config import TubeConfig
    cfg=TubeConfig(**spawn_cfg.tube_config)
    prim=sim.spawn_cuboid(path,spawn_cfg,translation,orientation,**kwargs)
    stage,base,body,physics=_materials(prim,(.64,.76,.78))
    bottom=-cfg.tube_length/2+cfg.tube_radius
    top=cfg.tube_length/2-cfg.cap_height
    _cylinder(stage,base+'/geometry/tube_body',cfg.tube_radius,top-bottom,(top+bottom)/2,body,physics)
    sphere=UsdGeom.Sphere.Define(stage,base+'/geometry/round_bottom')
    sphere.CreateRadiusAttr(cfg.tube_radius);sphere.AddTranslateOp().Set(Gf.Vec3d(0,0,bottom))
    _collider(sphere,body,physics)
    cap=UsdShade.Material.Define(stage,base+'/cap_material')
    shader=UsdShade.Shader.Define(stage,base+'/cap_material/Shader');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.10,.39,.68))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.55)
    cap.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    _cylinder(stage,base+'/geometry/blue_cap',cfg.cap_radius,cfg.cap_height,
              cfg.tube_length/2-cfg.cap_height/2,cap,physics)
    return prim
