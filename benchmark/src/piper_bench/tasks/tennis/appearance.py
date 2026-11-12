"""Smooth felt shell and a closed white seam, independent of ball collision."""
import numpy as np

BALL_APPEARANCE = 'tennis_felt_seam_v2'


def sphere_mesh(radius, rings=96, segments=192):
    """Watertight sphere with analytic smooth normals and real pole vertices."""
    theta = np.arange(1, rings)*np.pi/rings
    phi = np.arange(segments)*2*np.pi/segments
    normals = np.column_stack((np.outer(np.sin(theta), np.cos(phi)).ravel(),
                               np.outer(np.sin(theta), np.sin(phi)).ravel(),
                               np.repeat(np.cos(theta), segments)))
    normals = np.vstack(([0., 0., 1.], normals, [0., 0., -1.]))
    faces = []
    for j in range(segments):
        k = (j+1) % segments
        faces.append([0, 1+j, 1+k])
        for i in range(rings-2):
            a, b = 1+i*segments+j, 1+i*segments+k
            faces.extend(([a, a+segments, b+segments], [a, b+segments, b]))
        a, b = 1+(rings-2)*segments+j, 1+(rings-2)*segments+k
        faces.append([a, len(normals)-1, b])
    return normals*radius, np.asarray(faces), normals


def seam_curve(t):
    # One closed, sinuous seam separates two felt panels on the sphere.
    latitude = .72*np.sin(2*t)
    return np.column_stack((np.cos(latitude)*np.cos(t),
                            np.cos(latitude)*np.sin(t), np.sin(latitude)))


def seam_mesh(radius, samples=768, width=.0021):
    t = np.arange(samples)*2*np.pi/samples
    center = seam_curve(t)
    tangent = seam_curve(t+1e-4)-seam_curve(t-1e-4)
    side = np.cross(center, tangent); side /= np.linalg.norm(side, axis=1)[:, None]
    offsets = np.linspace(-width/2, width/2, 5)
    points = center[:, None, :]*radius+side[:, None, :]*offsets[None, :, None]
    points /= np.linalg.norm(points, axis=2)[..., None]
    normals = points.copy()
    points *= radius+.00006
    faces = [[5*i+j, 5*((i+1) % samples)+j, 5*((i+1) % samples)+j+1, 5*i+j+1]
             for i in range(samples) for j in range(4)]
    return points.reshape(-1, 3), np.asarray(faces), normals.reshape(-1, 3)


def add_felt_visual(prim, radius):
    from pxr import UsdGeom
    from pxr import UsdShade
    from pxr import Sdf
    from pxr import Gf
    from pxr import Vt
    stage, base = prim.GetStage(), str(prim.GetPath())
    # Keep the original analytic sphere and all its physics bindings untouched.
    UsdGeom.Imageable(stage.GetPrimAtPath(base+'/geometry/mesh')).MakeInvisible()
    material = UsdShade.Material.Define(stage, base+'/FeltMaterial')
    shader = UsdShade.Shader.Define(stage, base+'/FeltMaterial/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    for name, value in [('roughness', 1.), ('metallic', 0.), ('clearcoat', 0.)]:
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
    reader = UsdShade.Shader.Define(stage, base+'/FeltMaterial/Color')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), 'result')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    rng = np.random.default_rng(731)
    def mesh(name, points, faces, normals, colors):
        shape = UsdGeom.Mesh.Define(stage, base+'/'+name)
        shape.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        shape.CreateFaceVertexCountsAttr([faces.shape[1]]*len(faces))
        shape.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        shape.CreateSubdivisionSchemeAttr('none')
        shape.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        shape.SetNormalsInterpolation('vertex')
        shape.CreateDisplayColorPrimvar('vertex').Set(Vt.Vec3fArray.FromNumpy(colors.astype(np.float32)))
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    points, faces, normals = sphere_mesh(radius)
    green = np.array([.38, .49, .014])
    colors = green[None, :]*rng.uniform(.78, 1.12, (len(points), 1))
    mesh('FeltShell', points, faces, normals, colors)
    points, faces, normals = seam_mesh(radius)
    # Narrow olive shoulders make the ivory seam read as an inset, not paint.
    strip = np.array([[.28, .32, .075], [.72, .73, .59], [.84, .84, .70], [.72, .73, .59], [.28, .32, .075]])
    mesh('WhiteSeam', points, faces, normals, np.tile(strip, (len(points)//5, 1)))
    # Tiny non-colliding felt fibres soften the silhouette and catch diffuse light.
    directions = rng.normal(size=(18000, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
    lat = np.arcsin(directions[:, 2]); lon = np.arctan2(directions[:, 1], directions[:, 0])
    directions = directions[np.abs(lat-.72*np.sin(2*lon)) > .045]
    tangent = np.cross(directions, rng.normal(size=directions.shape))
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-9)
    roots = directions*(radius+.000015)
    tips = roots+directions*rng.uniform(.00006, .00018, (len(roots), 1))+tangent*.00016
    fibres = UsdGeom.BasisCurves.Define(stage, base+'/FeltFibres')
    fibres.CreateTypeAttr('linear'); fibres.CreateWrapAttr('nonperiodic')
    fibres.CreateCurveVertexCountsAttr([2]*len(roots))
    fibres.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.stack((roots, tips), axis=1).reshape(-1, 3).astype(np.float32)))
    fibres.CreateWidthsAttr([.000035]); fibres.SetWidthsInterpolation('constant')
    fibres.CreateDisplayColorPrimvar('constant').Set([Gf.Vec3f(*green)])
    UsdShade.MaterialBindingAPI.Apply(fibres.GetPrim()).Bind(material)
    prim.CreateAttribute('piper:ball_appearance', Sdf.ValueTypeNames.String).Set(BALL_APPEARANCE)
