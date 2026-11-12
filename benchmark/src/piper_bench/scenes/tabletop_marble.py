"""Optional grey-green marble tabletop; visual changes only, in metres."""
import math
import random

MATERIAL_ID = 'piper_grey_green_gold_marble_v2'
STONE_COLOR = (.12, .13, .11)
VEIN_COLOR = (.50, .44, .32)
LOW_CONTRAST_MATERIAL_ID = 'piper_low_contrast_matte_stone_v1'
LOW_CONTRAST_STONE = (.30, .31, .30)
LOW_CONTRAST_VEIN = (.33, .34, .33)
# SeattleLabTable's existing top reaches y=-.45974 and +.45015 in the
# CleanTabletop frame; the .9 m overlay otherwise leaves its dark rim exposed.
SIDE_COVER_NEGATIVE_Y = .010
SIDE_COVER_POSITIVE_Y = .001


def metadata(*, low_contrast=False):
    if low_contrast:
        return dict(material_id=LOW_CONTRAST_MATERIAL_ID, stone_linear_rgb=LOW_CONTRAST_STONE,
                    vein_linear_rgb=LOW_CONTRAST_VEIN, roughness=1., specular_rgb=(0., 0., 0.),
                    seed=20261020, changes_collision=False, changes_physical_table_dimensions=False,
                    visual_side_cover_m=[0., 0.])
    return dict(material_id=MATERIAL_ID, stone_linear_rgb=STONE_COLOR,
                vein_linear_rgb=VEIN_COLOR, roughness=.65, seed=20261020,
                changes_collision=False, changes_physical_table_dimensions=False,
                visual_side_cover_m=[SIDE_COVER_NEGATIVE_Y, SIDE_COVER_POSITIVE_Y])


def marble_meshes(size, *, low_contrast=False):
    """Clouded stone and sparse, branching mineral veins, without image assets."""
    sx, sy, sz = size
    ymin, ymax = ((-sy/2, sy/2) if low_contrast else
                  (-sy/2-SIDE_COVER_NEGATIVE_Y, sy/2+SIDE_COVER_POSITIVE_Y))
    stone_color = LOW_CONTRAST_STONE if low_contrast else STONE_COLOR
    vein_color = LOW_CONTRAST_VEIN if low_contrast else VEIN_COLOR
    cloud, grain = (.02, .008) if low_contrast else (.07, .025)
    z = sz/2 + .00002
    points, counts, indices, colors = [], [], [], []
    nx, ny = 96, 72
    for j in range(ny+1):
        y = ymin+(ymax-ymin)*j/ny
        for i in range(nx+1):
            x = sx*(i/nx-.5)
            tone = 1.+cloud*math.sin(9*x+4*y)*math.cos(6*y-2*x)+grain*math.sin(37*x+19*y)
            points.append((x,y,z))
            colors.append(tuple(c*tone for c in stone_color))
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            indices.extend((a,a+1,a+nx+2,a+nx+1))
            counts.append(4)
    stone = points, counts, indices, colors

    points, counts, indices, colors = [], [], [], []
    rng = random.Random(20261020)
    for line in range(11):
        offset = sy*(-.65+line*.13)+rng.uniform(-.025,.025)
        phase, width = rng.uniform(0,2*math.pi), rng.uniform(.0006,.0015)
        branch_start = rng.uniform(-.35,.05)*sx

        def center(x):
            return (offset+.32*x+.018*math.sin(9*x+phase)
                    +.005*math.sin(31*x+phase)+.0015*math.sin(109*x+phase))

        for branch in (False, True):
            start = branch_start if branch else -sx/2
            for segment in range(240):
                x0 = start+(sx/2-start)*segment/240
                x1 = start+(sx/2-start)*(segment+1)/240
                ys = [center(x)+(.12*(x-start) if branch else 0.) for x in (x0,x1)]
                w = width*(.55 if branch else 1.)*(.75+.25*math.sin(23*x0+phase))
                if min(ys)-w < ymin or max(ys)+w > ymax:
                    continue
                a = len(points)
                points.extend(((x0,ys[0]-w/2,z+.00002),(x1,ys[1]-w/2,z+.00002),
                               (x1,ys[1]+w/2,z+.00002),(x0,ys[0]+w/2,z+.00002)))
                indices.extend(range(a,a+4))
                counts.append(4)
                tone = ((.98+.02*math.sin(13*x0+phase)**2) if low_contrast else
                        (.8+.2*math.sin(13*x0+phase)**2))
                colors.extend([tuple(c*tone for c in vein_color)]*4)
    return stone, (points, counts, indices, colors)


def spawn_marble_tabletop(prim_path, spawn_cfg, translation=None, orientation=None, *, low_contrast=False, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdShade

    root = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage, base = root.GetStage(), str(root.GetPath())
    profile = metadata(low_contrast=low_contrast)
    specular = 0. if low_contrast else .025
    # Keep the original cube, transforms and any physical properties intact.
    shader = UsdShade.Shader.Get(stage, base+'/geometry/material/Shader')
    shader.GetInput('diffuseColor').Set(Gf.Vec3f(*profile['stone_linear_rgb']))
    shader.GetInput('roughness').Set(profile['roughness'])
    if low_contrast:
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
        shader.CreateInput('clearcoat', Sdf.ValueTypeNames.Float).Set(0.)
    visual = UsdGeom.Scope.Define(stage, base+'/MarbleVisual')
    visual.GetPrim().CreateAttribute('piper:material_id', Sdf.ValueTypeNames.String).Set(profile['material_id'])
    material = UsdShade.Material.Define(stage, base+'/MarbleVisual/Material')
    shader = UsdShade.Shader.Define(stage, str(material.GetPath())+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(profile['roughness'])
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(specular))
    shader.CreateInput('clearcoat', Sdf.ValueTypeNames.Float).Set(0.)
    reader = UsdShade.Shader.Define(stage, str(material.GetPath())+'/Color')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), 'result')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    for name, data in zip(('Stone','Veins'), marble_meshes(spawn_cfg.size, low_contrast=low_contrast)):
        points, counts, indices, colors = data
        mesh = UsdGeom.Mesh.Define(stage, base+'/MarbleVisual/'+name)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorPrimvar('vertex').Set(colors)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return root


def spawn_low_contrast_tabletop(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    return spawn_marble_tabletop(prim_path, spawn_cfg, translation, orientation, low_contrast=True, **kwargs)


def configure_marble_tabletop(scene, *, low_contrast=False):
    """Opt in after the task config is built, before constructing the environment."""
    tabletop = getattr(scene, 'clean_tabletop', None)
    if tabletop is None:
        raise ValueError('marble requires the task clean_tabletop asset')
    import isaaclab.sim as sim
    default_spawner = sim.CuboidCfg(size=tabletop.spawn.size).func
    if tabletop.spawn.func not in (default_spawner, sim.spawn_cuboid, spawn_marble_tabletop, spawn_low_contrast_tabletop):
        raise ValueError('marble cannot replace a custom tabletop spawner')
    tabletop.spawn.func = spawn_low_contrast_tabletop if low_contrast else spawn_marble_tabletop
