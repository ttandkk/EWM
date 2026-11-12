"""Warm white walls and brown wood floor by default; visual-only room profiles."""
import math
import random

BACKGROUND_ID = 'piper_warm_white_brown_planks_matte_v4'
ORIGINAL_BACKGROUND_ID = 'piper_original_open_grid_v1'
FLOOR_BACKGROUND_ID = 'piper_brown_planks_floor_only_v1'
# Linear diffuse colors; no lights, exposure, cameras or task physics are changed.
WALL_COLOR = (.72, .70, .66)
FLOOR_COLOR = (.15, .125, .10)


def floor_metadata():
    return dict(background_id=FLOOR_BACKGROUND_ID, wall=None,
                floor='muted brown staggered wood planks', floor_linear_rgb=FLOOR_COLOR,
                roughness=1., specular_rgb=(0., 0., 0.), seed=731,
                collision=False, changes_table=False, changes_lights=False,
                changes_cameras=False, split='background_ood')


def floor_boards():
    """Deterministic staggered planks, independent of episode random seeds."""
    rng = random.Random(731)
    for row in range(20):
        x = -3.7
        first = True
        while x < 3.7 - 1e-8:
            length = min((.4 + (row % 3)*.35) if first else 1.2, 3.7-x)
            yield (x+length/2, -2.+(row+.5)*.2, length-.0015, .1985,
                   rng.uniform(.83, 1.17), rng.uniform(0., 2*math.pi))
            x += length
            first = False


def spawn_room(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdShade

    # Use the stock spawner to retain its environment cloning/transform handling.
    root = sim.spawn_cuboid(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage, base = root.GetStage(), str(root.GetPath())

    def material(name, color):
        mat = UsdShade.Material.Define(stage, base+'/Looks/'+name)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
        shader.CreateInput('clearcoat', Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        return mat

    floor_base = material('FloorBase', (.065, .055, .045))
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(base+'/geometry')).Bind(
        floor_base, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    wall = material('WarmWhite', WALL_COLOR)
    trim = material('Skirting', (.59, .57, .53))

    def box(name, pos, size, mat):
        geom = UsdGeom.Cube.Define(stage, base+'/'+name)
        geom.CreateSizeAttr(1.)
        geom.AddTranslateOp().Set(Gf.Vec3d(*pos))
        geom.AddScaleOp().Set(Gf.Vec3d(*size))
        UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)

    # Keep the ceiling open; all existing cameras remain inside the room.
    box('BackWall', (-3.74, 0., 1.4), (.08, 4.16, 2.8), wall)
    box('FrontWall', (3.74, 0., 1.4), (.08, 4.16, 2.8), wall)
    box('LeftWall', (0., -2.04, 1.4), (7.4, .08, 2.8), wall)
    box('RightWall', (0., 2.04, 1.4), (7.4, .08, 2.8), wall)
    box('BackSkirting', (-3.693, 0., .055), (.014, 4., .11), trim)
    box('FrontSkirting', (3.693, 0., .055), (.014, 4., .11), trim)
    box('LeftSkirting', (0., -1.993, .055), (7.4, .014, .11), trim)
    box('RightSkirting', (0., 1.993, .055), (7.4, .014, .11), trim)

    # Native USD vertex colors provide restrained longitudinal wood grain without
    # external texture downloads or fine/high-contrast patterns in event streams.
    points, counts, indices, colors = [], [], [], []
    for x, y, length, width, shade, phase in floor_boards():
        for band in range(12):
            for segment in range(8):
                tone = shade*(1.+.045*math.sin(band*1.8+phase+segment*.35))
                color = tuple(c*tone for c in FLOOR_COLOR)
                x0, x1 = x-length/2+length*segment/8, x-length/2+length*(segment+1)/8
                y0, y1 = y-width/2+width*band/12, y-width/2+width*(band+1)/12
                first = len(points)
                points.extend([(x0,y0,.011),(x1,y0,.011),(x1,y1,.011),(x0,y1,.011)])
                indices.extend(range(first, first+4)); counts.append(4)
                colors.extend([color]*4)
    mesh = UsdGeom.Mesh.Define(stage, base+'/WoodFloor')
    mesh.CreatePointsAttr(points); mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices); mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDisplayColorPrimvar('vertex').Set(colors)
    mat = material('Wood', FLOOR_COLOR)
    shader = UsdShade.Shader.Get(stage, str(mat.GetPath())+'/Shader')
    reader = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Color')
    reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('displayColor')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float3)
    shader.GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(), 'result')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    root.CreateAttribute('piper:background_id', Sdf.ValueTypeNames.String).Set(BACKGROUND_ID)
    return root


def spawn_wood_floor(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    """Reuse the validated wood floor, removing enclosure geometry before rendering."""
    root = spawn_room(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage = root.GetStage()
    for child in list(root.GetChildren()):
        if child.GetName().endswith(('Wall', 'Skirting')):
            stage.RemovePrim(child.GetPath())
    root.GetAttribute('piper:background_id').Set(FLOOR_BACKGROUND_ID)
    return root


def configure_room_background(scene, *, profile=BACKGROUND_ID):
    """Select before env construction; default to the matte walls and wood floor.

    The original open grid and floor-only OOD remain explicit profile options.
    """
    if profile == ORIGINAL_BACKGROUND_ID:
        scene.room_background = None
        return
    from piper_bench.scenes.background_ood import BACKGROUND_ID as OOD_ID
    from piper_bench.scenes.background_ood import spawn_ood_room
    if profile not in (BACKGROUND_ID, FLOOR_BACKGROUND_ID, OOD_ID):
        raise ValueError(f'unknown room background profile: {profile}')
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg
    ground = getattr(scene, 'plane', None) or getattr(scene, 'ground', None)
    floor_z = ground.init_state.pos[2] if ground is not None else -1.05
    spawner = {BACKGROUND_ID: spawn_room, FLOOR_BACKGROUND_ID: spawn_wood_floor,
               OOD_ID: spawn_ood_room}[profile]
    scene.room_background = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/RoomBackground',
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-.9, 0., floor_z)),
        spawn=sim.CuboidCfg(func=spawner, size=(7.4, 4., .02),
            visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.065, .055, .045), roughness=1.)),
    )
