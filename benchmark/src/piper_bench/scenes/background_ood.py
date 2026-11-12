"""Visual-only OOD room: matte grey plaster and pale beige oak herringbone.

Deterministic USD vertex colours preserve the physical scene and need no image
textures. All dimensions are metres, colours are linear diffuse reflectances.
"""
import math
import random

BACKGROUND_ID = 'piper_gray_plaster_light_herringbone_matte_v2'
WALL_COLOR = (.36, .36, .36)
FLOOR_COLOR = (.62, .56, .47)
BOARD_WIDTH = .12
BOARD_LENGTH = .48
ROOM_BOUNDS = (-3.7, 3.7, -2., 2.)


def metadata():
    return dict(background_id=BACKGROUND_ID, wall='neutral grey matte plaster',
                floor='pale beige oak herringbone', wall_linear_rgb=WALL_COLOR,
                floor_linear_rgb=FLOOR_COLOR, board_width_m=BOARD_WIDTH,
                board_length_m=BOARD_LENGTH, board_rotation_deg=45,
                roughness=1., specular_rgb=(0., 0., 0.), seed=20261020,
                collision=False, changes_table=False, changes_lights=False,
                changes_cameras=False, split='background_ood')


def herringbone_boards():
    """Tile the plane with perpendicular 4:1 rectangles, then rotate by 45°.

    On the integer grid, (x+y) modulo 8 selects a horizontal run of four
    cells or a vertical run of four. The two runs partition the plane.
    """
    rng = random.Random(20261020)
    angle = math.pi/4
    c, s = math.cos(angle), math.sin(angle)
    for row in range(-40, 41):
        for cycle in range(-12, 13):
            for x, y, vertical in ((8*cycle-row, row, False),
                                    (row, 4-row+8*cycle, True)):
                dx, dy = (1, 4) if vertical else (4, 1)
                u, v = (x+dx/2)*BOARD_WIDTH, (y+dy/2)*BOARD_WIDTH
                cx, cy = c*u-s*v, s*u+c*v
                if abs(cx) > 4. or abs(cy) > 2.3:
                    continue
                yield cx, cy, angle+(math.pi/2 if vertical else 0.), rng.uniform(.95, 1.04), rng.uniform(0., 2*math.pi)


def clip_to_floor(polygon):
    """Clip a convex 2-D polygon to the unchanged rectangular room footprint."""
    for axis, edge, sign in ((0, -3.7, 1), (0, 3.7, -1),
                              (1, -2., 1), (1, 2., -1)):
        if not polygon:
            break
        result = []
        a = polygon[-1]
        a_in = sign*(a[axis]-edge) >= 0
        for b in polygon:
            b_in = sign*(b[axis]-edge) >= 0
            if a_in != b_in:
                t = (edge-a[axis])/(b[axis]-a[axis])
                result.append(tuple(a[i]+t*(b[i]-a[i]) for i in range(2)))
            if b_in:
                result.append(b)
            a, a_in = b, b_in
        polygon = result
    return polygon


def floor_mesh():
    points, counts, indices, colors = [], [], [], []
    gap = .0008
    length, width = BOARD_LENGTH-gap, BOARD_WIDTH-gap
    for cx, cy, angle, shade, phase in herringbone_boards():
        c, s = math.cos(angle), math.sin(angle)
        for segment in range(10):
            for band in range(8):
                u0, u1 = -length/2+length*segment/10, -length/2+length*(segment+1)/10
                v0, v1 = -width/2+width*band/8, -width/2+width*(band+1)/8
                polygon = clip_to_floor([(cx+c*u-s*v, cy+s*u+c*v)
                                         for u, v in ((u0,v0),(u1,v0),(u1,v1),(u0,v1))])
                if len(polygon) < 3:
                    continue
                tone = shade*(1.+.020*math.sin(band*1.9+phase+.30*segment)
                              +.006*math.sin(band*4.2+phase-.12*segment))
                first = len(points)
                points.extend((x, y, .011) for x, y in polygon)
                counts.append(len(polygon))
                indices.extend(range(first, len(points)))
                colors.extend([tuple(v*tone for v in FLOOR_COLOR)]*len(polygon))
    return points, counts, indices, colors


def wall_mesh(height=2.8):
    """Subtle continuous plaster variation on the four existing inside faces."""
    points, counts, indices, colors = [], [], [], []
    for axis, fixed, span in ((0,-3.699,2.),(0,3.699,2.),
                               (1,-1.999,3.7),(1,1.999,3.7)):
        nx, nz = math.ceil(2*span/.10), math.ceil(height/.10)
        first = len(points)
        for iz in range(nz+1):
            z = height*iz/nz
            for ix in range(nx+1):
                u = -span+2*span*ix/nx
                points.append((fixed,u,z) if axis == 0 else (u,fixed,z))
                tone = (1.+.018*math.sin(3.1*u+1.8*z)*math.cos(2.7*z-.8*u)
                        +.008*math.sin(7.3*u-4.5*z))
                colors.append(tuple(v*tone for v in WALL_COLOR))
        for iz in range(nz):
            for ix in range(nx):
                a = first+iz*(nx+1)+ix
                indices.extend((a,a+1,a+nx+2,a+nx+1))
                counts.append(4)
    return points, counts, indices, colors


def spawn_ood_room(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    """Reuse the standard room; modify only its visual materials and surfaces."""
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdShade
    from piper_bench.scenes.background import spawn_room
    root = spawn_room(prim_path, spawn_cfg, translation, orientation, **kwargs)
    stage, base = root.GetStage(), str(root.GetPath())
    for name, color in (('WarmWhite', WALL_COLOR), ('Skirting', (.64,.64,.61)),
                         ('FloorBase', (.42,.38,.32))):
        shader = UsdShade.Shader.Get(stage, base+'/Looks/'+name+'/Shader')
        shader.GetInput('diffuseColor').Set(Gf.Vec3f(*color))
    # The inherited material reads displayColor and has zero specular/clearcoat.
    material = UsdShade.Material.Get(stage, base+'/Looks/Wood')
    for name, data in (('WoodFloor', floor_mesh()), ('PlasterSurface', wall_mesh())):
        points, counts, indices, colors = data
        mesh = UsdGeom.Mesh.Define(stage, base+'/'+name)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorPrimvar('vertex').Set(colors)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    root.CreateAttribute('piper:background_id', Sdf.ValueTypeNames.String).Set(BACKGROUND_ID)
    root.CreateAttribute('piper:background_split', Sdf.ValueTypeNames.String).Set('background_ood')
    return root
