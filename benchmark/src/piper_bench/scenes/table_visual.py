"""Remove SeattleLabTable's side camera stand while retaining the physical table."""
import numpy as np

VISUAL_ID = 'piper_table_without_camera_stand_v1'


def _stand_faces(points, counts, indices):
    """Identify whole disconnected stand parts in the table's local metre frame."""
    # Weld coincident vertices only for connectivity; keep the original mesh data.
    _, welded = np.unique(np.round(points, 6), axis=0, return_inverse=True)
    parent = np.arange(welded.max() + 1)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    starts = np.r_[0, np.cumsum(counts)[:-1]]
    for start, count in zip(starts, counts):
        vertices = welded[indices[start:start + count]]
        first = root(vertices[0])
        for vertex in vertices[1:]:
            parent[root(vertex)] = first
    groups = np.array([root(index) for index in welded])
    low = np.full((len(parent), 3), np.inf)
    high = np.full((len(parent), 3), -np.inf)
    np.minimum.at(low, groups, points)
    np.maximum.at(high, groups, points)
    # The stand runs out from the +X side of this asset, in a 10 cm centre strip.
    # Select whole parts, including the rail below the tabletop and its fasteners;
    # clipping at the table edge would leave the inner end of the rail behind.
    stand = (high[:, 0] > .46) & (low[:, 1] > -.05) & (high[:, 1] < .05)
    return stand[groups[indices[starts]]]


def remove_camera_stand(table):
    """Author only a visual-mesh override; leave Collisions and source USD intact."""
    from pxr import Sdf
    from pxr import UsdGeom

    marker = table.GetAttribute('piper:table_visual_id')
    if marker and marker.Get() == VISUAL_ID:
        return
    visuals = table.GetChild('Visuals')
    if not visuals:
        raise ValueError('expected SeattleLabTable Visuals')
    # TableGeom is an instance proxy in the stock asset, so edits need a local
    # Visuals instance. The separate Collisions instance remains untouched.
    visuals.SetInstanceable(False)
    mesh = UsdGeom.Mesh(visuals.GetChild('TableGeom'))
    if not mesh:
        raise ValueError('expected SeattleLabTable Visuals/TableGeom mesh')
    points = np.asarray(mesh.GetPointsAttr().Get())
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(mesh.GetPrim(), table)
    matrix = np.asarray(transform)
    local = points @ matrix[:3, :3] + matrix[3, :3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    keep = ~_stand_faces(local, counts, indices)
    if keep.all() or not keep.any():
        raise ValueError('SeattleLabTable camera stand geometry was not recognized')
    corners = np.repeat(keep, counts)
    remap = np.cumsum(keep) - 1

    def filter_values(attribute, mask):
        values = attribute.Get()
        attribute.Set(type(values).FromNumpy(np.ascontiguousarray(np.asarray(values)[mask])))

    # Face-varying UVs/tangents/normals and per-face materials must follow the
    # retained faces. Vertex data can stay unchanged because indices are stable.
    normals = mesh.GetNormalsAttr()
    if normals.HasValue():
        interpolation = mesh.GetNormalsInterpolation()
        if interpolation in ('faceVarying', 'uniform'):
            filter_values(normals, corners if interpolation == 'faceVarying' else keep)
    for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        interpolation = primvar.GetInterpolation()
        if primvar.HasValue() and interpolation in ('faceVarying', 'uniform'):
            attribute = primvar.GetIndicesAttr() if primvar.IsIndexed() else primvar.GetAttr()
            filter_values(attribute, corners if interpolation == 'faceVarying' else keep)
    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        if subset.GetElementTypeAttr().Get() != 'face':
            raise ValueError('expected face material subsets on SeattleLabTable')
        old = np.asarray(subset.GetIndicesAttr().Get(), dtype=np.int64)
        subset.GetIndicesAttr().Set(remap[old[keep[old]]].tolist())
    holes = np.asarray(mesh.GetHoleIndicesAttr().Get(), dtype=np.int64)
    mesh.GetHoleIndicesAttr().Set(remap[holes[keep[holes]]].tolist())
    mesh.GetFaceVertexCountsAttr().Set(counts[keep].tolist())
    mesh.GetFaceVertexIndicesAttr().Set(indices[corners].tolist())
    used = points[np.unique(indices[corners])]
    mesh.GetExtentAttr().Set([used.min(axis=0).tolist(), used.max(axis=0).tolist()])
    table.CreateAttribute('piper:table_visual_id', Sdf.ValueTypeNames.String).Set(VISUAL_ID)


def spawn_table_without_camera_stand(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim
    from isaaclab.sim.utils import find_matching_prims

    table = sim.spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)
    for instance in find_matching_prims(prim_path, stage=table.GetStage()):
        remove_camera_stand(instance)
    return table
