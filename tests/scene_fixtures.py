"""Small architectural scenes built in code, for the mcvoxel tests (no files, no network)."""
import numpy as np

from mcvoxel.scene import KIND_GLASS, KIND_SURFACE, Material, SceneMesh

WALL = (0.85, 0.82, 0.75, 1.0)
ROOF = (0.45, 0.25, 0.15, 1.0)


def mesh_to_scene(parts):
    """parts: list of (trimesh, rgba 0..1, kind) -> SceneMesh with one material per part."""
    verts, faces, face_mat, mats = [], [], [], []
    off = 0
    for i, (m, rgba, kind) in enumerate(parts):
        verts.append(np.asarray(m.vertices, float))
        faces.append(np.asarray(m.faces, np.int64) + off)
        face_mat.append(np.full(len(m.faces), i, np.int32))
        mats.append(Material(name="p%d" % i, base_rgba=np.asarray(rgba, float), image=None,
                             image_alpha=None, alpha_mode="BLEND" if kind == KIND_GLASS else "OPAQUE",
                             kind=kind))
        off += len(m.vertices)
    v = np.concatenate(verts)
    return SceneMesh(verts=v, faces=np.concatenate(faces), uv=np.full((len(v), 2), np.nan), vcolor=None,
                     face_mat=np.concatenate(face_mat), materials=mats)


def _box(extents, centre):
    import trimesh
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(centre)
    return b


def _prism(profile_yz, length):
    """A closed prism: a triangle in the YZ plane extruded from x = 0 to x = length."""
    import trimesh
    tri = np.asarray(profile_yz, dtype=float)
    tri = np.c_[np.zeros(3), tri]
    v = np.vstack([tri, tri + [length, 0, 0]])
    f = np.array([[0, 1, 2], [3, 5, 4], [0, 3, 4], [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 3], [2, 3, 0]])
    m = trimesh.Trimesh(v, f, process=True)
    m.fix_normals()
    return m


def gable_house(width=10.0, depth=8.0, wall_h=6.0, roof_h=4.0, window=True, overhang=0.5):
    """Box walls + a gable roof whose ridge runs along X, so its slopes face +/-Z.

    The pitch is roof_h / (depth / 2 + overhang) rise per block.
    """
    box = _box((width, wall_h, depth), (width / 2, wall_h / 2, depth / 2))
    roof = _prism([[wall_h, -overhang], [wall_h, depth + overhang], [wall_h + roof_h, depth / 2]], width)
    parts = [(box, WALL, KIND_SURFACE), (roof, ROOF, KIND_SURFACE)]
    if window:
        pane = _box((2.0, 2.0, 0.1), (width / 2, wall_h / 2, depth + 0.02))
        parts.append((pane, (0.6, 0.8, 0.9, 0.3), KIND_GLASS))
    return mesh_to_scene(parts)


def hip_house(size=12.0, wall_h=5.0, roof_h=6.0):
    """Square box with a pyramid roof: slopes face all four directions."""
    import trimesh
    box = _box((size, wall_h, size), (size / 2, wall_h / 2, size / 2))
    v = np.array([[0, wall_h, 0], [size, wall_h, 0], [size, wall_h, size], [0, wall_h, size],
                  [size / 2, wall_h + roof_h, size / 2]], float)
    f = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 1], [0, 3, 2]])
    pyr = trimesh.Trimesh(v, f, process=True)
    pyr.fix_normals()
    return mesh_to_scene([(box, WALL, KIND_SURFACE), (pyr, ROOF, KIND_SURFACE)])


def flat_block(width=10.0, height=5.3, depth=8.0, glass_roof=False):
    """A box with a flat top at a non-integer height; optionally a glass skylight slab on top."""
    parts = [(_box((width, height, depth), (width / 2, height / 2, depth / 2)), WALL, KIND_SURFACE)]
    if glass_roof:
        parts.append((_box((width, 0.2, depth), (width / 2, height + 0.1, depth / 2)),
                      (0.6, 0.8, 0.9, 0.3), KIND_GLASS))
    return mesh_to_scene(parts)


def lean_to(pitch=.5):
    """A single roof plane, with a one-block eave on both long sides."""
    wall = _box((10, 5, 6), (5, 2.5, 4))
    roof = _prism([[5, 0], [5, 8], [5 + pitch * 8, 8]], 10)
    return mesh_to_scene([(wall, WALL, KIND_SURFACE), (roof, ROOF, KIND_SURFACE)])


def compound_house():
    """Intersecting perpendicular gables: concave valleys and convex outer ridges."""
    wall = _box((12, 5, 8), (6, 2.5, 4))
    main = _prism([[5, 0], [5, 8], [9, 4]], 12)
    cross = _prism([[5, 0], [5, 6], [8, 3]], 12)
    cross.vertices = cross.vertices[:, [2, 1, 0]] + [3, 0, -2]
    wing = _box((6, 5, 12), (6, 2.5, 4))
    return mesh_to_scene([(wall, WALL, KIND_SURFACE), (wing, WALL, KIND_SURFACE),
                          (main, ROOF, KIND_SURFACE), (cross, ROOF, KIND_SURFACE)])
