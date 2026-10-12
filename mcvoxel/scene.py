"""Load a glTF/GLB scene as one mesh that remembers its materials.

`mesh_normalize` reads a TRELLIS GLB: one mesh, one texture. Architectural models from
asset libraries are not like that -- one Sketchfab cottage in the pilot probe has 419
parts, some textured, most coloured only by a base-colour factor, one glass pane marked
BLEND. Concatenating them the way `trimesh.load(force="mesh")` does throws away exactly
what a training target wants: which faces are glass, which are a door, which colour each
surface really is.

`load_scene` keeps it. Every face knows its material; every material knows its base
colour, its texture (downsampled to `max_texture`), its alpha mode and a coarse KIND
(surface / glass / door) derived from those and from its name. `scene_color_fns` turns that
into the two colour callbacks `voxel_sampling.sample_voxel_surface` needs.

Units are NOT trusted. On the pilot probe, five buildings of ordinary size spanned 0.003 to
12,600 file units, so scale is decided by the caller from what the object is.

Meshes are never decimated here: trimesh's quadric decimation drops UVs, which would turn a
textured facade into its flat base colour without saying so. The vectorised rasteriser makes
large meshes affordable instead; `max_faces` only rejects outright.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Material", "SceneMesh", "load_scene", "scene_color_fns", "split_clutter", "clutter_kind",
           "KIND_SURFACE", "KIND_GLASS", "KIND_DOOR", "KIND_NAMES"]

KIND_SURFACE, KIND_GLASS, KIND_DOOR = 0, 1, 2
KIND_NAMES = ("surface", "glass", "door")

# Material or node names that say what the surface is, in the languages the probe met.
_GLASS_NAME = re.compile(r"glass|glazing|window|pane\b|vidrio|verre|vetro|glas\b|glas_|szk|стекл|окно|"
                         r"cristal|mirror|transparent|translucent", re.I)
_DOOR_NAME = re.compile(r"\bdoor|_door|door_|puerta|porte\b|t[üu]r\b|tuer|дверь|deur", re.I)


class SceneTooLarge(ValueError):
    """The scene has more faces than the caller is willing to voxelise."""


@dataclass
class Material:
    name: str
    base_rgba: np.ndarray              # (4,) floats 0..1 (glTF baseColorFactor)
    image: np.ndarray | None           # (H,W,3) uint8, factor NOT applied
    image_alpha: float | None          # mean texture alpha 0..1, if the texture has one
    alpha_mode: str                    # OPAQUE | MASK | BLEND
    kind: int = KIND_SURFACE


@dataclass
class SceneMesh:
    verts: np.ndarray                  # (V,3) float64, scene-graph transforms applied
    faces: np.ndarray                  # (F,3) int64
    uv: np.ndarray                     # (V,2) float64; NaN where the part had none
    vcolor: np.ndarray | None          # (V,3) float64 0..255, or None
    face_mat: np.ndarray               # (F,) int32 -> materials
    materials: list = field(default_factory=list)
    part_names: list = field(default_factory=list)
    face_part: np.ndarray | None = None

    @property
    def face_kind(self):
        kinds = np.array([m.kind for m in self.materials], dtype=np.int32)
        return kinds[self.face_mat]

    @property
    def bounds(self):
        return self.verts.min(axis=0), self.verts.max(axis=0)

    def transformed(self, matrix3, offset):
        """Copy with verts -> verts @ matrix3.T + offset (rotation/scale then shift)."""
        out = SceneMesh(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        out.verts = self.verts @ np.asarray(matrix3, dtype=np.float64).T + np.asarray(offset, dtype=np.float64)
        return out


def _to_rgba01(value, default=(1.0, 1.0, 1.0, 1.0)):
    if value is None:
        return np.asarray(default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 3:
        arr = np.r_[arr, 1.0 if arr.max() <= 1.0 else 255.0]
    if arr.max() > 1.0:                          # trimesh stores factors as uint8 0..255
        arr = arr / 255.0
    return np.clip(arr[:4], 0.0, 1.0)


def _image_array(img, max_texture):
    """PIL image -> ((H,W,3) uint8, mean alpha or None), downsampled to max_texture."""
    if img is None:
        return None, None
    try:
        from PIL import Image
        if max(img.size) > max_texture:
            scale = max_texture / float(max(img.size))
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                             Image.BILINEAR)
        alpha = None
        if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
            rgba = np.asarray(img.convert("RGBA"))
            alpha = float(rgba[..., 3].mean()) / 255.0
            rgb = rgba[..., :3]
        else:
            rgb = np.asarray(img.convert("RGB"))
        return np.ascontiguousarray(rgb, dtype=np.uint8), alpha
    except Exception:
        return None, None


def _material_of(geom, max_texture):
    """Material for one trimesh geometry (TextureVisuals or ColorVisuals)."""
    visual = geom.visual
    mat = getattr(visual, "material", None)
    name = str(getattr(mat, "name", None) or "")
    alpha_mode = str(getattr(mat, "alphaMode", None) or "OPAQUE").upper()
    base = getattr(mat, "baseColorFactor", None)
    if base is None:
        base = getattr(mat, "diffuse", None)             # SimpleMaterial
    image = getattr(mat, "baseColorTexture", None)
    if image is None:
        image = getattr(mat, "image", None)
    arr, alpha = _image_array(image, max_texture)
    m = Material(name=name, base_rgba=_to_rgba01(base), image=arr, image_alpha=alpha,
                 alpha_mode=alpha_mode)
    m.kind = _kind(m, str(getattr(geom, "metadata", {}).get("name") or ""))
    return m


def _kind(m: Material, node_name: str) -> int:
    label = m.name + " " + node_name
    if _DOOR_NAME.search(label):
        return KIND_DOOR
    see_through = m.base_rgba[3] < 0.95 or (m.image_alpha is not None and m.image_alpha < 0.9)
    if (m.alpha_mode == "BLEND" and see_through) or m.base_rgba[3] < 0.8 or _GLASS_NAME.search(label):
        return KIND_GLASS
    return KIND_SURFACE


def load_scene(path, max_texture: int = 1024, max_faces: int | None = None) -> SceneMesh:
    """Read a GLB/glTF (or anything trimesh opens) into one material-aware mesh."""
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    geoms = [g for g in scene.dump(concatenate=False)
             if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not geoms:
        raise ValueError("no triangle geometry in {}".format(path))
    total = sum(len(g.faces) for g in geoms)
    if max_faces and total > max_faces:
        raise SceneTooLarge("{:,} faces exceeds the {:,} limit".format(total, max_faces))

    verts, faces, uvs, vcols, face_mat, face_part = [], [], [], [], [], []
    materials, mat_index, part_names = [], {}, []
    offset = 0
    any_vcol = False
    for gi, g in enumerate(geoms):
        v = np.asarray(g.vertices, dtype=np.float64)
        f = np.asarray(g.faces, dtype=np.int64)
        mat = _material_of(g, max_texture)
        key = (mat.name, tuple(np.round(mat.base_rgba, 4)), mat.alpha_mode, mat.kind,
               None if mat.image is None else (mat.image.shape, int(mat.image[::17, ::17].sum())))
        mi = mat_index.get(key)
        if mi is None:
            mi = mat_index[key] = len(materials)
            materials.append(mat)
        uv = getattr(g.visual, "uv", None)
        if uv is None or len(uv) != len(v):
            uv = np.full((len(v), 2), np.nan)
        vc = None
        if getattr(g.visual, "kind", None) == "vertex":
            try:
                vc = np.asarray(g.visual.vertex_colors, dtype=np.float64)[:, :3]
                any_vcol = True
            except Exception:
                vc = None
        verts.append(v)
        faces.append(f + offset)
        uvs.append(np.asarray(uv, dtype=np.float64))
        vcols.append(vc if vc is not None else np.full((len(v), 3), np.nan))
        face_mat.append(np.full(len(f), mi, dtype=np.int32))
        face_part.append(np.full(len(f), gi, dtype=np.int32))
        part_names.append(str(g.metadata.get("name", "")))
        offset += len(v)

    return SceneMesh(
        verts=np.concatenate(verts), faces=np.concatenate(faces), uv=np.concatenate(uvs),
        vcolor=np.concatenate(vcols) if any_vcol else None,
        face_mat=np.concatenate(face_mat), materials=materials, part_names=part_names,
        face_part=np.concatenate(face_part))


# Words that mark a part or material as site clutter rather than building, matched as whole
# words of a name split at punctuation, digits and camelCase ("Palm_Tree01", "carJeep").
# Terrain words are deliberately absent: "ground" names ground-floor walls as often as lawns,
# "grass" names turf roofs, and a flat site is removed geometrically instead
# (`grid.remove_site`). So are tree species and "bark": "oak", "pine", "birch" and "bark" name
# wood materials and log textures.
CLUTTER_WORDS = {
    "vegetation": {"tree", "trees", "leaf", "leaves", "foliage", "bush", "bushes", "hedge", "hedges", "shrub",
                   "shrubs", "plant", "plants", "palm", "palms", "flower", "flowers", "ivy", "vine", "vines",
                   "conifer", "topiary", "branch", "branches", "twig", "twigs",
                   "vegetation", "fern"},
    "vehicle": {"car", "cars", "vehicle", "vehicles", "truck", "trucks", "bus", "bike", "bicycle", "motorcycle",
                "scooter", "jeep", "taxi", "carroceria", "sedan", "suv", "van"},
    "figure": {"person", "people", "human", "humans", "character", "characters", "npc", "man", "woman",
               "men", "women", "child", "figure", "figures", "mannequin"},
}
_WORD_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
MAX_CLUTTER_SHARE = 0.6


def name_words(name: str) -> set:
    """Lower-case words of a part or material name ("Palm_Tree01" -> {"palm", "tree"})."""
    return {w.lower() for w in _WORD_SPLIT.findall(name or "")}


def clutter_kind(name: str):
    words = name_words(name)
    for kind, vocab in CLUTTER_WORDS.items():
        if words & vocab:
            return kind
    return None


def split_clutter(sm: SceneMesh):
    """(building-only SceneMesh, info): faces whose part or material name marks them as
    vegetation, a vehicle or a figure are dropped. If that would drop more than
    MAX_CLUTTER_SHARE of the faces, the names are not trusted and nothing is dropped."""
    mat_kind = [clutter_kind(m.name) for m in sm.materials]
    part_kind = [clutter_kind(p) for p in sm.part_names]
    drop = np.zeros(len(sm.faces), dtype=bool)
    kinds = {}
    for i, k in enumerate(mat_kind):
        if k:
            sel = sm.face_mat == i
            drop |= sel
            kinds[k] = kinds.get(k, 0) + int(sel.sum())
    if sm.face_part is not None:
        for i, k in enumerate(part_kind):
            if k:
                sel = (sm.face_part == i) & ~drop
                drop |= sel
                kinds[k] = kinds.get(k, 0) + int(sel.sum())
    info = {"faces": int(len(sm.faces)), "dropped": int(drop.sum()), "by_kind": kinds}
    if not drop.any():
        return sm, info
    if drop.mean() > MAX_CLUTTER_SHARE:
        info.update(dropped=0, skipped="names would drop {:.0%} of the model".format(drop.mean()))
        return sm, info
    keep = ~drop
    faces = sm.faces[keep]
    # Drop the clutter's vertices too: the scene's bounds (its height, hence its scale, and
    # its grid) must be the building's, not the tree's beside it.
    used = np.unique(faces)
    remap = np.full(len(sm.verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    out = SceneMesh(**{k: getattr(sm, k) for k in sm.__dataclass_fields__})
    out.verts = sm.verts[used]
    out.uv = sm.uv[used]
    out.vcolor = None if sm.vcolor is None else sm.vcolor[used]
    out.faces = remap[faces]
    out.face_mat = sm.face_mat[keep]
    if sm.face_part is not None:
        out.face_part = sm.face_part[keep]
    return out, info


def scene_color_fns(sm: SceneMesh):
    """(group_colors, point_colors) callbacks for `voxel_sampling.sample_voxel_surface`.

    Colour of a sample = texture texel (if the material has a texture and the part has
    UVs), else vertex colour (if present), else white -- times the material's base colour.
    That is the glTF base-colour rule without lighting. Values are 0..255 floats.
    """
    from .voxel_sampling import _bilinear_sample

    faces = sm.faces
    face_mat = sm.face_mat
    mats = sm.materials
    base = np.array([m.base_rgba[:3] for m in mats], dtype=np.float64)   # (n_mat, 3), 0..1
    uv_ok = ~np.isnan(sm.uv[faces][:, :, 0]).any(axis=1)                  # (F,)
    vcol = sm.vcolor

    def group_colors(sel, bary):
        tri = np.flatnonzero(sel)
        n_pts = len(bary)
        out = np.empty((len(tri), n_pts, 3), dtype=np.float64)
        mat_here = face_mat[tri]
        for mi in np.unique(mat_here):
            rows = np.flatnonzero(mat_here == mi)
            t = tri[rows]
            col = np.full((len(rows), n_pts, 3), 255.0)
            m = mats[mi]
            if m.image is not None:
                ok = uv_ok[t]
                if ok.any():
                    uvs = np.einsum("mk,tkd->tmd", bary, sm.uv[faces[t[ok]]])
                    col[ok] = _bilinear_sample(m.image, uvs.reshape(-1, 2)).reshape(-1, n_pts, 3)
            elif vcol is not None:
                vc = np.nan_to_num(vcol[faces[t]], nan=255.0)
                col = np.einsum("mk,tkd->tmd", bary, vc)
            out[rows] = col * base[mi]
        return out.reshape(-1, 3)

    def point_colors(tri_id, bary):
        tri = np.asarray(tri_id)
        out = np.empty((len(tri), 3), dtype=np.float64)
        mat_here = face_mat[tri]
        for mi in np.unique(mat_here):
            rows = np.flatnonzero(mat_here == mi)
            t = tri[rows]
            b = bary[rows]
            col = np.full((len(rows), 3), 255.0)
            m = mats[mi]
            if m.image is not None:
                ok = uv_ok[t]
                if ok.any():
                    uvs = (b[ok][:, :, None] * sm.uv[faces[t[ok]]]).sum(axis=1)
                    col[ok] = _bilinear_sample(m.image, uvs)
            elif vcol is not None:
                vc = np.nan_to_num(vcol[faces[t]], nan=255.0)
                col = (b[:, :, None] * vc).sum(axis=1)
            out[rows] = col * base[mi]
        return out

    return group_colors, point_colors
