"""mcvoxel.raster must mark exactly the voxels the original per-triangle loop marks."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcvoxel.raster import fine_solid_aligned, rasterize_surface, solid_from_surface  # noqa: E402


def _voxelize_surface_sat(verts, faces):
    """REFERENCE ORACLE: the original per-triangle loop from voxelize.py (2026-07-08),
    kept verbatim here so the vectorised rasteriser is always compared against it."""
    origin = np.floor(verts.min(axis=0)).astype(np.int64)
    upper = np.floor(verts.max(axis=0)).astype(np.int64)
    sx, sy, sz = (upper - origin + 1).tolist()
    surf = np.zeros((sx, sy, sz), dtype=bool)

    tris = (verts - origin)[faces]
    lo_axis = np.array([0, 0, 0])
    hi_axis = np.array([sx - 1, sy - 1, sz - 1])
    h = 0.5

    for tri in tris:
        v0, v1, v2 = tri
        lo = np.clip(np.floor(tri.min(axis=0)).astype(np.int64), lo_axis, hi_axis)
        hi = np.clip(np.floor(tri.max(axis=0)).astype(np.int64), lo_axis, hi_axis)
        gx, gy, gz = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1),
                                 np.arange(lo[2], hi[2] + 1), indexing="ij")
        idx = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        C = idx + 0.5
        u0, u1, u2 = v0 - C, v1 - C, v2 - C
        ok = np.ones(len(C), dtype=bool)
        for a in range(3):
            tmin = np.minimum(np.minimum(u0[:, a], u1[:, a]), u2[:, a])
            tmax = np.maximum(np.maximum(u0[:, a], u1[:, a]), u2[:, a])
            ok &= ~((tmin > h) | (tmax < -h))
        if not ok.any():
            continue
        f0, f1, f2 = v1 - v0, v2 - v1, v0 - v2
        n = np.cross(f0, f1)
        if float(n @ n) < 1e-18:
            continue
        ok &= np.abs(u0 @ n) <= h * np.abs(n).sum()
        if not ok.any():
            continue
        for f in (f0, f1, f2):
            fx, fy, fz = f
            pa = np.stack([-u0[:, 1]*fz + u0[:, 2]*fy, -u1[:, 1]*fz + u1[:, 2]*fy,
                           -u2[:, 1]*fz + u2[:, 2]*fy], axis=0)
            ra = h * (abs(fz) + abs(fy))
            ok &= ~((pa.min(0) > ra) | (pa.max(0) < -ra))
            pb = np.stack([u0[:, 0]*fz - u0[:, 2]*fx, u1[:, 0]*fz - u1[:, 2]*fx,
                           u2[:, 0]*fz - u2[:, 2]*fx], axis=0)
            rb = h * (abs(fz) + abs(fx))
            ok &= ~((pb.min(0) > rb) | (pb.max(0) < -rb))
            pc = np.stack([-u0[:, 0]*fy + u0[:, 1]*fx, -u1[:, 0]*fy + u1[:, 1]*fx,
                           -u2[:, 0]*fy + u2[:, 1]*fx], axis=0)
            rc = h * (abs(fy) + abs(fx))
            ok &= ~((pc.min(0) > rc) | (pc.max(0) < -rc))
            if not ok.any():
                break
        if ok.any():
            sel = idx[ok]
            surf[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return surf, origin


def _box(lo, hi):
    import trimesh
    m = trimesh.creation.box(extents=np.subtract(hi, lo))
    m.apply_translation(np.add(lo, hi) / 2.0)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


class RasterEquivalence(unittest.TestCase):
    def test_random_soups_match_loop(self):
        rng = np.random.default_rng(0)
        for trial in range(8):
            v = rng.normal(size=(250, 3)) * rng.uniform(2, 12)
            f = rng.integers(0, 250, size=(300, 3))
            a, oa = _voxelize_surface_sat(v, f)
            b, ob = rasterize_surface(v, f)
            self.assertTrue((oa == ob).all())
            self.assertEqual(a.shape, b.shape)
            self.assertTrue((a == b).all(), "trial %d differs" % trial)

    def test_degenerate_and_axis_aligned(self):
        v, f = _box((0.0, 0.0, 0.0), (5.0, 3.0, 4.0))            # faces on integer planes
        v2 = np.vstack([v, [[1, 1, 1], [1, 1, 1], [2, 2, 2]]])   # plus a degenerate sliver
        f2 = np.vstack([f, [[8, 9, 10]]])
        for verts, faces in ((v, f), (v2, f2)):
            a, oa = _voxelize_surface_sat(verts, faces)
            b, ob = rasterize_surface(verts, faces)
            self.assertTrue((a == b).all() and (oa == ob).all())

    def test_explicit_frame_contains_same_cells(self):
        rng = np.random.default_rng(3)
        v = rng.uniform(3, 20, size=(120, 3))
        f = rng.integers(0, 120, size=(160, 3))
        own, origin = rasterize_surface(v, f)
        big, big_origin = rasterize_surface(v, f, origin=origin - 4, shape=np.array(own.shape) + 9)
        off = origin - big_origin
        sub = big[off[0]:off[0] + own.shape[0], off[1]:off[1] + own.shape[1], off[2]:off[2] + own.shape[2]]
        self.assertTrue((sub == own).all())
        self.assertEqual(int(big.sum()), int(own.sum()))

    def test_solid_fill_of_closed_box(self):
        v, f = _box((0.2, 0.2, 0.2), (6.8, 4.8, 5.8))
        surf, _ = rasterize_surface(v, f)
        solid = solid_from_surface(surf)
        self.assertTrue(solid.all())                              # a closed box is all solid

    def test_fine_grid_is_coarse_aligned(self):
        v, f = _box((0.3, 0.1, 0.4), (4.7, 3.2, 2.9))
        surf, origin = rasterize_surface(v, f)
        fine, fine_origin = fine_solid_aligned(v, f, origin, surf.shape, 4)
        self.assertEqual(fine.shape, tuple(np.array(surf.shape) * 4))
        self.assertTrue((fine_origin == origin * 4).all())


@unittest.skipUnless(list((ROOT / "outputs" / "trellis").glob("*.glb")), "no cached TRELLIS meshes")
class RasterOnTrellisMeshes(unittest.TestCase):
    def test_real_meshes_coarse_and_fine(self):
        import trimesh
        for glb in sorted((ROOT / "outputs" / "trellis").glob("*.glb"))[:3]:
            m = trimesh.load(str(glb), force="mesh")
            v = np.asarray(m.vertices, float)
            f = np.asarray(m.faces, np.int64)
            s = 48.0 / float((v.max(0) - v.min(0)).max())
            for mult in (1, 4):
                vv = (v - v.min(0)) * s * mult
                a, oa = _voxelize_surface_sat(vv, f)
                b, ob = rasterize_surface(vv, f)
                self.assertTrue((a == b).all() and (oa == ob).all(), "%s x%d" % (glb.name, mult))


if __name__ == "__main__":
    unittest.main()
