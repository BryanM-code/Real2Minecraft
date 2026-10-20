"""photofirst: what a photo sees (ray casting), the completion model's inputs and windows."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from photofirst import data as P
from photofirst import views as V
from photofirst.visibility import EVAL_VIEW, IGNORE, cast

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def cube(n=12, lo=4, hi=8):
    ts = np.zeros((n, n, n), np.uint8)
    ts[lo:hi, lo:hi, lo:hi] = 1
    return ts


class Visibility(unittest.TestCase):
    def test_cube_seen_from_the_photo_corner(self):
        # yaw 35, pitch 12: the viewer is towards -x, +z and above, so the -x face, the +z
        # face and the top are seen; the opposite corner is not
        ts = cube()
        solid, air = cast(ts, *EVAL_VIEW)
        self.assertTrue(solid[4, 5, 6] and solid[6, 5, 7] and solid[6, 7, 6])
        self.assertFalse(solid[7, 4, 4])
        self.assertFalse(solid[5:7, 5:7, 5:7].any())                 # the inside of the cube
        self.assertFalse((solid & (ts == 0)).any())
        self.assertFalse((air & (ts != 0)).any())
        self.assertTrue(air[1, 6, 10])                               # in front of the build
        self.assertFalse(air[10, 5, 1])                              # behind it

    def test_interior_stops_rays_but_is_not_reported(self):
        ts = cube()
        ts[5:7, 5:7, 5:7] = IGNORE
        ts[4, 5:7, 5:7] = 0                                          # an opening in the -x face
        solid, air = cast(ts, *EVAL_VIEW)
        self.assertFalse(solid[ts == IGNORE].any())
        self.assertFalse(air[ts == IGNORE].any())
        self.assertFalse(solid[7, 5, 5])                             # the back wall stays hidden
        self.assertTrue(air[4, 5, 5])                                # the opening is seen air

    def test_seen_share_is_partial(self):
        ts = cube()
        solid, _ = cast(ts, *EVAL_VIEW)
        shell = int(ts.sum()) - 8                                    # the 2x2x2 core is never on a ray first
        self.assertTrue(0.3 < solid.sum() / shell < 0.7)


class Inputs(unittest.TestCase):
    def setUp(self):
        self.table = np.zeros((313, 3), np.uint8)
        self.table[5] = (200, 100, 50)

    def test_encode_channels(self):
        tm = np.full((2, 1, 1), 5, np.uint16)
        solid = np.array([True, False]).reshape(2, 1, 1)
        air = np.array([False, True]).reshape(2, 1, 1)
        x = P.encode(tm, solid, air, self.table)
        self.assertEqual(x.shape, (2, 1, 1, 5))
        self.assertEqual(x[0, 0, 0].tolist(), [1, 0, 200, 100, 50])
        self.assertEqual(x[1, 0, 0].tolist(), [0, 1, 0, 0, 0])

    def test_encode_varies_colour_only_where_seen(self):
        tm = np.full((3, 1, 1), 5, np.uint16)
        solid = np.array([True, True, False]).reshape(3, 1, 1)
        x = P.encode(tm, solid, ~solid, self.table, np.random.default_rng(1))
        self.assertEqual(x[2, 0, 0, 2:].tolist(), [0, 0, 0])
        self.assertNotEqual(x[0, 0, 0, 2:].tolist(), [200, 100, 50])

    def test_front_noise_keeps_the_layout(self):
        rng = np.random.default_rng(2)
        w = np.zeros((24, 24, 24, 5), np.uint8)
        w[8:16, 0:12, 8] = (1, 0, 120, 60, 30)                      # a wall of seen blocks
        w[8:16, 0:12, 0:8, 1] = 1                                    # the seen air in front of it
        out = P.front_noise(w.copy(), rng)
        solid, air = out[..., 0].astype(bool), out[..., 1].astype(bool)
        self.assertEqual(out.shape, w.shape)
        self.assertFalse((solid & air).any())
        self.assertTrue((out[..., 2:][~solid] == 0).all())
        self.assertTrue(solid.any() and air.any())
        self.assertLess((solid & w[..., 0].astype(bool)).sum(), w[..., 0].sum())   # some blocks moved or lost
        self.assertTrue(set(np.unique(out[..., 2:][solid].reshape(-1, 3), axis=0).ravel()) <= {120, 60, 30})

    def test_fuse_views_union_and_carving(self):
        a = np.zeros((10, 4, 10, 5), np.uint8)
        b = np.zeros_like(a)
        a[2, 1, 2] = (1, 0, 9, 9, 9)                                  # a block photo A saw ...
        a[5, 1, 5] = (1, 0, 7, 7, 7)                                  # ... and a block next to B's surface
        b[2, 1, 2, 1] = 1                                             # B saw clear through the first
        b[5, 1, 6] = (1, 0, 4, 4, 4)                                  # B's own surface
        b[5, 1, 5, 1] = 1                                             # B's air right in front of it
        u = P.fuse_views([a, b])
        self.assertTrue(u[2, 1, 2, 0] and u[5, 1, 5, 0] and u[5, 1, 6, 0])
        self.assertEqual(u[2, 1, 2, 2:].tolist(), [9, 9, 9])
        c = P.fuse_views([a, b], carve=True)
        self.assertFalse(c[2, 1, 2, 0])                               # carved: B sees through it
        self.assertTrue(c[2, 1, 2, 1])                                # ... and it is seen air now
        self.assertTrue(c[5, 1, 5, 0])                                # kept: B's air there is next to B's surface
        self.assertFalse((c[..., 0].astype(bool) & c[..., 1].astype(bool)).any())

    def test_upsample(self):
        w = np.arange(8, dtype=np.uint8).reshape(2, 2, 2, 1)
        u = P.upsample(w)
        self.assertEqual(u.shape, (4, 4, 4, 1))
        self.assertTrue((u[2:4, 0:2, 2:4, 0] == w[1, 0, 1, 0]).all())

    def test_material_table_matches_the_vocabulary(self):
        from mcvoxel.vocab import MATERIALS
        t = P.material_rgb()
        self.assertEqual(t.shape, (len(MATERIALS), 3))
        self.assertEqual(t[0].tolist(), [0, 0, 0])


class FrontGeometry(unittest.TestCase):
    """The front end's geometry, on synthetic points (no depth model needed)."""

    def test_map_normals_of_a_plane(self):
        from photofirst import front as F
        j, i = np.meshgrid(np.arange(8.0), np.arange(6.0))
        pts = np.stack([j, i, 5.0 + 0.5 * j], axis=-1)
        n, ok = F.map_normals(pts, np.ones(pts.shape[:2], bool))
        want = np.array([-0.5, 0.0, 1.0]) / np.linalg.norm([-0.5, 0.0, 1.0])
        self.assertTrue(ok.all())
        self.assertTrue(np.allclose(np.abs(n @ want), 1.0, atol=1e-6))

    def test_frame_from_wall_and_roof_normals(self):
        from photofirst import front as F
        from photofirst.visibility import view_axes
        R = F.known_frame(*view_axes(35.0, 12.0))          # grid = R @ camera; row i: build axis i in camera coords
        rng = np.random.default_rng(0)
        n = np.concatenate([np.repeat(-R[0][None], 400, 0), np.repeat(R[2][None], 400, 0),
                            np.repeat(R[1][None], 200, 0)]) + rng.normal(0, 0.02, (1000, 3))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        est = F.frame_from_normals(n)
        self.assertGreater(abs(est[1] @ R[1]), np.cos(np.radians(1.5)))
        yaw_err = min(np.degrees(np.arccos(min(1.0, abs(est[0] @ a)))) for a in (R[0], R[2]))
        self.assertLess(yaw_err, 1.5)
        self.assertAlmostEqual(abs(np.linalg.det(est)), 1.0, places=6)

    def test_to_blocks_wall(self):
        from photofirst import front as F
        y, z = np.meshgrid(np.linspace(0, 9.99, 60), np.linspace(0, 9.99, 60))
        pts = np.stack([np.full(y.size, 5.0), y.ravel(), z.ravel()], axis=1)
        normals = np.tile([-1.0, 0.0, 0.0], (len(pts), 1))             # facing the camera at -x
        colours = np.tile([200, 10, 10], (len(pts), 1))
        b = F.to_blocks(pts, normals, colours, np.array([-100.0, 5.0, 5.0]), 10.0, air_every=1)
        xs = np.nonzero(b["solid"].any((1, 2)))[0]
        self.assertEqual(len(xs), 1)                                    # one layer of blocks
        self.assertGreaterEqual(b["solid"].sum(), 90)
        self.assertTrue(b["air"][: xs[0]].any() and not b["air"][xs[0] + 1:].any())
        self.assertEqual(b["rgb"][b["solid"]].mean(0).round().tolist(), [200.0, 10.0, 10.0])

    def test_place_turns_like_rotate_y(self):
        from mcvoxel.blockstates import rotate_y
        from photofirst import front as F
        g = np.zeros((4, 2, 3), np.uint8)
        g[3, 1, 0] = 1
        s, _ = rotate_y(g, 1, np.zeros_like(g, np.uint16))
        placed = F.place(g.astype(bool), 1, (1, 0, 2), (6, 2, 7), False)
        want = np.zeros((6, 2, 7), bool)
        want[1:1 + s.shape[0], :, 2:2 + s.shape[2]] = s != 0
        self.assertTrue(np.array_equal(placed, want))


class Viewpoints(unittest.TestCase):
    def test_eval_builds_start_with_the_photo_view(self):
        v = V.viewpoints("x", "val")
        self.assertEqual(v[0], EVAL_VIEW)
        self.assertEqual(len(v), 4)

    def test_train_views_are_seeded_and_spread(self):
        a, b = V.viewpoints("ab-1", "train"), V.viewpoints("ab-1", "train")
        self.assertEqual(a, b)
        self.assertEqual(len(a), V.TRAIN_VIEWS)
        self.assertNotEqual(a, V.viewpoints("ab-2", "train"))
        self.assertTrue(all(V.PITCH[0] <= p <= V.PITCH[1] for _, p in a))
        sector = 360.0 / V.TRAIN_VIEWS
        self.assertTrue(all(k * sector <= y < (k + 1) * sector for k, (y, _) in enumerate(a)))


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "normalized").mkdir()
        ts = np.zeros((20, 10, 24), np.uint8)
        ts[3:17, 0:7, 4:20] = 1
        ts[5:15, 1:6, 6:18] = IGNORE
        tm = np.where(ts == 1, 5, 0).astype(np.uint16)
        tm[ts == IGNORE] = 65535
        np.savez(self.tmp / "normalized" / "b1.npz", target_shape=ts, target_material=tm)
        out = self.tmp / V.OUT_DIR
        (out / "targets").mkdir(parents=True)
        (out / "views").mkdir()
        bid, entry = V._one((str(self.tmp), {"id": "b1", "split": "val", "category_top": "x"}, str(out)))
        (out / "manifest.json").write_text(json.dumps({"builds": {bid: entry}}), encoding="utf-8")
        self.ts, self.tm = ts, tm

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_views_round_trip(self):
        store = P.Store(self.tmp)
        self.assertEqual(store.ids("val"), ["b1"])
        self.assertEqual(store.n_views("b1"), 4)
        solid, air = store.view("b1", 0)
        want_solid, want_air = cast(self.ts, *EVAL_VIEW)
        self.assertTrue(np.array_equal(solid, want_solid) and np.array_equal(air, want_air))

    def test_window(self):
        store = P.Store(self.tmp)
        w_i, w_s, w_m, start = P.window(store, "b1", 1, np.random.default_rng(0), P.material_rgb())
        self.assertEqual(w_i.shape, (64, 64, 64, 5))
        self.assertEqual((w_s.shape, w_m.shape), ((64, 64, 64), (64, 64, 64)))
        self.assertTrue((start == 0).all())                          # the build is smaller than a window
        self.assertTrue((w_s[20:] == 255).all() and (w_i[20:] == 0).all())
        seen = w_i[..., 0].astype(bool)
        self.assertTrue(seen.any() and (w_s[seen] == 1).all())
        self.assertTrue((w_i[..., 2:][~seen] == 0).all())

    def test_full_input(self):
        store = P.Store(self.tmp)
        x = P.full_input(store, "b1", 0)
        self.assertEqual(x.shape, (40, 20, 48, 5))

    @unittest.skipUnless(HAVE_TORCH, "needs torch (r2m_train)")
    def test_dataset_sample_with_several_photos(self):
        ds = P.CompletionDataset(self.tmp, ids=["b1"], length=4, seed=5, n_views=(2, 4), noise=P.FRONT_NOISE)
        x, s, m, start = ds[1]
        self.assertEqual(tuple(x.shape), (5, 128, 128, 128))
        solid, air = x[0].numpy().astype(bool), x[1].numpy().astype(bool)
        self.assertFalse((solid & air).any())
        one = P.CompletionDataset(self.tmp, ids=["b1"], length=4, seed=5)
        self.assertGreater(int(x[1].sum()), 0)
        self.assertGreater(int(one[1][1].sum()), 0)

    @unittest.skipUnless(HAVE_TORCH, "needs torch (r2m_train)")
    def test_dataset_sample(self):
        ds = P.CompletionDataset(self.tmp, ids=["b1"], length=4, seed=3)
        x, s, m, start = ds[0]
        self.assertEqual(tuple(x.shape), (5, 128, 128, 128))
        self.assertEqual(tuple(s.shape), (64, 64, 64))
        a, _, _, _ = ds[0]
        self.assertTrue(bool((a == x).all()))                        # every sample has its own seed


if __name__ == "__main__":
    unittest.main()
