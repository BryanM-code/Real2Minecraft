"""photofirst multi-photo front end (front_mv), build quality (training.quality), the masked
model's tokens and decoding (photofirst.masked), real front-end pairs in the dataset."""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from photofirst import data as P
from photofirst import front_mv as M
from photofirst import views as V
from photofirst.visibility import IGNORE
from training import quality as Q

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def wall_view(x_wall=5.0, size=96, dist=12.0):
    """A camera at x = x_wall - dist looking along +x at the wall x = x_wall (y, z in 0..10):
    the `front_mv` view dict (pts world, ray camera-frame, cam2world OpenCV, mask, rgb)."""
    f = size / 2.0
    v, u = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5, indexing="ij")
    d = np.stack([(u - f) / f, (v - f) / f, np.ones_like(u)], -1)          # camera frame: right, down, forward
    R = np.stack([np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])], 1)
    c = np.array([x_wall - dist, 5.0, 5.0])
    world = d @ R.T
    t = (x_wall - c[0]) / world[..., 0]
    pts = c + world * t[..., None]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, c
    inside = (pts[..., 1] > 0.2) & (pts[..., 1] < 9.8) & (pts[..., 2] > 0.2) & (pts[..., 2] < 9.8)
    rgb = np.full((size, size, 3), 120, np.uint8)
    return {"pts": pts.astype(np.float32), "ray": (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32),
            "cam2world": T.astype(np.float32), "mask": inside, "rgb": rgb}


class FrontMV(unittest.TestCase):
    def test_view_subsets(self):
        subs = M.view_subsets(8, np.random.default_rng(3))
        self.assertEqual([len(s) for s in subs], list(M.SUBSET_SIZES))
        self.assertEqual(len(set(subs)), len(subs))
        self.assertTrue(all(0 <= i < 8 for s in subs for i in s))
        self.assertEqual(subs, M.view_subsets(8, np.random.default_rng(3)))

    def test_rough_poses_look_at_the_centre(self):
        for T in M.rough_poses([0.0, 90.0, 215.0], 10.0):
            R, c = T[:3, :3], T[:3, 3]
            self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=1e-9))
            self.assertAlmostEqual(np.linalg.norm(c), 3.0)
            self.assertLess(float(R[:, 2] @ c), 0.0)                  # forward points back at the origin

    def test_render_hints(self):
        cam = {"focal_px": 100.0, "size": 200, "yaw": 35.0, "pitch": 12.0, "centre": [1.0, 2.0, 3.0],
               "right": [1.0, 0.0, 0.0], "down": [0.0, -1.0, 0.0], "near": [0.0, 0.0, 1.0]}
        cams = {"a": cam, "b": dict(cam, yaw=125.0)}
        self.assertEqual(M.render_hints(cams, ["a", "b"], "none", np.random.default_rng(0)), (None, None))
        K, poses = M.render_hints(cams, ["a", "b"], "true", np.random.default_rng(0))
        self.assertEqual(K[0][0, 2], 100.0)
        self.assertTrue(np.allclose(poses[0][:3, 3], [1, 2, 3]) and np.allclose(poses[0][:3, 2], [0, 0, -1]))
        K, poses = M.render_hints(cams, ["a", "b"], "coarse", np.random.default_rng(0))
        self.assertTrue(90.0 <= K[0][0, 0] <= 110.0 and len(poses) == 2)

    def test_views_round_trip(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            views = [wall_view(), wall_view(6.0)]
            M.save_views(tmp / "v.npz", views)
            back = M.load_views(tmp / "v.npz")
            self.assertEqual(len(back), 2)
            for a, b in zip(views, back):
                self.assertTrue(all(np.array_equal(a[k], b[k]) for k in M.VIEW_KEYS))
        finally:
            shutil.rmtree(tmp)

    def test_march_matches_the_plain_marcher(self):
        from photofirst.front import _march
        rng = np.random.default_rng(4)
        dims = np.array([20, 12, 25])
        start = rng.uniform(-15, 40, (3000, 3))
        d = rng.normal(size=(3000, 3))
        d[:50, 1] = 0.0                                                # rays parallel to a face
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t = np.arange(0.3, 80.0, 0.5)
        a, b = np.zeros(tuple(dims), bool), np.zeros(tuple(dims), bool)
        _march(start, d, t, dims, a)
        M.march(start, d, 0.3, 0.5, len(t), dims, b)
        self.assertTrue(a.any() and np.array_equal(a, b))

    def test_carving_drops_what_another_photo_saw_through(self):
        dims = (12, 10, 10)
        sim = {"s": 1.0, "R": np.eye(3), "t": np.zeros(3)}
        stray = wall_view(2.0)                       # a wall in the space the first photo sees through
        both = M.to_grid([wall_view(), stray], sim, dims)
        carved = M.to_grid([wall_view(), stray], sim, dims, carve=True)
        self.assertTrue(both[2, 2:8, 2:8, 0].all() and both[5, 2:8, 2:8, 0].all())
        self.assertFalse(carved[2, 2:8, 2:8, 0].any())                         # seen through: dropped
        self.assertTrue(carved[5, 2:8, 2:8, 0].all())                          # the far wall stays

    def test_depth_edges(self):
        d = np.full((10, 10), 5.0)
        d[:, 5:] = 10.0
        e = M.depth_edges(d)
        self.assertTrue(e[:, 4:6].all() and not e[:, :3].any() and not e[:, 7:].any())

    def test_to_grid_wall_and_half_blocks(self):
        dims = (12, 10, 10)
        sim = {"s": 1.0, "R": np.eye(3), "t": np.zeros(3)}
        one = M.to_grid([wall_view()], sim, dims)
        solid = one[..., 0].astype(bool)
        self.assertTrue(solid[5].sum() > 50 and not solid[:5].any() and not solid[6:].any())   # the wall's block
        self.assertGreater(one[2:5, 3:7, 3:7, 1].mean(), 0.9)                                  # air before it
        two = M.to_grid([wall_view()], sim, dims, res=2)
        self.assertEqual(two.shape, (24, 20, 20, 5))
        self.assertTrue(two[10, ..., 0].sum() > 200 and not two[11, ..., 0].any())             # the front half
        back = M.to_blocks(two, dims)[..., 0].astype(bool)
        self.assertGreater((back == solid).mean(), 0.99)
        self.assertEqual(M.model_input(one, dims).shape, (24, 20, 20, 5))
        self.assertIs(M.model_input(two, dims), two)

    def test_similarity_from_cameras(self):
        rng = np.random.default_rng(1)
        a = math.radians(30)
        R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        cams, centres, rots = [], [], []
        for k in range(4):
            Rk = np.linalg.qr(rng.normal(size=(3, 3)))[0]
            Rk *= np.sign(np.linalg.det(Rk))
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rk, rng.normal(size=3) * 4
            cams.append(T)
            centres.append(2.5 * (R @ T[:3, 3]) + np.array([10.0, 0.0, -3.0]))
            rots.append(R @ Rk)
        for rotations in (None, rots):
            sim = M.similarity_from_cameras(cams, centres, rotations)
            self.assertAlmostEqual(sim["s"], 2.5, places=6)
            self.assertTrue(np.allclose(sim["R"], R, atol=1e-6) and np.allclose(sim["t"], [10, 0, -3], atol=1e-6))

    def test_refine_finds_the_shift(self):
        tgt = np.zeros((30, 20, 30), bool)
        tgt[5:25, 0:15, 6:24] = True
        tgt[8:22, 0:13, 9:21] = False
        pts = (np.argwhere(tgt) + 0.5 + np.array([2.0, -1.0, 3.0])).astype(np.float32)
        views = [{"pts": pts.reshape(-1, 1, 3), "mask": np.ones((len(pts), 1), bool)}]
        sim = M.refine_to_target(views, {"s": 1.0, "R": np.eye(3), "t": np.zeros(3)}, tgt,
                                 yaws=[0.0], scales=[1.0], sample=10 ** 9)
        self.assertEqual(sim["refine"]["shift"], [-2, 1, -3])


class Quality(unittest.TestCase):
    def wall(self, mats):
        s = np.zeros((6, 8, 6), np.uint8)
        m = np.zeros((6, 8, 6), np.uint16)
        s[3, 1:7, 1:5] = 1
        m[3, 1:7, 1:5] = mats
        return s, m

    def test_one_material_wall(self):
        q = Q.quality(*self.wall(7))
        self.assertEqual((q["coherence"], q["speckle"], q["palette90"], q["fragments"]), (1.0, 0.0, 1, 0.0))
        self.assertEqual(q["roof_steps"], 0.0)

    def test_checkerboard_is_speckled(self):
        y, z = np.meshgrid(np.arange(6), np.arange(4), indexing="ij")
        q = Q.quality(*self.wall(np.where((y + z) % 2 == 0, 7, 9)))
        self.assertEqual((q["coherence"], q["speckle"]), (0.0, 1.0))

    def test_fragments_and_target_air(self):
        s, m = self.wall(7)
        s[0, 0, 0], m[0, 0, 0] = 1, 7                     # a floater
        self.assertGreater(Q.quality(s, m)["fragments"], 0.0)
        t = np.full(s.shape, IGNORE, np.uint8)            # a target: only its air is outside air
        t[:3] = 0
        t[3, 1:7, 1:5] = 1
        q = Q.quality(t, np.where(t == 1, 7, 0).astype(np.uint16), target=True)
        self.assertEqual(q["roughness"], 1.0)             # one face each, on the -x side

    def test_parts_of_identical_grids(self):
        s, m = self.wall(7)
        s[3, 6, 1:5] = 4                                  # a row of stairs on top
        r = Q.parts(s, m, s, m)
        self.assertEqual((r["stairs"]["f1"], r["stair_facing"]), (1.0, 1.0))
        self.assertTrue(math.isnan(r["glass"]["f1"]))


@unittest.skipUnless(HAVE_TORCH, "needs torch (r2m_train)")
class Masked(unittest.TestCase):
    def test_given_tokens(self):
        import torch
        from photofirst.masked import UNDECIDED_MATERIAL, UNDECIDED_SHAPE, given_tokens
        ts = torch.tensor([[[[0, 1, 255, 4]]]])
        tm = torch.tensor([[[[0, 9, 65535, 3]]]])
        s, m, g = given_tokens(ts, tm, torch.tensor([0.0]))
        self.assertEqual(s.tolist(), [[[[0, 1, UNDECIDED_SHAPE, 4]]]])
        self.assertEqual(m.tolist(), [[[[0, 9, UNDECIDED_MATERIAL, 3]]]])
        s, m, g = given_tokens(ts, tm, torch.tensor([1.0]))
        self.assertFalse(bool(g.any()))

    def test_decode_on_a_small_model(self):
        import torch
        from photofirst.masked import MaskedUNet3D, decode
        torch.manual_seed(0)
        model = MaskedUNet3D((8, 16), 1)
        inp = np.zeros((24, 20, 16, 5), np.uint8)
        inp[4:12, 0:8, 4:12, 0] = 1
        s, m = decode(model, inp, (12, 10, 8), passes=3, window=8, overlap=2, device="cpu")
        self.assertEqual((s.shape, s.dtype, m.dtype), ((12, 10, 8), np.uint8, np.uint16))
        self.assertTrue(((s == 0) == (m == 0)).all())

    def test_shape_prior_keeps_the_geometry(self):
        """predict_window's decoding calibration: a voxel is air or solid exactly as without the
        prior; among solid voxels the favoured shape wins more often."""
        import torch
        from training.model import UNet3D, predict_window, valid_table
        torch.manual_seed(0)
        model = UNet3D((8, 16), 1).eval()
        x = torch.rand(1, 6, 16, 16, 16)
        valid = valid_table("cpu")
        with torch.no_grad():
            p0, _ = predict_window(model, x, valid)
            prior = torch.ones(30)
            prior[5] = 50.0                                      # favour one stair shape
            p1, _ = predict_window(model, x, valid, prior=prior)
        a0, a1 = p0.argmax(1), p1.argmax(1)
        self.assertTrue(torch.equal(a0 == 0, a1 == 0))
        self.assertGreater(int((a1 == 5).sum()), int((a0 == 5).sum()))


@unittest.skipUnless(HAVE_TORCH, "needs torch (r2m_train)")
class RealPairs(unittest.TestCase):
    def test_dataset_sample_from_a_real_pair(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "normalized").mkdir()
            ts = np.zeros((20, 10, 24), np.uint8)
            ts[3:17, 0:7, 4:20] = 1
            ts[5:15, 1:6, 6:18] = IGNORE
            tm = np.where(ts == 1, 5, 0).astype(np.uint16)
            tm[ts == IGNORE] = 65535
            np.savez(tmp / "normalized" / "b1.npz", target_shape=ts, target_material=tm)
            out = tmp / V.OUT_DIR
            (out / "targets").mkdir(parents=True)
            (out / "views").mkdir()
            bid, entry = V._one((str(tmp), {"id": "b1", "split": "train", "category_top": "x"}, str(out)))
            (out / "manifest.json").write_text(json.dumps({"builds": {bid: entry}}), encoding="utf-8")
            pairs = out / "mv_pairs" / "t"
            pairs.mkdir(parents=True)
            inp = np.zeros((40, 20, 48, 5), np.uint8)              # 2 cells per block
            inp[6:34, 0:14, 8:40, 0] = 1
            np.savez_compressed(pairs / "b1_s0.npz", input=inp)
            (pairs / "index.json").write_text(json.dumps({"pairs": {"b1_s0": {"build": "b1"}}}), encoding="utf-8")
            ds = P.CompletionDataset(tmp, ids=["b1"], length=4, seed=1, real_pairs=pairs, real_share=1.0)
            x, s, m, start = ds[0]
            self.assertEqual((tuple(x.shape), tuple(s.shape)), ((5, 128, 128, 128), (64, 64, 64)))
            self.assertEqual(int(x[0].sum()), int(inp[..., 0].sum()))   # the whole input, at its own cells
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
