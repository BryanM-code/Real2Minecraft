"""training.data (the loader contract's windows and symmetries) and training.model (shapes,
the space-to-depth stem, valid materials at inference). The model tests need torch."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcvoxel import vocab as V  # noqa: E402
from training import data as D  # noqa: E402

HAVE_TORCH = importlib.util.find_spec("torch") is not None
SH = V.SHAPE_INDEX


def pair(dims=(40, 30, 50), seed=0):
    """A target with oriented shapes and IGNORE, and an input whose occupancy is exactly the
    target's solid upsampled 2x (so the two can be checked against each other) and whose
    colour channel carries each block's x index."""
    rng = np.random.default_rng(seed)
    ts = np.zeros(dims, np.uint8)
    ts[5:25, 0:12, 8:30] = SH["full"]
    ts[5:25, 12, 8:30] = SH["stairs_east"]
    ts[10:20, 0:10, 12:25] = D.IGNORE_SHAPE
    ts[30:35, 0:4, 35:45] = SH["full_axis_x"]
    tm = np.where(ts == 0, 0, V.MATERIAL_INDEX["minecraft:bricks"]).astype(np.uint16)
    tm[ts == D.IGNORE_SHAPE] = D.IGNORE_MATERIAL
    solid = (ts != 0) & (ts != D.IGNORE_SHAPE)
    up = solid.repeat(2, 0).repeat(2, 1).repeat(2, 2)
    inp = np.zeros(up.shape + (5,), np.uint8)
    inp[..., 0] = up
    inp[..., 1] = up
    xs = np.arange(dims[0]).repeat(2)[:, None, None]
    inp[..., 2] = np.broadcast_to(xs, up.shape) * up
    inp[..., 3] = rng.integers(0, 255, up.shape) * up             # colour only on the surface
    return inp, ts, tm


def down(occ):
    """2x occupancy -> blocks (all 8 cells set)."""
    X, Y, Z = (s // 2 for s in occ.shape)
    return occ[:2 * X, :2 * Y, :2 * Z].reshape(X, 2, Y, 2, Z, 2).all(axis=(1, 3, 5))


class Windows(unittest.TestCase):
    def test_target_block_s_pairs_with_input_cell_2s(self):
        inp, ts, tm = pair()
        w_i, w_s, w_m = D.crop(inp, ts, tm, (3, 2, 7), size=16)
        solid = (w_s != 0) & (w_s != D.IGNORE_SHAPE)
        self.assertTrue(np.array_equal(down(w_i[..., 0]) & solid, solid))
        cells = np.nonzero(w_i[..., 0])                                      # solid cells carry their block's x
        self.assertTrue(len(cells[0]) > 0)
        self.assertTrue(np.array_equal(w_i[..., 2][cells], cells[0] // 2 + 3))

    def test_padding_beyond_the_grid_is_ignore_and_zero(self):
        inp, ts, tm = pair(dims=(20, 14, 20))
        w_i, w_s, w_m = D.crop(inp, ts, tm, (0, 0, 0), size=32)
        self.assertTrue((w_s[20:] == D.IGNORE_SHAPE).all() and (w_s[:, 14:] == D.IGNORE_SHAPE).all())
        self.assertTrue((w_m[20:] == D.IGNORE_MATERIAL).all())
        self.assertTrue((w_i[40:] == 0).all() and (w_i[:, 28:] == 0).all())
        self.assertFalse((w_s[:20, :14, :20] == D.IGNORE_SHAPE).all())

    def test_starts_cover_the_grid_and_short_axes_start_at_zero(self):
        rng = np.random.default_rng(1)
        starts = np.array([D.sample_start((100, 30, 64), rng) for _ in range(500)])
        self.assertEqual(starts[:, 0].min(), 0)
        self.assertEqual(starts[:, 0].max(), 36)
        self.assertTrue((starts[:, 1] == 0).all() and (starts[:, 2] == 0).all())

    def test_sampled_windows_have_solid(self):
        inp, ts, tm = pair()
        rng = np.random.default_rng(2)
        for _ in range(20):
            w_i, w_s, w_m, start = D.sample_window(inp, ts, tm, rng, size=16, min_solid=8)
            self.assertGreaterEqual(D.supervised_solid(w_s), 8)


class Symmetry(unittest.TestCase):
    def test_all_eight_keep_input_and_target_together(self):
        inp, ts, tm = pair()
        w_i, w_s, w_m = D.crop(inp, ts, tm, (0, 0, 0), size=50)
        for k in range(4):
            for mirror in (False, True):
                a_i, a_s, a_m = D.symmetry(w_i, w_s, w_m, k, mirror)
                solid = (a_s != 0) & (a_s != D.IGNORE_SHAPE)
                self.assertTrue(np.array_equal(down(a_i[..., 0]) & solid, solid), (k, mirror))
                self.assertTrue(np.array_equal(a_m[solid], np.full(solid.sum(), V.MATERIAL_INDEX["minecraft:bricks"])))
                self.assertEqual(int((a_s == D.IGNORE_SHAPE).sum()), int((w_s == D.IGNORE_SHAPE).sum()))

    def test_stairs_and_pillars_turn_with_the_grid(self):
        inp, ts, tm = pair()
        w_i, w_s, w_m = D.crop(inp, ts, tm, (0, 0, 0), size=50)
        _, a_s, _ = D.symmetry(w_i, w_s, w_m, 1, False)
        labels = set(np.unique(a_s).tolist())
        self.assertNotIn(SH["stairs_east"], labels)            # every east stair faces another way
        self.assertIn(SH["full_axis_z"], labels)               # an x pillar lies along z
        _, m_s, _ = D.symmetry(w_i, w_s, w_m, 0, True)
        self.assertIn(SH["stairs_west"], set(np.unique(m_s).tolist()))


class Augment(unittest.TestCase):
    def window(self):
        inp, ts, tm = pair()
        return D.crop(inp, ts, tm, (0, 0, 0), size=50)[0]

    def test_off_is_the_identity(self):
        w = self.window()
        off = {"brightness": (1.0, 1.0), "channel_sd": 0.0, "noise_sd": (0.0, 0.0), "grey_p": 0.0,
               "blobs": (0, 0), "dents": (0, 0), "radius_cells": (2, 2), "close_p": 0.0, "shift_p": 0.0}
        self.assertTrue(np.array_equal(D.augment(w, np.random.default_rng(0), off), w))

    def test_keeps_the_layout_and_colours_only_the_surface(self):
        w = self.window()
        for seed in range(6):
            a = D.augment(w, np.random.default_rng(seed))
            self.assertEqual((a.shape, a.dtype), (w.shape, w.dtype))
            self.assertTrue(set(np.unique(a[..., 0]).tolist()) <= {0, 1})
            self.assertFalse((a[..., 1] & ~a[..., 0]).any())                   # surface lies on solid
            self.assertFalse(a[..., 2:][a[..., 1] == 0].any())                 # rgb only on the surface

    def test_blobs_add_and_dents_remove_solid_near_the_surface(self):
        w = self.window()
        base = {"brightness": (1.0, 1.0), "channel_sd": 0.0, "noise_sd": (0.0, 0.0), "grey_p": 0.0,
                "radius_cells": (3, 3), "close_p": 0.0, "shift_p": 0.0}
        more = D.augment(w, np.random.default_rng(1), {**base, "blobs": (3, 3), "dents": (0, 0)})
        less = D.augment(w, np.random.default_rng(1), {**base, "blobs": (0, 0), "dents": (3, 3)})
        self.assertGreater(int(more[..., 0].sum()), int(w[..., 0].sum()))
        self.assertLess(int(less[..., 0].sum()), int(w[..., 0].sum()))
        new = (more[..., 1] == 1) & (w[..., 1] == 0)
        self.assertTrue((more[..., 2:][new].sum(axis=-1) > 0).all())          # new surface has a colour


class Weights(unittest.TestCase):
    def test_big_builds_count_by_the_square_root_and_variants_share(self):
        pairs = [{"id": "small", "target_dims": [30, 20, 30]}] * 4 + [{"id": "big", "target_dims": [256, 64, 256]}] * 4
        w = D.build_weights(pairs)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(w[4:].sum() / w[:4].sum(), 4.0)    # 16 windows -> sqrt = 4


class TrellisInputs(unittest.TestCase):
    """data.add_trellis puts validated real-TRELLIS train inputs into the cache;
    WindowDataset draws them with probability trellis_share."""

    def dataset(self, tmp: Path):
        import json
        inp, ts, tm = pair(dims=(20, 14, 20))
        settings = {"version": "pairs-test"}
        index = {"version": "pairs-test", "settings": settings, "pairs": [
            {"id": b, "variant": v, "split": split, "status": "ok", "target_shape": list(ts.shape),
             "file": f"synthetic/{b}/v{v}.npz", "validation": {"flags": []}}
            for b, split in (("a", "train"), ("b", "val")) for v in (0, 1)]}
        (tmp / "pairs" / "synthetic").mkdir(parents=True)
        for b in ("a", "b"):
            (tmp / "pairs" / "synthetic" / b).mkdir()
            for v in (0, 1):
                np.savez_compressed(tmp / "pairs" / "synthetic" / b / f"v{v}.npz", input_occ=inp[..., 0],
                                    input_surface=inp[..., 1], input_rgb=inp[..., 2:], target_shape=ts,
                                    target_material=tm)
        (tmp / "pairs" / "index.json").write_text(json.dumps(index))
        D.build_cache(tmp, log=None)
        tr = tmp / "pairs" / D.TRELLIS_SET
        builds = [{"id": "a", "split": "train", "status": "ok", "validation": {"failures": [], "flags": []}},
                  {"id": "b", "split": "val", "status": "ok", "validation": {"failures": [], "flags": []}},
                  {"id": "c", "split": "train", "status": "ok",
                   "validation": {"failures": ["input_misses_building:0.3"], "flags": []}}]
        (tr / "a").mkdir(parents=True)
        trellis_inp = inp.copy()
        trellis_inp[..., 2:] = 7                                   # recognisable colour
        trellis_inp[..., 2:][trellis_inp[..., 1] == 0] = 0
        np.savez_compressed(tr / "a" / "input.npz", input_occ=trellis_inp[..., 0], input_surface=trellis_inp[..., 1],
                            input_rgb=trellis_inp[..., 2:])
        (tr / "index.json").write_text(json.dumps({"builds": builds}))
        return tmp

    def test_only_validated_train_builds_are_added(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            ds = self.dataset(Path(t))
            info = D.add_trellis(ds, log=None)
            self.assertEqual(info["added"], 1)
            self.assertEqual(set(info["skipped"]), {"b", "c"})
            cache = D.Cache(ds)                                  # still current: only synthetic rows counted
            rows = [p for p in cache.pairs if p.get("source") == "trellis"]
            self.assertEqual([(p["id"], p["variant"]) for p in rows], [("a", "trellis")])
            self.assertEqual(int(np.asarray(cache.input("a", "trellis"))[..., 2].max()), 7)
            self.assertEqual(D.add_trellis(ds, log=None)["added"], 1)    # idempotent

    @unittest.skipUnless(HAVE_TORCH, "torch not installed (the r2m_train environment has it)")
    def test_trellis_share_picks_the_real_input(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            ds = self.dataset(Path(t))
            D.add_trellis(ds, log=None)
            for share, real in ((1.0, True), (0.0, False)):
                data = D.WindowDataset(ds, "train", length=6, symmetries=False, trellis_share=share)
                self.assertEqual(data.builds, ["a"])
                for i in range(6):
                    x = data[i][0]                       # (5, 2w, 2w, 2w): the TRELLIS input's colour is 7
                    self.assertEqual(int(x[2:].max()) == 7, real, (share, i))


@unittest.skipUnless(HAVE_TORCH, "torch not installed (the r2m_train environment has it)")
class Model(unittest.TestCase):
    def test_space_to_depth_keeps_every_cell(self):
        import torch
        from training.model import space_to_depth
        x = torch.arange(2 * 3 * 8 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8, 8)
        y = space_to_depth(x)
        self.assertEqual(tuple(y.shape), (2, 24, 4, 4, 4))
        self.assertTrue(torch.equal(torch.sort(y.flatten()).values, torch.sort(x.flatten()).values))
        self.assertEqual(float(y[0, 0, 1, 2, 3]), float(x[0, 0, 2, 4, 6]))   # cell (0,0,0) of block (1,2,3)

    def test_heads_and_valid_materials(self):
        import torch
        from training.model import UNet3D, features, predict_window, valid_table
        m = UNet3D((8, 16), blocks=1).eval()
        x8 = torch.randint(0, 2, (1, 5, 16, 16, 16), dtype=torch.uint8)
        logits, f = m(features(x8, torch.tensor([3])))
        self.assertEqual(tuple(logits.shape), (1, 30, 8, 8, 8))
        where = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
        where[0, 1, 2, 3] = True
        self.assertEqual(tuple(m.material_logits(f, where).shape), (1, 313))
        probs, mat = predict_window(m, features(x8, torch.tensor([3])), valid_table("cpu"), air_below=1.01)
        shape = probs[:, 1:].argmax(dim=1) + 1
        ok = torch.as_tensor(V.VALID)[shape.flatten(), mat.flatten()]
        self.assertTrue(bool(ok.all()))

    def test_height_channel_counts_blocks_from_the_floor(self):
        import torch
        from training.model import HEIGHT_NORM, features
        x = features(torch.zeros(1, 5, 4, 4, 4, dtype=torch.uint8), torch.tensor([10]))
        self.assertAlmostEqual(float(x[0, 5, 0, 0, 0]), 10.0 / HEIGHT_NORM)
        self.assertAlmostEqual(float(x[0, 5, 0, 3, 0]), 11.5 / HEIGHT_NORM)


if __name__ == "__main__":
    unittest.main()
