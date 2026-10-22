"""photofirst's real-photo path: placement without a target (place), the .schem writer
(schem), the photo helpers (real: EXIF, focal, tilt guard)."""
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from photofirst import place as PL
from photofirst import real as RL
from photofirst import schem as SC


def turn(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def box_views(world: np.ndarray, size=(8.0, 5.0, 6.0), n=4):
    """Views of a box's four walls and roof (points only, no rendering), each from a camera on
    a circle, all turned into an arbitrary world frame by `world` (3x3)."""
    sx, sy, sz = size
    views = []
    for k in range(n):
        yaw = math.radians(30 + 90 * k)
        cam = np.array([math.sin(yaw) * 20, 4.0, math.cos(yaw) * 20])
        fwd = -cam / np.linalg.norm(cam)
        right = np.cross(fwd, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        g = np.linspace(0.0, 1.0, 40)
        u, v = np.meshgrid(g, g, indexing="ij")
        faces = [np.stack([u * sx, v * sy, np.zeros_like(u)], -1), np.stack([u * sx, v * sy, np.full_like(u, sz)], -1),
                 np.stack([np.zeros_like(u), v * sy, u * sz], -1), np.stack([np.full_like(u, sx), v * sy, u * sz], -1),
                 np.stack([u * sx, np.full_like(u, sy), v * sz], -1)]
        pts = np.concatenate(faces, 0) - np.array([sx / 2, 0.0, sz / 2])      # (5*40, 40, 3), ground at y = 0
        c2w = np.eye(4)
        c2w[:3, :3] = world @ np.stack([right, down, fwd], 1)
        c2w[:3, 3] = world @ cam
        views.append({"pts": (pts @ world.T).astype(np.float32), "mask": np.ones(pts.shape[:2], bool),
                      "cam2world": c2w, "ray": np.zeros(pts.shape, np.float32),
                      "rgb": np.full(pts.shape, 120, np.uint8)})
    return views


class Placement(unittest.TestCase):
    def test_up_from_level_cameras(self):
        world = turn((1.0, 0.0, 0.0), 25.0) @ turn((0.0, 1.0, 0.0), 40.0)
        up = PL.up_from_cameras(box_views(world))
        self.assertGreater(float(up @ (world @ [0.0, 1.0, 0.0])), math.cos(math.radians(0.5)))

    def test_level_up_and_wall_axes(self):
        world = turn((0.0, 0.0, 1.0), 9.0) @ turn((0.0, 1.0, 0.0), 20.0)
        P, N = PL.fused_points(box_views(world))
        true_up = world @ np.array([0.0, 1.0, 0.0])
        up = PL.level_up(N, world @ np.array([0.1, 1.0, 0.05]) / np.linalg.norm([0.1, 1.0, 0.05]))
        self.assertLess(math.degrees(math.acos(min(1.0, float(up @ true_up)))), 0.5)
        x_axis, z_axis, share = PL.wall_axes(N, up)
        walls = [world @ np.array(v) for v in ((1.0, 0, 0), (0, 0, 1.0), (-1.0, 0, 0), (0, 0, -1.0))]
        self.assertGreater(max(float(x_axis @ w) for w in walls), math.cos(math.radians(1.0)))
        self.assertGreater(share, 0.5)

    def test_free_placement_puts_the_ground_at_zero(self):
        world = turn((0.0, 1.0, 0.0), 33.0)
        views = box_views(world)
        sim, dims, info = PL.free_placement(views, height_blocks=10.0, margin=3, below=0)
        P, _ = PL.fused_points(views)
        q = sim["s"] * (P @ np.asarray(sim["R"]).T) + sim["t"]
        self.assertAlmostEqual(float(np.percentile(q[:, 1], 0.2)), 0.0, places=6)
        self.assertAlmostEqual(info["blocks_per_unit"] * info["height_units"], 10.0, places=6)
        self.assertTrue(4.5 < info["height_units"] <= 5.0)                          # the box is 5 tall
        self.assertTrue(all(d > 0 for d in dims) and dims[1] >= 10)
        self.assertTrue((q.min(0) >= -0.01).all() and (q.max(0) <= np.array(dims) + 0.01).all())


class OntoTarget(unittest.TestCase):
    def test_a_turned_shifted_input_comes_back(self):
        from photofirst import front_mv as M
        ts = np.zeros((14, 8, 10), np.uint8)
        ts[2:11, 0:5, 3:8] = 1                                   # an L-free box, not square: the turn matters
        ts[2:5, 5:7, 3:5] = 1                                    # a tower on one corner breaks the symmetry
        tm = np.where(ts == 1, 5, 0).astype(np.uint16)
        inp = np.zeros((28, 16, 20, 5), np.uint8)                # 2 cells per block
        inp[..., 0] = np.repeat(np.repeat(np.repeat(ts, 2, 0), 2, 1), 2, 2)
        free = np.rot90(inp, 1, axes=(0, 2))                     # the free frame: a quarter turn
        free = np.pad(free, ((4, 0), (0, 0), (6, 0), (0, 0)))    # and shifted by (2, 0, 3) blocks
        dims = (free.shape[0] // 2, free.shape[1] // 2, free.shape[2] // 2)
        back, info = M.onto_target(free, dims, ts, tm)
        self.assertEqual(back.shape, inp.shape)
        self.assertTrue(np.array_equal(back[..., 0], inp[..., 0]))


class Schem(unittest.TestCase):
    def test_round_trip_and_hollow(self):
        import nbtlib
        from mcvoxel.vocab import MATERIAL_INDEX, TARGET_DATA_VERSION
        s = np.zeros((6, 5, 7), np.uint8)
        m = np.zeros((6, 5, 7), np.uint16)
        s[1:5, 0:4, 1:6] = 1
        m[s == 1] = MATERIAL_INDEX["minecraft:stone_bricks"]
        tmp = Path(tempfile.mkdtemp())
        try:
            info = SC.write_schem(s, m, tmp / "b.schem")
            f = nbtlib.load(tmp / "b.schem")
            self.assertEqual(f.root_name, "Schematic")
            self.assertEqual(int(f["DataVersion"]), TARGET_DATA_VERSION)
            self.assertEqual((int(f["Width"]), int(f["Height"]), int(f["Length"])), (6, 5, 7))
            self.assertIn("minecraft:stone_bricks", [str(k) for k in f["Palette"]])
            self.assertEqual(info["blocks"], int((s != 0).sum()))
        finally:
            shutil.rmtree(tmp)
        hs, hm = SC.hollow(s, m)
        self.assertFalse(hs[2:4, 1:3, 2:5].any())                     # the inside is gone
        self.assertTrue(hs[1, 1, 3] and hs[2, 3, 3] and hs[2, 0, 3])  # walls, roof and floor stay
        self.assertTrue(((hs == 0) == (hm == 0)).all())


class Photos(unittest.TestCase):
    def test_focal_from_35mm_equivalent(self):
        f = RL.focal_px({"focal35_mm": 26.0}, 4000, 3000)
        self.assertAlmostEqual(f, 26.0 / 43.27 * 5000.0, places=3)
        self.assertIsNone(RL.focal_px({"focal35_mm": None}, 4000, 3000))

    def test_steady_pitches(self):
        p, fixed = RL.steady_pitches([10.0, 12.0, 60.0, 11.0])
        self.assertEqual(fixed, [False, False, True, False])
        self.assertAlmostEqual(p[2], 11.5)

    def test_exif_compass_heading(self):
        from PIL import Image
        tmp = Path(tempfile.mkdtemp())
        try:
            ex = Image.Exif()
            gps = ex.get_ifd(0x8825)
            gps[16] = "T"
            gps[17] = 123.5
            Image.new("RGB", (32, 24), (200, 10, 10)).save(tmp / "p.jpg", exif=ex.tobytes())
            info = RL.exif_info(tmp / "p.jpg")
            self.assertAlmostEqual(info["heading_deg"], 123.5, places=3)
            self.assertEqual(info["heading_ref"], "T")
            self.assertIsNone(RL.exif_info(tmp / "missing.jpg")["heading_deg"])
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
