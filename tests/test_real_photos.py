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

    def test_room_for_the_unseen_back(self):
        views = box_views(np.eye(3))                               # an 8 x 5 x 6 box
        front = [views[0], views[3]]                               # both cameras on its +z side
        sim, dims, info = PL.free_placement(front, metres_per_block=0.5, margin=3, below=2)
        self.assertEqual(len(info["back_extended"]), 1)            # the depth only, away from them
        sim_all, dims_all, info_all = PL.free_placement(views, metres_per_block=0.5, margin=3, below=2)
        self.assertEqual(info_all["back_extended"], [])           # round it: nothing unseen
        self.assertEqual(sorted(dims), sorted([dims_all[0], dims_all[1], max(dims_all[0], dims_all[2])]))

    def test_ground_from_cameras_under_a_hidden_base(self):
        world = turn((0.0, 1.0, 0.0), 21.0)
        views = box_views(world)                                   # cameras 4 above the ground
        masks = [(np.asarray(v["pts"]) @ world)[..., 1] >= 1.5 for v in views]   # the base hidden
        # eye height 4: the ground at the box's foot (y = 0) goes `below` (2) above the floor, 2 blocks a unit
        sim, dims, info = PL.free_placement(views, masks, metres_per_block=0.5, margin=3, below=2, eye_units=4.0)
        self.assertEqual(info["ground"], "cameras")
        self.assertAlmostEqual(float(sim["t"][1]), 2.0, delta=0.05)
        self.assertGreaterEqual(dims[1], 2 + 10 + 3)               # ground to top: 5 units = 10 blocks
        # an eye height that puts the ground above what the photos show gives way to the points
        sim, dims, info = PL.free_placement(views, masks, metres_per_block=0.5, margin=3, below=2, eye_units=1.0)
        self.assertEqual(info["ground"], "lowest_point")
        P, _ = PL.fused_points(views, masks)                       # lowest point with a normal, as placed
        lowest = float(np.percentile(P @ (world @ [0.0, 1.0, 0.0]), 0.2))
        self.assertTrue(1.5 <= lowest < 2.0)
        self.assertAlmostEqual(float(sim["t"][1]), 2.0 - 2 * lowest, delta=0.01)


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
        # and the target into the free frame (`target_into`, for pairs placed as on real
        # photos): the free input's own solid, stair facings turned with it
        from mcvoxel import vocab
        from mcvoxel.blockstates import rotate_y
        from training.score import _place
        ts2 = ts.copy()
        stair = vocab.SHAPES.index(next(s for s in vocab.SHAPES if "stair" in s))
        ts2[2:5, 4, 3:8] = stair                                 # facings that a turn must move
        tm2 = np.where(ts2 != 0, 5, 0).astype(np.uint16)
        f_s, f_m = M.target_into(ts2, tm2, info, dims)
        self.assertEqual(f_s.shape, dims)
        free_solid = np.asarray(free[0::2, 0::2, 0::2, 0] > 0)
        self.assertTrue(np.array_equal(f_s != 0, free_solid))
        k = int(info["turns"]) % 4                               # forward again: the target back
        r_s, r_m = rotate_y(f_s, k, f_m) if k else (f_s, f_m)
        self.assertTrue(np.array_equal(_place(r_s, info["shift"], ts2.shape, 0), ts2))
        self.assertTrue(np.array_equal(_place(r_m, info["shift"], ts2.shape, 0), tm2))


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
