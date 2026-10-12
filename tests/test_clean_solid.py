"""`raster.clean_solid`: the solid cleanup must depend on the shape, not on where the grid ends.

Regression tests for the boundary-layer bug fixed on 2026-09-25: scipy's binary_closing
works inside the array only, so its erosion stripped every solid voxel on the grid boundary
(a solid 6^3 cube came back 4^3), and every mesh voxelised in its own bounding box lost its
outer layer on all six sides.

Run from the repo root:  r2m_app\\Scripts\\python.exe -m unittest discover -s tests
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcvoxel.raster import clean_solid  # noqa: E402


def _blob(rng, shape, n_boxes=6):
    """A few overlapping random boxes, some of them running into the grid boundary."""
    g = np.zeros(shape, dtype=bool)
    for _ in range(n_boxes):
        lo = [int(rng.integers(0, max(1, s - 2))) for s in shape]
        hi = [int(min(s, a + rng.integers(2, max(3, s // 2 + 2)))) for a, s in zip(lo, shape)]
        g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return g


class BoundaryLayer(unittest.TestCase):
    def test_block_filling_the_whole_grid_survives(self):
        cube = np.ones((6, 6, 6), dtype=bool)
        out = clean_solid(cube)
        self.assertEqual(int(out.sum()), 216)            # was 64: a 4^3 core
        self.assertTrue(out.all())

    def test_every_face_of_the_grid_keeps_its_layer(self):
        slab = np.zeros((8, 5, 7), dtype=bool)
        slab[:, :2, :] = True                              # touches five faces
        self.assertTrue((clean_solid(slab) == slab).all())

    def test_result_does_not_depend_on_the_margin_of_air(self):
        rng = np.random.default_rng(7)
        for trial in range(25):
            shape = tuple(int(v) for v in rng.integers(5, 14, size=3))
            g = _blob(rng, shape)
            tight = clean_solid(g)
            for pad in (1, 2, 5):
                wide = clean_solid(np.pad(g, pad))[pad:-pad, pad:-pad, pad:-pad]
                self.assertTrue((wide == tight).all(), "trial {} pad {}".format(trial, pad))


class Cleanup(unittest.TestCase):
    def test_closing_bridges_a_one_voxel_slot(self):
        g = np.zeros((10, 10, 10), dtype=bool)
        g[2:8, 2:8, 2:8] = True
        g[5, 2:8, 2:8] = False                             # a crack right through the block
        out = clean_solid(g)
        # Bridged wherever solid surrounds it; its rim, open to the outside air, stays open.
        self.assertTrue(out[5, 3:7, 3:7].all())
        self.assertEqual(int(out.sum()), 6 ** 3 - 20)
        # Without the closing the two halves stay apart and only the larger one is kept.
        self.assertEqual(int(clean_solid(g, close=False).sum()), 3 * 6 * 6)

    def test_slot_running_into_the_grid_edge(self):
        g = np.ones((6, 6, 6), dtype=bool)
        g[:, :, 3] = False                                 # the slot reaches four faces
        out = clean_solid(g)
        # Both halves survive whole, outer layers included (the old closing stripped them)...
        self.assertTrue(out[:, :, :3].all() and out[:, :, 4:].all())
        # ...the slot is bridged where solid surrounds it, and stays open where it meets the
        # outside air at the grid's edge, as it would for the same shape anywhere else.
        self.assertTrue(out[1:5, 1:5, 3].all())
        self.assertEqual(int(out[:, :, 3].sum()), 16)

    def test_keeps_only_the_largest_component(self):
        g = np.zeros((12, 12, 12), dtype=bool)
        g[1:6, 1:6, 1:6] = True
        g[9:11, 9:11, 9:11] = True                         # a floating island
        out = clean_solid(g)
        self.assertEqual(int(out.sum()), 125)
        self.assertFalse(out[9:11, 9:11, 9:11].any())
        self.assertEqual(int(clean_solid(g, keep_largest=False).sum()), 125 + 8)

    def test_despur_is_opt_in(self):
        g = np.zeros((9, 9, 9), dtype=bool)
        g[2:6, 0:4, 2:6] = True
        g[3, 4:8, 3] = True                                # a 1-voxel spire on top
        self.assertTrue(clean_solid(g, close=False)[3, 7, 3])
        self.assertFalse(clean_solid(g, close=False, despur=True)[3, 7, 3])

    def test_input_is_not_modified(self):
        g = np.ones((4, 4, 4), dtype=bool)
        g[1, 1, 1] = False
        before = g.copy()
        clean_solid(g)
        self.assertTrue((g == before).all())


if __name__ == "__main__":
    unittest.main()
