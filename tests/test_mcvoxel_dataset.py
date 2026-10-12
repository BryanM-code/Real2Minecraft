"""The dataset-side additions to mcvoxel: scene voxelisation, target encoding, block-state
symmetry, stair corners and chunked reassembly."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcvoxel import block_shapes as bs  # noqa: E402
from mcvoxel import vocab  # noqa: E402
from mcvoxel.assign import assign_blocks  # noqa: E402
from mcvoxel.blockstates import mirror_x, rotate_y, stair_shapes  # noqa: E402
from mcvoxel.grid import occluded_interior, remove_ground_plate, voxelize_scene  # noqa: E402
from mcvoxel.scene import KIND_GLASS, KIND_SURFACE, Material, SceneMesh  # noqa: E402
from mcvoxel.tiling import assemble, iter_windows  # noqa: E402
from scene_fixtures import gable_house as _gable_house, mesh_to_scene as _mesh_to_scene  # noqa: E402


class VocabRoundTrip(unittest.TestCase):
    def test_encode_decode(self):
        cases = ["minecraft:oak_planks", "minecraft:oak_stairs[facing=east,half=bottom]",
                 "minecraft:stone_brick_slab[type=top]", "minecraft:brick_slab[type=bottom]",
                 "minecraft:quartz_stairs[facing=north,half=bottom]", "minecraft:glass"]
        for s in cases:
            sh, mat = vocab.encode_state(s)
            self.assertIsNotNone(mat, s)
            self.assertEqual(vocab.encode_state(vocab.decode(sh, mat)), (sh, mat), s)

    def test_shape_without_variant_decodes_to_a_real_block(self):
        mat = vocab.MATERIAL_INDEX["minecraft:white_concrete"]
        state = vocab.decode(vocab.SHAPES.index("stairs_west"), mat)
        self.assertIn("_stairs[facing=west", state)


class Symmetry(unittest.TestCase):
    def test_rotation_matches_classifier_on_rotated_mesh(self):
        """Rotating the voxel labels must equal voxelising the rotated mesh."""
        sm = _gable_house(window=False)
        vm = voxelize_scene(sm, 1.0, offset=(0.3, 0.0, 0.3), drop_ground=False, with_colors=False)
        base = np.full(vm.solid.shape, 0, np.uint8)
        for code, idx in vocab.FROM_BLOCK_SHAPES.items():
            base[vm.shapes == code] = idx
        stairs_before = {vocab.SHAPES[i] for i in np.unique(base) if i >= 4}
        self.assertTrue(stairs_before <= {"stairs_north", "stairs_south"}, stairs_before)
        self.assertTrue(stairs_before, "gable roof produced no stairs")
        (rot,) = rotate_y(base, 1)
        rotated_labels = {vocab.SHAPES[i] for i in np.unique(rot) if i >= 4}
        self.assertTrue(rotated_labels <= {"stairs_east", "stairs_west"}, rotated_labels)
        # Every stair must still face its high side after rotation: the voxel one up
        # toward `facing` is solid more often than the one up and away.
        solid = rot > 0
        good = bad = 0
        for f, (dx, dz) in {"east": (1, 0), "west": (-1, 0)}.items():
            xs, ys, zs = np.nonzero(rot == vocab.SHAPES.index("stairs_" + f))
            for x, y, z in zip(xs, ys, zs):
                def s(xx, yy, zz):
                    return 0 <= xx < solid.shape[0] and 0 <= yy < solid.shape[1] and 0 <= zz < solid.shape[2] \
                        and solid[xx, yy, zz]
                good += s(x + dx, y + 1, z + dz)
                bad += s(x - dx, y + 1, z - dz)
        self.assertGreater(good, bad)

    def test_mirror_swaps_east_west_only(self):
        g = np.zeros((3, 1, 3), np.uint8)
        g[0, 0, 0] = vocab.SHAPES.index("stairs_east")
        g[2, 0, 2] = vocab.SHAPES.index("stairs_north")
        (m,) = mirror_x(g)
        self.assertEqual(m[2, 0, 0], vocab.SHAPES.index("stairs_west"))
        self.assertEqual(m[0, 0, 2], vocab.SHAPES.index("stairs_north"))

    def test_four_rotations_are_identity(self):
        rng = np.random.default_rng(0)
        g = rng.integers(0, 8, size=(5, 3, 7)).astype(np.uint8)
        out = g
        for _ in range(4):
            (out,) = rotate_y(out, 1)
        self.assertTrue((out == g).all())


class StairCorners(unittest.TestCase):
    def test_outer_and_inner_corner(self):
        # Two stair runs meeting at a corner, as on a hip roof seen from above.
        names = ["minecraft:air", "minecraft:oak_stairs[facing=north,half=bottom]",
                 "minecraft:oak_stairs[facing=west,half=bottom]"]
        grid = np.zeros((1, 3, 3), np.uint16)          # (y, z, x)
        grid[0, 1, 1] = 1                               # faces north
        grid[0, 0, 1] = 2                               # in front (north) faces west
        out, new_names = stair_shapes(grid, names)
        state = new_names[out[0, 1, 1]]
        self.assertIn("shape=outer_", state)
        grid2 = np.zeros((1, 3, 3), np.uint16)
        grid2[0, 1, 1] = 1
        grid2[0, 2, 1] = 2                              # behind (south) faces west
        out2, names2 = stair_shapes(grid2, names)
        self.assertIn("shape=inner_", names2[out2[0, 1, 1]])


class SceneVoxelisation(unittest.TestCase):
    def test_house_target(self):
        sm = _gable_house()
        vm = voxelize_scene(sm, 1.0, offset=(0.3, 0.0, 0.3))
        self.assertTrue(vm.shell.any())
        tgt = assign_blocks(vm)
        states = set(tgt.states)
        self.assertTrue(any("stairs" in s for s in states), states)
        self.assertTrue(any("glass" in s for s in states), "window lost: %s" % states)
        # Supervision covers the shell and outside air; the interior is ignored.
        self.assertTrue((tgt.shape[vm.interior] == vocab.IGNORE_SHAPE).all())
        self.assertTrue((tgt.shape[~vm.solid] == 0).all())

    def test_open_bottom_is_still_enclosed(self):
        import trimesh
        box = trimesh.creation.box(extents=(8, 6, 8))
        box.apply_translation((4, 3, 4))
        keep = box.face_normals[:, 1] > -0.5              # drop the floor faces
        open_box = trimesh.Trimesh(box.vertices, box.faces[keep])
        sm = _mesh_to_scene([(open_box, (0.7, 0.7, 0.7, 1.0), KIND_SURFACE)])
        vm = voxelize_scene(sm, 1.0, offset=(0.25, 0.1, 0.25), drop_ground=False, with_colors=False)
        filled = vm.solid.sum()
        self.assertGreater(filled, 0.8 * np.prod(vm.solid.shape))

    def test_ground_plate_removed_podium_kept(self):
        solid = np.zeros((20, 12, 20), bool)
        solid[:, 0, :] = True                             # wide 1-layer lawn
        solid[6:14, 1:10, 6:14] = True                    # the building
        out, info = remove_ground_plate(solid)
        self.assertTrue(info["removed"])
        self.assertFalse(out[:, 0, :].any())
        podium = np.zeros((20, 20, 20), bool)
        podium[2:18, 0:5, 2:18] = True                    # a 5-storey-thick podium
        podium[7:13, 5:19, 7:13] = True
        out2, info2 = remove_ground_plate(podium)
        self.assertFalse(info2["removed"])

    def test_courtyard_stays_open(self):
        occ = np.zeros((9, 4, 9), bool)
        occ[:, :, 0] = occ[:, :, 8] = occ[0, :, :] = occ[8, :, :] = True   # four walls, no roof
        enclosed = occluded_interior(occ)
        self.assertFalse(enclosed.any())
        occ[:, 3, :] = True                                                  # add a roof
        self.assertTrue(occluded_interior(occ)[4, 1, 4])


class Tiling(unittest.TestCase):
    def test_windows_cover_grid(self):
        shape = (70, 150, 40)
        covered = np.zeros(shape, bool)
        for _, gsl, _ in iter_windows(shape, 64, 16):
            covered[gsl] = True
        self.assertTrue(covered.all())

    def test_oracle_reassembly_is_exact_even_with_bad_edges(self):
        rng = np.random.default_rng(1)
        shape = (90, 140, 70)
        truth_shape = rng.integers(0, 8, size=shape).astype(np.uint8)
        truth_mat = rng.integers(1, 130, size=shape).astype(np.int32)
        window, overlap = 64, 16

        def predict(start, gsl):
            w = window
            probs = np.zeros((8, w, w, w), np.float32)
            mat = np.zeros((w, w, w), np.int32)
            sub_shape = truth_shape[gsl]
            sub_mat = truth_mat[gsl]
            n = [g.stop - g.start for g in gsl]
            idx = np.indices(n)
            probs[(sub_shape,) + tuple(idx)] = 1.0
            mat[:n[0], :n[1], :n[2]] = sub_mat
            # Corrupt a 4-voxel band at every interior window edge: the model's
            # least-context region.
            band = np.zeros((w, w, w), bool)
            for ax in range(3):
                sl_lo = [slice(None)] * 3
                sl_hi = [slice(None)] * 3
                sl_lo[ax] = slice(0, 4)
                sl_hi[ax] = slice(n[ax] - 4, n[ax])
                if gsl[ax].start != 0:
                    band[tuple(sl_lo)] = True
                if gsl[ax].stop != shape[ax]:
                    band[tuple(sl_hi)] = True
            noise = rng.integers(0, 8, size=(w, w, w))
            probs[:, band] = 0.0
            probs[noise[band], np.nonzero(band)[0], np.nonzero(band)[1], np.nonzero(band)[2]] = 1.0
            mat[band] = 0
            return probs, mat

        got_shape, got_mat, _ = assemble(shape, window, overlap, predict, 8)
        self.assertTrue((got_shape == truth_shape).all())
        self.assertTrue((got_mat == truth_mat).all())


if __name__ == "__main__":
    unittest.main()
