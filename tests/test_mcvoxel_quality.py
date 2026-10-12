"""Regression tests for the target-quality stages of mcvoxel (0.2.0 rework, 2026-09-27): the
rasteriser's voxel/triangle pairs, sheet-aware sliver resolution, site and clutter removal,
colours seen from outside, and the builder palette. Small scenes built in code."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcvoxel import block_shapes as bs  # noqa: E402
from mcvoxel.assign import assign_blocks  # noqa: E402
from mcvoxel.block_palette import AUX_PALETTE, BUILDER_PALETTE, match_blocks  # noqa: E402
from mcvoxel.grid import pad_model, remove_site, voxelize_scene  # noqa: E402
from mcvoxel.raster import rasterize_surface  # noqa: E402
from mcvoxel.scene import KIND_SURFACE, clutter_kind, split_clutter  # noqa: E402
from scene_fixtures import WALL, _box, gable_house, mesh_to_scene  # noqa: E402

RED = (0.8, 0.1, 0.1, 1.0)
WHITE = (0.92, 0.92, 0.92, 1.0)


def _at(vm, world):
    """Grid index of the voxel containing world point `world` (blocks)."""
    return tuple(int(np.floor(w)) - int(o) for w, o in zip(world, vm.origin))


class RasterPairs(unittest.TestCase):
    def test_pairs_are_each_triangles_own_raster(self):
        import trimesh
        m = trimesh.creation.icosphere(subdivisions=2, radius=3.3)
        m.apply_translation((4.2, 3.9, 4.6))
        v, f = np.asarray(m.vertices), np.asarray(m.faces)
        pairs = []
        surf, origin = rasterize_surface(v, f, pairs=pairs)
        cells = np.concatenate([p[0] for p in pairs])
        tris = np.concatenate([p[1] for p in pairs])
        self.assertEqual(set(np.unique(cells).tolist()), set(np.flatnonzero(surf.ravel()).tolist()))
        for t in range(0, len(f), 7):
            one, _ = rasterize_surface(v, f[t:t + 1], origin=origin, shape=surf.shape)
            self.assertEqual(set(cells[tris == t].tolist()), set(np.flatnonzero(one.ravel()).tolist()), t)


class ThinSheets(unittest.TestCase):
    """A thin surface must come out as ONE layer of voxels: never doubled, never holed, and
    never deleted because something solid is next to it."""

    def test_thin_wall_across_a_voxel_boundary_is_one_layer(self):
        wall = _box((0.2, 4.6, 7.6), (3.0, 2.5, 4.0))        # x 2.9..3.1 straddles x = 3
        vm = voxelize_scene(mesh_to_scene([(wall, WALL, KIND_SURFACE)]), 1.0, with_colors=False,
                            drop_ground=False)
        per_yz = vm.solid.sum(axis=0)
        self.assertEqual(int(per_yz.max()), 1)                # not doubled
        self.assertEqual(int((per_yz == 1).sum()), 5 * 8)     # not holed

    def test_parapet_on_a_roof_survives(self):
        # A flat-roofed block with a 0.2 m parapet flush with its front face. The old rule
        # deleted the parapet as a "sliver" of the solid roof below it.
        block = _box((9.4, 5.4, 7.4), (5.0, 2.7, 4.0))        # x 0.3..9.7, y 0..5.4, z 0.3..7.7
        parapet = _box((9.4, 1.3, 0.2), (5.0, 6.05, 7.6))     # y 5.4..6.7, z 7.5..7.7
        vm = voxelize_scene(mesh_to_scene([(block, WALL, KIND_SURFACE), (parapet, WALL, KIND_SURFACE)]),
                            1.0, with_colors=False, drop_ground=False)
        x0, y6, z7 = _at(vm, (0.3, 6.2, 7.6))
        x1 = _at(vm, (9.6, 6.2, 7.6))[0]
        self.assertTrue(vm.solid[x0:x1 + 1, y6, z7].all(), "parapet course missing")
        self.assertFalse(vm.solid[:, y6, :z7].any(), "parapet thickened into the roof")

    def test_porch_posts_and_porch_air_survive(self):
        # Posts 0.2 m square, 3 m apart, between a porch floor and roof, 3 m deep, in front
        # of a closed house -- voxelised the way clean source models are (no closing). The
        # pilot-v2 settings broke the inner posts (the sliver rule deleted them next to the
        # solid floor / roof, the closing filled the gaps between them) and filled the porch.
        width, depth, z0 = 9.4, 3.0, 6.7
        house = _box((width, 5.4, 6.4), (0.3 + width / 2, 2.7, 3.5))
        floor = _box((width, 0.3, depth), (0.3 + width / 2, 0.15, z0 + depth / 2))
        roof = _box((width, 0.3, depth), (0.3 + width / 2, 3.15, z0 + depth / 2))
        xs = np.arange(0.6, 0.3 + width, 3.0)
        zp = z0 + depth - 0.3
        posts = [_box((0.2, 2.7, 0.2), (x, 1.65, zp)) for x in xs]
        parts = [(m, WALL, KIND_SURFACE) for m in [house, floor, roof] + posts]
        vm = voxelize_scene(mesh_to_scene(parts), 1.0, drop_ground=False, with_colors=False, clean=False)
        for x in xs:
            xi, _, zi = _at(vm, (x, 0.0, zp))
            self.assertTrue(vm.solid[xi, 1:3, zi].all(), "post at x={} broken: {}".format(
                x, vm.solid[xi, :, zi].astype(int)))
        _, y1, zb = _at(vm, (0.0, 1.5, zp - 1.0))
        self.assertFalse(vm.solid[1:-1, y1:y1 + 2, zb].any(), "porch air filled")

    def test_gable_roof_has_no_holes_from_above(self):
        vm = voxelize_scene(gable_house(width=11.3, depth=8.6), 1.0, (0.35, 0.0, 0.2), with_colors=False)
        from scipy import ndimage
        cover = vm.solid.any(axis=1)
        self.assertTrue((ndimage.binary_fill_holes(cover) == cover).all(), "holes in the roof")
        walls = vm.solid[:, 1, :]                             # footprint just above the ground
        self.assertTrue(cover[walls].all())


class Bridges(unittest.TestCase):
    def test_long_link_and_originally_separate_piece(self):
        from mcvoxel.grid import keep_bridges
        from scipy import ndimage
        solid = np.zeros((25, 5, 5), bool)
        solid[:4, :3, :3] = solid[20:, :3, :3] = True
        solid[4:20, 1, 1] = True
        solid[10:13, 4, 4] = True  # already separate: do not grow or join it
        removed = np.zeros_like(solid)
        removed[4:20, 1, 1] = True
        removed[10, 4, 4] = True
        out = keep_bridges(solid, removed)
        self.assertFalse(out[4:20, 1, 1].any())
        self.assertTrue(out[10, 4, 4])
        self.assertEqual(ndimage.label(solid & ~out)[1], 2)

    def test_removal_never_cuts_a_piece_off(self):
        from mcvoxel.grid import keep_bridges
        solid = np.zeros((10, 5, 5), bool)
        solid[0:4, 0:3, 1:4] = True                           # the body
        solid[4:6, 1, 2] = True                               # a two-voxel link
        solid[6:10, 0:3, 1:4] = True                          # a wing
        removed = np.zeros_like(solid)
        removed[4, 1, 2] = removed[5, 1, 2] = True            # both link voxels marked as repeats
        removed[0, 0, 1] = True                               # a harmless one
        out = keep_bridges(solid, removed)
        self.assertFalse(out[4, 1, 2] or out[5, 1, 2])        # the link is kept, over two rounds
        self.assertTrue(out[0, 0, 1])                         # the harmless removal stands


class SiteRemoval(unittest.TestCase):
    def test_lawn_wall_and_car_go_building_and_eaves_stay(self):
        solid = np.zeros((40, 12, 40), bool)
        solid[:, 0, :] = True                                 # lawn everywhere, one layer
        solid[15:25, 0:8, 15:25] = True                       # the house, 8 tall
        solid[14:26, 7, 14:26] = True                         # eaves one block out
        solid[2:38, 0:2, 2] = True                            # a low garden wall
        solid[30:33, 0:2, 30:34] = True                       # a parked car
        out, info = remove_site(solid, site_height=3, margin=2)
        self.assertTrue(out[15:25, 0:8, 15:25].all())
        self.assertTrue(out[14:26, 7, 14:26].all())
        self.assertTrue(out[12, 0, 20])                       # ground within the margin stays
        self.assertFalse(out[:, :, :12].any())                # wall and far lawn go
        self.assertFalse(out[30:33, :, 30:34].any())          # the car goes
        self.assertEqual(info["removed"], int(solid.sum() - out.sum()))

    def test_boundary_wall_goes_where_it_leaves_the_building(self):
        solid = np.zeros((40, 12, 40), bool)
        solid[15:25, 0:8, 15:25] = True                       # the house
        solid[2:38, 0:3, 2] = solid[2:38, 0:3, 37] = True     # a 3-high boundary wall, one thick
        solid[2, 0:3, 2:38] = solid[37, 0:3, 2:38] = True
        solid[18:20, 0:3, 2:15] = True                        # a garden wall meeting the house
        out, info = remove_site(solid, site_height=2, margin=2)
        self.assertTrue(out[15:25, 0:8, 15:25].all())
        self.assertFalse(out[:, :, 2].any() or out[:, :, 37].any() or out[2].any() or out[37].any())
        self.assertTrue(out[18:20, 0:3, 13:15].all())         # its first metres stay attached
        self.assertFalse(out[18:20, :, 3:12].any())
        tower = np.zeros((10, 30, 10), bool)
        tower[4:6, :, 4:6] = True                             # slender: no 3 x 3 mass at all
        self.assertTrue((remove_site(tower, site_height=2)[0] == tower).all())

    def test_nothing_tall_means_nothing_removed(self):
        flat = np.zeros((10, 3, 10), bool)
        flat[:, 0:2, :] = True
        out, info = remove_site(flat, site_height=3)
        self.assertTrue((out == flat).all())
        self.assertTrue(info.get("no_tall_columns"))

    def test_low_building_with_rooftop_sign_stays(self):
        # A three-block bank roof is below the site threshold after its ground plate
        # is stripped. Its taller sign must not become the whole building footprint.
        solid = np.zeros((16, 6, 20), bool)
        solid[3:12, 1:4, 3:17] = True
        solid[5:7, 4, 5:14] = True
        out, info = remove_site(solid, site_height=3, margin=2)
        np.testing.assert_array_equal(out, solid)
        self.assertTrue(info.get("no_broad_tall_core"))


class Frame(unittest.TestCase):
    def test_grid_is_cropped_to_what_is_kept(self):
        # A house on a wide lawn: after the site goes, the grid must hold the house only
        # (pilot v1-v3 never cropped, so every target kept the whole plot's frame).
        house = _box((6.4, 5.4, 6.4), (20.0, 3.0, 20.0))
        lawn = _box((39.6, 0.3, 39.6), (20.0, 0.15, 20.0))
        vm = voxelize_scene(mesh_to_scene([(house, WALL, KIND_SURFACE), (lawn, WALL, KIND_SURFACE)]), 1.0,
                            with_colors=False, site_height=2)
        self.assertIn("cropped_from", vm.info)
        self.assertLessEqual(max(vm.solid.shape[0], vm.solid.shape[2]), 8)
        self.assertTrue(vm.solid[:, :, 0].any() and vm.solid[:, :, -1].any())   # tight
        # A caller-fixed frame is kept as given.
        fixed = voxelize_scene(mesh_to_scene([(house, WALL, KIND_SURFACE)]), 1.0, origin=(10, 0, 10),
                               shape=(20, 10, 20), with_colors=False)
        self.assertEqual(fixed.solid.shape, (20, 10, 20))


class Clutter(unittest.TestCase):
    def test_names(self):
        for name in ("Palm_Tree01", "carJeep", "Leaves", "people_walking", "BUSH_3"):
            self.assertIsNotNone(clutter_kind(name), name)
        # Wood species, turf roofs, ground floors and words that merely contain "car" are not clutter.
        for name in ("Oak_planks_floor", "Wall_Plaster", "Roof_Grass", "Manor_Door", "Ground_Floor", "Carpet",
                     "Cartouche", "pine_bark_log"):
            self.assertIsNone(clutter_kind(name), name)

    def test_split_drops_faces_and_their_vertices(self):
        house = _box((10, 6, 8), (5, 3, 4))
        tree = _box((2, 12, 2), (15, 6, 4))
        sm = mesh_to_scene([(house, WALL, KIND_SURFACE), (tree, (0.2, 0.5, 0.1, 1.0), KIND_SURFACE)])
        sm.materials[1].name = "Tree_Leaves"
        out, info = split_clutter(sm)
        self.assertEqual(info["dropped"], len(tree.faces))
        lo, hi = out.bounds                                    # the scale must see the house only
        self.assertAlmostEqual(float(hi[1]), 6.0)
        self.assertAlmostEqual(float(hi[0]), 10.0)
        self.assertEqual(int(out.faces.max()), len(out.verts) - 1)
        self.assertEqual(len(out.uv), len(out.verts))

    def test_names_that_would_drop_most_of_the_model_are_not_trusted(self):
        sm = mesh_to_scene([(_box((10, 6, 8), (5, 3, 4)), WALL, KIND_SURFACE)])
        sm.materials[0].name = "Tree_House"
        out, info = split_clutter(sm)
        self.assertEqual(info["dropped"], 0)
        self.assertIn("skipped", info)
        self.assertIs(out, sm)


class ExposedColours(unittest.TestCase):
    def test_panel_in_front_of_a_wall_is_what_the_wall_shows(self):
        block = _box((9.6, 5.6, 7.6), (5.0, 2.8, 4.0))        # z 0.2..7.8
        panel = _box((4.0, 2.0, 0.04), (5.0, 3.0, 7.82))      # just in front of the +z face
        vm = voxelize_scene(mesh_to_scene([(block, WHITE, KIND_SURFACE), (panel, RED, KIND_SURFACE)]),
                            1.0, drop_ground=False, color_mode="exposed")
        x, y, z = _at(vm, (5.5, 3.5, 7.5))
        r, g, b = (int(c) for c in vm.rgb[x, y, z])
        self.assertTrue(r > 150 and g < 80, (r, g, b))
        x, y, z = _at(vm, (1.5, 3.5, 7.5))                    # beside the panel: the wall
        self.assertGreater(int(vm.rgb[x, y, z].min()), 180)

    def test_facade_beats_roof_at_the_eave(self):
        # White roof top, red fascia band on the front at the top: the eave voxel shows the
        # facade (what the street sees), not the roof.
        block = _box((9.6, 5.6, 7.6), (5.0, 2.8, 4.0))
        fascia = _box((9.6, 0.8, 0.04), (5.0, 5.2, 7.82))
        vm = voxelize_scene(mesh_to_scene([(block, WHITE, KIND_SURFACE), (fascia, RED, KIND_SURFACE)]),
                            1.0, drop_ground=False, color_mode="exposed")
        top = int(np.flatnonzero(vm.solid.any(axis=(0, 2)))[-1])
        x, _, z = _at(vm, (5.5, 0.0, 7.5))
        r, g, b = (int(c) for c in vm.rgb[x, top, z])
        self.assertTrue(r > 150 and g < 80, (r, g, b))
        x, _, z = _at(vm, (5.5, 0.0, 4.5))                    # the middle of the roof is white
        self.assertGreater(int(vm.rgb[x, top, z].min()), 180)


class RoofMaterial(unittest.TestCase):
    def test_roof_takes_the_block_nearest_its_median_colour(self):
        # Speckled shingles: 40% of the caps near-white, 35% light grey, 25% darker grey. The
        # most frequent block is the white one; the roof as a whole (its median) reads grey.
        from mcvoxel import roofs
        sm = gable_house(width=12.3, depth=9.6, roof_h=4.8)
        offset = np.array([0.4, 0.0, 0.3])
        vm = pad_model(voxelize_scene(sm, 1.0, offset), ((2, 2), (0, 2), (2, 2)))
        vb, plan = roofs.builder_roofs(vm, sm.verts + offset - vm.origin, sm.faces)
        cols = np.argwhere(plan.group > 0)
        self.assertGreater(len(cols), 20)
        for x, z in cols:
            y = int(plan.level[x, z])
            k = (x + 3 * z) % 20
            vb.rgb[x, y, z] = (224, 224, 220) if k < 8 else ((172, 172, 172) if k < 15 else (104, 104, 108))
        t = assign_blocks(vb, palette=BUILDER_PALETTE)
        roofs.harmonize_roof_materials(t, plan, palette=BUILDER_PALETTE, rgb=vb.rgb)
        index = {tuple(int(v) for v in c): i for i, c in enumerate(t.occ)}
        bases = {bs.base_block(t.states[index[(int(x), int(plan.level[x, z]), int(z))]])
                 for x, z in cols if (int(x), int(plan.level[x, z]), int(z)) in index}
        self.assertEqual(len(bases), 1, bases)
        self.assertNotIn(bases.pop(), ("minecraft:quartz_block", "minecraft:calcite"))


class BuilderPalette(unittest.TestCase):
    def test_palette_holds_building_materials_only(self):
        names = {b.split(":", 1)[1] for b in BUILDER_PALETTE}
        self.assertFalse(any(n.endswith("_wool") for n in names))
        for bad in ("snow_block", "gold_block", "dirt", "grass_block", "purpur_block", "crimson_planks",
                    "warped_planks", "iron_block", "packed_mud", "clay"):
            self.assertNotIn(bad, names)
        for good in ("stone_bricks", "bricks", "oak_planks", "white_concrete", "sandstone", "quartz_block"):
            self.assertIn(good, names)

    def test_assignment_and_shape_swaps_stay_in_the_palette(self):
        sm = gable_house()
        sm.materials[0].base_rgba = np.array([0.98, 0.98, 0.98, 1.0])   # snow white walls
        self.assertEqual(match_blocks([(250, 250, 250)]), ["minecraft:snow_block"])   # full palette would
        vm = voxelize_scene(sm, 1.0, (0.3, 0.0, 0.3))
        vm = pad_model(vm, ((2, 2), (0, 2), (2, 2)))
        t = assign_blocks(vm, palette=BUILDER_PALETTE)
        allowed = set(BUILDER_PALETTE) | set(AUX_PALETTE)
        used = {bs.base_block(s) for s in t.states}
        self.assertTrue(used <= allowed, used - allowed)
        self.assertTrue(any("stairs" in s or "slab" in s for s in t.states))


if __name__ == "__main__":
    unittest.main()
