"""Builder-style roofs (`mcvoxel.roofs`, Approach B) on roofs built in code."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcvoxel import block_shapes as bs  # noqa: E402
from mcvoxel import roofs  # noqa: E402
from mcvoxel.assign import assign_blocks  # noqa: E402
from mcvoxel.grid import pad_model, voxelize_scene  # noqa: E402
from scene_fixtures import flat_block, gable_house, hip_house  # noqa: E402

PITCHED = roofs.SLOPE_CLASSES.index("pitched")
LOW = roofs.SLOPE_CLASSES.index("low")
FLAT = roofs.SLOPE_CLASSES.index("flat")


def build(sm, offset=(0.3, 0.0, 0.3), colors=False):
    """(geometric model, builder model, plan) the way the dataset pipeline makes them."""
    vm = voxelize_scene(sm, 1.0, offset=offset, drop_ground=False, with_colors=colors)
    vm = pad_model(vm, ((0, 0), (0, 2), (0, 0)))              # room for a raised cap
    verts = sm.verts + np.asarray(offset) - vm.origin
    vm_b, plan = roofs.builder_roofs(vm, verts, sm.faces)
    return vm, vm_b, plan


class Pitched(unittest.TestCase):
    def test_45_degree_gable_becomes_up_slope_stair_courses(self):
        sm = gable_house(depth=8.0, roof_h=4.0, overhang=0.0, window=False)   # rise 1 per block
        vm, vm_b, plan = build(sm)
        cols = np.argwhere(plan.cls == PITCHED)
        self.assertGreater(len(cols), 20)
        ridge_z = 4.0 - vm.origin[2]
        want = np.where(cols[:, 1] + 0.5 < ridge_z, bs.SHAPE_STAIR_S, bs.SHAPE_STAIR_N)
        got = plan.shape[cols[:, 0], cols[:, 1]]
        self.assertGreaterEqual(float((got == want).mean()), 0.95)
        # One block up per block across: along each slope the cap level climbs by 1.
        x = int(np.median(cols[:, 0]))
        south = sorted(int(plan.level[x, z]) for z in range(plan.level.shape[1])
                       if plan.cls[x, z] == PITCHED and z + 0.5 < ridge_z)
        self.assertEqual(np.diff(south).tolist(), [1] * (len(south) - 1))

    def test_courses_do_not_depend_on_where_the_grid_falls(self):
        sm = gable_house(depth=8.0, roof_h=4.0, overhang=0.0, window=False)
        for oy in (0.0, 0.3, 0.55, 0.8):
            for oz in (0.0, 0.5):
                _, _, plan = build(sm, offset=(0.3, oy, oz))
                shapes = plan.shape[plan.cls == PITCHED]
                stairs = np.isin(shapes, [bs.SHAPE_STAIR_N, bs.SHAPE_STAIR_S])
                self.assertGreaterEqual(float(stairs.mean()), 0.95, (oy, oz))

    def test_the_model_really_gets_the_caps(self):
        sm = gable_house(depth=8.0, roof_h=4.0, overhang=0.0, window=False)
        vm, vm_b, plan = build(sm)
        for x, z in np.argwhere(plan.level >= 0):
            b = int(plan.level[x, z])
            self.assertTrue(vm_b.solid[x, b, z])
            self.assertFalse(vm_b.solid[x, b + 1:, z].any())    # nothing above the cap
            self.assertEqual(int(vm_b.shapes[x, b, z]), int(plan.shape[x, z]))

    def test_hip_roof_faces_all_four_ways(self):
        _, _, plan = build(hip_house(size=12.0, wall_h=5.0, roof_h=6.0))   # rise 1 per block
        shapes = set(np.unique(plan.shape[plan.cls == PITCHED]).tolist())
        self.assertTrue({bs.SHAPE_STAIR_E, bs.SHAPE_STAIR_W, bs.SHAPE_STAIR_N, bs.SHAPE_STAIR_S} <= shapes)


class LowAndFlat(unittest.TestCase):
    def test_half_pitch_is_a_slab_roof(self):
        sm = gable_house(depth=8.0, roof_h=2.0, overhang=0.0, window=False)   # rise 0.5 per block
        vm, _, plan = build(sm)
        m = plan.cls == LOW
        self.assertGreater(int(m.sum()), 20)
        shapes = plan.shape[m]
        ok = np.isin(shapes, [bs.SHAPE_SLAB_BOTTOM, bs.SHAPE_FULL])
        self.assertTrue(ok.all())
        share = float((shapes == bs.SHAPE_SLAB_BOTTOM).mean())
        self.assertTrue(0.3 <= share <= 0.7, share)

    def test_flat_roof_is_one_level_of_one_block_kind(self):
        _, vm_b, plan = build(flat_block(height=5.3))
        m = plan.cls == FLAT
        self.assertGreater(int(m.sum()), 50)
        self.assertEqual(len(set(zip(plan.level[m].tolist(), plan.shape[m].tolist()))), 1)

    def test_glass_caps_are_left_geometric(self):
        vm, vm_b, plan = build(flat_block(height=5.0, glass_roof=True), colors=True)
        X, Y, Z = vm.solid.shape
        top = Y - 1 - np.argmax(vm.solid[:, ::-1, :], axis=1)
        xs, zs = np.nonzero(vm.solid.any(axis=1))
        glass = np.zeros((X, Z), bool)
        glass[xs, zs] = vm.kind_frac[xs, top[xs, zs], zs, 1].astype(float) > 0.5
        self.assertGreater(int(glass.sum()), 10)
        self.assertEqual(plan.info["glass_columns_skipped"], int(glass.sum()))
        self.assertTrue((plan.level[glass] == -1).all())
        cols_b, cols_a = vm_b.solid.transpose(0, 2, 1), vm.solid.transpose(0, 2, 1)   # (X, Z, Y)
        self.assertTrue((cols_b[glass] == cols_a[glass]).all())


class Materials(unittest.TestCase):
    def test_each_roof_gets_one_stair_capable_material(self):
        sm = gable_house(depth=8.0, roof_h=4.0, overhang=0.0, window=False)
        vm, vm_b, plan = build(sm, colors=True)
        tgt = assign_blocks(vm_b)
        info = roofs.harmonize_roof_materials(tgt, plan)
        self.assertGreaterEqual(info["roof_groups"], 1)
        index = {tuple(int(v) for v in c): i for i, c in enumerate(tgt.occ)}
        for g in np.unique(plan.group[plan.group > 0]):
            bases = {bs.base_block(tgt.states[index[(int(x), int(plan.level[x, z]), int(z))]])
                     for x, z in np.argwhere(plan.group == g)}
            self.assertEqual(len(bases), 1, bases)
            self.assertIsNotNone(bs.BLOCK_VARIANTS[bases.pop()][1])

    def test_geometric_model_is_not_modified(self):
        sm = gable_house(depth=8.0, roof_h=4.0, overhang=0.0, window=False)
        vm = voxelize_scene(sm, 1.0, offset=(0.3, 0.0, 0.3), drop_ground=False, with_colors=False)
        vm = pad_model(vm, ((0, 0), (0, 2), (0, 0)))
        before = (vm.solid.copy(), vm.shapes.copy())
        roofs.builder_roofs(vm, sm.verts + np.array([0.3, 0.0, 0.3]) - vm.origin, sm.faces)
        self.assertTrue((vm.solid == before[0]).all() and (vm.shapes == before[1]).all())


if __name__ == "__main__":
    unittest.main()
