"""mcvox-v3: append-only IDs, VALID pairs against the 1.21.11 block table, encode / decode
round trips, the training-target encoding, rotation and mirror of every shape, and the
states derived on the assembled grid (stair corners, panes, doors, fence gates)."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcvoxel import vocab as V  # noqa: E402
from mcvoxel.blockstates import (finish_states, mirror_state, mirror_x, rotate_state,  # noqa: E402
                                 rotate_y)

FIXTURES = Path(__file__).parent / "fixtures"
S = V.SHAPE_INDEX
M = V.MATERIAL_INDEX


def predicted(state):
    """(block id, the properties v3 predicts); derived ones (corners, connections, hinge,
    upper half, in_wall, waterlogged, powered, leaf distance...) are left out."""
    c = V.classify_state(state)
    name, props = state.split("[", 1)[0], V._props(state)
    kind = c["kind"]
    out = {k: props.get(k) for k in {"slab": ("type",), "stairs": ("facing", "half"),
                                     "door": ("facing", "half")}.get(kind, ())}
    if kind == "full" and name in V.AXIS_MATERIALS:
        out["axis"] = props.get("axis", "y")
    if kind == "trapdoor":
        out["open"] = props.get("open", "false")
        if out["open"] == "true":
            out["facing"] = props.get("facing", "north")
        else:
            out["half"] = props.get("half", "bottom")
    if kind == "gate":
        out["axis"] = "x" if props.get("facing", "north") in ("north", "south") else "z"
    return name, out


class AppendOnly(unittest.TestCase):
    def test_v2_ids_unchanged(self):
        v2 = json.loads((FIXTURES / "vocab_mcvox_v2.json").read_text(encoding="utf-8"))
        self.assertEqual(V.SHAPES[:V.V2_SHAPE_COUNT], v2["shapes"])
        self.assertEqual(V.MATERIALS[:V.V2_MATERIAL_COUNT], v2["materials"])
        self.assertEqual((V.IGNORE_SHAPE, V.IGNORE_MATERIAL), (v2["ignore_shape"], v2["ignore_material"]))
        self.assertEqual(V.VOCAB_VERSION, "mcvox-v3")
        self.assertEqual(len(set(V.MATERIALS)), len(V.MATERIALS))

    def test_v2_pairs_still_decode(self):
        v2 = json.loads((FIXTURES / "vocab_mcvox_v2.json").read_text(encoding="utf-8"))
        changed = []
        for key, old in v2["decode_v2"].items():
            sh, mat = map(int, key.split(","))
            new = V.decode(sh, mat)
            if new != old:
                changed.append((V.SHAPES[sh], V.MATERIALS[mat]))
                self.assertTrue(V.VALID[sh, mat], (sh, mat, old, new))   # only newly valid pairs
        self.assertEqual(sorted({m for _, m in changed}),
                         ["minecraft:nether_bricks", "minecraft:tuff"])

    def test_v3_ids_unchanged_by_the_append(self):
        v3 = json.loads((FIXTURES / "vocab_mcvox_v3_0.json").read_text(encoding="utf-8"))
        self.assertEqual(V.SHAPES, v3["shapes"])
        self.assertEqual(V.MATERIALS[:len(v3["materials"])], v3["materials"])
        for shape, mats in v3["valid"].items():           # every 0.6.0 pair is still valid
            self.assertTrue(V.VALID[S[shape], mats].all(), shape)
        appended = ["gold_ore", "emerald_ore", "redstone_ore", "emerald_block",
                    "chiseled_polished_blackstone", "suspicious_gravel", "rooted_dirt", "red_sand",
                    "shroomlight", "honey_block", "bee_nest", "bamboo_mosaic"]
        self.assertEqual(V.MATERIALS[len(v3["materials"]):], ["minecraft:" + n for n in appended])
        self.assertEqual(int(V.VALID.sum()), v3["valid_pairs"] + 22)  # 12 full, mosaic slab + stairs

    def test_appended_materials_encode(self):
        for state, shape, material, status in (
                ("minecraft:gold_ore", "full", "gold_ore", "exact"),
                ("minecraft:suspicious_gravel[dusted=2]", "full", "suspicious_gravel", "exact"),
                ("minecraft:redstone_ore[lit=true]", "full", "redstone_ore", "lossy"),
                ("minecraft:bee_nest[facing=east,honey_level=3]", "full", "bee_nest", "lossy"),
                ("minecraft:bamboo_mosaic_slab[type=top,waterlogged=false]", "slab_top",
                 "bamboo_mosaic", "exact"),
                ("minecraft:bamboo_mosaic_stairs[facing=west,half=bottom,shape=straight]",
                 "stairs_west", "bamboo_mosaic", "exact")):
            c = V.classify_state(state)
            self.assertEqual((c["shape"], c["material"], c["status"]),
                             (S[shape], M["minecraft:" + material], status), state)

    def test_app_palettes_untouched(self):
        from mcvoxel.block_palette import AUX_PALETTE, BLOCK_PALETTE, BUILDER_PALETTE
        self.assertEqual((len(BLOCK_PALETTE), len(AUX_PALETTE), len(BUILDER_PALETTE)), (110, 9, 72))
        for m in V.V3_MATERIALS:
            self.assertNotIn(m, BLOCK_PALETTE)
            self.assertNotIn(m, AUX_PALETTE)


class Valid(unittest.TestCase):
    def test_every_valid_pair_decodes_to_a_1_21_11_state_and_round_trips(self):
        n = 0
        for sh in range(len(V.SHAPES)):
            for mat in np.flatnonzero(V.VALID[sh]):
                state = V.decode(sh, int(mat), strict=True)
                V.validate_state(state, V.TARGET_DATA_VERSION)
                self.assertEqual(V.encode_state(state), (sh, int(mat)), state)
                n += 1
        self.assertEqual(n, int(V.VALID.sum()))

    def test_invalid_pairs_are_rejected(self):
        for shape, material in (("stairs_east", "white_concrete"), ("door_north", "white_wool"),
                                ("pane", "oak_planks"), ("full_axis_x", "stone"),
                                ("carpet", "glass"), ("fence", "stone_bricks"),
                                ("trapdoor_bottom", "bricks"), ("stairs_top_north", "smooth_stone")):
            sh, mat = S[shape], M["minecraft:" + material]
            self.assertFalse(V.VALID[sh, mat], (shape, material))
            with self.assertRaises(ValueError):
                V.decode(sh, mat, strict=True)
            self.assertEqual(V.decode(S["air"], mat), V.AIR)
        self.assertTrue(V.VALID[S["air"], 0])
        self.assertFalse(V.VALID[:, 0][1:].any())

    def test_validation_against_the_target_table(self):
        V.validate_state("minecraft:pale_oak_door[facing=east,half=upper,hinge=right,open=true,powered=false]",
                         V.TARGET_DATA_VERSION)
        with self.assertRaises(ValueError):
            V.validate_state("minecraft:oak_door[facing=up]", V.TARGET_DATA_VERSION)
        with self.assertRaises(ValueError):
            V.validate_state("minecraft:chain", V.TARGET_DATA_VERSION)     # iron_chain since 1.21.9
        with self.assertRaises(ValueError):
            V.validate_state("minecraft:tuff_stairs[facing=east,half=bottom]")   # not in v2 / 1.20.1
        schema = json.loads(V.SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual((schema["minecraft_version"], schema["data_version"]),
                         (V.TARGET_VERSION, V.TARGET_DATA_VERSION))
        self.assertEqual(len(schema["source_sha256"]), 64)


class Encoding(unittest.TestCase):
    def check(self, state, shape, material, status, reason=None):
        c = V.classify_state(state)
        self.assertEqual((c["shape"], c["material"], c["status"]),
                         (S[shape] if shape else c["shape"],
                          M["minecraft:" + material] if material else c["material"], status), state)
        if reason:
            self.assertIn(reason, c["reasons"], state)

    def test_new_shapes(self):
        self.check("minecraft:oak_stairs[facing=west,half=top,shape=outer_left,waterlogged=false]",
                   "stairs_top_west", "oak_planks", "exact")
        self.check("minecraft:iron_trapdoor[facing=south,half=bottom,open=true,powered=false,waterlogged=false]",
                   "trapdoor_open_north", "iron_block", "exact")
        self.check("minecraft:iron_trapdoor[facing=south,half=top,open=true]",
                   "trapdoor_open_north", "iron_block", "lossy", "trapdoor_half")
        self.check("minecraft:spruce_trapdoor[facing=north,half=top,open=false]",
                   "trapdoor_top", "spruce_planks", "exact")
        self.check("minecraft:spruce_trapdoor[facing=east,half=bottom,open=false]",
                   "trapdoor_bottom", "spruce_planks", "lossy", "trapdoor_facing")
        self.check("minecraft:dark_oak_door[facing=north,half=lower,hinge=left,open=false,powered=false]",
                   "door_south", "dark_oak_planks", "exact")
        self.check("minecraft:dark_oak_door[facing=north,half=lower,hinge=right,open=true,powered=false]",
                   "door_east", "dark_oak_planks", "lossy", "door_open")
        self.check("minecraft:dark_oak_door[facing=north,half=lower,hinge=left,open=true,powered=false]",
                   "door_west", "dark_oak_planks", "lossy", "door_open")
        self.assertEqual(V.classify_state("minecraft:oak_door[facing=north,half=upper,hinge=left,open=false]")
                         ["status"], "derived")
        self.check("minecraft:stripped_oak_log[axis=x]", "full_axis_x", "stripped_oak_log", "exact")
        self.check("minecraft:quartz_pillar[axis=z]", "full_axis_z", "quartz_pillar", "exact")
        self.check("minecraft:basalt[axis=x]", "full_axis_x", "basalt", "exact")
        self.check("minecraft:oak_fence_gate[facing=west,in_wall=true,open=false,powered=false]",
                   "fence_gate_z", "oak_planks", "exact")
        self.check("minecraft:oak_fence_gate[facing=south,in_wall=false,open=true,powered=false]",
                   "fence_gate_x", "oak_planks", "lossy", "gate_open")
        self.check("minecraft:lime_carpet", "carpet", "lime_wool", "exact")
        self.check("minecraft:moss_carpet", "carpet", "moss_block", "exact")
        self.check("minecraft:iron_bars[east=true,north=false,south=false,waterlogged=false,west=true]",
                   "pane", "iron_block", "exact")
        self.check("minecraft:purple_stained_glass_pane[east=false,north=true,south=true,waterlogged=false,west=false]",
                   "pane", "purple_stained_glass", "exact")
        self.check("minecraft:nether_brick_fence[east=true,north=false,south=false,waterlogged=false,west=false]",
                   "fence", "nether_bricks", "exact")
        self.check("minecraft:cobbled_deepslate_stairs[facing=east,half=bottom]",
                   "stairs_east", "cobbled_deepslate", "exact")
        self.check("minecraft:oak_leaves[distance=3,persistent=false,waterlogged=false]",
                   "full", "oak_leaves", "exact")
        self.check("minecraft:smooth_quartz_slab[type=double,waterlogged=false]",
                   "full", "smooth_quartz", "lossy", "double_slab")
        self.check("minecraft:gray_glazed_terracotta[facing=west]", "full", "gray_glazed_terracotta",
                   "lossy", "facing")

    def test_aliases_and_ignored_blocks(self):
        self.check("minecraft:waxed_weathered_cut_copper_stairs[facing=north,half=bottom]",
                   "stairs_north", "weathered_cut_copper", "lossy", "alias")
        self.check("minecraft:infested_stone_bricks", "full", "stone_bricks", "lossy", "alias")
        self.check("minecraft:petrified_oak_slab[type=top]", "slab_top", "oak_planks", "lossy", "alias")
        for decor in ("minecraft:torch", "minecraft:oak_wall_sign[facing=north]", "minecraft:snow[layers=1]",
                      "minecraft:water[level=0]", "minecraft:chest[facing=north]", "minecraft:rail",
                      "minecraft:poplar_fence", "minecraft:gilded_blackstone"):
            self.assertEqual(V.classify_state(decor)["status"], "ignore", decor)
            self.assertEqual(V.encode_state(decor), (V.SHAPE_FULL, None))

    def test_exact_states_round_trip_their_predicted_properties(self):
        samples = [
            "minecraft:oak_stairs[facing=west,half=top,shape=outer_left,waterlogged=false]",
            "minecraft:polished_tuff_stairs[facing=south,half=bottom,shape=straight,waterlogged=true]",
            "minecraft:smooth_quartz_slab[type=top,waterlogged=false]",
            "minecraft:iron_trapdoor[facing=east,half=bottom,open=true,powered=false,waterlogged=false]",
            "minecraft:birch_trapdoor[facing=north,half=top,open=false,powered=false,waterlogged=false]",
            "minecraft:spruce_door[facing=west,half=lower,hinge=right,open=false,powered=true]",
            "minecraft:cherry_log[axis=z]", "minecraft:hay_block[axis=y]",
            "minecraft:black_stained_glass_pane[east=true,north=false,south=false,waterlogged=false,west=true]",
            "minecraft:warped_fence_gate[facing=south,in_wall=true,open=false,powered=false]",
            "minecraft:red_carpet", "minecraft:mossy_stone_brick_wall[east=low,north=none,south=none,up=true,waterlogged=false,west=tall]",
            "minecraft:mushroom_stem[down=false,east=true,north=false,south=true,up=false,west=true]",
            "minecraft:redstone_lamp[lit=false]", "minecraft:grass_block[snowy=true]",
            "minecraft:weathered_copper_door[facing=south,half=lower,hinge=left,open=false,powered=false]",
            "minecraft:copper_bars[east=false,north=true,south=true,waterlogged=false,west=false]",
        ]
        for state in samples:
            c = V.classify_state(state)
            self.assertEqual(c["status"], "exact", state)
            back = V.decode(c["shape"], c["material"], strict=True)
            self.assertEqual(predicted(back), predicted(state), state)


class TargetEncoding(unittest.TestCase):
    def test_policy(self):
        palette = [V.AIR, "minecraft:stone_bricks", "minecraft:torch",
                   "minecraft:oak_door[facing=north,half=lower,hinge=left,open=false,powered=false]",
                   "minecraft:oak_door[facing=north,half=upper,hinge=left,open=false,powered=false]",
                   "minecraft:snow[layers=1]", "minecraft:oak_stairs[facing=east,half=top]"]
        blocks = np.array([[[0, 1, 2, 3, 4, 5, 6, 1]]])
        shell = np.array([[[0, 1, 1, 1, 1, 1, 1, 0]]], bool)
        exterior = np.array([[[1, 0, 0, 0, 0, 0, 0, 0]]], bool)
        interior = ~(shell | exterior)
        shape, material = V.encode_target(blocks, palette, shell, exterior, interior)
        self.assertEqual(shape.dtype, np.uint8)
        self.assertEqual(material.dtype, np.uint16)
        self.assertEqual(shape[0, 0].tolist(),
                         [0, 1, 255, S["door_south"], 255, 255, S["stairs_top_east"], 255])
        self.assertEqual(material[0, 0, 1], M["minecraft:stone_bricks"])
        self.assertEqual(material[0, 0, 0], 0)
        self.assertTrue(((shape == 255) == (material == V.IGNORE_MATERIAL)).all())
        self.assertFalse(((shape == 0) & (blocks != 0)).any())      # nothing became air


class Symmetry(unittest.TestCase):
    def test_four_rotations_and_two_mirrors_are_identity_for_every_shape(self):
        rng = np.random.default_rng(3)
        g = rng.integers(0, len(V.SHAPES), size=(5, 3, 7)).astype(np.uint8)
        g[0, 0, 0] = V.IGNORE_SHAPE
        out = g
        for _ in range(4):
            (out,) = rotate_y(out, 1)
        np.testing.assert_array_equal(out, g)
        np.testing.assert_array_equal(mirror_x(mirror_x(g)[0])[0], g)
        np.testing.assert_array_equal(rotate_y(g, 2)[0], rotate_y(rotate_y(g, 1)[0], 1)[0])

    def test_spot_checks(self):
        def rot(label, k=1):
            return V.SHAPES[int(rotate_y(np.array([[[S[label]]]], np.uint8), k)[0][0, 0, 0])]

        def mir(label):
            return V.SHAPES[int(mirror_x(np.array([[[S[label]]]], np.uint8))[0][0, 0, 0])]
        self.assertEqual(rot("stairs_top_east"), "stairs_top_south")
        self.assertEqual(rot("stairs_east"), "stairs_south")
        self.assertEqual(rot("trapdoor_open_north"), "trapdoor_open_east")
        self.assertEqual(rot("door_west", 2), "door_east")
        self.assertEqual(rot("full_axis_x"), "full_axis_z")
        self.assertEqual(rot("full_axis_x", 2), "full_axis_x")
        self.assertEqual(rot("fence_gate_z"), "fence_gate_x")
        self.assertEqual(mir("stairs_top_east"), "stairs_top_west")
        self.assertEqual(mir("trapdoor_open_west"), "trapdoor_open_east")
        self.assertEqual(mir("door_north"), "door_north")
        self.assertEqual(mir("full_axis_x"), "full_axis_x")
        self.assertEqual(rot("trapdoor_top"), "trapdoor_top")

    def test_state_transforms_match_the_grid_transforms(self):
        for sh in range(len(V.SHAPES)):
            for mat in np.flatnonzero(V.VALID[sh])[:3]:
                state = V.decode(sh, int(mat), strict=True)
                for k in (1, 2, 3):
                    want = rotate_y(np.array([[[sh]]], np.uint8), k)[0][0, 0, 0]
                    self.assertEqual(V.encode_state(rotate_state(state, k))[0], want, (state, k))
                self.assertEqual(V.encode_state(mirror_state(state))[0],
                                 mirror_x(np.array([[[sh]]], np.uint8))[0][0, 0, 0], state)
                self.assertEqual(rotate_state(rotate_state(state, 3), 1), state)
                self.assertEqual(mirror_state(mirror_state(state)), state)

    def test_mirror_flips_door_hinge_and_stair_corner(self):
        door = "minecraft:oak_door[facing=east,half=lower,hinge=left,open=false,powered=false]"
        self.assertEqual(mirror_state(door),
                         "minecraft:oak_door[facing=west,half=lower,hinge=right,open=false,powered=false]")
        self.assertIn("hinge=right", mirror_state(door.replace("facing=east", "facing=north")))
        self.assertIn("shape=inner_right", mirror_state("minecraft:oak_stairs[facing=north,half=top,shape=inner_left]"))
        self.assertEqual(rotate_state("minecraft:oak_log[axis=x]"), "minecraft:oak_log[axis=z]")
        self.assertEqual(rotate_state("minecraft:glass_pane[east=true,north=false,south=false,waterlogged=false,west=false]"),
                         "minecraft:glass_pane[east=false,north=false,south=true,waterlogged=false,west=false]")


class DerivedStates(unittest.TestCase):
    def test_stair_corners_on_both_halves(self):
        for half in ("bottom", "top"):
            names = [V.AIR, f"minecraft:oak_stairs[facing=north,half={half}]",
                     f"minecraft:oak_stairs[facing=west,half={half}]"]
            grid = np.zeros((1, 3, 3), np.uint16)
            grid[0, 1, 1], grid[0, 0, 1] = 1, 2                      # in front faces west
            out, ns = finish_states(grid, names)
            self.assertIn("shape=outer_", ns[out[0, 1, 1]], half)
            grid[0, 0, 1], grid[0, 2, 1] = 0, 2                      # behind faces west
            out, ns = finish_states(grid, names)
            self.assertIn("shape=inner_", ns[out[0, 1, 1]], half)
        mixed = [V.AIR, "minecraft:oak_stairs[facing=north,half=top]",
                 "minecraft:oak_stairs[facing=west,half=bottom]"]
        grid = np.zeros((1, 3, 3), np.uint16)
        grid[0, 1, 1], grid[0, 0, 1] = 1, 2
        out, ns = finish_states(grid, mixed)
        self.assertIn("shape=straight", ns[out[0, 1, 1]])            # halves never join

    def test_pane_connections(self):
        pane = V.decode(S["pane"], M["minecraft:glass"])
        names = [V.AIR, pane, "minecraft:stone", "minecraft:glass",
                 V.decode(S["fence"], M["minecraft:oak_planks"])]
        grid = np.zeros((1, 1, 6), np.uint16)                        # (y, z, x)
        grid[0, 0] = [2, 1, 1, 1, 3, 4]
        out, ns = finish_states(grid, names)
        states = [ns[i] for i in out[0, 0]]
        self.assertIn("east=true", states[1])
        self.assertIn("west=true", states[1])                       # stone face
        self.assertIn("east=true", states[3])                       # glass block: panes join it
        self.assertIn("north=false", states[2])
        self.assertIn("west=false", states[5])                      # a fence does not join glass
        for s in states:
            V.validate_state(s, V.TARGET_DATA_VERSION)

    def test_door_upper_half_and_hinge(self):
        door = V.decode(S["door_south"], M["minecraft:oak_planks"])   # facing north
        names = [V.AIR, door, "minecraft:stone"]
        for wall_side, hinge in ((+1, "right"), (-1, "left"), (0, "left")):
            grid = np.zeros((4, 3, 3), np.uint16)                    # (y, z, x)
            grid[1, 1, 1] = 1
            if wall_side:
                grid[1:3, 1, 1 + wall_side] = 2                     # east of a north door = right
            out, ns = finish_states(grid, names)
            lower, upper = ns[out[1, 1, 1]], ns[out[2, 1, 1]]
            self.assertIn("half=lower", lower)
            self.assertIn("hinge=" + hinge, lower, wall_side)
            self.assertIn("half=upper", upper)
            self.assertIn("hinge=" + hinge, upper)
            self.assertIn("facing=north", upper)
            for s in (lower, upper):
                V.validate_state(s, V.TARGET_DATA_VERSION)
        grid = np.zeros((4, 3, 4), np.uint16)                        # a double door
        grid[1, 1, 1] = grid[1, 1, 2] = 1
        out, ns = finish_states(grid, names)
        self.assertEqual({_hinge(ns[out[1, 1, 1]]), _hinge(ns[out[1, 1, 2]])}, {"left", "right"})

    def test_fence_gate_in_wall_and_fence_joins(self):
        gate = V.decode(S["fence_gate_x"], M["minecraft:oak_planks"])
        wall = V.decode(S["wall"], M["minecraft:cobblestone"])
        fence = V.decode(S["fence"], M["minecraft:oak_planks"])
        names = [V.AIR, gate, wall, fence]
        grid = np.zeros((1, 1, 3), np.uint16)
        grid[0, 0] = [2, 1, 2]
        out, ns = finish_states(grid, names)
        self.assertIn("in_wall=true", ns[out[0, 0, 1]])
        self.assertIn("east=low", ns[out[0, 0, 0]])                  # walls join the gate
        grid[0, 0] = [3, 1, 3]
        out, ns = finish_states(grid, names)
        self.assertIn("in_wall=false", ns[out[0, 0, 1]])
        self.assertIn("east=true", ns[out[0, 0, 0]])                 # fences join it along X
        grid = np.zeros((1, 3, 1), np.uint16)
        grid[0, :, 0] = [3, 1, 3]                                     # across the gate: no join
        out, ns = finish_states(grid, names)
        self.assertIn("south=false", ns[out[0, 0, 0]])


def _hinge(state):
    return V._props(state)["hinge"]


if __name__ == "__main__":
    unittest.main()
