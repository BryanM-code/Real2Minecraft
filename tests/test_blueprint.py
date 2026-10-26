"""blueprint: the format's checks (spec) and the builder (build): where walls, roofs and
openings land, that parts do not cut into each other, and that every block is one the
vocabulary (and so the schematic writer) can write."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blueprint import BlueprintError, build, load
from mcvoxel import vocab as V
from photofirst.schem import palette_grid, write_schem

EXAMPLE = Path(__file__).resolve().parents[1] / "blueprint" / "examples" / "cottage.json"


def house(**roof):
    """A 9 x 7, one-storey (5-row) box with a roof; walls x 0-8, z 0-6, rows 0-4."""
    return {"masses": [{"id": "main", "x": 0, "z": 0, "w": 9, "d": 7, "y": 0, "storeys": 1, "storey_h": 5,
                        "corner_trim": False, "roof": {"type": "gable", "ridge": "x", "pitch": 1.0,
                                                       "overhang": 1, **roof}}]}


def cell(b, x, y, z):
    """(shape name, material name) at blueprint coordinates."""
    ox, oy, oz = b["origin"]
    s, m = b["shape"][x - ox, y - oy, z - oz], b["material"][x - ox, y - oy, z - oz]
    return V.SHAPES[s], V.MATERIALS[m].removeprefix("minecraft:")


def top(b):
    return b["origin"][1] + np.nonzero(b["shape"].any(axis=(0, 2)))[0].max()


class Spec(unittest.TestCase):
    def test_defaults_are_filled(self):
        bp = load(house())
        m = bp["masses"][0]
        self.assertEqual(m["floors"], [0])
        self.assertEqual(m["plinth"], {"south": 0, "north": 0, "east": 0, "west": 0})
        self.assertEqual(bp["site"]["ground"], [])

    def test_every_problem_is_reported(self):
        bp = house(type="onion")
        bp["materials"] = {"roof": "glass", "colour": "red"}
        bp["openings"] = [{"mass": "main", "side": "south", "kind": "window", "u": 7, "y": 1, "w": 4, "h": 2},
                          {"mass": "nope", "side": "south", "kind": "door", "u": 0, "y": 1}]
        with self.assertRaises(BlueprintError) as e:
            load(bp)
        msg = str(e.exception)
        for part in ("roof.type", "'glass' has no", "materials.colour", "off the 9-block wall", "no mass 'nope'"):
            self.assertIn(part, msg)


class Walls(unittest.TestCase):
    def test_ring_floor_and_hollow(self):
        b = build(house())
        self.assertEqual(cell(b, 0, 2, 0), ("full", "birch_planks"))
        self.assertEqual(cell(b, 4, 0, 3), ("full", "spruce_planks"))       # the floor plate
        self.assertEqual(cell(b, 4, 2, 3)[0], "air")
        self.assertEqual(b["warnings"], [])

    def test_plinth_per_side(self):
        bp = house()
        bp["masses"][0]["plinth"] = {"south": 2}
        b = build(bp)
        self.assertEqual(cell(b, 4, 1, 6)[1], "stone_bricks")
        self.assertEqual(cell(b, 4, 2, 6)[1], "birch_planks")
        self.assertEqual(cell(b, 4, 1, 0)[1], "birch_planks")


class Roofs(unittest.TestCase):
    def test_gable_pitch_one(self):
        # roof z -1..7 (9 rows): stairs climb from both eaves, a block and a slab cap on the ridge
        b = build(house())
        self.assertEqual(cell(b, 4, 4, 7), ("stairs_north", "deepslate_tiles"))   # the eave overhang
        self.assertEqual(cell(b, 4, 5, 6), ("stairs_north", "deepslate_tiles"))   # over the wall
        self.assertEqual(cell(b, 4, 5, 0), ("stairs_south", "deepslate_tiles"))
        self.assertEqual(cell(b, 4, 7, 3), ("full", "deepslate_tiles"))
        self.assertEqual(cell(b, 4, 8, 3), ("slab_bottom", "deepslate_tiles"))
        self.assertEqual(top(b), 8)
        self.assertEqual(cell(b, 0, 6, 3), ("full", "birch_planks"))              # the gable end

    def test_half_pitch_slabs_and_no_slit_over_walls(self):
        b = build(house(pitch=0.5))
        self.assertEqual(cell(b, 4, 4, 7), ("slab_bottom", "deepslate_tiles"))
        self.assertEqual(cell(b, 4, 4, 6), ("full", "deepslate_tiles"))           # top slab over the wall
        self.assertEqual(cell(b, 4, 5, 5), ("slab_bottom", "deepslate_tiles"))
        self.assertEqual(top(b), 6)

    def test_hip_and_flat(self):
        b = build(house(type="hip"))
        self.assertEqual(cell(b, -1, 4, 3), ("stairs_east", "deepslate_tiles"))
        b = build(house(type="flat"))
        self.assertEqual(cell(b, 0, 5, 3), ("full", "birch_planks"))              # parapet
        self.assertEqual(cell(b, 4, 5, 3), ("slab_bottom", "deepslate_tiles"))    # deck

    def test_a_lower_roof_never_cuts_a_taller_wall(self):
        bp = house()
        bp["masses"].append({"id": "low", "x": 9, "z": 0, "w": 6, "d": 7, "y": 0, "walls": 3,
                             "roof": {"type": "gable", "ridge": "x", "pitch": 1.0, "overhang": 1}})
        b = build(bp)
        for y in range(0, 5):
            self.assertEqual(cell(b, 8, y, 3), ("full", "birch_planks"))


class Openings(unittest.TestCase):
    def test_u_counts_from_the_left_seen_from_outside(self):
        bp = house()
        bp["openings"] = [{"mass": "main", "side": s, "kind": "window", "u": 1, "y": 2, "w": 1, "h": 1,
                           "frame": False, "sill": False} for s in ("south", "north", "east", "west")]
        b = build(bp)
        for x, z in ((1, 6), (7, 0), (8, 5), (0, 1)):         # south: x0+1; north: x1-1; east: z1-1; west: z0+1
            self.assertEqual(cell(b, x, 2, z), ("pane", "glass"), (x, z))

    def test_window_frame_mullion_sill_shutters(self):
        bp = house()
        bp["openings"] = [{"mass": "main", "side": "south", "kind": "window", "u": 2, "y": 1, "w": 3, "h": 2,
                           "mullions": [1], "shutters": True}]
        b = build(bp)
        self.assertEqual(cell(b, 2, 1, 6), ("pane", "glass"))
        self.assertEqual(cell(b, 3, 1, 6), ("full", "spruce_planks"))            # mullion
        self.assertEqual(cell(b, 1, 2, 6), ("full", "spruce_planks"))            # frame
        self.assertEqual(cell(b, 3, 0, 7), ("stairs_top_north", "spruce_planks"))  # sill, outside
        self.assertEqual(cell(b, 1, 1, 7), ("trapdoor_open_north", "spruce_planks"))

    def test_door_writes_both_halves_and_garage_is_recessed(self):
        bp = house()
        bp["openings"] = [{"mass": "main", "side": "south", "kind": "door", "u": 4, "y": 1},
                          {"mass": "main", "side": "east", "kind": "garage", "u": 1, "y": 0, "w": 3, "h": 3}]
        b = build(bp)
        grid, names = palette_grid(b["shape"], b["material"])
        doors = sorted(n for n in (names[i] for i in np.unique(grid)) if "door" in n)
        self.assertEqual([("half=lower" in n, "half=upper" in n) for n in doors], [(True, False), (False, True)])
        self.assertEqual(cell(b, 8, 1, 4)[0], "air")
        self.assertEqual(cell(b, 7, 1, 4), ("full", "white_concrete"))


class Features(unittest.TestCase):
    def test_chimney_and_skylight(self):
        bp = house(pitch=0.5)
        bp["features"] = [{"kind": "chimney", "x": 1, "z": 2, "w": 2, "d": 2, "top": 9},
                          {"kind": "skylight", "mass": "main", "side": "south", "u": 3, "v": 2, "w": 2, "h": 1}]
        b = build(bp)
        self.assertEqual(cell(b, 1, 8, 2), ("full", "bricks"))
        self.assertEqual(cell(b, 1, 9, 2), ("slab_bottom", "stone_bricks"))
        self.assertEqual(cell(b, 2, 5, 5), ("full", "glass"))                     # roof x -1.. (u 3 = x 2), k 2 = z 5
        self.assertEqual(cell(b, 1, 5, 5)[1], "spruce_planks")                    # its frame


class Site(unittest.TestCase):
    def test_ground_digs_and_masses_clear_their_insides(self):
        bp = house()
        bp["masses"][0]["y"] = -2
        bp["site"] = {"ground": [{"x0": -3, "z0": -3, "x1": 12, "z1": 10, "top": -1},
                                 {"x0": 10, "z0": -3, "x1": 12, "z1": 10, "top": -3}],
                      "bushes": [{"x": 4, "z": 9, "r": 1}]}
        b = build(bp)
        self.assertEqual(cell(b, 4, -1, 3)[0], "air")                             # dug out inside
        self.assertEqual(cell(b, 11, -1, 3)[0], "air")                            # the lower pad dug the higher
        self.assertEqual(cell(b, 11, -3, 3), ("full", "grass_block"))
        self.assertEqual(cell(b, 4, 0, 9), ("full", "azalea_leaves"))


class Describe(unittest.TestCase):
    """The helpers that read a vision model's answer (no model needed)."""

    def test_json_is_found_inside_fences_and_prose(self):
        from blueprint.describe import parse_json
        self.assertEqual(parse_json('Sure:\n```json\n{"storeys": 2, "dome": false}\n```'), {"storeys": 2, "dome": False})
        self.assertIsNone(parse_json("no json here"))
        self.assertIsNone(parse_json('{"storeys": 2,'))

    def test_check_names_the_problem_or_none(self):
        from blueprint.describe import check
        self.assertIsNone(check(load(EXAMPLE)))
        self.assertIn("off the 5-block wall", check({"masses": [{"x": 0, "z": 0, "w": 5, "d": 5}], "openings": [
            {"mass": "m0", "side": "south", "kind": "window", "u": 4, "y": 1, "w": 3, "h": 2}]}))
        self.assertEqual(check([1, 2]), "the answer is not a JSON object")


class Assemble(unittest.TestCase):
    """A filled-in form -> a valid blueprint, whatever the model left out."""
    HOUSE = {"wings": [
        {"position": "left", "storeys": 2, "windows_across": 2, "roof": "gable", "ridge": "parallel",
         "front": {"ground_windows": 1, "door": True}, "chimney": True},
        {"position": "right", "storeys": 1, "level": "half_below", "front_offset": "forward", "windows_across": 1,
         "front": {"wide_window": True}, "side_garage_doors": 2, "skylights": 2}],
        "wall": {"material": "wood_siding", "colour": "cream"}, "base": {"material": "brick"},
        "trim_colour": "dark brown", "roof_colour": "grey"}

    def test_house_form(self):
        from blueprint.assemble import assemble
        bp = assemble(self.HOUSE)
        load(bp)
        left, right = bp["masses"]
        self.assertEqual(right["x"], left["x"] + left["w"])                       # wings side by side
        self.assertEqual(right["z"] + right["d"], left["z"] + left["d"] + 4)      # the right wing stands forward
        self.assertEqual(right["y"], -4)                                          # split level
        kinds = sorted((o["mass"], o["side"], o["kind"]) for o in bp["openings"])
        self.assertIn(("w1", "east", "garage"), kinds)
        self.assertEqual(sum(k == ("w0", "south", "window") for k in kinds), 3)  # 2 upstairs + 1 beside the door
        self.assertEqual(bp["materials"]["wall"], "smooth_sandstone")
        self.assertEqual([f["kind"] for f in bp["features"]], ["chimney", "skylight", "skylight"])
        self.assertEqual(build(bp)["warnings"], [])

    def test_sparse_and_odd_forms_still_build(self):
        from blueprint.assemble import assemble
        for form in ({}, {"wings": [{"storeys": "three", "roof": "dome"}]}, {"wings": [{"storeys": 40}]}):
            build(assemble(form))


class Parts(unittest.TestCase):
    """blueprint-1's shapes, roofs, features and decor."""

    def test_round_tower_with_cone_and_finial(self):
        b = build({"masses": [{"id": "t", "x": 0, "z": 0, "w": 7, "d": 7, "shape": "round", "walls": 6, "floors": [0],
                               "roof": {"type": "cone"}}]})
        self.assertEqual(cell(b, 0, 2, 0)[0], "air")                    # a round shape leaves the corners
        self.assertEqual(cell(b, 3, 2, 0)[0], "full")                   # the middle of its edge is wall
        self.assertEqual(cell(b, 3, top(b), 3)[0], "fence")             # the finial
        self.assertEqual(b["warnings"], [])

    def test_dome_shell_rises_and_is_hollow(self):
        b = build({"masses": [{"id": "m", "x": 0, "z": 0, "w": 13, "d": 13, "walls": 5, "floors": [0],
                               "roof": {"type": "flat", "overhang": 0}}],
                   "features": [{"kind": "dome", "mass": "m"}]})
        self.assertGreaterEqual(top(b), 5 + 1 + 4)
        self.assertEqual(cell(b, 6, 8, 6)[0], "air")                    # inside the shell

    def test_portico_columns_and_pediment(self):
        bp = house()
        bp["masses"][0]["walls"] = 6
        bp["features"] = [{"kind": "portico", "mass": "main", "side": "south", "u": 2, "w": 5, "depth": 2, "columns": 3}]
        b = build(bp)
        self.assertEqual(cell(b, 2, 2, 8), ("full", "smooth_quartz"))   # a column at the outer edge
        self.assertEqual(cell(b, 4, 2, 7)[0], "air")                    # the porch under the roof is open
        self.assertEqual(cell(b, 4, 7, 8), ("full", "spruce_planks"))   # the pediment's face
        self.assertEqual(cell(b, 4, 7, 9)[0], "slab_top")               # its slope, one block out

    def test_battlements_alternate(self):
        b = build({"masses": [{"id": "c", "x": 0, "z": 0, "w": 6, "d": 6, "walls": 4, "floors": [0],
                               "roof": {"type": "flat", "battlements": True, "overhang": 0}}]})
        row = [cell(b, x, 5, 0)[0] for x in range(6)]
        self.assertEqual(row, ["full", "air"] * 3)

    def test_curtain_wall_is_glass_in_a_grid(self):
        b = build({"materials": {"window": "light_blue_stained_glass"},
                   "masses": [{"id": "o", "x": 0, "z": 0, "w": 9, "d": 7, "storeys": 2, "storey_h": 4, "decor": ["curtain"],
                               "roof": {"type": "flat", "overhang": 0}}]})
        self.assertEqual(cell(b, 2, 2, 6)[1], "light_blue_stained_glass")
        self.assertEqual(cell(b, 4, 2, 6)[1], "spruce_planks")          # a mullion every 4

    def test_roof_types_and_parts_all_build(self):
        for t in ("pyramid", "cross_gable", "mansard", "gambrel", "sawtooth", "spire", "cone", "dome", "pagoda"):
            b = build({"masses": [{"id": "m", "x": 0, "z": 0, "w": 12, "d": 10, "walls": 5, "floors": [0],
                                   "roof": {"type": t}}]})
            self.assertGreater(b["blocks"], 0, t)
        bp = house()
        bp["features"] = [{"kind": k, "mass": "main", "side": "south", "u": 1, "w": 4} for k in ("porch", "balcony", "bay", "awning")]
        bp["features"] += [{"kind": "dormer", "mass": "main", "side": "south", "u": 3},
                           {"kind": "stack", "x": 12, "z": 3, "top": 14}, {"kind": "sails", "mass": "main", "side": "south"},
                           {"kind": "gallery", "mass": "main", "y": 4}, {"kind": "lantern", "mass": "main"},
                           {"kind": "canopy", "x0": 0, "z0": 9, "x1": 8, "z1": 14, "top": 4}]
        bp["openings"] = [{"mass": "main", "side": "east", "kind": "arch", "u": 2, "y": 1, "w": 3, "h": 3},
                          {"mass": "main", "side": "west", "kind": "window", "u": 2, "y": 1, "w": 3, "h": 3, "style": "lancet"}]
        self.assertGreater(build(bp)["blocks"], 0)


class Compose(unittest.TestCase):
    """The style-aware form + style + kind (+ a measurement) -> a valid blueprint-1."""

    def check(self, form, style, kind, measure=None):
        from blueprint.compose import compose
        bp = compose(form, style, kind, measure)
        load(bp)
        b = build(bp)
        self.assertEqual(b["warnings"], [], (style, kind))
        return bp, b

    def test_house_church_capitol_skyscraper_lighthouse(self):
        _, b = self.check({"parts": [{"part": "main", "share": 60, "storeys": 2, "roof": "gable", "windows_across": 2,
                                      "front": {"door": True, "ground_windows": 1}},
                                     {"part": "wing", "share": 40, "storeys": 1, "roof": "hip", "front_offset": "forward"}],
                           "porch": "front", "garage": {"doors": 2, "wall": "right"}, "chimneys": 1, "dormers": 2,
                           "wall": {"material": "wood_siding", "colour": "cream"}}, "suburban", "house",
                          {"width_m": 16, "height_m": 9, "depth_m": 10})
        bp, _ = self.check({"parts": [{"part": "main", "storeys": 1, "roof": "gable", "ridge": "perpendicular",
                                       "window_style": "lancet", "pitch": "steep"}],
                            "spire": True, "buttresses": True}, "gothic", "church")
        self.assertTrue(any(m["roof"]["type"] == "spire" for m in bp["masses"]))
        bp, _ = self.check({"parts": [{"part": "main", "share": 100, "storeys": 3, "roof": "dome"}],
                            "portico": {"columns": 6, "pediment": True}, "dome": {"size": "large", "drum": True},
                            "symmetric": True}, "neoclassical", "capitol")
        self.assertEqual(sorted(f["kind"] for f in bp["features"] if f["kind"] in ("dome", "portico")), ["dome", "portico"])
        bp, b = self.check({"parts": [{"part": "main", "storeys": 60, "roof": "flat", "window_style": "curtain"}],
                            "tiers": [40, 12, 8], "crown": "spire"}, "art_deco", "skyscraper", {"height_m": 240})
        self.assertGreaterEqual(len(bp["masses"]), 3)
        self.assertLessEqual(max(b["dims"]), 384)
        bp, _ = self.check({"parts": [{"part": "tower", "shape": "round", "storeys": 6}]}, "victorian", "lighthouse")
        self.assertEqual({f["kind"] for f in bp["features"]} >= {"gallery", "lantern"}, True)

    def test_aliases_and_junk(self):
        from blueprint.compose import kind_of, style_of
        self.assertEqual((kind_of("bank"), kind_of("Cathedral"), kind_of("zeppelin")), ("office", "church", "other"))
        self.assertEqual((style_of("Georgian"), style_of("queen-anne"), style_of("??")), ("colonial", "victorian", "other"))
        self.check({}, None, None)
        self.check({"parts": "lots", "towers": [1], "dome": "yes"}, "baroque", "palace")

    def test_style_palettes_are_real_blocks(self):
        from blueprint.spec import material_index
        from blueprint.styles import STYLES
        for style, st in STYLES.items():
            for role, block in st["palette"].items():
                self.assertIsNotNone(material_index(block), (style, role, block))


class NewParts(unittest.TestCase):
    """Pagoda tiers, crow-stepped gables, rows of houses, podiums and crowns, hidden roofs,
    roofs decided by their ends, and mirroring."""

    def masses(self, bp):
        return {m["id"]: m for m in bp["masses"]}

    def test_pagoda_tiers_show_their_walls(self):
        from blueprint.compose import compose
        bp = compose({"parts": [{"part": "main", "storeys": 4, "roof": "pagoda"}], "flared_eaves": True},
                     "chinese", "tower", {"width_m": 20, "height_m": 40, "depth_m": 20})
        ms = self.masses(bp)
        tiers = [ms["p0"]] + [ms[f"tier{k}"] for k in range(1, 4)]
        self.assertIn("platform", ms)
        self.assertTrue(all(t["roof"]["skirt"] > 0 for t in tiers[:-1]) and not tiers[-1]["roof"].get("skirt"))
        b = build(bp)
        self.assertEqual(b["warnings"], [])
        for t in tiers[1:]:                                  # each tier's wall, mid-height, is wall, not roof
            x, y, z = t["x"] + t["w"] // 2, t["y"] + t["walls"] // 2, t["z"] + t["d"] - 1
            self.assertNotEqual(cell(b, x, y, z)[1], bp["materials"]["roof"], t["id"])
        self.assertTrue(any(f["kind"] == "finial" for f in bp["features"]))
        last = tiers[-1]
        cx, cz = (2 * last["x"] + last["w"] - 1) // 2, (2 * last["z"] + last["d"] - 1) // 2
        self.assertEqual(cell(b, cx, top(b), cz)[0], "fence")

    def test_a_skirt_leaves_the_middle_open(self):
        bp = house(type="pagoda", pitch=0.5, overhang=2, skirt=3)
        b = build(bp)
        self.assertEqual(top(b), 6)
        self.assertEqual(cell(b, 4, 5, 3), ("air", "air"))   # no roof over the middle
        # a curled corner: a pagoda's eave stands out one more than its overhang
        self.assertEqual([cell(b, -3, y, -3)[0] for y in (4, 5, 6)], ["full", "full", "slab_bottom"])

    def test_stepped_gables_rise_above_the_roof(self):
        bp = {"masses": [{"id": "m", "x": 0, "z": 0, "w": 9, "d": 12, "y": 0, "walls": 8, "storey_h": 4,
                          "roof": {"type": "gable", "ridge": "z", "pitch": 1.0, "overhang": 0, "gable_end": "stepped"}}]}
        b = build(bp)
        self.assertEqual(b["warnings"], [])
        heights = []
        for x in range(9):                                   # the front wall's top in each column
            ys = [y for y in range(0, top(b) + 1) if cell(b, x, y, 11)[0] == "full"]
            heights.append(max(ys))
        self.assertEqual(heights, heights[::-1])            # symmetric
        self.assertEqual(heights[0], heights[1])            # steps two blocks wide
        self.assertGreater(heights[4], heights[0])
        self.assertEqual(cell(b, 4, heights[4] + 1, 11)[0], "slab_bottom")   # each step capped
        with self.assertRaises(BlueprintError):
            load(dict(bp, masses=[dict(bp["masses"][0], roof={"type": "gable", "gable_end": "curly"})]))

    def test_a_row_of_houses(self):
        from blueprint.compose import compose
        bp = compose({"parts": [{"part": "main", "storeys": s} for s in (4, 5, 4, 3, 5)], "units": 5,
                      "gables": "stepped", "wall": {"material": "brick"}}, "dutch", "townhouse")
        houses = sorted((m for m in bp["masses"] if m["id"][0] == "p"), key=lambda m: m["x"])
        self.assertEqual(len(houses), 5)
        for a, c in zip(houses, houses[1:]):
            self.assertEqual(c["x"], a["x"] + a["w"] - 1)     # a shared party wall
        self.assertTrue(all(m["roof"]["gable_end"] == "stepped" and m["roof"]["ridge"] == "z" for m in houses))
        self.assertGreater(len({(m.get("materials") or {}).get("wall") for m in houses}), 1)
        doors = [o for o in bp["openings"] if o["kind"] == "door"]
        self.assertEqual(sorted(o["mass"] for o in doors), sorted(m["id"] for m in houses))
        self.assertEqual(build(bp)["warnings"], [])
        terrace = compose({"parts": [{"part": "main", "storeys": 2, "roof": "gable"}], "units": 4, "gables": "side",
                           "chimneys": 1}, "victorian", "townhouse")
        self.assertEqual(sum(1 for f in terrace["features"] if f["kind"] == "chimney"), 3)   # one per party wall

    def test_podium_and_crowns(self):
        from blueprint.compose import compose
        form = {"parts": [{"part": "main", "storeys": 30, "roof": "not_visible"}], "podium": {"storeys": 3}}
        for crown, expect in (("spire", "spire"), ("flat", "crown"), ("pinnacles", "pinnacle3")):
            bp = compose(dict(form, crown=crown), "art_deco", "skyscraper")
            ms = self.masses(bp)
            self.assertIn(expect, ms, crown)
            self.assertGreater(ms["podium"]["w"], ms["p0"]["w"])
            self.assertEqual(ms["p0"]["y"], ms["podium"]["y"] + ms["podium"]["walls"])
            self.assertEqual(build(bp)["warnings"], [])
        doors = [o for o in bp["openings"] if o["kind"] == "door"]
        self.assertEqual([o["mass"] for o in doors], ["podium"])

    def test_hidden_roofs_and_roof_ends(self):
        from blueprint.compose import compose

        def roof(part, style="suburban", kind="house"):
            return compose({"parts": [dict(part, part="main")]}, style, kind)["masses"][0]["roof"]["type"]
        self.assertEqual(roof({"roof": "not_visible"}, "international", "office"), "flat")
        self.assertEqual(roof({"roof": "not_visible"}, "victorian", "house"), "cross_gable")
        self.assertEqual(roof({"roof": "gable", "roof_ends": "slope"}), "hip")
        self.assertEqual(roof({"roof": "hip", "roof_ends": "triangle"}), "gable")

    def test_mirror_flips_the_build(self):
        from blueprint.compose import mirror
        bp = {"masses": [{"id": "a", "x": 0, "z": 0, "w": 9, "d": 7, "storeys": 2, "storey_h": 5,
                          "roof": {"type": "gable", "ridge": "x"}},
                         {"id": "b", "x": 9, "z": 2, "w": 6, "d": 5, "storeys": 1, "storey_h": 5,
                          "roof": {"type": "shed", "high": "west", "pitch": 0.5}}],
              "openings": [{"mass": "a", "side": "south", "kind": "window", "u": 1, "y": 6, "w": 3, "h": 2, "mullions": [0]},
                           {"mass": "b", "side": "east", "kind": "door", "u": 1, "y": 1}],
              "features": [{"kind": "chimney", "x": 1, "z": 3, "top": 14}]}
        b1, b2 = build(bp), build(mirror(bp))
        self.assertTrue(np.array_equal(b1["shape"] != 0, (b2["shape"] != 0)[::-1]))
        self.assertEqual(mirror(bp)["openings"][1]["side"], "west")


class Segment(unittest.TestCase):
    """The building's pixels from a label map (no model needed)."""

    def test_boxed_building_hidden_ends_and_specks(self):
        from blueprint.segment import main_building
        lab = np.full((100, 200), 2, np.int16)            # sky
        lab[40:90, 50:130] = 1                            # the building
        lab[40:90, 0:20] = 1                              # a neighbour, outside the box
        lab[30:95, 130:150] = 4                           # a tree hiding its right end
        lab[10:12, 60:62] = 1                             # a speck
        out = main_building(lab, (240, 0, 750, 1000))     # x 48-150 px
        self.assertTrue(out["ok"])
        self.assertFalse(out["mask"][:, :40].any())       # not the neighbour
        self.assertFalse(out["mask"][10:12, 60:62].any())  # not the speck
        self.assertEqual(out["frame"][2], 150)            # carried out behind the tree
        self.assertTrue(out["occluders"][50, 140])
        self.assertGreater(out["occluded"], 0.1)

    def test_lab(self):
        from blueprint.segment import lab_colours
        white, black = lab_colours(np.array([[255, 255, 255], [0, 0, 0]]))
        self.assertAlmostEqual(white[0], 100, delta=0.5)
        self.assertAlmostEqual(black[0], 0, delta=0.5)


class Score(unittest.TestCase):
    """The computer measures on made-up silhouettes."""

    def side(self, mask, win=None, hidden=None):
        from blueprint.score import boxes_of, facts, frame_of
        rgb = np.full(mask.shape + (3,), 180, np.uint8)
        frame = frame_of(mask | (hidden if hidden is not None else False))
        out = facts(rgb, mask, frame, hidden, win)
        if win is not None:
            out["openings"] = {"window": boxes_of(win, frame), "door": [], "garage": []}
        return out

    def house(self, w=60, h=40, roof=15, flip=False):
        m = np.zeros((100, 100), bool)
        m[90 - h:90, 20:20 + w] = True
        for k in range(roof):                             # a roof leaning to one side
            m[90 - h - k, 20 + k * 2:20 + w // 2 + k] = True
        return m[:, ::-1] if flip else m

    def test_the_same_scores_one(self):
        from blueprint.score import compare
        m = self.house()
        win = np.zeros_like(m)
        win[60:70, 30:40] = win[60:70, 50:60] = True
        s = compare(self.side(m, win), self.side(m, win))
        for k in ("outline", "proportions", "roof_line", "windows", "colour"):
            self.assertAlmostEqual(s[k][0], 1.0, places=3, msg=k)

    def test_hidden_pixels_do_not_count(self):
        from blueprint.score import compare
        full = self.house()
        hid = np.zeros_like(full)
        hid[55:90, 60:78] = True                          # a tree in front
        seen = full & ~hid
        self.assertAlmostEqual(compare(self.side(seen, hidden=hid), self.side(full))["outline"][0], 1.0, places=2)

    def test_diagnose(self):
        from blueprint.score import diagnose
        photo = self.side(self.house())
        self.assertEqual(diagnose(photo, self.side(self.house(w=40, h=60)))["shape"], "taller")
        self.assertGreater(diagnose(photo, self.side(self.house(flip=True)))["mirror_gain"], 0.04)


class Edits(unittest.TestCase):
    """The evaluator's corrections stay inside what compose can build."""

    def state(self):
        from blueprint.edits import initial
        return initial({"style": "suburban", "kind": "house", "form": {
            "parts": [{"part": "main", "storeys": 2, "roof": "gable", "windows_across": 3, "front": {"door": True}},
                      {"part": "tower", "storeys": 1, "roof": "hip"}], "garage": {"doors": 1, "wall": "right"}}})

    def test_pitch_steps_to_a_limit(self):
        from blueprint.compose import compose
        from blueprint.edits import apply
        s = self.state()
        pitches = []
        for _ in range(3):
            bp = compose(s["form"], s["style"], s["kind"], adjust=s["adjust"])
            s, note = apply(s, {"op": "pitch", "part": 0, "delta": 1}, bp)
            pitches.append(note or s["adjust"]["pitch"]["0"])
        self.assertEqual(pitches[:2], [2.0, 3.0])
        self.assertIn("72 degrees", pitches[2])

    def test_every_edit_builds(self):
        from blueprint.compose import compose
        from blueprint.edits import FEATURE_ON, apply
        s0 = self.state()
        edits = [{"op": "style", "value": "tudor"}, {"op": "kind", "value": "cottage"}, {"op": "storeys", "part": 1, "delta": 1},
                 {"op": "roof", "part": 1, "value": "mansard"}, {"op": "ridge", "part": 0}, {"op": "width", "factor": 1.25},
                 {"op": "height", "factor": 0.8}, {"op": "depth", "factor": 1.25}, {"op": "share", "part": 0, "factor": 1.25},
                 {"op": "mirror"}, {"op": "add_part", "role": "annex", "side": "left"}, {"op": "remove_part", "part": 1},
                 {"op": "windows", "delta": -1}, {"op": "colour", "role": "wall", "value": "red"},
                 {"op": "colour", "role": "roof", "value": "green"}, {"op": "wall_material", "value": "stone"},
                 {"op": "skylights_on", "part": 1}, {"op": "units", "delta": 2}, {"op": "garage_wall", "value": "left"}]
        edits += [{"op": "feature", "name": n, "on": on} for n in FEATURE_ON for on in (True, False)]
        for e in edits:
            s, note = apply(s0, e)
            self.assertIsNone(note, e)
            bp = compose(s["form"], s["style"], s["kind"], adjust=s["adjust"])
            self.assertEqual(build(bp)["warnings"], [], e)
        s, _ = apply(s0, {"op": "feature", "name": "towers", "on": False})
        self.assertEqual([p["part"] for p in s["form"]["parts"]], ["main", "wing"])
        self.assertTrue(apply(s0, {"op": "roof", "part": 0, "value": "onion"})[1].startswith("invalid"))

    def test_critique_becomes_edits_and_gaps(self):
        from blueprint.edits import from_critique
        crit = {"style_ok": False, "style": "Tudor", "kind_ok": True, "proportions": {"width": "too_narrow"},
                "roofs": [{"part": 1, "type_ok": False, "type": "gambrel"}, {"part": 2, "pitch": "steeper"},
                          {"part": 1, "type_ok": False, "type": "onion dome"}],
                "missing": ["turret", "clock face"], "extra": ["balcony"], "mirrored": True, "windows": "too_few",
                "wall_colour": "pale yellow", "roof_colour": "ok", "other": "the roof should curve"}
        edits, gaps = from_critique(crit, self.state(), {1: 1, 2: 0})          # shown mirrored
        names = {(e["op"], e.get("part"), e.get("value"), e.get("name")) for e in edits}
        for want in (("style", None, "tudor", None), ("width", None, None, None), ("roof", 1, "gambrel", None),
                     ("feature", None, None, "towers"), ("feature", None, None, "balconies"),
                     ("mirror", None, None, None), ("windows", None, None, None), ("colour", None, "pale yellow", None)):
            self.assertIn(want, names)
        self.assertFalse([e for e in edits if e["op"] == "pitch"])    # a roof's pitch is the roof pass's
        self.assertEqual(len(gaps), 3)                     # the onion dome, the clock face, the curved roof
        self.assertTrue(all(e["why"] == "vision" for e in edits))


class Vision(unittest.TestCase):
    def test_boxes_survive_a_cut_off_answer(self):
        from blueprint.vision import boxes
        text = ('```json\n[{"bbox_2d": [10, 20, 30, 40], "label": "window"}, {"bbox_2d": [50, 20, 70, 60], '
                '"label": "door"}, {"bbox_2d": [80, 20, 9')
        self.assertEqual([b["label"] for b in boxes(text)], ["window", "door"])

    def test_camera_side_from_the_model(self):
        from blueprint.refine import camera
        m = {"camera_yaw_deg": 44.0, "camera_pitch_deg": 20.0}
        self.assertEqual(camera({"locate": {"side": "right"}}, m), (-44.0, 20.0))
        self.assertEqual(camera({"locate": {"side": "left"}}, m), (44.0, 20.0))
        self.assertEqual(camera({"locate": {"side": "none"}}, None), (0.0, 12.0))

    def test_undo_when_the_next_critique_contradicts(self):
        from blueprint.refine import _contradicts, _kept
        porch = {"op": "feature", "name": "porch", "on": True}
        self.assertTrue(_contradicts({"extra": ["veranda"]}, porch))           # a synonym counts
        self.assertFalse(_contradicts({"extra": ["dormers"], "missing": ["porch"]}, porch))
        self.assertTrue(_contradicts({"roofs": [{"part": 1, "pitch": "flatter"}]}, {"op": "pitch", "part": 0, "delta": 1}))
        self.assertTrue(_contradicts({"mirrored": True}, {"op": "mirror"}))
        self.assertFalse(_contradicts({"mirrored": True}, {"op": "width", "factor": 1.2}))
        hist = [{"accepted": "a"}, {"accepted": "b"}, {"reverted": "b"}, {"accepted": "c"}]
        self.assertEqual(_kept(hist), ["a", "c"])

    def test_a_correction_is_checked_by_critiquing_it(self):
        from blueprint.refine import _verify
        porch = {"op": "feature", "name": "porch", "on": True, "why": "vision"}
        before = {"resemblance": 6, "missing": ["porch"]}
        self.assertEqual(_verify(porch, before, {"resemblance": 6, "missing": []}, 0.0), "kept")
        self.assertEqual(_verify(porch, before, {"resemblance": 7, "missing": ["porch"]}, 0.0), "complaint still there")
        self.assertEqual(_verify(porch, before, {"resemblance": 6, "extra": ["porch"]}, 0.0), "contradicted")
        self.assertTrue(_verify(porch, before, {"resemblance": 4}, 0.0).startswith("rated lower"))
        mirror = {"op": "mirror", "why": "measures"}                        # theirs: the model must like it more
        self.assertEqual(_verify(mirror, {"resemblance": 6}, {"resemblance": 6}, 0.2), "not rated higher")
        self.assertEqual(_verify(mirror, {"resemblance": 6}, {"resemblance": 7}, 0.0), "kept")
        pitch = {"op": "pitch", "part": 0, "delta": 1, "why": "measures"}
        self.assertEqual(_verify(pitch, {"resemblance": 6}, {"resemblance": 6}, 0.06), "kept")
        self.assertEqual(_verify(pitch, {"resemblance": 6}, {"resemblance": 6}, 0.01), "too small a gain")


class PerPart(unittest.TestCase):
    """The form's version 3: each part's size, level, offset, roof slope and garage doors."""

    FORM = {"parts": [
        {"part": "main", "width_m": 8, "depth_m": 8, "storeys": 2, "eave_m": 6, "floor_m": 0, "forward_m": 0,
         "roof": "gable", "ridge": "parallel", "pitch_deg": 30, "windows_across": 2,
         "front": {"door": True, "ground_windows": 1, "garage_doors": 0}, "side_garage_doors": 0, "base": "none"},
        {"part": "wing", "width_m": 12, "depth_m": 6, "storeys": 2, "eave_m": 4.5, "floor_m": -1.5, "forward_m": 1,
         "roof": "gable", "ridge": "parallel", "pitch_deg": 14, "windows_across": 1,
         "front": {"door": False, "wide_window": True, "garage_doors": 0}, "side_garage_doors": 2, "base": "brick"}],
        "wall": {"material": "wood_siding", "colour": "white"}}

    def compose(self, form=None, **kw):
        import copy as _copy

        from blueprint.compose import compose
        bp = compose(_copy.deepcopy(form or self.FORM), "suburban", "house", **kw)
        load(bp)
        b = build(bp)
        self.assertEqual(b["warnings"], [])
        return bp

    def test_parts_take_their_metres_and_degrees(self):
        bp = self.compose()
        main, wing = bp["masses"][:2]
        self.assertAlmostEqual(wing["w"] / main["w"], 12 / 8, delta=0.2)          # widths by their metres
        self.assertAlmostEqual(wing["d"] / wing["w"], 0.5, delta=0.1)             # depth by its own ratio
        self.assertLess(wing["y"], 0)                                             # a split level, lower
        self.assertLess(wing["walls"], main["walls"])
        self.assertGreater(wing["z"] + wing["d"], main["z"] + main["d"])          # a metre forward
        self.assertEqual((main["roof"]["pitch"], wing["roof"]["pitch"]), (0.5, 0.25))
        self.assertEqual(wing["materials"]["wall_base"], "bricks")
        garages = [o for o in bp["openings"] if o["kind"] == "garage"]
        self.assertEqual([(o["mass"], o["side"]) for o in garages], [(wing["id"], "east")] * 2)
        self.assertEqual(bp["meta"]["form_version"], 3)

    def test_what_does_not_fit_is_reported_or_grown(self):
        import copy as _copy
        form = _copy.deepcopy(self.FORM)
        form["parts"][1].update(depth_m=1.5, side_garage_doors=3)                # three doors on a short end wall
        form["parts"][0].update(windows_across=30, ridge="perpendicular")
        form["dormers"] = 2
        bp = self.compose(form)
        conflicts = " | ".join(bp["meta"]["conflicts"])
        self.assertIn("garage doors need a wall", conflicts)
        self.assertIn("windows across", conflicts)
        self.assertIn("dormers", conflicts)
        wing = bp["masses"][1]
        self.assertEqual(sum(1 for o in bp["openings"] if o["kind"] == "garage" and o["mass"] == wing["id"]), 3)

    def test_pitch_a_quarter_is_half_a_block_every_two(self):
        b = build(house(pitch=0.25))
        self.assertEqual(cell(b, 4, 4, 7), ("slab_bottom", "deepslate_tiles"))
        self.assertEqual(cell(b, 4, 4, 5), ("slab_top", "deepslate_tiles"))
        self.assertEqual(top(b), 5)


class Placement(unittest.TestCase):
    """The photo's openings, placed on the building by MapAnything, on the build's walls."""

    def bp(self, wing_d=10):
        from blueprint.compose import compose
        return compose({"parts": [{"part": "main", "share": 60, "storeys": 2, "roof": "gable", "windows_across": 2,
                                   "front": {"door": True}},
                                  {"part": "wing", "share": 40, "storeys": 1, "roof": "gable"}]}, "suburban", "house",
                       adjust={"parts": {"1": {"depth": wing_d / 10}}})

    OPENINGS = [{"label": "window", "wall": "front", "along": 0.2, "up": 0.6, "w": 0.1, "h": 0.12, "up_wall": 0.75,
                 "h_wall": 0.2, "into": 0.0},
                {"label": "door", "wall": "front", "along": 0.35, "up": 0.15, "w": 0.04, "h": 0.15, "up_wall": 0.2,
                 "h_wall": 0.3, "w_m": 0.9, "h_m": 2.0, "into": 0.0},
                {"label": "door", "wall": "right", "along": 0.3, "up": 0.1, "w": 0.2, "h": 0.15, "along_wall": 0.3,
                 "w_wall": 0.3, "up_wall": 0.25, "h_wall": 0.45, "w_m": 2.6, "h_m": 2.2},
                {"label": "door", "wall": "right", "along": 0.7, "up": 0.1, "w": 0.2, "h": 0.15, "along_wall": 0.72,
                 "w_wall": 0.3, "up_wall": 0.25, "h_wall": 0.45, "w_m": 2.6, "h_m": 2.2}]

    def test_kinds(self):
        from blueprint.facade import kind_of
        self.assertEqual([kind_of(o) for o in self.OPENINGS], ["window", "door", "garage", "garage"])
        self.assertEqual(kind_of({"label": "garage_door", "up": 0.1, "h": 0.2}), "garage")

    def test_openings_land_on_their_walls(self):
        from blueprint.facade import fractions, photo_fractions, place_openings_3d
        from blueprint.score import match3d
        bp, notes = place_openings_3d(self.bp(), self.OPENINGS, "right")
        main, wing = bp["masses"][:2]
        placed = [o for o in bp["openings"] if (o["mass"], o["side"]) in ((main["id"], "south"), (wing["id"], "east"))]
        win = [o for o in placed if o["kind"] == "window" and o["mass"] == main["id"]]
        self.assertTrue(win and all(o["y"] >= main["storey_h"] for o in win))     # on the upper floor
        self.assertEqual([o["y"] for o in placed if o["kind"] == "door"], [1])
        self.assertEqual(sum(1 for o in placed if o["kind"] == "garage" and o["mass"] == wing["id"]), 2)
        b = build(bp)
        s, pairs, missing, extra = match3d(photo_fractions(self.OPENINGS), fractions(bp, b, "right"))
        self.assertEqual(missing, [])

    def test_unseen_windows_stay_and_upstairs_doors_become_windows(self):
        from blueprint.facade import place_openings_3d
        bp0 = self.bp()
        wing = bp0["masses"][1]["id"]
        before = [o for o in bp0["openings"] if o["mass"] == wing and o["side"] == "south" and o["kind"] == "window"]
        ops = [dict(self.OPENINGS[0]),                                            # a window on the main block only
               {"label": "door", "wall": "front", "along": 0.15, "up": 0.6, "w": 0.05, "h": 0.12, "up_wall": 0.75,
                "h_wall": 0.2, "w_m": 0.9, "h_m": 2.0, "into": 0.0}]               # a balcony's door upstairs
        bp, notes = place_openings_3d(bp0, ops, "right")
        after = [o for o in bp["openings"] if o["mass"] == wing and o["side"] == "south" and o["kind"] == "window"]
        self.assertEqual(len(after), len(before))          # the photo shows no window on the wing: it keeps its own
        main = bp["masses"][0]
        self.assertFalse([o for o in bp["openings"] if o["kind"] == "door" and o["mass"] == main["id"] and o["y"] > 1])
        self.assertEqual(notes["placed"]["door"], 0)

    def test_garage_doors_that_do_not_fit_lengthen_the_wall(self):
        from blueprint.facade import place_openings_3d
        bp, notes = place_openings_3d(self.bp(wing_d=5), self.OPENINGS, "right")
        wing = bp["masses"][1]
        self.assertTrue(notes["grown"])
        self.assertEqual(sum(1 for o in bp["openings"] if o["kind"] == "garage" and o["mass"] == wing["id"]), 2)
        load(bp)


class PartEdits(unittest.TestCase):
    """The loop's corrections of one part, and the critique's answers that ask for them."""

    def state(self):
        from blueprint.edits import initial
        return initial({"style": "suburban", "kind": "house", "form": {"parts": [
            {"part": "main", "storeys": 2, "roof": "gable", "front": {"door": True}},
            {"part": "wing", "storeys": 1, "roof": "gable", "side_garage_doors": 1}]}})

    def test_each_correction_moves_its_part(self):
        from blueprint.compose import compose
        from blueprint.edits import apply
        s0 = self.state()
        base = compose(s0["form"], s0["style"], s0["kind"], adjust=s0["adjust"])["masses"][1]
        for edit, check in (({"op": "part_height", "part": 1, "factor": 1.5}, lambda m: m["walls"] > base["walls"]),
                            ({"op": "part_depth", "part": 1, "factor": 1.5}, lambda m: m["d"] > base["d"]),
                            ({"op": "part_level", "part": 1, "delta": -2}, lambda m: m["y"] == base["y"] - 2),
                            ({"op": "part_forward", "part": 1, "delta": 2},
                             lambda m: m["z"] + m["d"] == base["z"] + base["d"] + 2)):
            s, note = apply(s0, edit)
            self.assertIsNone(note, edit)
            m = compose(s["form"], s["style"], s["kind"], adjust=s["adjust"])["masses"][1]
            self.assertTrue(check(m), edit)
        s, _ = apply(s0, {"op": "garage_doors", "part": 1, "value": 2})
        bp = compose(s["form"], s["style"], s["kind"], adjust=s["adjust"])
        self.assertEqual(sum(1 for o in bp["openings"] if o["kind"] == "garage"), 2)

    def test_conflicts_become_edits(self):
        from blueprint.compose import compose
        from blueprint.edits import apply, from_conflicts
        s0 = self.state()
        s0["form"]["parts"][1]["windows_across"] = 12                            # more than its wall holds
        bp = compose(s0["form"], s0["style"], s0["kind"])
        edits = from_conflicts(bp["meta"]["conflicts"], s0, bp)
        widen = [e for e in edits if e["op"] == "share" and e["part"] == 1]
        self.assertTrue(widen and widen[0]["factor"] > 1 and widen[0]["why"] == "conflict")
        s1, _ = apply(s0, widen[0])
        self.assertGreater(compose(s1["form"], s1["style"], s1["kind"])["masses"][1]["w"], bp["masses"][1]["w"])

    def test_part_answers_become_part_edits(self):
        from blueprint.edits import from_critique
        crit = {"parts": [{"part": 1, "height": "ok", "level": "ok", "roof_ok": True},
                          {"part": 2, "height": "too_tall", "level": "should_be_lower", "depth": "too_shallow",
                           "roof_ok": False, "roof_type": "gable", "pitch": "flatter", "garage_doors": 2}]}
        edits, gaps = from_critique(crit, self.state(), {1: 0, 2: 1})
        got = {(e["op"], e.get("part"), e.get("shown")) for e in edits}
        for want in (("part_height", 1, 2), ("part_level", 1, 2), ("part_depth", 1, 2), ("garage_doors", 1, 2),
                     ("roof", 1, None)):
            self.assertIn(want, got)
        self.assertNotIn(("pitch", 1, None), got)


class Check(unittest.TestCase):
    """The photo-only check: its answers against the build become corrections."""

    def setup(self):
        from blueprint.compose import compose
        from blueprint.edits import check_parts, initial
        state = initial({"style": "suburban", "kind": "house", "form": {"parts": [
            {"part": "main", "share": 50, "storeys": 2, "roof": "gable", "ridge": "perpendicular", "front": {"door": True}},
            {"part": "wing", "share": 50, "storeys": 1, "roof": "hip"}]}})
        bp = compose(state["form"], state["style"], state["kind"])
        descs, numbers = check_parts(bp, state)
        return state, bp, descs, numbers

    def test_answers_that_differ_become_edits(self):
        from blueprint.edits import from_check
        state, bp, descs, numbers = self.setup()
        self.assertEqual(descs, ["the main block", "a wing"])
        check = {"parts": [{"part": 1, "roof_ends": "triangle", "ridge": "along_front", "floor": "same", "storeys": 2,
                            "garage_doors": 0},
                           {"part": 2, "roof_ends": "triangle", "ridge": "along_front", "floor": "lower", "storeys": 2,
                            "garage_doors": 2, "garage_wall": "end"}], "tallest": 1}
        got = {(e["op"], e.get("part"), e.get("value")) for e in from_check(check, state, bp, numbers)}
        for want in (("ridge", 0, None), ("roof", 1, "gable"), ("part_level", 1, None), ("storeys", 1, None),
                     ("garage_doors", 1, 2)):
            self.assertIn(want, got)
        self.assertNotIn(("ridge", 1, None), got)          # the hip wing's ridge already runs left to right
        self.assertTrue(all(e["why"] == "check" for e in from_check(check, state, bp, numbers)))

    def test_the_check_keeps_what_both_askings_agree_on(self):
        import json as _json

        from blueprint import vision
        answers = iter([_json.dumps({"parts": [{"part": 1, "ridge": "along_front", "storeys": 2, "roof_ends": "slope"}],
                                     "tallest": 1}),
                        _json.dumps({"parts": [{"part": 1, "ridge": "along_front", "storeys": 3, "roof_ends": "slope"}],
                                     "tallest": 1})])
        saved = vision.ask
        vision.ask = lambda *a, **k: next(answers)
        try:
            got = vision.check((None, None), None, ["the main block"])
        finally:
            vision.ask = saved
        self.assertEqual(got["parts"], [{"part": 1, "ridge": "along_front", "roof_ends": "slope"}])
        self.assertEqual(got["disagreed"], 1)
        self.assertEqual(got["tallest"], 1)
        self.assertIn("front_to_back|along_front", vision._flipped(vision.CHECK))

    def test_only_answers_the_form_agrees_with_are_acted_on(self):
        from blueprint.edits import agreed_with_form
        state, bp, descs, numbers = self.setup()
        check = {"parts": [{"part": 1, "ridge": "along_front", "roof_ends": "triangle", "storeys": 2},
                           {"part": 2, "roof_ends": "slope", "storeys": 3}]}
        got = agreed_with_form(check, state["form"], numbers)["parts"]
        # the form read the main block's ridge perpendicular (front to back) and its wing hip with 1 storey
        self.assertEqual(got[0], {"part": 1, "roof_ends": "triangle", "storeys": 2})
        self.assertEqual(got[1], {"part": 2, "roof_ends": "slope"})

    def test_doors_size_a_level_they_agree_with(self):
        from blueprint.edits import from_check
        from blueprint.facade import levels
        state, bp, descs, numbers = self.setup()
        openings = [{"label": "door", "wall": "front", "along": 0.25, "up": 0.3, "w": 0.05, "h": 0.2, "w_m": 1, "h_m": 2.1},
                    {"label": "garage_door", "wall": "right", "along": 0.5, "along_wall": 0.5, "up": 0.08, "h": 0.2,
                     "w": 0.2, "w_wall": 0.4}]
        door_levels = levels(bp, openings)
        wing = bp["masses"][1]["id"]
        self.assertLessEqual(door_levels.get(wing, 0), -3)  # its garage door stands well below the entrance
        check = {"parts": [{"part": 2, "floor": "lower"}]}
        e = [e for e in from_check(check, state, bp, numbers, door_levels) if e["op"] == "part_level"][0]
        self.assertTrue(e["doors_agree"])
        self.assertEqual(e["delta"], door_levels[wing])


class RoofPass(unittest.TestCase):
    """Roofs as the photo shows them: the concrete questions' answers, and the parts they build."""

    def form(self, **seen):
        return {"parts": [{"part": "main", "width_m": 14, "depth_m": 9, "storeys": 2, "eave_m": 6, "roof": "gable",
                           "ridge": "perpendicular", "pitch_deg": 30, "front": {"door": True}, "roof_seen": seen}]}

    def compose(self, form, style="suburban", kind="house"):
        from blueprint.compose import compose
        bp = compose(form, style, kind)
        load(bp)
        b = build(bp)
        self.assertEqual(b["warnings"], [])
        return bp

    def test_three_quarter_pitch_climbs_stairs_and_slabs(self):
        b = build(house(pitch=0.75))
        shapes = [cell(b, 4, y, z)[0] for z, y in ((7, 4), (6, 5), (5, 5))]
        self.assertEqual(shapes[0][:6], "stairs")                 # the eave: a full step
        self.assertEqual(shapes[1], "slab_bottom")                # then half a step
        self.assertEqual(top(b), 7)

    def test_the_front_top_sets_the_ridge_and_the_pitch_class_the_slope(self):
        bp = self.compose(self.form(front_top="eave", pitch="medium"))
        self.assertEqual((bp["masses"][0]["roof"]["ridge"], bp["masses"][0]["roof"]["pitch"]), ("x", 0.75))
        bp = self.compose(self.form(front_top="triangle", pitch="low"))
        self.assertEqual((bp["masses"][0]["roof"]["ridge"], bp["masses"][0]["roof"]["pitch"]), ("z", 0.5))
        bp = self.compose(self.form(front_top="flat"))
        self.assertEqual(bp["masses"][0]["roof"]["type"], "flat")
        f = self.form(front_top="eave", pitch="medium")
        f["parts"][0]["pitch_deg"] = 50                          # the form's steep kept where the pass said medium
        self.assertEqual(self.compose(f)["masses"][0]["roof"]["pitch"], 1.0)
        for deg, seen, want in ((45, "medium", 1.0), (60, "steep", 2.0), (30, "steep", 2.0)):
            f = self.form(front_top="eave", pitch=seen)          # a steep style: the form's degrees first
            f["parts"][0]["pitch_deg"] = deg
            self.assertEqual(self.compose(f, "victorian")["masses"][0]["roof"]["pitch"], want, deg)

    def test_a_gable_rising_from_the_eave_is_its_own_mass(self):
        bp = self.compose(self.form(front_top="both", cross_at="left", pitch="medium"))
        main, xg = bp["masses"][0], bp["masses"][1]
        self.assertEqual(xg["id"], "x0")
        self.assertEqual((xg["roof"]["type"], xg["roof"]["ridge"]), ("gable", "z"))
        self.assertEqual(xg["z"] + xg["d"], main["z"] + main["d"] + 1)          # a block proud of the front
        self.assertLess(xg["x"] - main["x"], main["w"] // 3)                     # at the left
        self.assertTrue(any(o["mass"] == "x0" and o["kind"] == "window" for o in bp["openings"]))

    def test_tower_tops(self):
        for answer, roof, lantern in (("spire", "spire", False), ("lantern", "flat", True),
                                      ("battlements", "flat", False), ("cone", "pyramid", False)):
            form = {"parts": [{"part": "main", "storeys": 2, "roof": "gable"},
                              {"part": "tower", "storeys": 5, "roof": "flat", "roof_seen": {"tower_top": answer}}]}
            bp = self.compose(form, "gothic", "church")
            tower = bp["masses"][1]
            self.assertEqual(tower["roof"]["type"], roof, answer)
            self.assertEqual(any(f["kind"] == "lantern" and f["mass"] == tower["id"] for f in bp["features"]), lantern)
            if answer == "battlements":
                self.assertTrue(tower["roof"]["battlements"])

    def test_a_lighthouse_lantern_crowns_its_tower_not_the_house(self):
        for seen in ({"tower_top": "lantern"}, {}):
            form = {"parts": [{"part": "tower", "storeys": 4, "roof": "flat", "roof_seen": seen},
                              {"part": "main", "storeys": 1, "roof": "gable", "pitch_deg": 40}]}
            bp = self.compose(form, "victorian", "lighthouse")
            tower, house = bp["masses"][0], bp["masses"][1]
            self.assertEqual(house["roof"]["type"], "gable")
            self.assertEqual([f["mass"] for f in bp["features"] if f["kind"] in ("lantern", "gallery")],
                             [tower["id"], tower["id"]])

    def test_open_lantern_has_posts_and_a_cap(self):
        bp = {"masses": [{"id": "t", "x": 0, "z": 0, "w": 7, "d": 7, "y": 0, "walls": 10, "floors": [0],
                          "roof": {"type": "flat", "overhang": 0}}],
              "features": [{"kind": "lantern", "mass": "t", "h": 4, "open": True, "cap": "spire"}]}
        b = build(bp)
        self.assertEqual(cell(b, 1, 13, 1)[0], "full")           # a corner post
        self.assertEqual(cell(b, 3, 12, 1)[0], "air")            # open between the posts
        self.assertGreater(top(b), 15)                           # the spire on top

    def test_the_roof_pass_asks_each_part_what_it_shows(self):
        from blueprint import refine, vision
        asked = []

        def fake(vlm, photo, which, key):
            asked.append((which, key))
            return {"front_top": "both", "ends": "slope", "pitch": "steep", "cross_at": "right",
                    "tower_top": "lantern"}[key]
        saved = vision.roof_question, vision.roof_yes
        vision.roof_question = fake
        vision.roof_yes = lambda vlm, photo, which, key: asked.append((which, key)) or True
        try:
            read = {"form": {"parts": [{"part": "main", "roof": "gable"}, {"part": "tower", "roof": "flat"}]}}
            out = refine.roof_pass(None, None, read)
        finally:
            vision.roof_question, vision.roof_yes = saved
        self.assertEqual(out[0], {"front_top": "both", "ends": "slope", "pitch": "steep", "cross_at": "right"})
        self.assertEqual(out[1], {"tower_top": "lantern"})
        self.assertEqual(read["form"]["parts"][1]["roof_seen"], {"tower_top": "lantern"})
        self.assertIn("its tower, the part at the right of the 2 parts it is made of", [w for w, _ in asked])
        self.assertNotIn("slope", [k for _, k in asked])         # no flat roof against a pitched one: no tie-break

    def test_a_disagreement_is_settled_by_a_yes_no_question(self):
        from blueprint import refine, vision
        saved = vision.roof_yes
        try:
            for seen, form, yes, key, want in (("eave", "flat", False, "slope", "flat"),
                                               ("eave", "flat", True, "slope", "eave"),
                                               ("flat", "triangle", True, "slope", "triangle"),
                                               ("eave", "flat", None, "slope", "eave"),       # askings differ
                                               ("both", "eave", False, None, "both"),          # no question
                                               ("eave", "triangle", True, None, "eave")):
                asked = []
                vision.roof_yes = lambda vlm, photo, which, k, yes=yes: asked.append(k) or yes
                self.assertEqual(refine.settle_front(None, None, "it", seen, form), want, (seen, form, yes))
                self.assertEqual(asked, [key] if key else [])
        finally:
            vision.roof_yes = saved


class Example(unittest.TestCase):
    def test_example_writes_a_schematic(self):
        b = build(EXAMPLE)
        self.assertEqual(b["warnings"], [])
        shape, material = b["shape"], b["material"]
        solid = shape != 0
        self.assertTrue(V.VALID[shape[solid], material[solid]].all())
        d = Path(tempfile.mkdtemp())
        try:
            info = write_schem(shape, material, d / "build.schem")
            self.assertEqual(info["dims"], list(shape.shape))
            self.assertEqual(info["blocks"], int(solid.sum()) + 1)              # + the door's upper half
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
