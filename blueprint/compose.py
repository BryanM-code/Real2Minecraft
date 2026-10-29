"""The style-aware form (`describe.form2`) + the building's style and kind + its measured
size (`measure`) -> a blueprint-1. The model chooses and counts; this module decides every
size and position: metres from the measurement (or the kind's typical sizes), blocks at an
adaptive scale, the parts the style and kind call for when the model says nothing, and the
style's palette under the colours the model reports.

A version-3 form (`describe.form2(..., 3)`) gives each part its size, level and offset in
metres, its roof's slope in degrees and its garage doors; they are honoured as ratios (the
model's metres are rough, its proportions better): widths by width_m, a part's depth by its
depth_m over its width_m, heights, levels and offsets by the main block's walls over its
eave_m. Whatever a form leaves out, the rules below fill in.

What the form asks and the build cannot do is reported in meta["conflicts"], never dropped
silently: more windows than a wall holds, garage doors that need a longer wall (the wall
grows), dormers on a roof that cannot take them.

The evaluator (`refine`) corrects a build only through the form and through `adjust`, which
this module honours: {"scale": {"width" | "depth" | "height": factor}, "mirror": bool,
"pitch": {part number: 0.25 | 0.5 | 1 | 2 | 3}, "skylights_on": part number, "materials": {role:
block}, "parts": {part number: {"height" | "depth": factor, "level" | "forward": blocks}}};
part numbers count the form's parts from 0, left to right as the photo shows them.

    compose(form, style="victorian", kind="house", measure={"width_m": 14, "height_m": 9, ...})
"""
from __future__ import annotations

import copy
import math

from .assemble import BASES, DOORS, ROOFS, TRIMS, WALLS, _int, pick
from .spec import PITCHES, material_index, shape_ok
from .styles import KINDS, STYLES

KIND_ALIASES = {"bank": "office", "museum": "library_museum", "library": "library_museum", "temple": "mosque_temple",
                "mosque": "mosque_temple", "pagoda": "mosque_temple", "cathedral": "church", "chapel": "church",
                "capitol": "government", "courthouse": "government", "city_hall": "government", "hall": "government",
                "villa": "mansion", "cabin": "cottage", "bungalow": "house", "row_house": "townhouse",
                "terraced": "townhouse", "fort": "castle", "keep": "castle", "chateau": "palace", "motel": "hotel",
                "store": "shop", "market": "shop", "cafe": "restaurant", "diner": "restaurant", "silo": "barn",
                "airport": "hangar_airport", "hangar": "hangar_airport", "terminal": "station", "mill": "windmill",
                "power_plant": "factory", "plant": "factory", "industrial": "factory", "garage": "parking"}
STYLE_ALIASES = {"georgian": "colonial", "cape_cod": "colonial", "federal": "colonial", "italianate": "victorian",
                 "queen_anne": "victorian", "gothic_revival": "gothic", "spanish": "mediterranean",
                 "mission": "mediterranean", "contemporary": "modern", "minimalist": "modern",
                 "cabin": "rustic", "log": "rustic", "bauhaus": "modern", "classical": "neoclassical",
                 "greek_revival": "neoclassical", "chalet": "alpine", "nordic": "scandinavian", "ottoman": "islamic",
                 "pueblo": "adobe", "pueblo_revival": "adobe", "southwestern": "adobe", "mud_brick": "adobe",
                 "sudano_sahelian": "adobe"}
WALLS2 = dict(WALLS, metal=[("", "light_gray_concrete")], logs=[("", "spruce_log")],
              timber_frame=[("", "white_terracotta")])
# (pink: the game's pink terracotta is a dull brick red; painted pink is pale, white terracotta's pink beige)
WALL_COLOURS = [("green", "green_terracotta"), ("blue", "light_blue_terracotta"), ("yellow", "yellow_terracotta"),
                ("pink", "white_terracotta"), ("red", "red_terracotta"), ("orange", "orange_terracotta"),
                ("purple", "purple_terracotta"),
                # earth tones (adobe, Spanish and Mexican stucco painted tan, ochre, brown)
                ("ochre", "yellow_terracotta"), ("terracotta", "terracotta"), ("adobe", "packed_mud"),
                ("dark brown", "brown_terracotta"), ("brown", "terracotta"), ("tan", "white_terracotta"),
                ("sand", "smooth_sandstone"), ("khaki", "smooth_sandstone"), ("salmon", "pink_terracotta"),
                ("peach", "white_terracotta"), ("olive", "green_terracotta")]
ROOF_KINDS = ("gable", "cross_gable", "hip", "flat", "shed", "mansard", "gambrel", "dome", "cone", "spire", "pyramid",
              "sawtooth", "pagoda", "barrel", "butterfly", "onion_dome", "clipped_gable", "hip_and_gable")
# styles whose hip roofs end in small gables at the top (irimoya, xieshan)
GABLET_STYLES = ("japanese", "chinese")
# how far each answer to "eaves" puts a pitched roof's edge out from the walls, in blocks
EAVES = {"flush": 0, "normal": 1, "deep": 2}
# kinds whose every part stands on posts, not walls
OPEN_KINDS = ("parking", "pavilion")
# kinds whose one tower in the middle of the roof, read beside a cupola, is that cupola (a belvedere)
CUPOLA_KINDS = ("house", "mansion", "cottage", "townhouse", "barn")
# kinds that may carry a dome on their main block besides a domed tower (a capitol, a church)
DOME_KINDS = ("government", "church", "mosque_temple", "palace", "library_museum", "theatre", "station")
# styles whose columned front is a portico where it was read as a porch as well
PORTICO_STYLES = ("neoclassical", "beaux_arts", "colonial", "renaissance", "baroque")
# styles whose jettied floors step out on every side (elsewhere a jetty is a street front's)
MODERN_STYLES = ("modern", "brutalist", "international", "futuristic")
BIG = 150                    # blocks: the longest side of a build at the adaptive scale
# kinds a photo shows end-on (their long side runs away from the camera)
END_ON = ("church", "barn", "lighthouse", "windmill", "tower", "mosque_temple")
# kinds with a podium and a crown
TALL_KINDS = ("skyscraper", "office", "hotel", "apartment", "hospital")
# where a part the model calls a tower really is one (elsewhere it is often a taller block)
TOWER_KINDS = ("church", "castle", "government", "palace", "station", "school", "mansion", "lighthouse", "tower",
               "mosque_temple", "library_museum")
TOWER_STYLES = ("victorian", "gothic", "romanesque", "medieval", "baroque", "beaux_arts")
# "steep" means 63 degrees only where steep roofs are the style; the model says steep often
STEEP_STYLES = ("gothic", "victorian", "tudor", "medieval", "scandinavian", "romanesque", "dutch")
# kinds whose roof, when the photo cannot see it, is flat (tall or commercial buildings from the street)
FLAT_KINDS = ("skyscraper", "office", "apartment", "hotel", "hospital", "shop", "warehouse", "factory", "station",
              "stadium", "parking", "hangar_airport", "school", "gas_station", "library_museum", "government")
# kinds whose stacked roofs make a pagoda's tiers (elsewhere upturned eaves are one roof)
PAGODA_KINDS = ("mosque_temple", "tower", "other")
# a row's houses in turn when they differ: the form's wall block first, then these
ROW_WALLS = ("bricks", "brown_terracotta", "deepslate_bricks", "smooth_quartz", "mud_bricks", "white_terracotta")
UNIT_M = 6.0                 # metres: one house of a row
# what the roof pass (`refine.roof_pass`: one concrete question at a time about each part's roof,
# a part's "roof_seen") overrides the form on: where it read the 50 test photos' main blocks better
# (outputs/vlm/truth50_roofs.json). What tops the front wall: 37 of 48 right, the form 29; the
# roof's slope as low / medium / steep: 15 of 21, the form 11 (17 with the form's "steep" kept where
# the pass said medium: it called 5 of 6 steep roofs medium); a tower's top: 6 of 7, the form 5. Not
# the roof's ends (gable or hip): the form 31 of 33, the pass 27.
ROOF_TRUST = ("front_top", "pitch", "tower_top")
TOWER_TOPS = {"spire": "spire", "cone": "cone", "pyramid": "pyramid", "dome": "dome", "flat": "flat",
              "battlements": "flat", "lantern": "flat", "onion": "dome"}


def kind_of(kind) -> str:
    k = str(kind or "other").lower().replace(" ", "_")
    return k if k in KINDS else KIND_ALIASES.get(k, "other")


def style_of(style) -> str:
    s = str(style or "other").lower().replace(" ", "_").replace("-", "_")
    return s if s in STYLES else STYLE_ALIASES.get(s, "other")


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _choice(v, options, default):
    v = str(v or "").lower()
    return v if v in options else default


def _wall_material(form: dict) -> str:
    """The form's wall material word, lower case ('' when it names none)."""
    wall = form.get("wall") if isinstance(form.get("wall"), dict) else {}
    return str(wall.get("material") or "").lower()


def palette(form: dict, style: str, overrides: dict | None = None) -> dict:
    """Block per role: the style's palette, then what the model reports, then `overrides`."""
    st = STYLES[style]["palette"]
    mats = {"wall": "smooth_sandstone", "wall_base": "stone_bricks", "trim": "dark_oak_planks", "roof": "stone_bricks",
            "window": "glass", "door": "spruce_planks", "garage": "smooth_quartz", "chimney": "bricks", "skylight": "glass",
            "column": "smooth_quartz", "dome": "oxidized_cut_copper", "railing": "dark_oak_planks", "floor": "spruce_planks"}
    mats.update({k: v for k, v in st.items() if k in ("wall", "trim", "roof", "window", "column", "dome", "shutter")})
    if st.get("base"):
        mats["wall_base"] = st["base"]
    wall = form.get("wall") if isinstance(form.get("wall"), dict) else {}
    material, colour = str(wall.get("material", "")).lower(), str(wall.get("colour", "")).lower()
    if material in WALLS2:
        block = pick(WALLS2[material], colour)
        # painted walls in a colour their table lacks (pink concrete): the coloured terracottas
        if material in ("wood_siding", "stucco", "concrete", "metal") and not any(
                k for k, _ in WALLS2[material] if k and k in colour):
            block = next((b for k, b in WALL_COLOURS if k in colour), block)
        mats["wall"] = block
    base = form.get("base") if isinstance(form.get("base"), dict) else {}
    if str(base.get("material", "")).lower() in BASES:
        mats["wall_base"] = BASES[str(base["material"]).lower()]
    if form.get("trim_colour"):
        mats["trim"] = pick(TRIMS, form.get("trim_colour"))
    if form.get("roof_colour"):
        mats["roof"] = pick(ROOFS, form.get("roof_colour"))
    if form.get("door_colour"):
        mats["door"] = pick(DOORS, form.get("door_colour"))
    for role, block in (overrides or {}).items():
        if role in mats and material_index(str(block)) is not None:
            mats[role] = str(block)
    for role, need in (("roof_edge", "slab_bottom"), ("sill", "stairs_north"), ("shutter", "trapdoor_open_north")):
        mats[role] = mats["trim"] if shape_ok(need, mats["trim"]) else ("dark_oak_planks" if role != "roof_edge"
                                                                        else mats["roof"])
    if not shape_ok("fence", mats["railing"]):
        mats["railing"] = "dark_oak_planks"
    return mats


def _parts(form: dict) -> list:
    raw = form.get("parts") or form.get("wings") or []
    parts = [p for p in raw if isinstance(p, dict)] or [{"part": "main"}]
    return parts


def _row(form: dict, parts: list, style: str):
    """A row of joined houses (terraced or canal) as one part per house: the model's parts
    when it listed one per house, else copies of its main part. (parts, row) with row None
    when the form describes a single building."""
    n = _int(form.get("units"), 1, 1, 30)
    if n < 2:
        return parts, None
    gables = _choice(form.get("gables"), ("side", "front", "stepped", "curved"),
                     "stepped" if style == "dutch" else "side")
    own = len(parts) == n
    main = next((p for p in parts if p.get("part") == "main"), parts[0])
    out = []
    for i in range(n):
        p = dict(parts[i] if own else main)
        p.update(part="main" if i == 0 else "wing", share=100.0 / n, front_offset="flush", level="ground")
        p["windows_across"] = max(1, min(3, _int(p.get("windows_across"), 2, 1, 40) if own else 2))
        front = dict(p.get("front") if isinstance(p.get("front"), dict) else {})
        front.update(door=True, wide_window=False)
        p["front"] = front
        if p.get("roof") in (None, "", "not_visible", "flat") and gables != "side":
            p["roof"] = "gable"
        if gables != "side":
            p.update(ridge="perpendicular", pitch=p.get("pitch") or "steep")
        out.append(p)
    varied = own and len({_int(p.get("storeys"), 1, 1, 120) for p in out}) > 1 or style == "dutch"
    return out, {"n": n, "gables": gables, "varied": varied}


def _main_index(parts: list) -> int:
    """The part the form calls main; where it calls none, the biggest that is not a tower (by its
    width in metres, else its share, times its storeys): a reading of "wing, tower, wing" puts the
    porch, the portico and the steps on the bigger wing, not on whichever came first."""
    i = next((i for i, p in enumerate(parts) if p.get("part") == "main"), None)
    if i is not None:
        return i
    size = [(_f(p.get("width_m"), 0.0) or _f(p.get("share"), 1.0)) * _int(p.get("storeys"), 1, 1, 120)
            if p.get("part") != "tower" else -1.0 for p in parts]
    return max(range(len(parts)), key=lambda k: size[k]) if parts and max(size) > 0 else 0


def compose(form: dict, style=None, kind=None, measure: dict | None = None, blocks_per_m: float | None = None,
            site: bool = False, adjust: dict | None = None) -> dict:
    """A blueprint-1 dict for the building the form describes (see the module docstring)."""
    form = form if isinstance(form, dict) else {}
    adjust = adjust if isinstance(adjust, dict) else {}
    style, kind = style_of(style), kind_of(kind)
    st, kd = STYLES[style], KINDS[kind]
    parts, row = _row(form, _parts(form), style)
    main_i = _main_index(parts)
    storeys = [_int(p.get("storeys"), 1, 1, 120) for p in parts]
    top_storeys = max(storeys)
    storey_m = kd["storey_m"] if not row else KINDS["townhouse"]["storey_m"]
    kd_w = kd["width_m"] if not row else (UNIT_M * 0.7 * row["n"], UNIT_M * 1.4 * row["n"])
    # --- size in metres: measured, else typical for the kind -------------------------------
    m = measure or {}
    across = sum(max(1, _int(p.get("windows_across"), 2, 0, 40)) for p in parts)
    typical = min(max(kd_w[0], across * 3.0), kd_w[1]) if not row else UNIT_M * row["n"]
    # version 3: the parts' widths in metres, measured against doors and storeys: a better
    # absolute size than MapAnything's from one photo (9.6 m for an 18 m house), whose
    # proportions are kept
    read_w = sum(_f(p.get("width_m")) for p in parts) if not row and all(_f(p.get("width_m")) > 0 for p in parts) else 0.0
    if read_w > 0:
        typical = max(kd_w[0] * 0.5, min(kd_w[1] * 2.0, read_w))
    width_m, height_m, depth_m = typical, 0.0, 0.0
    if _f(m.get("width_m")) > 0:
        # a single photo's scale is rough, its proportions less so: keep the measured ratios,
        # and the measured width only within reach of what such buildings measure
        mw, md = _f(m["width_m"]), _f(m.get("depth_m"))
        if md > 1.3 * mw and kind not in END_ON:     # a view along the street: the long side is the front
            mw, md = md, mw
        # (a row: at least 4 m a house; one photo of a canal row measured 15 m for five houses)
        width_m = max(4.0 * row["n"], min(9.0 * row["n"], mw)) if row else max(kd_w[0] * 0.7, min(kd_w[1] * 1.5, mw))
        if read_w > 0:
            width_m = typical
        height_m = _f(m.get("height_m")) / mw * width_m
        r_dw = md / mw
        if r_dw >= 0.3:
            depth_m = min(3.0, r_dw) * width_m
    height_used = False
    main_roof = str(parts[main_i].get("roof", "gable"))
    if height_m:
        roofy = main_roof not in ("flat", "dome", "not_visible")
        # walls ~80% of the measured height (tested: a share by what crowns the building, a dome
        # 50%, a pitched roof 65%, scored worse against the photos' outlines: 0.529 against 0.544)
        implied = height_m * (0.8 if roofy else 0.95) / max(top_storeys, 1)
        # else the photo cannot give the height (a view up a tower); a pagoda's storeys are tall
        if 2.4 <= implied <= (14.0 if main_roof == "pagoda" else 8.5):
            storey_m, height_used = implied, True
    top_storeys = min(top_storeys, 80)
    total_h_m = storey_m * top_storeys
    roofy = main_roof not in ("flat", "dome", "not_visible")
    aspect = _f(m.get("mask_aspect"))
    if not height_used and aspect > 0 and kind != "skyscraper" and not row:
        # MapAnything gave no usable size: the cut-out's height / width does, within a factor of two of
        # the parts' own metres where the form has them (a cut-out of a mall's entrance only, a dome
        # making a capitol's tall, a storey too few: a 23 m house came out 6 m wide, a 49 m mall 7 m)
        full = total_h_m * (1.25 if roofy else 1.0)
        width_m = max(kd_w[0] * 0.5, min(kd_w[1] * 2.0, full / aspect))
        if read_w > 0:
            width_m = max(typical / 2, min(typical * 2, width_m))
    if kind == "skyscraper":
        width_m = max(width_m, total_h_m / 6.0)        # towers are seldom slimmer than 6 : 1
    if row:
        depth_m = max(6.0, min(14.0, UNIT_M * 1.6))   # a row is as deep as one of its houses
    elif not depth_m or depth_m < 0.35 * width_m * kd["depth"]:
        depth_m = width_m * kd["depth"]
    if kind == "skyscraper":
        depth_m = max(depth_m, 0.6 * width_m)          # a tower, not a slab
    scale = adjust.get("scale") if isinstance(adjust.get("scale"), dict) else {}
    width_m *= _f(scale.get("width"), 1.0) or 1.0
    depth_m *= _f(scale.get("depth"), 1.0) or 1.0
    storey_m *= _f(scale.get("height"), 1.0) or 1.0
    total_h_m = storey_m * top_storeys
    bpm = blocks_per_m or min(2.0, BIG / max(width_m, depth_m, total_h_m * 1.25, 1.0))
    if storey_m * bpm < 4:                             # a storey needs 4 blocks: scale up, keeping proportions
        bpm = 4.0 / storey_m
    storey_h = max(4, min(8, round(storey_m * bpm)))
    width = max(7, round(width_m * bpm))
    depth = max(7, round(depth_m * bpm))
    # --- parts: widths by share (or by their metres), left to right ---------------------------
    shares = [max(5.0, _f(p.get("share"), 100.0 / len(parts))) for p in parts]
    if not row and all(_f(p.get("width_m")) > 0 for p in parts):
        shares = [max(5.0, 100.0 * _f(p["width_m"]) / sum(_f(q["width_m"]) for q in parts)) for p in parts]
    total = sum(shares)
    conflicts = []
    part_adj = adjust.get("parts") if isinstance(adjust.get("parts"), dict) else {}

    def padj(i, key, default):
        v = (part_adj.get(str(i)) or {}).get(key, default) if isinstance(part_adj.get(str(i)), dict) else default
        return _f(v, default)
    # the model's metres as blocks, for heights, levels and offsets: the main block's walls over
    # its eave height (the build's scale comes from the measurement, the model's metres are rough)
    eave_main = _f(parts[main_i].get("eave_m"))
    bpm_v = storeys[main_i] * storey_h / eave_main if 2.0 <= eave_main <= 400 else bpm
    garage_w = max(3, min(6, round(2.5 * bpm_v)))
    masses, openings, features = [], [], []
    base_form = form.get("base") if isinstance(form.get("base"), dict) else {}
    has_base = str(base_form.get("material", "none")).lower() in BASES
    pitches = adjust.get("pitch") if isinstance(adjust.get("pitch"), dict) else {}
    mats0 = palette(form, style, adjust.get("materials") if isinstance(adjust.get("materials"), dict) else None)
    x = 0
    main_depth = depth
    rows = []

    def width_of(i):
        role = _choice(parts[i].get("part"), ("main", "wing", "tower", "annex"), "wing")
        if row:
            return max(5, round(width / row["n"]) + 1)  # + 1: neighbours share their party wall
        return max(3 if role == "tower" else 5, round(width * shares[i] / total))

    def depth_of(i, w):
        """A part's depth: its own depth over width when the form gives both (version 3), else
        the main block's (measured, or typical for the kind) and 0.8 of it for the others."""
        p = parts[i]
        role = _choice(p.get("part"), ("main", "wing", "tower", "annex"), "wing")
        shape = {"round": "round", "octagonal": "octagon", "octagon": "octagon"}.get(str(p.get("shape")).lower(), "box")
        if role == "tower" or shape != "box":
            return w
        dm, wm = _f(p.get("depth_m")), _f(p.get("width_m"))
        if dm > 0 and wm > 0 and not row:
            # w carries the width scale already: the depth follows the depth scale alone
            ratio = dm / wm * (_f(scale.get("depth"), 1.0) or 1.0) / (_f(scale.get("width"), 1.0) or 1.0)
            return max(5, min(4 * w, max(round(0.25 * w), round(w * ratio))))
        return depth if i == main_i or row else max(5, round(main_depth * 0.8))
    main_depth = max(5, round(depth_of(main_i, width_of(main_i)) * padj(main_i, "depth", 1.0)))
    for i, p in enumerate(parts):
        role = _choice(p.get("part"), ("main", "wing", "tower", "annex"), "wing")
        shape = {"round": "round", "octagonal": "octagon", "octagon": "octagon"}.get(str(p.get("shape")).lower(), "box")
        w = width_of(i)
        d = main_depth if i == main_i else max(5, round(depth_of(i, w) * padj(i, "depth", 1.0)))
        front_g, side_g = 0, 0
        if not row and shape == "box":                  # the part's own garage doors (version 3)
            front = p.get("front") if isinstance(p.get("front"), dict) else {}
            front_g = _int(front.get("garage_doors"), 0, 0, 6)
            side_g = _int(p.get("side_garage_doors"), 0, 0, 6)
            for n_g, along, name in ((front_g, "w", "wider"), (side_g, "d", "deeper")):
                need = n_g * (3 + 1) + 1               # 3 blocks a door at least, frames between and at the ends
                have = w if along == "w" else d
                if n_g and have < need:
                    conflicts.append(f"part {i + 1}: {n_g} garage doors need a wall of {need} blocks, it had {have}: "
                                     f"made {need - have} {name}")
                    if along == "w":
                        w = need
                    else:
                        d = need
        if "floor_m" in p or "eave_m" in p:             # version 3: the part's own level and height
            y0 = 0 if i == main_i else round(_f(p.get("floor_m")) * bpm_v)
            below = max(0, -y0)
            eave = _f(p.get("eave_m"))
            walls = (storeys[i] * storey_h if i == main_i or eave <= 0 else max(3, round(eave * bpm_v)))
        else:
            below = 3 if p.get("level") == "half_below" else 0
            y0 = -below
            walls = storeys[i] * storey_h + below
        walls = max(3, round(walls * padj(i, "height", 1.0)))
        y0 += round(padj(i, "level", 0.0))
        below = max(0, -y0)
        others = max([sv for j, sv in enumerate(storeys) if j != i] or [0])
        if role == "tower" and i != main_i and storeys[i] <= others and (kind in TOWER_KINDS or style in TOWER_STYLES):
            if walls < top_storeys * storey_h + storey_h:
                conflicts.append(f"part {i + 1} is a tower no taller than the rest: raised above them")
            walls = max(walls, top_storeys * storey_h + storey_h)       # a tower rises above the rest
        if "forward_m" in p and i != main_i:
            offset = round(_f(p.get("forward_m")) * bpm_v)
        else:
            offset = {"forward": 3, "back": -3}.get(p.get("front_offset"), 0) if i != main_i else 0
        offset += round(padj(i, "forward", 0.0))
        z = main_depth - d + offset
        raw_roof = str(p.get("roof") or "").lower()
        if raw_roof in ("not_visible", "hidden", "none", "unknown", ""):
            roof_kind = "flat" if kind in FLAT_KINDS else (st["roofs"][0] if st["roofs"] else "gable")
        else:
            roof_kind = _choice(raw_roof, ROOF_KINDS, st["roofs"][0] if st["roofs"] else "gable")
        ends = str(p.get("roof_ends") or "").lower()
        if roof_kind == "hip" and ends == "triangle":       # the ends are what the model sees best
            roof_kind = "gable"
        elif roof_kind == "gable" and ends == "slope":
            roof_kind = "hip"
        if form.get("flared_eaves") and roof_kind == "hip":
            roof_kind = "pagoda"
        if form.get("sawtooth") and i == main_i:
            roof_kind = "sawtooth"
        profile, clip, gabled = "round", 0, style in GABLET_STYLES
        if roof_kind == "onion_dome":
            roof_kind, profile = "dome", "onion"
        elif roof_kind == "clipped_gable":
            roof_kind, clip = "gable", 2
        elif roof_kind == "hip_and_gable":
            roof_kind, gabled = "hip", True
        deg = _f(p.get("pitch_deg"), -1.0)
        if deg >= 0:                                      # version 3: the slope in degrees, the nearest the builder has
            pitch = min(PITCHES, key=lambda v: abs(math.degrees(math.atan(v)) - deg))
        else:
            pitch = {"low": 0.5, "medium": 1.0, "steep": 2.0 if style in STEEP_STYLES or kind == "church" else 1.0}.get(
                p.get("pitch"), 1.0)
            if roof_kind in ("gable", "hip", "cross_gable") and pitch == 1.0 and w > 24:
                pitch = 0.5                               # wide roofs read better gentle
        if roof_kind == "barrel":                         # how high the arc rises: a half circle from 40 degrees
            pitch = 1.0 if deg >= 40 else 0.5
        elif roof_kind == "butterfly":
            pitch = 0.25 if 0 <= deg < 18 else 0.5
        ridge = "z" if p.get("ridge") == "perpendicular" else "x"
        seen = p.get("roof_seen") if isinstance(p.get("roof_seen"), dict) else {}
        battlements = bool(form.get("battlements"))
        front_seen = seen.get("front_top") if "front_top" in ROOF_TRUST else None
        if front_seen == "flat" and roof_kind not in ("dome",):
            roof_kind = "flat"
        elif front_seen in ("triangle", "eave", "both") and roof_kind in ("gable", "hip", "gambrel", "mansard", "flat",
                                                                           "shed", "cross_gable"):
            if roof_kind == "flat":                       # the photo shows a pitched roof the form missed
                roof_kind = "hip" if str(seen.get("ends")) == "slope" else "gable"
            ridge = "z" if front_seen == "triangle" else "x"
            if roof_kind == "cross_gable":                # its front gable is built on its own (`_cross_gable`)
                roof_kind = "gable"
        elif front_seen in ("triangle", "eave", "other") and roof_kind in ("barrel", "butterfly"):
            # its curved or V end facing the camera (an arch is "something else" to the question: 5 of 5 barrels
            # answered so faced it, the 3 answered "eave" showed their side), or its side
            ridge = "x" if front_seen == "eave" else "z"
        ends_seen = seen.get("ends") if "ends" in ROOF_TRUST else None
        if roof_kind in ("gable", "hip") and ends_seen in ("triangle", "slope"):
            roof_kind = "gable" if ends_seen == "triangle" else "hip"
        pitch_seen = seen.get("pitch") if "pitch" in ROOF_TRUST else None
        if pitch_seen == "medium" and deg > 40:            # the form's steep where the pass said medium
            pitch_seen = "steep"
        if pitch_seen in ("low", "medium", "steep"):
            # a steep roof as steep as the form's own degrees where they say steep too (45 is 1.0,
            # 55 and more 2.0); the style's guess only where they do not
            steep = min((1.0, 2.0, 3.0), key=lambda v: abs(math.degrees(math.atan(v)) - deg)) if deg > 40 else \
                2.0 if style in STEEP_STYLES else 1.0
            pitch = {"low": 0.25 if 0 <= deg < 18 else 0.5, "medium": 0.75, "steep": steep}[pitch_seen]
        top_seen = seen.get("tower_top") if "tower_top" in ROOF_TRUST else None
        lantern_top = False
        if role == "tower" and top_seen in TOWER_TOPS:
            roof_kind = TOWER_TOPS[top_seen]
            if roof_kind == "cone" and shape == "box":
                roof_kind = "pyramid"
            battlements = top_seen == "battlements"
            lantern_top = top_seen == "lantern"
            profile = "onion" if top_seen == "onion" else profile
        if _f(pitches.get(str(i))) in PITCHES:
            pitch = _f(pitches[str(i)])
        eave = EAVES.get(str(form.get("eaves") or "normal").lower(), 1)
        gablet = max(1, round((d if ridge == "x" else w) / 2 * 0.4)) if gabled and roof_kind in ("hip", "pagoda") else 0
        # a low round roof (an apse's, a rotunda's, a gazebo's) keeps its own slope; a turret's cone is steep
        keep = roof_kind == "cone" and role != "tower" and 0 <= deg < 35
        roof = {"type": roof_kind, "ridge": ridge, "pitch": pitch, "gablet": gablet, "keep_pitch": keep,
                "overhang": (2 if roof_kind == "flat" and eave == 2 and shape == "box" and not row else 0)
                if roof_kind in ("flat", "dome", "spire", "barrel") or shape != "box" or row else eave,
                "battlements": battlements and roof_kind == "flat", "clip": clip, "profile": profile}
        gables = row["gables"] if row else (str(form.get("gables") or "").lower() if i == main_i else "")
        if gables in ("stepped", "curved") and roof_kind == "gable":       # a row's, or one building's front gable
            roof["gable_end"] = "stepped"
        decor = []
        for flag, dec in (("half_timbering", "half_timber"), ("pilasters", "pilasters"), ("buttresses", "buttresses"),
                          ("curtain_wall", "curtain")):
            if form.get(flag) and (role != "tower" or dec in ("curtain",)):
                decor.append(dec)
        if p.get("window_style") == "curtain" and "curtain" not in decor:
            decor.append("curtain")
        if style == "adobe" and roof_kind == "flat":
            decor.append("vigas")
        mid = f"p{i}"
        base_p = str(p.get("base") or "").lower()
        own_base = base_p in BASES
        sh_i = storey_h
        if "eave_m" in p and i != main_i and _f(p.get("eave_m")) > 0:      # its own storeys in its own height
            sh_i = max(4, min(8, walls // max(1, storeys[i])))
        # (the floors under its walls' top: a part the loop made lower kept its upper floors, a car
        # park's decks floating over it)
        floors = sorted({0, *(f for f in (below + k * storey_h for k in range(storeys[i])) if f < walls - 2)}) \
            if sh_i == storey_h else sorted({k * sh_i for k in range(storeys[i]) if k * sh_i < walls - 2})
        plinth = (below + 1) if (own_base or (has_base and base_p in ("", "none") and "base" not in p)) else 0
        if plinth and ("base_m" in p or "base_end_m" in p):
            # version 3: the band's height on its front and on its end wall the photo shows (the
            # far walls as the front); a split level's lower floor often shows on one side only
            end = "west" if i < len(parts) / 2 and len(parts) > 1 else "east"
            front_r = max(1, round(_f(p.get("base_m"), (below + 1) / bpm_v) * bpm_v))
            end_r = max(1, round(_f(p.get("base_end_m"), _f(p.get("base_m"), (below + 1) / bpm_v)) * bpm_v))
            plinth = {"south": front_r, "north": front_r, "east": front_r, "west": front_r, end: end_r}
        mass = {"id": mid, "x": x, "z": z, "w": w, "d": d, "y": y0, "walls": walls, "shape": shape,
                "storey_h": sh_i, "floors": floors, "plinth": plinth, "decor": decor, "roof": roof}
        if own_base:
            mass["materials"] = {"wall_base": BASES[base_p]}
        if row and row["varied"]:
            walls_cycle = [mats0["wall"]] + [b for b in ROW_WALLS if b != mats0["wall"]]
            mass["materials"] = dict(mass.get("materials") or {}, wall=walls_cycle[i % len(walls_cycle)])
        masses.append(mass)
        if lantern_top:                                    # an open lantern under a little dome on the tower's top
            features.append({"kind": "lantern", "mass": mid, "h": max(3, storey_h // 2 + 1), "open": True,
                             "cap": "dome", "shrink": max(1, w // 4)})
        rows.append({"i": i, "p": p, "role": role, "below": below, "storeys": storeys[i], "garages": (front_g, side_g),
                     "cross_at": seen.get("cross_at") if front_seen == "both" else None})
        x += w - 1 if row else w
    # --- cross gables: a gable rising from a part's eave over part of its front --------------------
    for r, mass in zip(rows, list(masses)):
        if r["cross_at"] and mass["roof"]["type"] in ("gable", "hip") and mass["roof"].get("ridge") == "x":
            xg = _cross_gable(mass, r["cross_at"], r["i"])
            if xg:
                masses.append(xg)
                ww = 3 if xg["w"] >= 7 else 1               # a window a storey on its front, centred
                for f in xg["floors"]:
                    sill = 2 if xg["storey_h"] >= 6 else 1
                    if f + sill + 2 < xg["walls"]:
                        openings.append({"mass": xg["id"], "side": "south", "kind": "window", "u": (xg["w"] - ww) // 2,
                                         "y": f + sill, "w": ww, "h": max(2, min(3, xg["storey_h"] - sill - 1))})
            else:
                conflicts.append(f"part {r['i'] + 1}: a gable rising from its eave asked, the part is too narrow for one")
    # --- pagoda: a platform, tiers that step in, each under its own eave, a finial on top ------
    # a Japanese castle's keep stacks whichever hipped roof it was read with (pagoda, hip, hip-and-gable)
    main_type = masses[main_i]["roof"]["type"]
    pagoda = top_storeys >= 3 and ((main_type == "pagoda" and kind in PAGODA_KINDS) or (
        main_type in ("pagoda", "hip") and style in GABLET_STYLES and kind in ("castle", "palace")))
    tier_masses = []
    if pagoda:                                         # tiers as tall as the storeys, past the 8-row cap
        tier_masses = _pagoda(masses, main_i, top_storeys, max(4, round(storey_m * bpm * 0.7)), features, mats0)
    # --- podium: a tall building's wider base ----------------------------------------------------
    pod = form.get("podium") if isinstance(form.get("podium"), dict) else {}
    n_pod = _int(pod.get("storeys"), 0, 0, 12) if kind in TALL_KINDS and not pagoda else 0
    if 0 < n_pod < storeys[main_i] and not row:
        _podium(masses, main_i, rows[main_i], n_pod, storey_h, openings)
    # --- tiers (setbacks) on the main part -------------------------------------------------------
    tiers = [t for t in (form.get("tiers") or []) if _int(t, 0, 0, 120) > 0] if isinstance(form.get("tiers"), list) else []
    if [_int(t) for t in tiers] == list(range(1, len(tiers) + 1)):     # numbered, not counted: equal tiers
        tiers = [max(1, top_storeys // max(len(tiers), 1))] * len(tiers)
    if pagoda:                                         # a pagoda's tiers are its own (`_pagoda`)
        tiers = []
    if len(tiers) >= 2:
        base = masses[main_i]
        total_tiers = sum(_int(t, 1, 1, 120) for t in tiers)
        scale_t = base["walls"] / max(total_tiers, 1)
        y, inset = base["y"], 0
        base["walls"] = max(storey_h, round(_int(tiers[0], 1, 1, 120) * scale_t))
        y += base["walls"]
        for k, t in enumerate(tiers[1:], 1):
            inset += max(1, round(min(base["w"], base["d"]) * 0.12))
            if base["w"] - 2 * inset < 5 or base["d"] - 2 * inset < 5:
                break
            tw = max(storey_h, round(_int(t, 1, 1, 120) * scale_t))
            tier = {"id": f"tier{k}", "x": base["x"] + inset, "z": base["z"] + inset, "w": base["w"] - 2 * inset,
                    "d": base["d"] - 2 * inset, "y": y, "walls": tw, "storey_h": storey_h,
                    "floors": list(range(0, tw, storey_h)), "decor": list(base["decor"]), "roof": dict(base["roof"])}
            masses.append(tier)
            tier_masses.append(tier)
            base["roof"] = {"type": "flat", "overhang": 0, "parapet": 1}
            y += tw
    crown = str(form.get("crown") or "").lower()
    if crown and not pagoda:
        _crown(masses, tier_masses[-1] if tier_masses else masses[main_i], crown, storey_h, features)
    # --- towers, minarets, silos the form lists beyond its parts ---------------------------
    main = masses[main_i]
    towers = form.get("towers") if isinstance(form.get("towers"), dict) else {}
    have = sum(1 for r in rows if r["role"] == "tower")
    extra = max(0, _int(towers.get("count"), 0, 0, 8) - have)
    if extra == 1 and str(towers.get("where") or "").lower() == "centre":
        # one thing on the roof's top read twice: a stacked keep's top tier, or a house's or barn's
        # belvedere also read as a cupola
        if pagoda:
            extra = 0
            conflicts.append("a tower in the middle asked: the stacked roofs' top tier is it")
        elif form.get("cupola") and kind in CUPOLA_KINDS:
            extra = 0
            conflicts.append("a tower in the middle and a cupola asked: one thing on the roof, the cupola kept")
    if form.get("spire") and not have and not extra:
        extra = 1
        towers = dict(towers, where=towers.get("where") or "front", roof="spire")
    tw = max(3, round(min(main["w"], main["d"]) * 0.28))
    tower_h = main["walls"] + max(storey_h, main["walls"] // 2)
    if str(towers.get("where") or "").lower() == "centre" and main["roof"]["type"] not in ("flat", "dome"):
        # a tower in the middle of the main block (a clock tower on a hall, a crossing tower) rises
        # out of its roof: clear of the ridge by a storey, or it is hidden inside the roof
        span = main["d"] if main["roof"].get("ridge", "x") == "x" else main["w"]
        ridge = round((span / 2 + 1) * min(float(main["roof"].get("pitch", 1.0)), 3.0))
        tower_h = max(tower_h, main["walls"] + ridge + max(3, storey_h))
    tshape = {"round": "round", "octagonal": "octagon"}.get(str(towers.get("shape")).lower(), "box")
    troof = _choice(towers.get("roof"), ("cone", "spire", "pyramid", "flat", "dome"), "cone" if tshape != "box" else "pyramid")
    if form.get("spire"):
        troof = "spire"
    where = str(towers.get("where") or "corners").lower()
    spots = {"left": [(main["x"] - tw // 2, main["z"] + main["d"] - tw)],
             "right": [(main["x"] + main["w"] - tw + tw // 2, main["z"] + main["d"] - tw)],
             "centre": [(main["x"] + (main["w"] - tw) // 2, main["z"] + (main["d"] - tw) // 2)],
             "front": [(main["x"] + (main["w"] - tw) // 2, main["z"] + main["d"] - tw + tw // 2)],
             "corners": [(main["x"] - 1, main["z"] + main["d"] - tw + 1), (main["x"] + main["w"] - tw + 1, main["z"] + main["d"] - tw + 1),
                         (main["x"] - 1, main["z"] - 1), (main["x"] + main["w"] - tw + 1, main["z"] - 1)]}.get(where, [])
    if where == "corners" and extra <= 2:
        spots = spots[:2]
    if extra and not spots:
        conflicts.append(f"{extra} towers asked where {where!r}: no place for them, so none were built")
    for k in range(extra):
        if not spots:
            break
        tx, tz = spots[k % len(spots)]
        masses.append({"id": f"tower{k}", "x": tx, "z": tz, "w": tw, "d": tw, "y": main["y"], "walls": tower_h,
                       "shape": tshape, "storey_h": storey_h, "floors": [0],
                       "roof": {"type": troof, "overhang": 0 if troof in ("flat", "spire") else 1,
                                "battlements": bool(form.get("battlements")) and troof == "flat", "pitch": 1.0}})
        openings.append({"mass": f"tower{k}", "side": "south", "kind": "window", "u": tw // 2, "y": tower_h - storey_h + 1,
                         "w": 1, "h": min(3, storey_h - 2), "sill": False,
                         "style": "lancet" if style == "gothic" else "rect"})
    for k in range(_int(form.get("minarets"), 0, 0, 6)):
        mx = main["x"] - 4 if k % 2 == 0 else main["x"] + main["w"] + 1
        mz = main["z"] + main["d"] - 3 if k < 2 else main["z"]
        mh = max(main["walls"] * 2, storey_h * 6)
        masses.append({"id": f"minaret{k}", "x": mx, "z": mz, "w": 3, "d": 3, "shape": "round", "y": main["y"],
                       "walls": mh, "floors": [0], "roof": {"type": "spire", "overhang": 0}})
        features.append({"kind": "gallery", "mass": f"minaret{k}", "y": mh - 3})
    if form.get("silo"):
        sx = main["x"] + main["w"] + 1
        masses.append({"id": "silo", "x": sx, "z": main["z"] + 1, "w": 5, "d": 5, "shape": "round", "y": main["y"],
                       "walls": main["walls"] * 2, "floors": [0], "roof": {"type": "dome"}})
    # --- openings on each part -----------------------------------------------------------------
    win_h = max(2, storey_h - 3)
    door_done = False
    for r, mass in zip(rows, masses):
        p, mid, w = r["p"], mass["id"], mass["w"]
        front = p.get("front") if isinstance(p.get("front"), dict) else {}
        n_top = _int(p.get("windows_across"), 2, 0, 40)
        style_w = _choice(p.get("window_style"), ("rect", "arched", "lancet", "ribbon", "curtain"),
                          st["windows"][0] if st["windows"] else "rect")
        if style_w == "curtain":
            continue                                       # the curtain wall decor is the window
        win_w = 2 if w < 9 else 3
        if style_w == "ribbon":
            win_w = max(3, w - 4)
        n = 1 if style_w == "ribbon" else n_top
        floors_n = max(1, min(r["storeys"], (mass["walls"] - r["below"]) // storey_h))
        for k in range(floors_n):
            row_y = r["below"] + k * storey_h + (2 if storey_h >= 6 else 1)
            if row_y + win_h > mass["walls"]:
                break
            count = n if k > 0 else _int(front.get("ground_windows"), n, 0, 40)
            if k == 0 and (front.get("wide_window") or front.get("door")):
                continue                                   # the ground floor gets its door / wide window below
            if k == 0 and form.get("arcade"):
                continue
            # narrower, so the count the photo shows fits, before fewer
            ww = win_w if style_w == "ribbon" else min(win_w, max(1, (w - 2 - (count + 1)) // max(1, count)))
            for u in _spread(count, ww, w, f"part {r['i'] + 1}, floor {k + 1}: windows across", conflicts):
                o = {"mass": mid, "side": "south", "kind": "window", "u": u, "y": row_y, "w": ww, "h": win_h,
                     "style": style_w if style_w in ("arched", "lancet") else "rect",
                     "shutters": bool(form.get("shutters")) and style_w == "rect"}
                if style_w == "ribbon":
                    o["mullions"] = list(range(3, ww - 1, 3))
                openings.append(o)
        g0 = r["below"] + (2 if storey_h >= 6 else 1)
        g_h = min(win_h, mass["walls"] - g0 - 1)           # short walls (a pagoda tier's) get short windows
        if form.get("arcade"):
            # arches in plaster are cut in the wall (a trim frame round each one darkened a whitewashed front)
            plain = _wall_material(form) in ("stucco", "concrete")
            for u in _positions(max(2, w // 4), 3, w):
                openings.append({"mass": mid, "side": "south", "kind": "arch", "u": u, "y": r["below"] + 1, "w": 3,
                                 "h": min(storey_h - 1, 4), **({"frame": False} if plain else {})})
        elif front.get("door") or front.get("wide_window"):
            u_door = None
            if front.get("door") and (not door_done or row):
                if row:                                    # neighbours' doors in mirrored pairs
                    u_door = 1 if r["i"] % 2 == 0 else w - 2
                else:
                    u_door = (w - 2) // 2 if form.get("symmetric") or len(rows) == 1 else (w - 3 if r["i"] < len(rows) / 2 else 2)
                openings.append({"mass": mid, "side": "south", "kind": "door", "u": max(1, min(u_door, w - 2)),
                                 "y": r["below"] + 1, "w": 2 if form.get("symmetric") and w >= 9 and not row else 1})
                door_done = True
            if front.get("wide_window") and g_h >= 1:
                ww = max(3, min(w - 6, 13))
                u = 1 if u_door is not None and u_door > w / 2 else w - ww - 1 if u_door is not None else (w - ww) // 2
                u = max(1, u)
                door = next((o for o in openings if o["mass"] == mid and o["kind"] == "door"), None)
                if u + ww > w or (door is not None and u < door["u"] + door["w"] and door["u"] < u + ww):
                    # a narrow part (it ran off the wall, or over the door): the longer stretch beside the door
                    spans = [(1, w - 1)] if door is None else [(1, door["u"]), (door["u"] + door["w"], w - 1)]
                    lo_u, hi_u = max(spans, key=lambda s: s[1] - s[0])
                    ww = min(ww, hi_u - lo_u)
                    u = lo_u + max(0, (hi_u - lo_u - ww) // 2)
                if ww >= 1:
                    openings.append({"mass": mid, "side": "south", "kind": "window", "u": u, "y": g0, "w": ww,
                                     "h": g_h, "mullions": list(range(4, ww - 1, 4))})
                else:
                    conflicts.append(f"part {r['i'] + 1}: no room for a wide window beside the door on a wall "
                                     f"{w} blocks long")
            elif g_h >= 1:
                count = _int(front.get("ground_windows"), n_top, 0, 40)
                door_w = next((o["w"] for o in reversed(openings) if o["mass"] == mid and o["kind"] == "door"), 1)
                if u_door is not None and abs(u_door + door_w / 2 - w / 2) <= w / 6:
                    segs = [(0, u_door - 1), (u_door + door_w + 1, w)]      # a door in the middle: both sides
                elif u_door is not None:
                    segs = [(0, u_door - 1) if u_door > w / 2 else (u_door + door_w + 2, w)]
                else:
                    segs = [(0, w)]
                for k, (lo_u, hi_u) in enumerate(segs):
                    n_seg = count // len(segs) + (1 if k < count % len(segs) else 0)
                    # narrower, so the count the photo shows fits, before fewer
                    ww = min(win_w, max(1, (hi_u - lo_u - 2 - (n_seg + 1)) // max(1, n_seg)))
                    for u in _spread(n_seg, ww, hi_u - lo_u, f"part {r['i'] + 1}: ground-floor windows", conflicts):
                        openings.append({"mass": mid, "side": "south", "kind": "window", "u": lo_u + u, "y": g0,
                                         "w": ww, "h": g_h, "shutters": bool(form.get("shutters"))})
        sides = [s for s, keep in (("west", r["i"] == 0), ("east", r["i"] == len(rows) - 1)) if keep] + ["north"]
        for side in sides:                                 # the walls the photo barely shows
            length = mass["d"] if side != "north" else w
            for k in range(floors_n):
                row_y = r["below"] + k * storey_h + (2 if storey_h >= 6 else 1)
                if row_y + win_h > mass["walls"]:
                    break
                for u in _positions(max(1, length // 6), win_w, length):
                    openings.append({"mass": mid, "side": side, "kind": "window", "u": u, "y": row_y, "w": win_w,
                                     "h": win_h, "style": style_w if style_w in ("arched", "lancet") else "rect"})
    # --- the form's features -------------------------------------------------------------------------
    garage = form.get("garage") if isinstance(form.get("garage"), dict) else {}
    gn = _int(garage.get("doors"), 0, 0, 6) + _int(form.get("big_doors"), 0, 0, 6)
    own_garages = [(r, mass) for r, mass in zip(rows, masses) if any(r["garages"])]
    for r, mass in own_garages:
        for n_g, side in ((r["garages"][0], "south"),
                          (r["garages"][1], "west" if r["i"] < len(rows) / 2 and len(rows) > 1 else "east")):
            if not n_g:
                continue
            length = mass["w"] if side == "south" else mass["d"]
            gw = garage_w
            while gw > 3 and n_g * (gw + 1) + 1 > length:
                gw -= 1
            gh = max(3, min(round(2.2 * bpm_v), mass["walls"] - 1, mass.get("storey_h", storey_h) - 1
                            if len(mass["floors"]) > 1 else mass["walls"] - 1))
            us = _spread(n_g, gw, length, f"part {r['i'] + 1}: garage doors on its {side} wall", conflicts)
            openings[:] = [o for o in openings if not (o["mass"] == mass["id"] and o["side"] == side and o["y"] < gh + 1)]
            for u in us:
                openings.append({"mass": mass["id"], "side": side, "kind": "garage", "u": u, "y": 0, "w": gw, "h": gh})
    if gn and own_garages:
        gn = 0                                         # the parts said where theirs are
    if gn:
        gwall = _choice(garage.get("wall"), ("front", "left", "right"), "front")
        big = bool(form.get("big_doors"))
        gw, gh = (6, min(6, masses[main_i]["walls"] - 1)) if big else (5, 4)
        target = rows[0] if gwall == "left" else rows[-1] if gwall == "right" else min(rows, key=lambda r: r["storeys"])
        mass = masses[target["i"]]
        gh = max(2, min(gh, mass["walls"] - 1))            # as tall as the part it is in allows
        side = {"left": "west", "right": "east"}.get(gwall, "south")
        length = mass["d"] if side != "south" else mass["w"]
        if gn * (gw + 1) + 1 > length and side != "south":           # a side wall grows to take them
            need = gn * (gw + 1) + 1
            conflicts.append(f"{gn} garage doors need a wall of {need} blocks, part {target['i'] + 1}'s {side} wall had "
                             f"{length}: made {need - length} deeper")
            mass["z"] -= need - length
            mass["d"] = need
            length = need
        us = _spread(gn, gw, length, f"part {target['i'] + 1}: garage doors on its {side} wall", conflicts)
        openings[:] = [o for o in openings if not (o["mass"] == mass["id"] and o["side"] == side and o["y"] < gh + 1)]
        for u in us:
            openings.append({"mass": mass["id"], "side": side, "kind": "garage", "u": u, "y": 0, "w": gw, "h": gh})
    main_row = rows[main_i]
    porch = str(form.get("porch") or "none").lower()
    if porch in ("front", "wraparound"):
        features.append({"kind": "porch", "mass": main["id"], "side": "south", "u": 0, "w": main["w"],
                         "depth": max(2, round(2.4 * bpm)), "height": storey_h, "wrap": porch == "wraparound"})
    portico = form.get("portico") if isinstance(form.get("portico"), dict) else {}
    if _int(portico.get("columns"), 0, 0, 20) >= 2 and (kind in OPEN_KINDS or form.get("open_sides")):
        portico = {}                                       # its posts are the colonnade (a bandstand's ring)
        conflicts.append("a portico asked on a building open on its sides: its posts are the columns")
    if _int(portico.get("columns"), 0, 0, 20) >= 2 and any(f["kind"] == "porch" and f.get("mass") == main["id"]
                                                            and f["side"] == "south" for f in features):
        # one front read as both: a classical building's is its portico, a house's veranda its porch
        if style in PORTICO_STYLES or kind in ("government", "library_museum", "palace"):
            features[:] = [f for f in features if not (f["kind"] == "porch" and f.get("mass") == main["id"]
                                                       and f["side"] == "south")]
            conflicts.append("a porch and a portico asked on the main front: the portico kept")
        else:
            portico = {}
            conflicts.append("a porch and a portico asked on the main front: the porch kept")
    if _int(portico.get("columns"), 0, 0, 20) >= 2:
        n = _int(portico["columns"], 4, 2, 20)
        pw = min(main["w"] - 2, max(5, n * 3 - 2))
        features.append({"kind": "portico", "mass": main["id"], "side": "south", "u": (main["w"] - pw) // 2, "w": pw,
                         "depth": max(3, round(main["d"] * 0.18)), "columns": n,
                         "pediment": bool(portico.get("pediment", True)),
                         "height": min(main["walls"] - 1, max(storey_h, main["walls"] - (storey_h if main_row["storeys"] > 2 else 0)))})
    dome = form.get("dome") if isinstance(form.get("dome"), dict) else {}
    domed_tower = any(r["role"] == "tower" and mm["roof"]["type"] == "dome" for r, mm in zip(rows, masses))
    if domed_tower and kind not in DOME_KINDS and main["roof"]["type"] != "dome":
        dome = {}                                          # the dome the form names is its tower's (a corner turret's)
    if str(dome.get("size", "none")).lower() in ("small", "large") or main["roof"]["type"] == "dome":
        own = main["roof"]["type"] == "dome"
        profile = main["roof"].get("profile", "round") if own else "round"
        if own:
            main["roof"] = {"type": "flat", "overhang": 0}
        large = str(dome.get("size")).lower() == "large"
        r = max(2, round(min(main["w"], main["d"]) * (0.36 if large else 0.22)))
        drum, squash = (max(2, r // 2) if dome.get("drum") else 0), 1.1
        if own and (kind in OPEN_KINDS or form.get("open_sides")):
            # a dome over posts (a bandstand, a gazebo) is the roof itself: over the whole ring, low,
            # no drum (one read "large, on a drum" stood taller than its columns)
            r, drum, squash = max(2, min(main["w"], main["d"]) // 2), 0, 0.5
        cx, cz = main["x"] + main["w"] // 2, main["z"] + main["d"] // 2
        features.append({"kind": "dome", "x": cx, "z": cz, "r": r, "y": main["y"] + main["walls"] + 1,
                         "drum": drum, "squash": squash, "lantern": profile == "round", "profile": profile})
    n_dormers = _int(form.get("dormers"), 0, 0, 8)
    if n_dormers and not (main["roof"]["type"] in ("gable", "hip", "mansard", "cross_gable")
                          and main["roof"].get("ridge") == "x"):
        conflicts.append(f"{n_dormers} dormers asked: the main block's {main['roof']['type']} roof "
                         f"{'runs front to back' if main['roof'].get('ridge') == 'z' else 'has no slope facing the front'}"
                         ", so none were built")
    elif n_dormers:
        for u in _spread(n_dormers, 2, main["w"] - 2, "dormers", conflicts):
            features.append({"kind": "dormer", "mass": main["id"], "side": "south", "u": u + 1, "w": 2, "h": 2})
    n_bays = _int(form.get("bay_windows"), 0, 0, 4)
    if row and n_bays:                                     # one per house, beside its door
        for r, mass in zip(rows, masses):
            u = mass["w"] - 4 if r["i"] % 2 == 0 else 1
            if 0 <= u <= mass["w"] - 3:
                features.append({"kind": "bay", "mass": mass["id"], "side": "south", "u": u, "w": 3,
                                 "y": r["below"] + 2, "h": max(2, storey_h - 3), "depth": 1})
    else:
        for u in _spread(n_bays, 3, main["w"], "bay windows", conflicts):
            features.append({"kind": "bay", "mass": main["id"], "side": "south", "u": u, "w": 3, "y": main_row["below"] + 2,
                             "h": max(2, storey_h - 3), "depth": 1})
    n_sky = _int(form.get("skylights"), 0, 0, 6)
    sky_on = adjust.get("skylights_on")
    if sky_on is None and form.get("skylights_part") is not None:
        sky_on = _int(form.get("skylights_part"), 1, 1, 40) - 1
    order = sorted(zip(rows, masses), key=lambda rm: rm[0]["i"] != _int(sky_on, -1, -1, 40))
    for r, mass in order:
        if n_sky and mass["roof"]["type"] in ("gable", "hip") and r["role"] != "tower":
            side = "south" if mass["roof"].get("ridge") == "x" else ("east" if r["i"] >= len(rows) / 2 else "west")
            length = mass["w"] if side == "south" else mass["d"]
            for u in _spread(n_sky, 3, length, f"part {r['i'] + 1}: skylights", conflicts):
                features.append({"kind": "skylight", "mass": mass["id"], "side": side, "u": u + 1, "v": 3, "w": 3,
                                 "h": 3})
            n_sky = 0
    if n_sky:
        conflicts.append(f"{n_sky} skylights asked: no gable or hip roof to set them in, so none were built")
    roof_t = main["roof"]["type"]
    span = main["d"] if main["roof"].get("ridge", "x") == "x" else main["w"]
    rise = 1 if roof_t in ("flat", "dome", "sawtooth") else max(2, round(span / 2 * min(main["roof"].get("pitch", 1.0), 2.0)))
    n_chim = _int(form.get("chimneys"), 0, 0, 6)
    if row and n_chim:                                     # a terrace's stacks stand on its party walls
        for r, mass in list(zip(rows, masses))[1:]:
            cz = mass["z"] + mass["d"] // 2 - 1
            features.append({"kind": "chimney", "x": mass["x"] - 1, "z": cz, "w": 2, "d": 2,
                             "top": mass["y"] + mass["walls"] + rise + 2})
    else:
        for k in range(n_chim):
            cx = main["x"] + 1 if k % 2 == 0 else main["x"] + main["w"] - 3
            ridge = main["y"] + main["walls"] + rise
            features.append({"kind": "chimney", "x": cx, "z": main["z"] + main["d"] // 2 - 2, "w": 2, "d": 2, "top": ridge + 2})
    for k in range(_int(form.get("smokestacks"), 0, 0, 6)):
        features.append({"kind": "stack", "x": main["x"] + main["w"] + 3 + 4 * k, "z": main["z"] + 2, "r": 1,
                         "top": main["y"] + main["walls"] * 2 + 6})
    balconies = str(form.get("balconies") or "none").lower()
    if balconies in ("some", "every_floor"):
        for o in [o for o in openings if o["kind"] == "window" and o["side"] == "south" and o["y"] > storey_h]:
            mass = next(mm for mm in masses if mm["id"] == o["mass"])
            if balconies == "some" and (o["u"] < mass["w"] / 3 or o["u"] > 2 * mass["w"] / 3):
                continue
            features.append({"kind": "balcony", "mass": o["mass"], "side": "south", "u": max(0, o["u"] - 1),
                             "w": min(o["w"] + 2, mass["w"] - max(0, o["u"] - 1)), "y": o["y"] - 2, "depth": 1})
    if form.get("awning"):
        for mass in masses[:len(rows)]:
            features.append({"kind": "awning", "mass": mass["id"], "side": "south", "u": 0, "w": mass["w"],
                             "y": storey_h - 1, "depth": 2})
    if form.get("canopy"):
        features.append({"kind": "canopy", "x0": main["x"], "x1": main["x"] + main["w"] - 1, "z0": main["z"] + main["d"] + 3,
                         "z1": main["z"] + main["d"] + 3 + max(6, main["d"]), "top": main["y"] + storey_h})
    if form.get("sails") or kind == "windmill":
        features.append({"kind": "sails", "mass": main["id"], "side": "south"})
    _structures(form, kind, rows, masses, main, main_row, openings, features, storey_h, bpm, conflicts, style)
    # a gallery and a lantern crown the tallest tower where the building has one beside its main
    # block (a lighthouse's keeper's house keeps its roof); one lantern a tower
    towers_m = [mm for r, mm in zip(rows, masses) if r["role"] == "tower"] + \
        [mm for mm in masses if str(mm["id"]).startswith("tower")]
    top_m = max(towers_m, key=lambda mm: mm["y"] + mm["walls"], default=main)
    if form.get("gallery") or kind == "lighthouse":
        features.append({"kind": "gallery", "mass": top_m["id"], "y": top_m["walls"] - 1})
    if (form.get("lantern") or kind == "lighthouse") and not any(
            f["kind"] == "lantern" and f.get("mass") == top_m["id"] for f in features):
        top_m["roof"] = {"type": "flat", "overhang": 0}
        features.append({"kind": "lantern", "mass": top_m["id"], "h": max(2, storey_h // 2)})
    open_ids = {mm["id"] for mm in masses if mm.get("open")}
    openings = [o for o in openings if o["mass"] not in open_ids]          # posts have no windows
    openings = _fit_openings(openings, masses, conflicts)
    bp = {"format": "blueprint-1", "name": f"{style} {kind}", "materials": mats0, "masses": masses,
          "openings": openings, "features": features,
          "meta": {"style": style, "kind": kind, "blocks_per_m": round(bpm, 3), "storey_h": storey_h,
                   "size_m": [round(width_m, 1), round(depth_m, 1), round(total_h_m, 1)], "measured": bool(measure),
                   "measured_height_used": height_used, "parts": len(rows), "row": row,
                   "form_version": 3 if any("eave_m" in p or "width_m" in p for p in parts) else 2,
                   "conflicts": conflicts, "adjust": copy.deepcopy(adjust) if adjust else None}}
    if site:
        xs = [mm["x"] for mm in masses] + [mm["x"] + mm["w"] for mm in masses]
        zs = [mm["z"] for mm in masses] + [mm["z"] + mm["d"] for mm in masses]
        bp["site"] = {"ground": [{"x0": min(xs) - 4, "x1": max(xs) + 4, "z0": min(zs) - 4, "z1": max(zs) + 6,
                                  "top": -1 - max(r["below"] for r in rows)}]}
    if adjust.get("mirror"):
        bp = mirror(bp)
    return bp


def _structures(form: dict, kind: str, rows: list, masses: list, main: dict, main_row: dict, openings: list,
                features: list, storey_h: int, bpm: float, conflicts: list, style: str = "other") -> None:
    """The structures round and on a building the form names: open sides (posts instead of
    walls), a deck, verandas on several floors, entrance steps (the door raised over them), a
    clock face, jettied upper storeys, corner turrets, fire escapes, a roof deck, a rose window,
    a water tank."""
    parts_m = masses[:len(rows)]
    if kind in OPEN_KINDS or form.get("open_sides"):
        for r, mm in zip(rows, parts_m):                   # a car park's stair tower stays a closed core
            if (kind in OPEN_KINDS and r["role"] != "tower") or mm is main:
                mm.update(open=True, bay=max(2, round(3.5 * bpm)), rail=r["storeys"] > 1 or kind == "parking")
    deck = str(form.get("deck") or "none").lower()
    if deck in ("front", "side", "back"):
        side = {"front": "south", "side": "east", "back": "north"}[deck]
        if side == "south" and any(f["kind"] in ("porch", "portico") and f.get("mass") == main["id"] for f in features):
            side = "east"                                  # a porch has the front: the deck goes to the side
        length = main["w"] if side in ("south", "north") else main["d"]
        w = max(4, min(length - 1, round(length * 0.6)))
        features.append({"kind": "deck", "mass": main["id"], "side": side, "u": (length - w) // 2, "w": w,
                         "depth": max(2, round(2.5 * bpm)), "y": main_row["below"],
                         "stairs": "front" if side == "south" else "right"})
    ver = form.get("veranda") if isinstance(form.get("veranda"), dict) else {}
    levels = min(_int(ver.get("levels"), 0, 0, 8), main_row["storeys"])
    if levels >= 2 and any(f["kind"] == "portico" and f.get("mass") == main["id"] for f in features):
        conflicts.append("a veranda and a portico asked on the main front: the portico kept")
        levels = 0
    if levels >= 2:
        features[:] = [f for f in features if not (f["kind"] == "porch" and f.get("mass") == main["id"])]
        depth = max(2, round(2.4 * bpm))
        features.append({"kind": "veranda", "mass": main["id"], "side": "south", "u": 0, "w": main["w"],
                         "depth": depth, "levels": levels})
        if ver.get("wraparound"):
            features.append({"kind": "veranda", "mass": main["id"], "side": "east", "u": 0, "w": main["d"],
                             "depth": depth, "levels": levels})
    elif levels == 1 and not any(f["kind"] in ("porch", "portico") and f.get("mass") == main["id"] for f in features):
        features.append({"kind": "porch", "mass": main["id"], "side": "south", "u": 0, "w": main["w"],
                         "depth": max(2, round(2.4 * bpm)), "height": storey_h, "wrap": bool(ver.get("wraparound"))})
    steps = str(form.get("steps") or "none").lower()
    if steps in ("stoop", "wide"):
        # a row's houses each their own stoop; otherwise the main block's front door
        hosts = [mm for mm in parts_m] if len(rows) > 1 and steps == "stoop" and form.get("units") else [main]
        rise = max(2, round((1.6 if steps == "stoop" else 1.0) * bpm))     # a stoop: a raised parlour floor
        built = 0
        if hosts == [main] and not any(o["mass"] == main["id"] and o["kind"] == "door" and o["side"] == "south"
                                       for o in openings):              # the part that has the front door
            hosts = [mm for mm in parts_m if any(o["mass"] == mm["id"] and o["kind"] == "door" and o["side"] == "south"
                                                 for o in openings)][:1] or [main]
        for host in hosts:
            door = next((o for o in openings if o["mass"] == host["id"] and o["kind"] == "door"
                         and o["side"] == "south"), None)
            if door is None or door["y"] + rise - 1 + 3 > host["walls"]:
                continue
            door["y"] += rise - 1                          # the door stands on the top step
            top = door["y"] - 1
            portico = next((f for f in features if f["kind"] == "portico" and f.get("mass") == host["id"]), None)
            if steps == "wide" and portico is not None:
                u, w = portico["u"], portico["w"]
                portico["lift"] = top
            elif steps == "wide":
                u = max(0, door["u"] - 3)
                w = min(host["w"] - u, door["w"] + 6)
            else:
                u, w = door["u"], door["w"]
            features.append({"kind": "steps", "mass": host["id"], "side": "south", "u": u, "w": w, "top": top,
                             "rise": rise, "rail": steps == "stoop"})
            plinth = host["plinth"] if isinstance(host["plinth"], dict) else {s: host["plinth"] for s in
                                                                              ("south", "north", "east", "west")}
            host["plinth"] = dict(plinth, south=max(int(plinth.get("south", 0)), rise - 1))
            built += 1
        if not built:
            conflicts.append(f"{steps} steps asked: no front door with room to raise it over them, so none were built")
    towers_m = [mm for r, mm in zip(rows, parts_m) if r["role"] == "tower"] + \
        [mm for mm in masses if str(mm["id"]).startswith("tower")]
    top_m = max(towers_m, key=lambda mm: mm["y"] + mm["walls"], default=main)
    if form.get("clock"):
        features.append({"kind": "clock", "mass": top_m["id"], "side": "south", "r": 2 if top_m["w"] >= 7 else 1})
    if form.get("cupola") and main["roof"]["type"] not in ("dome", "spire", "cone"):
        features.append({"kind": "lantern", "mass": main["id"], "h": max(3, storey_h // 2 + 1), "open": False,
                         "cap": "pyramid", "w": 5 if min(main["w"], main["d"]) >= 16 else 3, "on_roof": True})
    if form.get("bell_gable"):                             # over the front gable's peak, or the front wall's top
        features.append({"kind": "bell_gable", "mass": main["id"], "side": "south", "w": max(5, min(main["w"] - 2, 9)),
                         "h": max(5, storey_h + 1), "bells": 2 if main["w"] >= 9 else 1})
    if form.get("marquee"):                                # over the main door, a few blocks either side
        door = next((o for o in openings if o["mass"] == main["id"] and o["kind"] == "door" and o["side"] == "south"),
                    None)
        mu = max(0, (door["u"] if door else main["w"] // 2) - 3)
        mw = min(main["w"] - mu, (door["w"] if door else 2) + 6)
        features.append({"kind": "marquee", "mass": main["id"], "side": "south", "u": mu, "w": mw,
                         "y": (door["y"] if door else main_row["below"] + 1) + 3, "depth": 3, "h": 2})
    if kind == "greenhouse" or form.get("glass_roof"):     # glass in a frame: the roof, and a glasshouse's walls
        glass_parts = parts_m if kind == "greenhouse" else [main]
        for mm in glass_parts:
            mm["materials"] = dict(mm.get("materials") or {}, roof="glass", roof_edge="quartz_block")
            if kind == "greenhouse" and "curtain" not in mm.get("decor", []):
                mm["decor"] = list(mm.get("decor") or []) + ["curtain"]
    if form.get("jetty") and main_row["storeys"] >= 2 and main.get("shape", "box") == "box":
        # a timber-framed house over its street; a modern block's floors step out all round
        sides = ["south", "north", "east", "west"] if style in MODERN_STYLES else ["south"]
        main.update(jetty=1, jetty_sides=sides)
    n_turrets = _int(form.get("corner_turrets"), 0, 0, 4)
    if n_turrets:
        corners = ["front_right"] if n_turrets == 1 else ["front_left", "front_right", "back_left", "back_right"][:n_turrets]
        for c in corners:
            features.append({"kind": "turret", "mass": main["id"], "corner": c, "r": 2 if main["w"] >= 12 else 1,
                             "start": max(1, main["walls"] - storey_h - 1), "top": max(3, storey_h // 2 + 2)})
    if form.get("fire_escapes") and main["walls"] > 2 * storey_h:
        w = min(5, main["w"] - 2)
        features.append({"kind": "fire_escape", "mass": main["id"], "side": "south", "u": max(1, main["w"] // 4 - w // 2),
                         "w": w, "depth": 2})
    if form.get("roof_deck"):
        rt = main["roof"]["type"]
        if rt == "flat":
            main["roof"]["rail"] = True
        elif rt in ("hip", "pyramid"):
            main["roof"].update(deck=2, rail=True)
        else:
            conflicts.append(f"a roof deck asked: the main block's {rt} roof has no flat top for one")
    if form.get("rose_window"):
        s = 5 if main["w"] >= 11 else 3
        gable = main["roof"]["type"] in ("gable", "cross_gable") and main["roof"].get("ridge") == "z"
        y = main["walls"] - s + 2 if gable else max(1, main["walls"] - s - 1)
        openings.append({"mass": main["id"], "side": "south", "kind": "round", "u": (main["w"] - s) // 2, "y": y,
                         "w": s, "h": s})
    if form.get("tank"):
        if kind == "tower":                                # a water tower: the tank on its top
            if _wall_material(form) not in ("brick", "stone", "concrete", "stucco"):
                # on legs (steel or timber); a brick or stone shaft carries its tank itself
                main.update(open=True, bay=max(main["w"], main["d"]), rail=False, plinth=0, floors=[0])
            main["roof"] = {"type": "flat", "overhang": 0, "parapet": 0}
            # the tank is the tower's top: a crown or a cupola asked there would stand inside it
            n_top = len(masses) + len(features)
            masses[:] = [mm for mm in masses if mm["id"] not in ("crown", "spire")]
            features[:] = [f for f in features if not (f["kind"] == "lantern" and f.get("mass") == main["id"])]
            if len(masses) + len(features) < n_top:
                conflicts.append("a crown or a cupola asked on a water tower: its tank is its top")
            features.append({"kind": "tank", "mass": main["id"], "r": max(2, min(main["w"], main["d"]) // 2 + 1),
                             "h": max(3, storey_h), "y": 0, "legs": 4})
        else:                                              # one standing on the roof
            features.append({"kind": "tank", "mass": main["id"], "r": 2, "h": 3, "y": 2, "legs": 4})


def _cross_gable(mass: dict, at: str, i: int) -> dict | None:
    """A front gable on a part whose ridge runs along its front: a narrow mass a block proud of
    the front wall, as tall as the part's walls, under a gable roof running back, deep enough
    that its ridge dies into the part's roof; at its left, centre or right. None when the part
    is too narrow for one."""
    w, d = mass["w"], mass["d"]
    gw = max(5, min(w - 4, round(w * 0.36)))
    gw -= (gw + 1) % 2                                 # odd: a peak, not a flat ridge cap
    if gw < 5 or w < 9:
        return None
    u = {"left": 1, "right": w - gw - 1}.get(str(at), (w - gw) // 2)
    pm = max(0.25, float(mass["roof"].get("pitch", 1.0)))
    pg = max(1.0, pm)
    dz = min(d - 2, max(3, int(-(-(gw / 2.0 * pg) // pm)) + 1))   # ceil: the gable's ridge dies into the roof
    return {"id": f"x{i}", "x": mass["x"] + u, "z": mass["z"] + d - dz, "w": gw, "d": dz + 1, "y": mass["y"],
            "walls": mass["walls"], "storey_h": mass.get("storey_h", 5), "floors": list(mass.get("floors") or [0]),
            "plinth": mass.get("plinth", 0), "decor": list(mass.get("decor") or []),
            **({"materials": dict(mass["materials"])} if mass.get("materials") else {}),
            "roof": {"type": "gable", "ridge": "z", "pitch": pg, "overhang": 1}}


def _pagoda(masses: list, main_i: int, top_storeys: int, th: int, features: list, mats: dict) -> list:
    """A pagoda in place of the main mass: a stone platform, then up to 9 tiers stepping in,
    each with walls `th` rows high under a wide eave whose slope runs in to the next tier's
    walls (the `skirt`), a railed gallery round each upper tier, a full roof and a finial on
    the last."""
    base = masses[main_i]
    n = max(2, min(top_storeys, 9))
    o = 2                                              # the eaves stand out this far
    shrink = max(1, round(base["w"] * 0.05))
    run = o + shrink                                   # from an eave's edge to the next tier's wall
    platform = {"id": "platform", "x": base["x"] - 2, "z": base["z"] - 2, "w": base["w"] + 4, "d": base["d"] + 4,
                "y": base["y"], "walls": 2, "floors": [0], "plinth": 2, "corner_trim": False,
                "roof": {"type": "flat", "overhang": 0, "parapet": 0}}
    y = base["y"] + 2
    tiers = []
    column = {"trim": mats["column"]} if "column" in mats else {}
    for k in range(n):
        s = k * shrink
        w, d = base["w"] - 2 * s, base["d"] - 2 * s
        if min(w, d) < 5:
            break
        tier = base if k == 0 else {"id": f"tier{k}", "storey_h": th, "shape": "box"}
        tier.update(x=base["x"] + s if k else base["x"], z=base["z"] + s if k else base["z"], w=w, d=d, y=y, walls=th,
                    floors=[0], plinth=0, decor=["pilasters"], storey_h=th, shape="box",
                    roof={"type": "pagoda", "pitch": 0.5, "overhang": o, "skirt": run + 1})
        if column:
            tier["materials"] = dict(tier.get("materials") or {}, **column)
        if k:
            masses.append(tier)
            features.append({"kind": "gallery", "mass": tier["id"], "y": 0})
        tiers.append(tier)
        y += th + run // 2
    top = tiers[-1]
    top["roof"] = {"type": "pagoda", "pitch": 1.0 if top["w"] <= 15 else 0.5, "overhang": o}
    features.append({"kind": "finial", "mass": top["id"]})
    masses.append(platform)
    return tiers[1:]


def _podium(masses: list, main_i: int, row: dict, n_pod: int, storey_h: int, openings: list) -> None:
    """The main part's lowest storeys as a wider base, the tower standing on it, with shop
    windows and a door in the base's front."""
    base = masses[main_i]
    g = max(2, round(min(base["w"], base["d"]) * 0.12))
    ph = n_pod * storey_h
    podium = {"id": "podium", "x": base["x"] - g, "z": base["z"] - g, "w": base["w"] + 2 * g, "d": base["d"] + 2 * g,
              "y": base["y"], "walls": ph, "storey_h": storey_h, "floors": list(range(0, ph, storey_h)),
              "plinth": base["plinth"], "decor": [], "roof": {"type": "flat", "overhang": 0, "parapet": 1}}
    base.update(y=base["y"] + ph, walls=max(storey_h, base["walls"] - ph), plinth=0)
    base["floors"] = sorted({0, *range(0, base["walls"], storey_h)})
    row["storeys"] = max(1, row["storeys"] - n_pod)
    row["below"] = 0
    row["p"] = dict(row["p"], front={})                # the door is the podium's; the tower has windows all the way
    masses.append(podium)
    w = podium["w"]
    win_h = max(2, storey_h - 2)
    openings.append({"mass": "podium", "side": "south", "kind": "door", "u": w // 2 - 1, "y": 1, "w": 2})
    for k in range(n_pod):
        for u in _positions(max(1, w // 5), 3, w):
            if k == 0 and w // 2 - 3 <= u <= w // 2 + 1:
                continue
            openings.append({"mass": "podium", "side": "south", "kind": "window", "u": u, "y": k * storey_h + 1, "w": 3,
                             "h": win_h})


def _crown(masses: list, top: dict, crown: str, storey_h: int, features: list) -> None:
    """How a tall building ends, on its top mass: flat with a plant room, stepped, a spire
    on a stepped top, a pyramid, a dome, an antenna, or pinnacles on the corners."""
    ey = top["y"] + top["walls"] + 1
    if crown in ("pyramid", "dome", "flat", "antenna", "stepped", "spire", "pinnacles"):
        top["roof"] = {"pyramid": {"type": "pyramid", "pitch": 2.0, "overhang": 0}, "dome": {"type": "dome"}}.get(
            crown, {"type": "flat", "overhang": 0, "parapet": 1})
    if crown == "flat" and min(top["w"], top["d"]) >= 12:      # the plant room on a modern tower's roof
        bw, bd = top["w"] // 3, top["d"] // 3
        masses.append({"id": "crown", "x": top["x"] + (top["w"] - bw) // 2, "z": top["z"] + (top["d"] - bd) // 2,
                       "w": bw, "d": bd, "y": ey, "walls": max(2, storey_h // 2 + 1), "floors": [0],
                       "corner_trim": False, "roof": {"type": "flat", "overhang": 0}})
    if crown in ("stepped", "spire") and top["w"] > 9 and top["d"] > 9:
        step = {"id": "crown", "x": top["x"] + 2, "z": top["z"] + 2, "w": top["w"] - 4, "d": top["d"] - 4, "y": ey,
                "walls": storey_h if crown == "stepped" else storey_h + storey_h // 2, "floors": [0],
                "roof": {"type": "flat", "overhang": 0}}
        masses.append(step)
        if crown == "spire":
            sw = max(3, round(step["w"] * 0.3)) | 1
            masses.append({"id": "spire", "x": step["x"] + (step["w"] - sw) // 2, "z": step["z"] + (step["d"] - sw) // 2,
                           "w": sw, "d": sw, "y": step["y"] + step["walls"] + 1, "walls": storey_h, "floors": [0],
                           "roof": {"type": "spire", "overhang": 0}})
    elif crown == "spire":                             # too slim to step: a spire on the top itself
        top["roof"] = {"type": "spire", "overhang": 0}
    if crown == "antenna":
        cx, cz = top["x"] + top["w"] // 2, top["z"] + top["d"] // 2
        features.append({"kind": "stack", "x": cx, "z": cz, "r": 0, "y": top["y"] + top["walls"],
                         "top": top["y"] + top["walls"] + max(8, top["walls"] // 4)})
    if crown == "pinnacles" and min(top["w"], top["d"]) >= 9:
        for k, (px, pz) in enumerate(((top["x"], top["z"]), (top["x"] + top["w"] - 3, top["z"]),
                                      (top["x"], top["z"] + top["d"] - 3), (top["x"] + top["w"] - 3, top["z"] + top["d"] - 3))):
            masses.append({"id": f"pinnacle{k}", "x": px, "z": pz, "w": 3, "d": 3, "y": ey,
                           "walls": max(2, storey_h // 2), "floors": [0], "roof": {"type": "spire", "overhang": 0}})


def mirror(bp: dict) -> dict:
    """The blueprint flipped left to right (along x; the front stays south): masses,
    openings, features and site trade sides, and walls east and west swap."""
    bp = copy.deepcopy(bp)
    ms = bp["masses"]
    s = min(m["x"] for m in ms) + max(m["x"] + m["w"] - 1 for m in ms)      # x -> s - x
    swap = {"east": "west", "west": "east"}
    by_id = {}
    for m in ms:
        m["x"] = s - (m["x"] + m["w"] - 1)
        r = m.get("roof") or {}
        if r.get("high") in swap:
            r["high"] = swap[r["high"]]
        if isinstance(m.get("plinth"), dict):
            m["plinth"] = {swap.get(k, k): v for k, v in m["plinth"].items()}
        by_id[m["id"]] = m
    for o in bp.get("openings", []):
        m = by_id.get(o.get("mass"))
        if m is None or "u" not in o:
            continue
        length = m["w"] if o["side"] in ("south", "north") else m["d"]
        o["u"] = length - o["u"] - o.get("w", 1)
        if o.get("mullions"):
            o["mullions"] = sorted(o["w"] - 1 - t for t in o["mullions"])
        o["side"] = swap.get(o["side"], o["side"])
    for f in bp.get("features", []):
        m = by_id.get(f.get("mass"))
        if "side" in f:
            if m is not None and "u" in f and "w" in f:
                length = m["w"] if f["side"] in ("south", "north") else m["d"]
                f["u"] = length - f["u"] - f["w"]
            f["side"] = swap.get(f["side"], f["side"])
        if f["kind"] == "chimney":
            f["x"] = s - (f["x"] + f.get("w", 2) - 1)
        elif f["kind"] == "stack" or (f["kind"] == "dome" and "x" in f):
            f["x"] = s - f["x"]
        elif f["kind"] == "canopy":
            f["x0"], f["x1"] = s - f["x1"], s - f["x0"]
        if f["kind"] == "porch" and f.get("wrap"):
            f["wrap_side"] = "left" if f.get("wrap_side", "right") == "right" else "right"
    site = bp.get("site") or {}
    for g in site.get("ground", []) + site.get("boxes", []):
        g["x0"], g["x1"] = s - g["x1"], s - g["x0"]
    for b in site.get("bushes", []) + site.get("trees", []):
        b["x"] = s - b["x"]
    if bp.get("meta") is not None:
        bp["meta"]["mirrored"] = True
    return bp


def _fit_openings(openings: list, masses: list, conflicts: list) -> list:
    """The openings that fit their walls as `spec.load` checks them; one that does not (a part a
    few blocks wide, its door and shop window crowding it, when a photo's size is small) is dropped
    and noted, so no reading makes a blueprint that fails to load."""
    by_id = {mm["id"]: mm for mm in masses}
    kept = []
    for o in openings:
        mm = by_id.get(o.get("mass"))
        if mm is None or any(not isinstance(o.get(k), int) for k in ("u", "y", "w", "h")):
            kept.append(o)                             # spec.load names it
            continue
        length = mm["w"] if o["side"] in ("south", "north") else mm["d"]
        rows = mm.get("walls")
        off = o["u"] < 0 or o["w"] < 1 or o["u"] + o["w"] > length or o["y"] < 0 or o["h"] < 1 or (
            rows is not None and o["kind"] != "round" and o["y"] + o["h"] > rows)
        if off:
            conflicts.append(f"a {o['kind']} {o['w']} wide at {o['u']} does not fit the {mm['id']} {o['side']} "
                             f"wall ({length} blocks)")
        else:
            kept.append(o)
    return kept


def _spread(n: int, w: int, length: int, what: str, conflicts: list) -> list:
    """`_positions`, with a conflict noted when fewer than `n` fit."""
    us = _positions(n, w, length)
    if n > len(us):
        conflicts.append(f"{what}: {n} asked, {len(us)} fit on a wall {length} blocks long")
    return us


def _positions(n: int, w: int, length: int, margin: int = 1) -> list:
    """Left ends of `n` openings `w` wide spread evenly along a wall, at least `margin` in."""
    room = length - 2 * margin
    if n <= 0 or room < w:
        return []
    n = min(n, max(1, (room + 1) // (w + 1)))
    gap = (room - n * w) / (n + 1)
    return [margin + round(gap * (i + 1) + w * i) for i in range(n)]
