"""The evaluator's corrections, and only these. Each one changes the form or compose's
`adjust` (`compose` turns both into blocks), so a correction can never ask the builder for
something it cannot build. The vision model's critique is asked in this list's terms
(`roof_menu`, `FEATURE_ON`, the styles and kinds) and its answers are mapped onto it
(`from_critique`); the computer measures propose their own (`from_measures`). What the model
asks for beyond the list, a correction that hits a limit (no roof steeper than 72 degrees),
and one that changes no block are logged as capability gaps: the parts to build next.

A state is {"form", "style", "kind", "adjust"}; an edit is {"op": ..., ...}. Corrections of one
part (its height, depth, level, how far forward, its garage doors) say which: "part" is the
form's part number, "shown" the number the critique saw it under.
"""
from __future__ import annotations

import copy
from collections import Counter

from .compose import ROOF_KINDS, WALLS2, kind_of, style_of
from .spec import wall_rows
from .styles import KINDS, STYLES

PITCH_STEPS = (0.25, 0.5, 0.75, 1.0, 2.0, 3.0)
PITCHED = ("gable", "cross_gable", "hip", "shed", "mansard", "gambrel", "pyramid", "pagoda")   # roofs a pitch changes
PITCH_NAMES = {0.25: "very low (14 degrees)", 0.5: "low (27 degrees)", 0.75: "medium (37 degrees)",
               1.0: "steep (45 degrees)", 2.0: "very steep (63 degrees)", 3.0: "very steep (72 degrees)"}
SCALE_RANGE = (0.4, 2.5)
SHIFT_RANGE = 16          # blocks: how far a part's level or front may move
# a feature switched on (the values a form gives it), and off
FEATURE_ON = {"porch": "front", "portico": {"columns": 4, "pediment": True}, "garage": {"doors": 1, "wall": "front"},
              "dormers": 2, "bay_windows": 1, "chimneys": 1, "shutters": True, "skylights": 2, "balconies": "some",
              "towers": {"count": 1, "shape": "square", "roof": "pyramid", "where": "corners"}, "spire": True,
              "buttresses": True, "dome": {"size": "small", "drum": False}, "minarets": 2, "pilasters": True,
              "arcade": True, "curtain_wall": True, "tiers": "setbacks", "crown": "stepped", "half_timbering": True,
              "battlements": True, "smokestacks": 1, "sawtooth": True, "big_doors": 1, "silo": True,
              "flared_eaves": True, "awning": True, "canopy": True, "sails": True, "gallery": True, "lantern": True,
              "podium": {"storeys": 3}, "units": 4, "gables": "front", "deck": "side",
              "veranda": {"levels": 2, "wraparound": False}, "steps": "stoop", "clock": True, "corner_turrets": 2,
              "eaves": "deep", "fire_escapes": True, "roof_deck": True, "open_sides": True, "rose_window": True,
              "tank": True, "jetty": True, "marquee": True, "glass_roof": True, "bell_gable": True, "cupola": True}
FEATURE_OFF = {"porch": "none", "portico": {"columns": 0}, "garage": {"doors": 0, "wall": "none"},
               "balconies": "none", "towers": {"count": 0}, "dome": {"size": "none"}, "tiers": [], "crown": "flat",
               "podium": {"storeys": 0}, "units": 1, "gables": "side", "deck": "none", "veranda": {"levels": 0},
               "steps": "none", "eaves": "normal"}
# what the model may call a feature, as the list's name
FEATURE_SYNONYMS = {"turret": "towers", "turrets": "towers", "tower": "towers", "steeple": "spire", "columns": "portico",
                    "colonnade": "portico", "veranda": "porch", "verandah": "porch", "balcony": "balconies",
                    "dormer": "dormers", "chimney": "chimneys", "bay_window": "bay_windows", "bay": "bay_windows",
                    "skylight": "skylights", "minaret": "minarets", "setbacks": "tiers", "arches": "arcade",
                    "garage_door": "garage", "garage_doors": "garage", "cupola": "cupola", "belvedere": "cupola",
                    "smokestack": "smokestacks",
                    "glass_facade": "curtain_wall", "crenellations": "battlements", "pinnacles": "crown",
                    "row_of_houses": "units", "row": "units", "terrace": "units", "terraced_houses": "units",
                    "stoop": "steps", "stairs": "steps", "staircase": "steps", "entrance_steps": "steps",
                    "clock_face": "clock", "clocks": "clock", "corbelled_turrets": "corner_turrets",
                    "widows_walk": "roof_deck", "roof_terrace": "roof_deck", "rooftop_terrace": "roof_deck",
                    "fire_escape": "fire_escapes", "water_tank": "tank", "water_tower": "tank",
                    "rose": "rose_window", "oculus": "rose_window", "round_window": "rose_window",
                    "overhang": "eaves", "overhangs": "eaves", "deep_eaves": "eaves", "decking": "deck",
                    "open_walls": "open_sides", "posts": "open_sides", "gallery_porch": "veranda",
                    "two_storey_porch": "veranda", "walkways": "veranda", "jettied": "jetty",
                    "overhanging_floors": "jetty", "jetties": "jetty", "espadana": "bell_gable", "belfry_wall": "bell_gable",
                    "bell_wall": "bell_gable", "bell_tower_wall": "bell_gable"}


def initial(read: dict) -> dict:
    """The state the local model's reading gives."""
    return {"form": copy.deepcopy(read.get("form") or {}), "style": style_of(read.get("style")),
            "kind": kind_of(read.get("kind")), "adjust": {}}


def _parts(form: dict) -> list:
    raw = form.get("parts")
    if not isinstance(raw, list) or not any(isinstance(p, dict) for p in raw):
        form["parts"] = [{"part": "main"}]
    form["parts"] = [p for p in form["parts"] if isinstance(p, dict)]
    return form["parts"]


def built_parts(bp: dict) -> list:
    """[(part number, its mass)] in the order a viewer sees them, left to right."""
    out = [(int(m["id"][1:]), m) for m in bp["masses"] if m["id"][:1] == "p" and m["id"][1:].isdigit()]
    out.sort(key=lambda im: im[1]["x"])
    return out


def describe_state(state: dict, bp: dict) -> tuple:
    """(text, numbers): the build as the critique sees it, its parts numbered 1, 2, ... left to
    right as drawn, and {shown number: part number}."""
    numbers, lines = {}, []
    form = state["form"]
    fparts = [p for p in (form.get("parts") or []) if isinstance(p, dict)]
    main = next((i for i, p in enumerate(fparts) if p.get("part") == "main"), 0)
    row = (bp.get("meta") or {}).get("row")
    for k, (i, m) in enumerate(built_parts(bp), 1):
        # a row built from one described house: every house is that part
        numbers[k] = main if row and len(fparts) != row["n"] else i
        p = fparts[numbers[k]] if numbers[k] < len(fparts) else {}
        r = m["roof"]
        pitch = PITCH_NAMES.get(float(r.get("pitch", 1.0)), "") if r["type"] not in ("flat", "dome") else ""
        storeys = max(1, len(m.get("floors") or []) or wall_rows(m) // max(1, int(m.get("storey_h", 5))))
        garages = Counter(o["side"] for o in bp.get("openings", []) if o.get("mass") == m["id"] and o["kind"] == "garage")
        where = {"south": "front", "east": "right end", "west": "left end", "north": "back"}
        lines.append(f"- part {k}: {'house in the row' if row else p.get('part', 'main')}, {storeys} storeys, "
                     f"{r['type']} roof" + (f", {pitch}" if pitch else "") +
                     f"; {m['w']} wide, {m['d']} deep, walls {wall_rows(m)} high (blocks)" +
                     (f", its floor {-m['y']} below the others'" if m["y"] < 0 else
                      f", its floor {m['y']} above the others'" if m["y"] > 0 else "") +
                     "".join(f", {n} garage door{'s' if n > 1 else ''} in its {where[s]}" for s, n in garages.items()))
    feats = [f for f in FEATURE_ON if _is_on(f, form.get(f))]
    text = (f"Style {state['style']}, kind {state['kind']}.\n" + "\n".join(lines) +
            f"\nFeatures: {', '.join(feats) if feats else 'none'}.")
    return text, numbers


def _is_on(name: str, v) -> bool:
    """Whether a form's feature field says the building has it."""
    if isinstance(v, dict):
        return any(_is_on(name, x) for k, x in v.items() if k in ("columns", "doors", "count", "size", "storeys"))
    if isinstance(v, str):
        return v.lower() not in ("", "none", "no", "false", "flat", "side", "0")
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    if isinstance(v, bool) or v is None:
        return bool(v)
    try:
        return float(v) > (1 if name == "units" else 0)
    except (TypeError, ValueError):
        return False


def roof_menu() -> str:
    return "|".join(ROOF_KINDS)


def _roof_word(t) -> str:
    """A roof type as the critique writes it ("onion dome", "barrel roof") in the builder's words."""
    t = str(t or "").strip().lower().replace(" ", "_").replace("-", "_")
    return t[:-5] if t.endswith("_roof") and t[:-5] in ROOF_KINDS else t


def feature_menu() -> list:
    """The features the critique may call missing or extra: the form's, a row of houses by
    a plain name, and not "gables" (how a row's roofs meet the street, not a thing to add)."""
    return ["row_of_houses" if f == "units" else f for f in FEATURE_ON if f != "gables"]


def apply(state: dict, edit: dict, bp: dict | None = None) -> tuple:
    """(new state, note): note is None, or why the edit could not do all it asked (a limit),
    or "invalid: ..." when it cannot apply at all."""
    s = copy.deepcopy(state)
    form, adj = s["form"], s["adjust"]
    op = edit.get("op")
    parts = _parts(form)
    i = edit.get("part", 0)
    if op in ("storeys", "roof", "pitch", "ridge", "share", "remove_part", "window_style", "skylights_on", "part_height",
              "part_depth", "part_level", "part_forward", "garage_doors", "part_base"):
        if not isinstance(i, int) or not 0 <= i < len(parts):
            return state, f"invalid: no part {i}"
    if op == "style":
        if edit.get("value") not in STYLES:
            return state, f"invalid: no style {edit.get('value')!r}"
        s["style"] = edit["value"]
    elif op == "kind":
        if edit.get("value") not in KINDS:
            return state, f"invalid: no kind {edit.get('value')!r}"
        s["kind"] = edit["value"]
    elif op == "storeys":
        n0 = int(parts[i].get("storeys") or 1)
        n = n0 + int(edit.get("delta", 0))
        if n < 1:
            return state, "limit: a part needs at least 1 storey"
        parts[i]["storeys"] = min(120, n)
        if isinstance(parts[i].get("eave_m"), (int, float)) and parts[i]["eave_m"] > 0:
            # its height in metres with it (version 3): compose sizes a part by its eave, and the
            # main block's eave is the ruler for the others
            parts[i]["eave_m"] = round(float(parts[i]["eave_m"]) * min(120, n) / max(1, n0), 2)
    elif op == "roof":
        if edit.get("value") not in ROOF_KINDS:
            return state, f"invalid: no roof {edit.get('value')!r}"
        parts[i]["roof"] = edit["value"]
        parts[i].pop("roof_ends", None)
    elif op == "pitch":
        cur = 1.0
        if bp is not None:
            cur = next((float(m["roof"].get("pitch", 1.0)) for k, m in built_parts(bp) if k == i), 1.0)
        k = min(range(len(PITCH_STEPS)), key=lambda j: abs(PITCH_STEPS[j] - cur))
        k2 = max(0, min(len(PITCH_STEPS) - 1, k + int(edit.get("delta", 0))))
        if k2 == k:
            return state, ("limit: no pitched roof steeper than 72 degrees" if edit.get("delta", 0) > 0
                           else "limit: no pitched roof gentler than 14 degrees")
        adj.setdefault("pitch", {})[str(i)] = PITCH_STEPS[k2]
    elif op == "ridge":
        parts[i]["ridge"] = "parallel" if parts[i].get("ridge") == "perpendicular" else "perpendicular"
    elif op in ("width", "depth", "height"):
        sc = adj.setdefault("scale", {})
        v = float(sc.get(op, 1.0)) * float(edit.get("factor", 1.0))
        if not SCALE_RANGE[0] <= v <= SCALE_RANGE[1]:
            return state, f"limit: {op} already scaled {sc.get(op, 1.0):.2f}x"
        sc[op] = round(v, 3)
    elif op == "footprint":                            # width and depth together: the plan keeps its shape
        sc = adj.setdefault("scale", {})
        vs = {k: float(sc.get(k, 1.0)) * float(edit.get("factor", 1.0)) for k in ("width", "depth")}
        if not all(SCALE_RANGE[0] <= v <= SCALE_RANGE[1] for v in vs.values()):
            return state, f"limit: the footprint already scaled {sc.get('width', 1.0):.2f}x"
        sc.update({k: round(v, 3) for k, v in vs.items()})
    elif op == "part_base":                            # a part's base band, metres on its front and end walls
        p = parts[i]
        if not str(p.get("base") or "").lower() or str(p.get("base")).lower() == "none":
            p["base"] = (form.get("base") or {}).get("material", "brick") if isinstance(form.get("base"), dict) else "brick"
            if str(p["base"]).lower() in ("", "none"):
                p["base"] = "brick"
        p["base_m"] = round(float(edit.get("front", 0.0)), 2)
        p["base_end_m"] = round(float(edit.get("end", edit.get("front", 0.0))), 2)
        if p["base_m"] <= 0 and p["base_end_m"] <= 0:
            p["base"] = "none"
    elif op == "share":
        parts[i]["share"] = max(5.0, float(parts[i].get("share") or 100.0 / len(parts)) * float(edit.get("factor", 1.0)))
        if all(isinstance(p.get("width_m"), (int, float)) and p["width_m"] > 0 for p in parts):
            parts[i]["width_m"] = round(float(parts[i]["width_m"]) * float(edit.get("factor", 1.0)), 2)
    elif op in ("part_height", "part_depth"):          # this part only, as a factor
        pa = adj.setdefault("parts", {}).setdefault(str(i), {})
        key = op.split("_")[1]
        v = float(pa.get(key, 1.0)) * float(edit.get("factor", 1.0))
        if not SCALE_RANGE[0] <= v <= SCALE_RANGE[1]:
            return state, f"limit: part {i + 1}'s {key} already scaled {pa.get(key, 1.0):.2f}x"
        pa[key] = round(v, 3)
    elif op in ("part_level", "part_forward"):         # this part only, in blocks
        pa = adj.setdefault("parts", {}).setdefault(str(i), {})
        key = op.split("_")[1]
        v = float(pa.get(key, 0.0)) + float(edit.get("delta", 0))
        if abs(v) > SHIFT_RANGE:
            return state, f"limit: part {i + 1} already moved {pa.get(key, 0.0):+.0f} blocks ({key})"
        pa[key] = v
    elif op == "garage_doors":                         # how many this part has, on the wall they are on
        n = max(0, min(6, int(edit.get("value", 0))))
        p = parts[i]
        front = p["front"] if isinstance(p.get("front"), dict) else {}
        g = form.get("garage") if isinstance(form.get("garage"), dict) else {}
        on_side = int(p.get("side_garage_doors") or 0) > 0 or (
            not int(front.get("garage_doors") or 0) and g.get("wall") in ("left", "right"))
        if edit.get("wall") in ("front", "end"):           # the photo's answer says which wall
            on_side = edit["wall"] == "end"
            p["front"] = dict(front, garage_doors=0)
            p["side_garage_doors"] = 0
            front = p["front"]
        if on_side:
            p["side_garage_doors"] = n
        else:
            p["front"] = dict(front, garage_doors=n)
        if isinstance(form.get("garage"), dict):       # the parts now say where theirs are
            form["garage"] = dict(g, doors=0)
    elif op == "mirror":
        adj["mirror"] = not adj.get("mirror", False)
    elif op == "add_part":
        role = edit.get("role", "wing")
        new = {"part": role if role in ("wing", "tower", "annex") else "wing", "share": 30,
               "storeys": int(edit.get("storeys") or 1), "roof": edit.get("roof") or "gable"}
        if edit.get("side") == "left":
            parts.insert(0, new)
        else:
            parts.append(new)
    elif op == "remove_part":
        if len(parts) < 2:
            return state, "limit: the building's only part"
        parts.pop(i)
        if not any(p.get("part") == "main" for p in parts):
            parts[0]["part"] = "main"
    elif op == "feature":
        name, on = edit.get("name"), edit.get("on", True)
        if name not in FEATURE_ON:
            return state, f"invalid: no feature {name!r}"
        if name == "tiers":
            n = max(int(p.get("storeys") or 1) for p in parts)
            form["tiers"] = [max(1, round(n * 0.7)), max(1, n - round(n * 0.7))] if on else []
        elif on:
            form[name] = copy.deepcopy(FEATURE_ON[name])
        else:
            form[name] = copy.deepcopy(FEATURE_OFF.get(name, 0 if isinstance(FEATURE_ON[name], int) and
                                                      not isinstance(FEATURE_ON[name], bool) else False))
            if name == "towers":                        # the parts called towers no longer rise
                for p in parts:
                    if p.get("part") == "tower":
                        p["part"] = "wing"
    elif op == "windows":                              # by half again (at least one): 3 -> 5 -> 7 -> 11
        for p in parts:
            n = int(p.get("windows_across") or 2)
            p["windows_across"] = max(n + 1, round(n * 1.5)) if edit.get("delta", 0) > 0 else max(0, min(
                n - 1, round(n / 1.5)))
    elif op == "window_style":
        parts[i]["window_style"] = edit.get("value", "rect")
    elif op == "colour":
        role, word = edit.get("role"), str(edit.get("value") or "")
        if role == "wall":
            wall = form["wall"] if isinstance(form.get("wall"), dict) else {}
            form["wall"] = dict(wall, colour=word)
        elif role in ("roof", "trim", "door"):
            form[f"{role}_colour"] = word
        else:
            return state, f"invalid: no colour role {role!r}"
    elif op == "wall_material":
        if edit.get("value") not in WALLS2:
            return state, f"invalid: no wall material {edit.get('value')!r}"
        wall = form["wall"] if isinstance(form.get("wall"), dict) else {}
        form["wall"] = dict(wall, material=edit["value"])
    elif op == "skylights_on":
        adj["skylights_on"] = i
        form.setdefault("skylights", 2)
    elif op == "units":
        form["units"] = max(1, int(form.get("units") or 1) + int(edit.get("delta", 0)))
    elif op == "garage_wall":
        g = form["garage"] if isinstance(form.get("garage"), dict) else {"doors": 1}
        form["garage"] = dict(g, wall=edit.get("value", "front"))
    else:
        return state, f"invalid: no edit {op!r}"
    return s, None


def label(edit: dict) -> str:
    """A short name for an edit (logs, the report)."""
    keys = [k for k in ("part", "value", "delta", "factor", "name", "role", "on", "side", "wall", "front", "end")
            if k in edit]
    return edit["op"] + "(" + ", ".join(f"{k}={edit[k]}" for k in keys) + ")"


def _feature_name(word) -> str | None:
    w = str(word or "").lower().strip().replace(" ", "_").replace("-", "_")
    w = FEATURE_SYNONYMS.get(w, w)
    if w in FEATURE_ON:
        return w
    if w.endswith("s") and w[:-1] in FEATURE_ON:
        return w[:-1]
    return FEATURE_SYNONYMS.get(w.rstrip("s"))


def from_critique(crit: dict, state: dict, numbers: dict) -> tuple:
    """(edits, gaps) from the vision model's critique (`vision.critique`); `numbers` maps the
    critique's part numbers to the form's (`describe_state`)."""
    edits, gaps = [], []

    def part(k):
        try:
            return numbers.get(int(k))
        except (TypeError, ValueError):
            return None
    if crit.get("style_ok") is False:
        s = style_of(crit.get("style"))
        if s != state["style"] and s != "other":
            edits.append({"op": "style", "value": s})
    if crit.get("kind_ok") is False:
        k = kind_of(crit.get("kind"))
        if k != state["kind"] and k != "other":
            edits.append({"op": "kind", "value": k})
    prop = crit.get("proportions") if isinstance(crit.get("proportions"), dict) else {}
    for axis, small, big in (("width", "too_narrow", "too_wide"), ("height", "too_low", "too_tall"),
                             ("depth", "too_shallow", "too_deep")):
        v = str(prop.get(axis, "ok")).lower()
        # (not a storey more or less for "too low": the loop's three such corrections on the
        # 50-photo run were all wrong against the photos' storey counts)
        if v == small:
            edits.append({"op": axis, "factor": 1.25})
        elif v == big:
            edits.append({"op": axis, "factor": 0.8})
    for r in crit.get("parts") or []:                  # each part's size, level and garage doors
        if not isinstance(r, dict):
            continue
        i = part(r.get("part"))
        if i is None:
            continue
        shown = int(r["part"]) if str(r.get("part")).isdigit() else r.get("part")
        for key, op, small, big, up, down in (("height", "part_height", "too_low", "too_tall", 1.25, 0.8),
                                              ("depth", "part_depth", "too_shallow", "too_deep", 1.3, 0.77),
                                              ("width", "share", "too_narrow", "too_wide", 1.25, 0.8)):
            v = str(r.get(key) or "ok").lower()
            if v in (small, big):
                edits.append({"op": op, "part": i, "factor": up if v == small else down, "shown": shown})
        for key, op, lower, higher in (("level", "part_level", "should_be_lower", "should_be_higher"),
                                       ("forward", "part_forward", "should_go_back", "should_come_forward")):
            v = str(r.get(key) or "ok").lower()
            if v in (lower, higher):
                edits.append({"op": op, "part": i, "delta": -2 if v == lower else 2, "shown": shown})
        g = r.get("garage_doors")
        fp = (state["form"].get("parts") or [{}])[i] if i < len(state["form"].get("parts") or []) else {}
        had = int((fp.get("front") or {}).get("garage_doors") or 0) + int(fp.get("side_garage_doors") or 0)             if isinstance(fp, dict) and isinstance(fp.get("front") or {}, dict) else 0
        if isinstance(g, (int, float)) and not isinstance(g, bool) and 0 <= g <= 6 and (g > 0 or had):
            edits.append({"op": "garage_doors", "part": i, "value": int(g), "shown": shown})
        t = _roof_word(r.get("roof_type"))
        if r.get("roof_ok") is False or r.get("roof_type_ok") is False:
            if t in ROOF_KINDS:
                edits.append({"op": "roof", "part": i, "value": t})
            elif t:
                gaps.append(f"roof type {t!r} (part {r.get('part')})")
    roofs = crit.get("roofs") or []
    for r in [roofs] if isinstance(roofs, dict) else roofs:
        if not isinstance(r, dict):
            continue
        i = part(r.get("part"))
        if i is None:
            continue
        t = _roof_word(r.get("type"))
        if r.get("type_ok") is False:
            if t in ROOF_KINDS:
                edits.append({"op": "roof", "part": i, "value": t})
            elif t:
                gaps.append(f"roof type {t!r} (part {r.get('part')})")
    for key, on in (("missing", True), ("extra", False)):
        words = crit.get(key) or []
        if isinstance(words, str):                     # "porch, dormers" rather than a list
            words = [w for w in words.split(",") if w.strip().lower() not in ("", "none")]
        for word in words if isinstance(words, list) else []:
            name = _feature_name(word)
            if name:
                edits.append({"op": "feature", "name": name, "on": on})
            elif str(word).strip():
                gaps.append(f"{'missing' if on else 'extra'} {str(word).strip()!r}")
    if crit.get("mirrored") is True:
        edits.append({"op": "mirror"})
    extra_parts = crit.get("extra_parts") or []
    for k in extra_parts if isinstance(extra_parts, list) else [extra_parts]:
        i = part(k)
        if i is not None and len({numbers[n] for n in numbers}) > 1:     # not the only part
            edits.append({"op": "remove_part", "part": i})
    side = str(crit.get("missing_part") or "none").lower()
    if side in ("left", "right"):
        edits.append({"op": "add_part", "role": "wing", "side": side, "storeys": 1})
    garage = str(crit.get("garage") or "ok").lower()
    if garage == "other_side":
        wall = (state["form"].get("garage") or {}).get("wall") if isinstance(state["form"].get("garage"), dict) else None
        edits.append({"op": "garage_wall", "value": {"left": "right", "right": "left"}[wall]} if wall in ("left", "right")
                     else {"op": "mirror"})
    elif garage == "missing":
        edits.append({"op": "feature", "name": "garage", "on": True})
    elif garage == "not_in_photo":
        edits.append({"op": "feature", "name": "garage", "on": False})
    w = str(crit.get("windows") or "ok").lower()
    if w in ("too_few", "too_many"):
        edits.append({"op": "windows", "delta": 1 if w == "too_few" else -1})
    for role in ("wall", "roof"):
        c = str(crit.get(f"{role}_colour") or "ok").strip().lower()
        if c and c not in ("ok", "same", "correct", "none"):
            edits.append({"op": "colour", "role": role, "value": c})
    other = str(crit.get("other") or "").strip()
    if other and other.lower() not in ("none", "nothing", "n/a", "-", "no"):
        gaps.append(other[:200])
    for e in edits:
        e["why"] = "vision"
    return edits, gaps


ENDS = {"gable": "triangle", "cross_gable": "triangle", "gambrel": "triangle", "shed": "triangle", "hip": "slope",
        "pyramid": "slope", "mansard": "slope", "pagoda": "slope", "flat": "flat", "sawtooth": "flat"}


def check_parts(bp: dict, state: dict) -> tuple:
    """(descriptions, numbers): the build's parts left to right as `vision.check` asks about
    them ("the main block", "a wing", ...), and {shown number: form part number}."""
    text, numbers = describe_state(state, bp)
    fparts = [p for p in (state["form"].get("parts") or []) if isinstance(p, dict)]
    row = (bp.get("meta") or {}).get("row")
    out = []
    for k in sorted(numbers):
        role = str((fparts[numbers[k]] if numbers[k] < len(fparts) else {}).get("part") or "wing")
        out.append(f"house {k} of the row" if row else
                   {"main": "the main block", "tower": "a tower", "annex": "a low annex"}.get(role, "a wing"))
    return out, numbers


def form_answers(p: dict) -> dict:
    """What a form's part says, in the check's terms (`vision.CHECK`)."""
    out = {"exists": True}
    roof = str(p.get("roof") or "").lower()
    ends = str(p.get("roof_ends") or "").lower()
    if roof == "gable" and ends == "slope":
        roof = "hip"
    elif roof == "hip" and ends == "triangle":
        roof = "gable"
    if roof in ENDS or roof == "dome":
        out["roof_ends"] = ENDS.get(roof, "flat")
    ridge = {"parallel": "along_front", "perpendicular": "front_to_back"}.get(str(p.get("ridge") or "").lower())
    seen = p.get("roof_seen") if isinstance(p.get("roof_seen"), dict) else {}
    if seen.get("front_top") in ("triangle", "eave", "both"):     # the roof pass's answer, when it ran
        ridge = "front_to_back" if seen["front_top"] == "triangle" else "along_front"
    if ridge and roof in ("gable", "hip", "gambrel", "mansard", "cross_gable"):
        out["ridge"] = ridge
    try:
        f = float(p.get("floor_m", 0) or 0)
        out["floor"] = "lower" if f <= -0.8 else "higher" if f >= 0.8 else "same"
    except (TypeError, ValueError):
        pass
    if p.get("level") == "half_below":
        out["floor"] = "lower"
    if isinstance(p.get("storeys"), int):
        out["storeys"] = p["storeys"]
    try:
        out["garage_doors"] = int((p.get("front") or {}).get("garage_doors") or 0) + int(p.get("side_garage_doors") or 0)
    except (TypeError, ValueError):
        pass
    return out


def agreed_with_form(check: dict, form: dict, numbers: dict) -> dict:
    """The check's answers the form's reading agrees with: where the build differs from both,
    it is the build that is wrong (compose's rules, an earlier correction). Where the two
    readings disagree, which is right was a coin toss on the 50 test photos (the main roof's
    ridge: the form 15 of 24, the check 8 of 16), so those are left alone."""
    parts = [p for p in (form.get("parts") or []) if isinstance(p, dict)]
    out = []
    for a in check.get("parts") or []:
        if not isinstance(a, dict):
            continue
        i = numbers.get(a.get("part"))
        if i is None or i >= len(parts):
            continue
        says = form_answers(parts[i])
        out.append({"part": a["part"], **{k: v for k, v in a.items()
                                          if k != "part" and k in says and str(says[k]).lower() == str(v).lower()}})
    return {"parts": out}


def from_check(check: dict, state: dict, bp: dict, numbers: dict, door_levels: dict | None = None) -> list:
    """Edits where the photo's answers (`vision.check`) differ from the build: a part the
    photo's building does not have, a part's roof ends (gable / hip / flat), its ridge, its
    floor's level against the main block's, its storeys, its garage doors and base band; the
    tallest part on the other side: a mirror image. A level
    change the photo's doors agree with (`facade.levels`: {mass id: rows}) takes their size
    and is marked "doors_agree"."""
    edits = []
    shown = dict(built_parts(bp))
    by_shown = {k: shown.get(i) for k, i in numbers.items()}
    fparts = [p for p in (state["form"].get("parts") or []) if isinstance(p, dict)]
    main_i = next((i for i, p in enumerate(fparts) if p.get("part") == "main"), 0)
    main = shown.get(main_i)
    for r in check.get("parts") or []:
        if not isinstance(r, dict):
            continue
        try:
            k = int(r.get("part"))
        except (TypeError, ValueError):
            continue
        m, i = by_shown.get(k), numbers.get(k)
        if m is None or i is None:
            continue
        if r.get("exists") is False:                   # a part the photo's building does not have
            if i != main_i and len(by_shown) > 1:
                edits.append({"op": "remove_part", "part": i})
            continue
        roof = m["roof"]["type"]
        ends = str(r.get("roof_ends") or "").lower()
        if ends in ("triangle", "slope", "flat") and roof in ENDS and ENDS[roof] != ends:
            edits.append({"op": "roof", "part": i, "value": {"triangle": "gable", "slope": "hip", "flat": "flat"}[ends]})
        ridge = str(r.get("ridge") or "").lower()
        if roof in ("gable", "hip", "gambrel", "mansard") and ridge in ("along_front", "front_to_back") and \
                (m["roof"].get("ridge", "x") == "x") != (ridge == "along_front"):
            edits.append({"op": "ridge", "part": i})
        floor = str(r.get("floor") or "").lower()
        if main is not None and i != main_i and floor in ("same", "lower", "higher"):
            dy = m["y"] - main["y"]
            have = "lower" if dy <= -2 else "higher" if dy >= 2 else "same"
            if have != floor:
                step = max(2, int(m.get("storey_h", 5)) // 2 + 1)
                delta = -dy if floor == "same" else (-step if floor == "lower" else step) - (dy if have != "same" else 0)
                e = {"op": "part_level", "part": i, "delta": int(delta)}
                doors = (door_levels or {}).get(m["id"])
                if doors and (doors > 0) == (delta > 0):
                    e.update(delta=int(doors), doors_agree=True)
                edits.append(e)
        try:
            n = int(r.get("storeys"))
        except (TypeError, ValueError):
            n = 0
        have_n = max(1, len(m.get("floors") or [0]))
        if 1 <= n <= 120 and n != have_n and abs(n - have_n) <= 2:
            edits.append({"op": "storeys", "part": i, "delta": n - have_n})
        try:
            g = int(r.get("garage_doors"))
        except (TypeError, ValueError):
            g = -1
        have_g = sum(1 for o in bp.get("openings", []) if o.get("mass") == m["id"] and o["kind"] == "garage")
        if 0 <= g <= 6 and g != have_g and (g > 0 or have_g):
            edits.append({"op": "garage_doors", "part": i, "value": g,
                          "wall": str(r.get("garage_wall") or "").lower()})
        # its base band on the front and the end wall, against the build's (rows over its storey's metres)
        try:
            bf, be = float(r.get("base_m")), float(r.get("base_end_m", r.get("base_m")))
        except (TypeError, ValueError):
            bf = be = -1.0
        if 0 <= bf <= 6 and 0 <= be <= 6:
            rows_m = max(1.0, float(m.get("storey_h", 5))) / 3.0          # blocks a metre, from a 3 m storey
            pl = m.get("plinth", 0)
            pl = pl if isinstance(pl, dict) else {s: int(pl or 0) for s in ("south", "north", "east", "west")}
            end = "west" if k - 1 < len(by_shown) / 2 and len(by_shown) > 1 else "east"     # as compose picks it
            want_f, want_e = round(bf * rows_m), round(be * rows_m)
            if abs(want_f - int(pl.get("south", 0))) >= 2 or abs(want_e - int(pl.get(end, 0))) >= 2:
                edits.append({"op": "part_base", "part": i, "front": round(bf, 2), "end": round(be, 2)})
    try:
        tallest = int(check.get("tallest"))
    except (TypeError, ValueError):
        tallest = 0
    tops = {k: m["y"] + wall_rows(m) for k, m in by_shown.items() if m is not None}
    if tops and tallest in tops and len(tops) >= 2:
        n = len(tops)
        built = max(tops, key=tops.get)
        if built != tallest and tallest == n + 1 - built and tops[built] > sorted(tops.values())[-2]:
            edits.append({"op": "mirror"})
    out, seen = [], set()
    for e in edits:                                    # once each: a row's houses can all be one form part
        e["why"] = "check"
        if label(e) not in seen:
            seen.add(label(e))
            out.append(e)
    return out


def from_conflicts(conflicts: list, state: dict, bp: dict) -> list:
    """Edits that would let the build do what its form asked (`compose`'s conflicts): a part
    wide enough for the windows the form gives it, the main block's ridge turned so its
    dormers have a slope facing the front."""
    import re
    edits = []
    fparts = [p for p in (state["form"].get("parts") or []) if isinstance(p, dict)]
    for c in conflicts or []:
        m = re.match(r"part (\d+)(?:, floor \d+)?: (?:windows across|ground-floor windows): (\d+) asked, (\d+) fit "
                     r"on a wall (\d+) blocks long", c)
        if m:
            i, asked, fit, length = (int(g) for g in m.groups())
            if 0 < i <= len(fparts) and len(fparts) > 1 and fit < asked:
                need = length * (asked + 1) / max(1, fit + 1)          # the same spacing for all of them
                edits.append({"op": "share", "part": i - 1, "factor": round(min(1.6, max(1.15, need / length)), 2)})
            continue
        if "dormers asked" in c and "front to back" in c:
            main = next((k for k, p in enumerate(fparts) if p.get("part") == "main"), 0)
            edits.append({"op": "ridge", "part": main})
    out, seen = [], set()
    for e in edits:
        e["why"] = "conflict"
        if label(e) not in seen:
            seen.add(label(e))
            out.append(e)
    return out


def from_measures(diag: dict, state: dict, bp: dict) -> list:
    """Edits the computer measures point to (`score.diagnose`)."""
    edits = []
    # proportions only: the measures cannot count storeys (the reading gets them right on 41
    # of 44 photos), so a storey more or less is the vision model's call. A small step, and
    # one sized to the error (to at most twice) when the shape is far off: in 15% steps a
    # 65-storey needle against a square photo never got there
    ratio = float(diag.get("aspect_ratio") or 1.0)
    if diag.get("shape") in ("taller", "wider"):
        # the footprint as a whole, not the width alone: widening alone fitted the outline and
        # left a house shallow (round 3: width x 1.47, depth unchanged)
        f = min(2.0, max(1.15, (ratio if ratio > 1 else 1 / ratio) ** 0.6))
        tall = diag["shape"] == "taller"
        edits += [{"op": "height", "factor": 0.85 if tall else 1.18}, {"op": "footprint", "factor": 1.15 if tall else 0.87}]
        if f > 1.25:
            edits += [{"op": "height", "factor": round(1 / f if tall else f, 2)},
                      {"op": "footprint", "factor": round(f if tall else 1 / f, 2)}]
    plan = float(diag.get("plan_ratio") or 1.0)        # the build's depth / width over the photo's
    if diag.get("plan") in ("deeper", "shallower"):
        f = min(1.6, max(1.15, (plan if plan > 1 else 1 / plan) ** 0.7))
        edits.append({"op": "depth", "factor": round(1 / f if diag["plan"] == "deeper" else f, 2)})
    shown = built_parts(bp)
    # no pitch from the roof line: on the 50 test photos the measures could not tell a right roof
    # change from a wrong one, pushed roofs flatter until they hit the 14-degree limit 9 times, and
    # no such change was ever kept; a roof's pitch is the roof pass's (`refine.roof_pass`)
    # each part's own roof line (the label view's columns of that part): its height, and how much
    # more its roof rises and falls than the photo's there
    masses = bp.get("masses") or []
    for code, (dh, dshape) in (diag.get("sections") or {}).items():
        k = int(code) - 1
        if not 0 <= k < len(masses) or not str(masses[k]["id"])[1:].isdigit() or masses[k]["id"][:1] != "p":
            continue
        i = int(masses[k]["id"][1:])
        if abs(dh) > 0.05:
            f = min(1.6, max(1.12, 1.0 + 2.5 * abs(dh)))
            edits.append({"op": "part_height", "part": i, "factor": round(1 / f if dh > 0 else f, 2)})
    ops = diag.get("openings") or {}
    g = ops.get("garage") or {}
    if diag.get("openings_on_building") and g.get("photo", 0) != g.get("build", 0):
        # the photo's garage doors on the part at their wall (MapAnything placed them)
        walls = Counter(o["wall"] for o in g.get("missing") or [])
        for wall, n in walls.items():
            side = {"front": "south", "right": "east", "left": "west"}[wall]
            cands = [(i, m) for i, m in shown]
            if not cands:
                continue
            i, m = (max(cands, key=lambda im: im[1]["x"] + im[1]["w"]) if side == "east" else
                    min(cands, key=lambda im: im[1]["x"]) if side == "west" else
                    min(cands, key=lambda im: len([o for o in bp["openings"] if o["mass"] == im[1]["id"]])))
            have = sum(1 for o in bp["openings"] if o["mass"] == m["id"] and o["kind"] == "garage")
            edits.append({"op": "garage_doors", "part": i, "value": have + n})
    if diag.get("windows") in ("more", "fewer"):
        edits.append({"op": "windows", "delta": -1 if diag["windows"] == "more" else 1})
    if diag.get("mirror_gain", 0) > 0.04:
        edits.append({"op": "mirror"})
    for role in ("wall", "roof"):
        if diag.get(f"{role}_colour"):
            edits.append({"op": "colour", "role": role, "value": diag[f"{role}_colour"]})
    for e in edits:
        e["why"] = "measures"
    out, seen = [], set()
    for e in edits:
        key = label(e)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out
