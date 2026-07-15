#!/usr/bin/env python3
"""Offline building-mode harness: run cached GLB+photo pairs, no TRELLIS/UI.

Usage:
    r2m_app\\Scripts\\python.exe dev_building.py <tag> [glb_stem ...]
        Full building runs.  Renders to outputs/dev_renders/<tag>/, dataset samples to
        outputs/dataset_dev/<tag>/ (NEVER outputs/dataset/ — dev runs must not pollute
        training data).  `glb_stem` filters to specific cached models; default = all
        stems that have a cutout archived.

    r2m_app\\Scripts\\python.exe dev_building.py --facing [glb_stem ...]
        The TRELLIS front-convention experiment: prints each model's per-direction
        facade ΔE table and the overall winner tally.  Used once to calibrate
        PARAMS["front_convention"] in building/params.py.

    r2m_app\\Scripts\\python.exe dev_building.py --diff <tagA> <tagB>
        IR regression: compares dataset_dev samples by CONTENT hash (canonical JSON
        minus volatile keys; npz by array bytes — the files themselves embed zip/gzip
        timestamps, so raw file hashes can never match).  Prints per-file verdicts.

    r2m_app\\Scripts\\python.exe dev_building.py --golden write|check <tag>
        Golden regression: `write` stores the content hashes of dataset_dev/<tag>
        into dev_golden.json (repo-tracked) together with PIPELINE_VERSION;
        `check` re-compares — differing hashes under an UNCHANGED PIPELINE_VERSION
        are flagged as unintended drift (bump the version when a change is meant).
"""

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from config import OUTPUTS_DIR
from mesh_normalize import normalize_mesh
from building.pipeline import run_building
from building.errors import BuildingModeError

TARGET_BLOCKS = 48
GLB_DIR = OUTPUTS_DIR / "trellis"
DEV_DATASET = OUTPUTS_DIR / "dataset_dev"

# Manifest keys that legitimately differ between identical runs.
_VOLATILE = {"created_utc", "elapsed_s", "env", "git_rev"}


def _pairs(only):
    """Cached (glb, input photo, cutout) triples, newest first per stem filter."""
    out = []
    for glb in sorted(GLB_DIR.glob("*.glb")):
        if only and glb.stem not in only:
            continue
        cutout = next(GLB_DIR.glob(glb.stem + "_cutout.png"), None)
        photo = next(GLB_DIR.glob(glb.stem + "_input.*"), None)
        out.append((glb, photo, cutout))
    return out


def run_tag(tag, only):
    render_dir = OUTPUTS_DIR / "dev_renders" / tag
    render_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = DEV_DATASET / tag

    pairs = [(g, p, c) for g, p, c in _pairs(only) if c is not None or only]
    if not pairs:
        print("No cached GLBs (with cutouts) found under", GLB_DIR)
        return 1

    for glb, photo, cutout in pairs:
        t0 = time.time()
        print("\n== {} ==".format(glb.stem))
        try:
            norm_path, norm_info = normalize_mesh(glb)
            voxel_data, state = run_building(
                image_path=photo, cutout_path=cutout, glb_path=glb,
                norm_info=norm_info, target_blocks=TARGET_BLOCKS,
                dataset_root=dataset_root)
        except BuildingModeError as exc:
            print("   SKIPPED: {}".format(exc))
            continue

        yaw = state["yaw_ir"]
        facing = state["walls_ir"]["photo_facing"]
        W, H, L = voxel_data["dims"]
        print("   yaw: theta={:+.1f} deg  conf={:.2f}  applied={}".format(
            yaw["theta_deg"], yaw["confidence"], yaw["applied"]))
        print("   massing: mean_iou={:.3f}  levels={}  roof={}".format(
            state["massing_ir"]["mean_iou"], len(state["massing_ir"]["levels"]),
            state["massing_ir"]["roof"]["type"]))
        if state.get("facade_bundle"):
            g = state["facade_bundle"]["ir"]["facades"][0]["grid"]
            print("   facade: mode={}  rows={} (conf {:.2f})  cols={} (conf {:.2f})"
                  .format(g["mode"], g["n_rows"], g["row_conf"],
                          g["n_cols"], g["col_conf"]))
        print("   facing: {} ({}, score {})".format(
            facing["dir"], facing["method"],
            "{:.1f}".format(facing["score"]) if facing["score"] else "n/a"))
        print("   {:,} blocks  {}x{}x{}  ({:.1f}s)".format(
            voxel_data["block_count"], W, H, L, time.time() - t0))
        for bid, n in Counter(voxel_data["block_ids"]).most_common(8):
            print("   {:5.1f}%  {}".format(
                100.0 * n / voxel_data["block_count"], bid))

        try:
            from dev_rebuild import render_screenshot
            png = render_dir / (glb.stem + ".png")
            render_screenshot(voxel_data, png)
            print("   ->", png)
        except Exception as exc:
            print("   (render skipped: {})".format(str(exc)[:70]))
    return 0


def run_facing(only):
    """Per-model ΔE table for all four directions + winner tally."""
    pairs = [(g, p, c) for g, p, c in _pairs(only) if c is not None]
    if not pairs:
        print("No cached GLB+cutout pairs found under", GLB_DIR)
        return 1
    tally = Counter()
    print("{:44s} {:>7s} {:>7s} {:>7s} {:>7s}  winner".format(
        "model", "-Z", "+Z", "-X", "+X"))
    for glb, photo, cutout in pairs:
        try:
            norm_path, norm_info = normalize_mesh(glb)
            _, state = run_building(
                image_path=photo, cutout_path=cutout, glb_path=glb,
                norm_info=norm_info, target_blocks=TARGET_BLOCKS,
                dataset_enabled=False)
        except BuildingModeError as exc:
            print("{:44s} SKIPPED: {}".format(glb.stem[:44], exc))
            continue
        scores = state["walls_ir"]["photo_facing"].get("scores_de") or {}
        if not scores:
            print("{:44s} no texture colors".format(glb.stem[:44]))
            continue
        best = min(scores, key=scores.get)
        tally[best] += 1
        print("{:44s} {:7.1f} {:7.1f} {:7.1f} {:7.1f}  {}".format(
            glb.stem[:44], scores["-Z"], scores["+Z"],
            scores["-X"], scores["+X"], best))
    total = sum(tally.values())
    print("\nWinner tally:", dict(tally))
    if total:
        top, n = tally.most_common(1)[0]
        print("-> {} wins {}/{} ({:.0%})".format(top, n, total, n / total),
              "- set PARAMS['front_convention'] accordingly" if n / total >= 0.8
              else "- no dominant convention; leave front_convention = None")
    return 0


# ── content-hash diff ─────────────────────────────────────────────────────────

def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _content_hash(path: Path) -> str | None:
    """Deterministic content hash: canonical JSON / npz array bytes / raw file bytes.
    Returns None for files that are legitimately nondeterministic (renders, gzip)."""
    name = path.name
    if name.startswith("preview") or name.endswith(".schem"):
        return None                       # VTK renders / gzip mtime: not comparable
    h = hashlib.sha256()
    if path.suffix == ".json":
        doc = _strip_volatile(json.loads(path.read_text(encoding="utf-8")))
        h.update(json.dumps(doc, sort_keys=True).encode())
    elif path.suffix == ".npz":
        with np.load(str(path)) as data:
            for key in sorted(data.files):
                arr = data[key]
                h.update(key.encode())
                h.update(str(arr.shape).encode())
                h.update(str(arr.dtype).encode())
                h.update(np.ascontiguousarray(arr).tobytes())
    else:
        h.update(path.read_bytes())
    return h.hexdigest()


def run_diff(tag_a, tag_b):
    root_a, root_b = DEV_DATASET / tag_a, DEV_DATASET / tag_b
    if not root_a.is_dir() or not root_b.is_dir():
        print("Missing dataset_dev tag dir:", root_a if not root_a.is_dir() else root_b)
        return 1
    clean = True
    samples = sorted({p.name for p in root_a.iterdir() if p.is_dir()}
                     | {p.name for p in root_b.iterdir() if p.is_dir()})
    for s in samples:
        da, db = root_a / s, root_b / s
        if not da.is_dir() or not db.is_dir():
            print("[{}] only in {}".format(s, tag_a if da.is_dir() else tag_b))
            clean = False
            continue
        files = sorted({str(p.relative_to(da)) for p in da.rglob("*") if p.is_file()}
                       | {str(p.relative_to(db)) for p in db.rglob("*") if p.is_file()})
        for f in files:
            fa, fb = da / f, db / f
            if not fa.exists() or not fb.exists():
                print("[{}] {}: only in {}".format(s, f, tag_a if fa.exists() else tag_b))
                clean = False
                continue
            ha, hb = _content_hash(fa), _content_hash(fb)
            if ha is None:
                continue
            if ha != hb:
                print("[{}] {}: DIFFERS".format(s, f))
                clean = False
    print("\n" + ("CLEAN — all comparable IRs identical" if clean else "DIFFERENCES FOUND"))
    return 0 if clean else 2


def _tag_hashes(tag):
    """{sample/file: content hash} over one dataset_dev tag (comparable files only)."""
    root = DEV_DATASET / tag
    out = {}
    for sample in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(sample.rglob("*")):
            if f.is_file():
                h = _content_hash(f)
                if h is not None:
                    out["{}/{}".format(sample.name,
                                       f.relative_to(sample).as_posix())] = h
    return out


def run_golden(action, tag):
    from building.params import PIPELINE_VERSION
    golden_path = Path(__file__).parent / "dev_golden.json"
    if action == "write":
        doc = {"pipeline_version": PIPELINE_VERSION, "tag": tag,
               "hashes": _tag_hashes(tag)}
        golden_path.write_text(json.dumps(doc, sort_keys=True, indent=1),
                               encoding="utf-8")
        print("Wrote {} hashes (pipeline {}) to {}".format(
            len(doc["hashes"]), PIPELINE_VERSION, golden_path.name))
        return 0
    doc = json.loads(golden_path.read_text(encoding="utf-8"))
    current = _tag_hashes(tag)
    golden = doc["hashes"]
    same_version = doc["pipeline_version"] == PIPELINE_VERSION
    drift = sorted(k for k in golden.keys() & current.keys()
                   if golden[k] != current[k])
    missing = sorted(golden.keys() - current.keys())
    added = sorted(current.keys() - golden.keys())
    for k in drift:
        print("DRIFT:", k)
    for k in missing:
        print("MISSING:", k)
    for k in added:
        print("NEW:", k)
    if not (drift or missing):
        print("GOLDEN CLEAN ({} files, pipeline {})".format(
            len(golden), PIPELINE_VERSION))
        return 0
    if same_version:
        print("\nFAIL: outputs changed but PIPELINE_VERSION is still {} — either an "
              "unintended regression, or bump the version in building/params.py and "
              "re-write the golden.".format(PIPELINE_VERSION))
        return 2
    print("\nOutputs changed under a version bump ({} -> {}): re-write the golden "
          "when the new outputs are verified.".format(doc["pipeline_version"],
                                                      PIPELINE_VERSION))
    return 0


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if args[0] == "--facing":
        return run_facing(set(args[1:]))
    if args[0] == "--diff":
        if len(args) != 3:
            print("usage: dev_building.py --diff <tagA> <tagB>")
            return 1
        return run_diff(args[1], args[2])
    if args[0] == "--golden":
        if len(args) != 3 or args[1] not in ("write", "check"):
            print("usage: dev_building.py --golden write|check <tag>")
            return 1
        return run_golden(args[1], args[2])
    return run_tag(args[0], set(args[1:]))


if __name__ == "__main__":
    sys.exit(main())
