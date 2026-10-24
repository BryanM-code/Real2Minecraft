"""Build a blueprint into a schematic, from this repository's root:

    r2m_app\\Scripts\\python -m blueprint house.json --out outputs\\blueprint\\house
    ..\\BuildDataCollector\\.venv\\Scripts\\python -m blueprint house.json --out outputs\\blueprint\\house --render

Writes <out>/build.schem (Sponge v2, 1.21.11), blocks.npz (shape, material, origin) and, with
--render (BuildDataCollector's environment), preview renders from four sides.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("blueprint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--render", action="store_true", help="preview renders (BuildDataCollector's environment)")
    ap.add_argument("--size", type=int, default=900)
    args = ap.parse_args(argv)
    from photofirst.schem import write_schem
    from .build import build
    from .spec import BlueprintError
    try:
        b = build(args.blueprint)
    except BlueprintError as e:
        print("blueprint error:", e, file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    info = write_schem(b["shape"], b["material"], args.out / "build.schem")
    np.savez_compressed(args.out / "blocks.npz", shape=b["shape"], material=b["material"], origin=np.array(b["origin"]))
    print(json.dumps({"schem": str(args.out / "build.schem"), "dims_xyz": b["dims"], "blocks": b["blocks"],
                      "palette": info.get("palette"), "warnings": b["warnings"]}, indent=1))
    if args.render:
        from .render import render_views
        for p in render_views(b["shape"], b["material"], args.out, size=args.size):
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
