"""One photo -> a Minecraft build (.schem), the whole chain in one command.

From the repository's root:

    r2m_measure\\Scripts\\python -m blueprint.photo house.jpg --out outputs\\blueprint\\house --model qwen3.8-27b

Each step runs in its own environment:

1. r2m_vlm: the vision model reads the photo (what kind of building, its style, its parts
   and roofs) and finds the main building and its windows and doors; the segmentation model
   finds the building's pixels (`refine prepare`)
                                -> <stem>.read.json, <stem>.photo.json, _work/<stem>/
2. r2m_measure: MapAnything measures the building and places each window and door on its
   walls (`measure`)            -> <stem>.measure.json
3. r2m_vlm: the blueprint is composed and built, compared with the photo and corrected
   (`refine run`)               -> <stem>/first.*, <stem>/final.*, <stem>.refine.json
4. here: blueprint.json and build.schem (the final build; Sponge v2, Minecraft 1.21.11), and
   preview renders from four sides (view_*.png).

--rounds 0 keeps the first build (no corrections); --no-measure skips MapAnything (typical
sizes for the kind of building); --model picks the vision model (`describe.load_model`):
qwen3.8-27b (through llama.cpp, recommended) or the default Qwen/Qwen3.5-9B.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLM = ROOT / "r2m_vlm" / "Scripts" / "python.exe"
MEASURE = ROOT / "r2m_measure" / "Scripts" / "python.exe"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4, help="correction rounds (0: none)")
    ap.add_argument("--no-measure", action="store_true", help="skip MapAnything (typical sizes for the kind)")
    ap.add_argument("--model", default=None, help="the vision model (describe.load_model)")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    photo = args.photo.resolve()
    stem = photo.parent.name if photo.name == "photo.jpg" else photo.stem      # as the steps name it
    model = ["--model", args.model] if args.model else []
    steps = [[str(VLM), "-m", "blueprint.refine", "prepare", str(photo), "--out", str(out), *model]]
    if not args.no_measure:
        steps.append([str(MEASURE), "-m", "blueprint.measure", str(photo), "--out", str(out)])
    steps.append([str(VLM), "-m", "blueprint.refine", "run", str(photo), "--out", str(out), "--rounds", str(args.rounds),
                  "--redo", *model])
    for cmd in steps:
        print(">", " ".join(cmd[1:]), flush=True)
        if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
            print("step failed:", " ".join(cmd[2:4]), file=sys.stderr)
            return 1
    from .build import build
    from .minecraft.schem import write_schem
    from .render import render_views
    rec = json.loads((out / f"{stem}.refine.json").read_text(encoding="utf-8"))
    shutil.copyfile(out / stem / "final.blueprint.json", out / "blueprint.json")
    bp = json.loads((out / "blueprint.json").read_text(encoding="utf-8"))
    b = build(bp)
    write_schem(b["shape"], b["material"], out / "build.schem")
    print(json.dumps({"schem": str(out / "build.schem"), "style": rec["final"]["state"]["style"],
                      "kind": rec["final"]["state"]["kind"], "measured": rec["measured"], "kept": rec["kept"],
                      "score": [rec["first"]["combined"], rec["final"]["combined"]],
                      "blocks_per_m": bp["meta"]["blocks_per_m"], "gaps": [g["gap"] for g in rec["gaps"]],
                      "conflicts": rec.get("conflicts", {}).get("final", [])},
                     indent=1), flush=True)
    previews = render_views(b["shape"], b["material"], out)
    print("previews:", ", ".join(p.name for p in previews), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
