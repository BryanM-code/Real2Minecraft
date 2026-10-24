"""One photo -> a .schem through the whole local chain, from this repository's root; each
step runs in its own environment:

    r2m_app\\Scripts\\python -m blueprint.photo house.jpg --out outputs\\blueprint\\house [--rounds 4]

1. r2m_vlm: the local model reads the photo (description, kind, style, form) and finds the
   main building, the side it shows and its windows; the segmentation model finds its
   pixels (`refine prepare`)                     -> <stem>.read.json, <stem>.photo.json, _work/<stem>/
2. r2m_train: MapAnything measures the building  -> <stem>.measure.json
3. r2m_vlm: the evaluator builds, compares the build with the photo and corrects it
   (`refine run`)                                -> <stem>/first.*, <stem>/final.* (blueprint, view), <stem>.refine.json
4. here: build.schem (the final build; Sponge v2, 1.21.11) and blueprint.json; with
   BuildDataCollector's environment present, preview renders from four sides.

--rounds 0 keeps the first reading's build (no corrections); --no-measure skips MapAnything.
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
TRAIN = ROOT / "r2m_train" / "Scripts" / "python.exe"
BDC = ROOT.parent / "BuildDataCollector" / ".venv" / "Scripts" / "python.exe"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=4, help="the evaluator's correction rounds (0: none)")
    ap.add_argument("--no-measure", action="store_true", help="skip MapAnything (typical sizes for the kind)")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    photo = args.photo.resolve()
    stem = photo.parent.name if photo.name == "photo.jpg" else photo.stem      # as the stages name it
    steps = [[str(VLM), "-m", "blueprint.refine", "prepare", str(photo), "--out", str(out)]]
    if not args.no_measure:
        steps.append([str(TRAIN), "-m", "blueprint.measure", str(photo), "--out", str(out)])
    steps.append([str(VLM), "-m", "blueprint.refine", "run", str(photo), "--out", str(out), "--rounds", str(args.rounds),
                  "--redo"])
    for cmd in steps:
        print(">", " ".join(cmd[1:]), flush=True)
        if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
            print("step failed:", " ".join(cmd[2:4]), file=sys.stderr)
            return 1
    from photofirst.schem import write_schem
    from .build import build
    rec = json.loads((out / f"{stem}.refine.json").read_text(encoding="utf-8"))
    shutil.copyfile(out / stem / "final.blueprint.json", out / "blueprint.json")
    bp = json.loads((out / "blueprint.json").read_text(encoding="utf-8"))
    b = build(bp)                                      # the schematic here: r2m_vlm has no nbtlib
    write_schem(b["shape"], b["material"], out / "build.schem")
    print(json.dumps({"schem": str(out / "build.schem"), "style": rec["final"]["state"]["style"],
                      "kind": rec["final"]["state"]["kind"], "measured": rec["measured"], "kept": rec["kept"],
                      "score": [rec["first"]["combined"], rec["final"]["combined"]],
                      "blocks_per_m": bp["meta"]["blocks_per_m"], "gaps": [g["gap"] for g in rec["gaps"]]},
                     indent=1), flush=True)
    if BDC.is_file():
        subprocess.run([str(BDC), "-m", "blueprint", str(out / "blueprint.json"), "--out", str(out), "--render"],
                       cwd=str(ROOT), capture_output=True)
        print("previews:", ", ".join(p.name for p in sorted(out.glob("view_*.png"))), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
