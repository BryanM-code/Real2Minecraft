"""Dataset sample writer: every building-mode run becomes a training sample.

Layout of one sample (outputs/dataset/<run_id>/, run_id = the TRELLIS stem so samples
cross-reference the outputs/trellis/ cache 1:1):

    inputs/input.<ext>, inputs/cutout.png
    mesh/raw.glb                      (copied: samples are self-contained)
    irs/yaw.json, massing.json, massing_grid.npz, facade_layout.json,
        facade_rect_<i>.png, facade_cells.npz, walls.json, wall_grids.npz,
        semantic_voxels.npz           (+ <name>.auto.json snapshots after human edits)
    final/blocks.npz, palette.json, build.schem, preview.png, preview_back.png
    manifest.json
    human_edits.json                  (only when the user refined the facade)

The refine UI doubles as human-in-the-loop labeling: on the FIRST edit the auto
predictions are snapshotted to *.auto.json / *.auto.npz, then the corrected results
overwrite the primary files — the sample permanently holds (prediction, correction).
"""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config import OUTPUTS_DIR
from building import ir
from building.params import PIPELINE_VERSION, SCHEMA_VERSION

DATASET_DIR = OUTPUTS_DIR / "dataset"

# IR files snapshotted to *.auto.* before the first human edit overwrites them.
_EDITABLE_IRS = ("facade_layout.json", "facade_cells.npz",
                 "walls.json", "wall_grids.npz", "semantic_voxels.npz")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=str(OUTPUTS_DIR.parent))
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _env_versions() -> dict:
    import numpy, scipy, PIL
    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
    }


class Sample:
    """One dataset sample directory; all pipeline writes go through this."""

    def __init__(self, run_id: str, root: Path | None = None, enabled: bool = True):
        self.run_id = run_id
        self.enabled = enabled
        self.dir = (root or DATASET_DIR) / run_id
        if enabled:
            for sub in ("inputs", "mesh", "irs", "final"):
                (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_rev": _git_rev(),
            "env": _env_versions(),
            "mode": "building",
            "run_id": run_id,
            "image": {},
            "trellis": {},
            "params": {},
            "stages": {},
            "result": {},
            "edited": False,
        }

    # ── inputs ────────────────────────────────────────────────────────────────

    def write_inputs(self, image_path: Path | None, cutout_path: Path | None,
                     glb_path: Path | None):
        if not self.enabled:
            return
        img_info = {}
        if image_path and Path(image_path).exists():
            dst = self.dir / "inputs" / ("input" + Path(image_path).suffix.lower())
            shutil.copyfile(image_path, dst)
            img_info["file"] = "inputs/" + dst.name
            img_info["sha256"] = sha256_file(dst)
        if cutout_path and Path(cutout_path).exists():
            dst = self.dir / "inputs" / "cutout.png"
            shutil.copyfile(cutout_path, dst)
            img_info["cutout"] = "inputs/cutout.png"
            img_info["cutout_sha256"] = sha256_file(dst)
        self.manifest["image"] = img_info

        trellis_info = {"seed": None}
        if glb_path and Path(glb_path).exists():
            dst = self.dir / "mesh" / "raw.glb"
            shutil.copyfile(glb_path, dst)
            trellis_info["glb"] = "mesh/raw.glb"
            trellis_info["glb_sha256"] = sha256_file(dst)
            # Sidecar written by trellis_client._generate_mesh since bldg-0.1.0;
            # older cached GLBs simply have no recorded seed.
            meta = Path(glb_path).with_name(Path(glb_path).stem + "_meta.json")
            if meta.exists():
                try:
                    trellis_info.update(json.loads(meta.read_text(encoding="utf-8")))
                except Exception:
                    pass
        self.manifest["trellis"] = trellis_info

    # ── IRs / stages ─────────────────────────────────────────────────────────

    def write_ir_json(self, name: str, obj: dict):
        if self.enabled:
            ir.save_json(self.dir / "irs" / name, obj)

    def write_ir_npz(self, name: str, **arrays):
        if self.enabled:
            ir.save_npz(self.dir / "irs" / name, **arrays)

    def ir_path(self, name: str) -> Path:
        return self.dir / "irs" / name

    def record_stage(self, name: str, elapsed_s: float, ok: bool, file: str | None = None):
        self.manifest["stages"][name] = {
            "elapsed_s": round(float(elapsed_s), 3), "ok": bool(ok),
            **({"file": "irs/" + file} if file else {}),
        }

    # ── final artifacts + manifest ───────────────────────────────────────────

    def write_final(self, voxel_data: dict, params: dict, warnings: list[str]):
        if not self.enabled:
            return
        import numpy as np
        from schematic import export_schematic

        blocks = np.asarray(voxel_data["blocks"], dtype=np.int16)
        block_ids = voxel_data["block_ids"]
        palette = list(dict.fromkeys(block_ids))            # unique, order-preserving
        index = {bid: i for i, bid in enumerate(palette)}
        colors = np.asarray(voxel_data["colors"] or [(0, 0, 0)] * len(block_ids),
                            dtype=np.uint8)
        shapes = voxel_data.get("shapes")
        shapes = (np.asarray(shapes, dtype=np.int8) if shapes is not None
                  else np.zeros(len(block_ids), dtype=np.int8))
        ir.save_npz(self.dir / "final" / "blocks.npz",
                    coords=blocks,
                    palette_index=np.asarray([index[b] for b in block_ids],
                                             dtype=np.uint16),
                    colors=colors, shapes=shapes)
        ir.save_json(self.dir / "final" / "palette.json", {"palette": palette})
        export_schematic(voxel_data, self.dir / "final" / "build.schem")

        self.manifest["params"] = dict(params)
        self.manifest["result"] = {
            "block_count": int(voxel_data["block_count"]),
            "dims": [int(d) for d in voxel_data["dims"]],
            "schem": "final/build.schem",
            "warnings": list(warnings),
        }
        self.save_manifest()

    def save_manifest(self):
        if self.enabled:
            ir.save_json(self.dir / "manifest.json", self.manifest)

    # ── human-in-the-loop edit capture ───────────────────────────────────────

    def snapshot_auto_ir(self):
        """Before the FIRST human edit: preserve the auto predictions as *.auto.*"""
        if not self.enabled:
            return
        for name in _EDITABLE_IRS:
            src = self.dir / "irs" / name
            if src.exists():
                dst = src.with_name(src.stem + ".auto" + src.suffix)
                if not dst.exists():                # first edit only
                    shutil.copyfile(src, dst)

    def append_human_edit(self, stage: str, facade_id: int,
                          before: dict, after: dict):
        if not self.enabled:
            return
        path = self.dir / "human_edits.json"
        doc = ir.load_json(path) if path.exists() else {"version": SCHEMA_VERSION,
                                                        "edits": []}
        changed = sorted(k for k in set(before) | set(after)
                         if before.get(k) != after.get(k))
        doc["edits"].append({
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage, "facade_id": int(facade_id),
            "before": before, "after": after, "fields_changed": changed,
        })
        ir.save_json(path, doc)
        self.manifest["edited"] = True
        self.save_manifest()


def write_sculpture_sample(run_id: str, image_path, cutout_path, glb_path,
                           voxel_data: dict, params: dict):
    """Minimal dataset sample for a Sculpture-mode run (inputs + manifest + final).

    Sculpture outputs are contrast/negative data for the future model — cheap to keep.
    Failures are swallowed: dataset capture must never break a user build.
    """
    try:
        sample = Sample(run_id)
        sample.manifest["mode"] = "sculpture"
        sample.write_inputs(image_path, cutout_path, glb_path)
        sample.write_final(voxel_data, params, warnings=[])
    except Exception:
        pass
