# Real2Minecraft

Turns one photo of a real object (mainly buildings) into a buildable Minecraft `.schem`.
Everything runs locally; no cloud APIs. The photo is a reference, not a template: the goal is
a recognisable, buildable version in real blocks at the scale the user picks.

**Training plan and decisions (D1–D18): `../BuildDataCollector/docs/mesh-to-schem-plan.md`,
Version 4 (2026-10-11).** That file is the source of truth for the learned model, the dataset
and evaluation. This file only describes the code in this repo.

## Pipeline

| # | Stage | Today (rule-based) | Target (after the model beats the baseline) |
| --- | --- | --- | --- |
| 1 | Background removal | rembg, client-side; the cutout is shown so a bad segmentation is visible (`trellis_client.py`) | same |
| 2 | Image → coloured mesh | TRELLIS over local HTTP, port 7960. Run `C:\TRELLIS\run.bat` (full precision: fits 16 GB, cleaner textures) | same |
| 3 | Normalize | ground at y = 0, XZ-centred; TRELLIS is already Y-up (`mesh_normalize.py`) | same |
| 4 | Voxelize | at block resolution: surface raster + exterior flood fill (`voxelize.py`, `mcvoxel.raster`) | at **2x block resolution** (each block = 2×2×2 cells), same frame as the output (D16) |
| 5 | Voxels → blocks | Sculpture: colour sampling, photo-guided colour repair, palette match, slabs/stairs (`voxelize.py`, `color_repair.py`, `mcvoxel/`). Building (beta): B1 yaw-align → B2 massing → B3 facade → B4 facade map → B5 assemble (`building/`) | **learned model** (3D U-Net, Phase 5). The rule-based output stays in the app as the baseline until the model beats it (D18) |
| 6 | Output | linked mesh / voxel preview (`voxel_view.py`), Sponge v2 `.schem` (`schematic.py`) | same |

Colour mode: "Enhanced" (default) runs the colour repairs; "Raw" keeps model colours and only
lifts brightness. Colour repair never changes geometry.

## Where things live

| Path | What |
| --- | --- |
| `app.py`, `config.py`, `trellis_client.py`, `mesh_normalize.py`, `voxelize.py`, `color_repair.py`, `schematic.py`, `voxel_view.py`, `block_textures.py` | the Tk app and its app-only stages (scripts run from this folder, not a package) |
| `building/` | Building mode B1–B5, IR-in/IR-out stages. Bump `PIPELINE_VERSION` in `building/params.py` on any behaviour change. Active as the baseline (D18) |
| `mcvoxel/` | shared voxel/block core, the only installable package (`pyproject.toml`); BDC installs it editable. See `mcvoxel/README.md` |
| `training/` | offline training tooling; never imported by the app |
| `dev_building.py`, `dev_rebuild.py`, `dev_golden.json` | offline harnesses on cached TRELLIS meshes in `outputs/trellis/`; the building-mode golden |
| `../BuildDataCollector/synth/` | training-pair generation: `degrade.py` (TRELLIS-like inputs), `trellis_compare.py`, `real_eval.py` |
| `outputs/` | gitignored: TRELLIS GLBs, renders, per-run `dataset/` samples, `dataset_dev/` |

## Environment

- Venv `r2m_app` (Python 3.10, untracked; recreate from `requirements.txt`). No torch here.
- Run: `r2m_app\Scripts\python.exe app.py` with the TRELLIS server up (`http://127.0.0.1:7960/ping`).
- TRELLIS lives in its own bundle at `C:\TRELLIS` (Python 3.11, torch, CUDA); the app only talks HTTP.

## Regression bar (any change to `mcvoxel/`, `building/` or the voxelizer)

```powershell
r2m_app\Scripts\python.exe -m unittest discover -s tests
r2m_app\Scripts\python.exe dev_building.py <tag> Easy_Tower_for_M2R_20260706-162025 Hard_Tower_For_M2R_20260706-162335 House_for_M2R_20260706-163500
r2m_app\Scripts\python.exe dev_building.py --golden check <tag>    # must print GOLDEN CLEAN (45 files)
```

Sculpture output must also stay identical on the cached TRELLIS meshes (`dev_rebuild.py`).

## History

InstantMesh, the LLaVA/Ollama vision pass and the PCA orientation system were removed; do not re-add them.
