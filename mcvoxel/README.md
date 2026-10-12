# mcvoxel

Mesh to Minecraft blocks: the voxel and block-processing core shared by Real2Minecraft (the
photo-to-build app), BuildDataCollector (the dataset pipeline) and the Mesh-to-Schem model's
training and inference.

It holds the parts both sides need and nothing app-specific: Real2Minecraft's own sculpture
orchestrator (`voxelize.voxelize_mesh`), its photo-guided TRELLIS colour repairs
(`color_repair`), its `.schem` export and its TRELLIS mesh normaliser live in the app, not here.

```
pip install -e <path to Real2Minecraft>        # editable: edits here are picked up
```

## Conventions (everything below relies on them)

* Voxel grids are `(X, Y, Z)` arrays with **Y up**, **+X east**, **+Z south**. Sponge `.schem`
  files are `(Y, Z, X)`; conversion happens only where schematics are read or written.
* Voxel `(i, j, k)` spans `[origin + (i, j, k), origin + (i, j, k) + 1)` in the *scaled* mesh
  frame (one unit = one block). `origin` travels with every grid, so two meshes voxelised with
  the same `origin` and `shape` land in the same frame voxel for voxel.
* Stair `facing` names the **high** side (verified on 42,233 stairs in real player builds:
  84.0% have solid one up-and-toward `facing`, 5.0% one up-and-away).
* Everything is deterministic: fixed seeds, index-ordered tie-breaks, no global state beyond
  caches of pure functions.

## Modules

| Module | Public API | What it does |
| --- | --- | --- |
| `raster` | `rasterize_surface`, `solid_from_surface`, `clean_solid`, `fine_solid_aligned`, `grid_frame` | Conservative (gap-free) surface voxelisation by the separating-axis test, exterior flood fill, and cleanup (closing + largest part). Explicit `origin` / `shape` pin the frame. |
| `scene` | `load_scene`, `SceneMesh`, `Material`, `scene_color_fns`, `KIND_*`, `split_clutter`, `clutter_kind`, `CLUTTER_WORDS` | Multi-material GLB loader that keeps each face's material, colour, texture and kind (surface / glass / door). Never decimates. `split_clutter` drops vegetation, vehicles and figures by part / material name (whole words; wood species, turf and terrain words excluded; skipped when names would drop over 60% of the model). |
| `voxel_sampling` | `sample_voxel_surface` (+ the app's texture sampler) | Per-voxel colour from every surface sample inside the voxel ("volume" mode): mode-seeking, side-aware, with per-voxel material-kind fractions. |
| `surface_color` | `exposed_colors` | Per-voxel colour as seen from outside ("exposed" mode, the default of `voxelize_scene`): axis rays through each air-facing face, nearest surface wins, facades before roofs, per-channel median; glass / door kinds from the same hits. |
| `grid` | `voxelize_scene`, `VoxelModel`, `pad_model`, `shell_of`, `occluded_interior`, `remove_ground_plate`, `remove_site`, `classify_shapes_banded`, `resolve_slivers` | A scene to a voxel model at any size and in any frame: solid, enclosed rooms, ground plate and site removal, visible shell, slab and stair shapes from a 4x-finer grid built chunk by chunk, surface-repeat (sliver) removal, colours and kinds. `trace=` records the grid after every stage. |
| `block_palette` | `BLOCK_PALETTE`, `BUILDER_PALETTE`, `AUX_PALETTE`, `match_blocks`, `match_blocks_restricted`, `srgb_to_lab`, `lab_to_srgb`, `match_space` | 110 uniform blocks + 9 glasses, matched in CIELAB with lightness down-weighted so hue wins. `BUILDER_PALETTE`: the 72 a builder makes walls and roofs from (no wool, soil, snow, ore or nether-wood blocks). |
| `color_quantize` | `quantize_colors`, `consolidate_block_ids` | Collapse noisy per-voxel colours into a few flat regions before matching. `shadow_repaint` (stage A2) is for TRELLIS bakes; turn it off for albedo sources. |
| `block_shapes` | shape codes, `BLOCK_VARIANTS`, `octant_counts`, `shapes_from_counts`, `classify_shapes`, `apply_shapes`, `base_block` | The octant classifier (full / slab / stairs from sub-voxel occupancy) and the block-variant table. |
| `roofs` | `builder_roofs`, `plan_roofs`, `apply_roofs`, `top_surface`, `harmonize_roof_materials`, `RoofParams` | Builder-style roofs (Approach B): roof caps rebuilt per slope class the way players build them, canonical repeated courses, shallow connected backing, one material per plane/source-material region. |
| `assign` | `assign_blocks`, `BlockTarget` | Voxel model to blocks: colour chain (optionally restricted to a `palette`, with `quantize` settings), glass and door rules, shapes, and the model-ready encodings. |
| `vocab` | `SHAPES`, `MATERIALS`, `encode_state`, `decode`, `IGNORE_*` | The factored vocabulary a model predicts: 10 shapes x 120 materials (`mcvox-v2`; v1 IDs unchanged). |
| `blockstates` | `rotate_y`, `mirror_x`, `finish_states` | Rotation / mirror remap stair facings; assembled-grid stair corners and fence/wall connections. |
| `tiling` | `window_starts`, `iter_windows`, `assemble` | Overlapping windows over a large grid and blended reassembly of predictions. |
| `yaw_align` | `estimate_yaw`, `apply_yaw` | Rotate a mesh about +Y so its facades lie on the grid axes. |

## Tests and the regression bar

```
r2m_app\Scripts\python.exe -m unittest discover -s tests       # from the Real2Minecraft root
```

Any change to `mcvoxel` must also keep Real2Minecraft's outputs where they were, or change them
on purpose: `dev_building.py --golden check <tag>` must say GOLDEN CLEAN (bump
`building/params.py` PIPELINE_VERSION and re-write the golden when a change is meant), and
sculpture mode must stay identical on the cached TRELLIS meshes.

## Builder targets and structural details (0.4.0)

`roofs` fits connected mesh planes to 1:1, 2:1, 3:1 and 1:2 horizontal:vertical
courses, with explicit ridge/hip/valley seams and shallow backing retained in the
exported shell. Long noncanonical planes use a bounded rational fit. Plane decisions,
phase, error and repair counts are returned with the plan. Roof colours come from
upward source hits, grouped by plane and source material. Flat parapets, glass and
source openings are protected; this pass does not close the whole building.

`details.structural_posts` uses fine occupancy width, vertical continuity, separation
from walls and base/overhead anchoring before material assignment can choose a fence
or wall. Explicitly named raised railings/parapets can also qualify. Broad walls,
floaters and named metal/glass/vegetation fail this evidence. Existing slab evidence
continues to describe thin horizontal surfaces. `occluded_interior` excludes isolated
column cross-sections from horizontal wall blockers; it remains a geometric heuristic.

`mcvox-v2` appends shape 8 (fence) and 9 (wall), preserving shape IDs 0–7 and every
material ID from v1. Fence/wall states are derived with `finish_states` after complete
grid assembly, never independently per window. New targets need a 10-class consumer;
v1 targets remain readable, but old 8-output checkpoints cannot predict the new classes
without adaptation. Geometry-only `block_shapes` codes stay unchanged; structural
classification enters the target vocabulary through `assign_blocks`.

The dataset writer validates Minecraft 1.20.1 (DataVersion 3465) states and checks
schematic readback. A 220-block property fixture in `tests/fixtures` was extracted from
[PrismarineJS minecraft-data](https://github.com/PrismarineJS/minecraft-data/blob/master/data/pc/1.20/blocks.json)
(MIT; source URL and SHA256 are retained in the fixture). The schema is an independent
test oracle, not a runtime network dependency.

Full visual audit, reproduction commands and known limitations are in the sibling
BuildDataCollector `docs/builder-roof-quality.md`. Its comparison and fixture galleries
preserve the earlier pilots. No training is part of this change.

## History

* 2026-09-30 (`0.4.0`): coherent roof lattices and exported backing; contextual fences
  and walls; append-only vocabulary v2; column-aware enclosure; parapet preservation.

* 2026-09-29 (`0.3.2`): resumed dataset audit. Bridge repair preserves long links without
  restoring unrelated surface repeats or joining originally separate parts. Explicit
  `clean=True` also closes gaps when `keep_largest=False`. Site removal preserves a low
  building when only a narrow rooftop sign/chimney clears the height threshold; it needs
  a broad tall core before cropping the site. The paired-data debugger and visual audit
  live in BuildDataCollector's `synth` package. Open colonnades can still be overfilled by
  `occluded_interior` even at high resolution; its axis-ray evidence does not establish
  that a space is a room.
* 2026-09-24: extracted from Real2Minecraft's flat modules (with `git mv`, so history follows).
* 2026-09-25: `clean_solid` fixed -- scipy's closing had stripped every grid's outer layer
  (sculpture builds came out 2 blocks short of the requested size); app-only modules moved back
  to the app; `roofs` added; `shadow_repaint` made explicit.
* 2026-09-27: target-quality rework after the pilot-v2 audit (`tests/test_mcvoxel_quality.py`
  holds a regression test per item, each failing under the old behaviour):
  * `boundary_slivers` -> `resolve_slivers`: a voxel is a surface repeat only if its content
    lies against the face of the neighbour that represents the surface (fine-layer masks from
    `classify_shapes_banded`). The old "touches a substantial neighbour" rule deleted
    parapets, roof courses and porch posts and doubled nothing less.
  * `surface_color.exposed_colors` (default `color_mode="exposed"`): colour from what is seen
    through a voxel's air-facing faces. Volume pooling mixed a balcony's floor and front,
    painted windows and walls, thatch and the beams over it.
  * `remove_site`: lawns, pools, paths, boundary walls, fences and posts around the building
    (a 2D opening keeps masses at least 3 columns wide; `site_height` in voxels).
  * `scene.split_clutter`, `block_palette.BUILDER_PALETTE`, `assign_blocks(palette=, quantize=)`,
    `harmonize_roof_materials(rgb=)`, `rasterize_surface(pairs=)` (voxel / triangle pairs).
  * Found, not changed: `clean_solid`'s closing fills 1-voxel gaps between columns and the air
    of shallow porches. Callers voxelising CLEAN models should pass `clean=False` to
    `voxelize_scene` (BuildDataCollector does); noisy TRELLIS meshes keep it. At 1 block = 1 m
    the enclosed-room fill still fills a gap of one voxel between two posts (posts about 2 m
    apart); posts 3 m or more apart keep their porch open.
