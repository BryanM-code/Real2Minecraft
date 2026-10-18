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
| `vocab` | `SHAPES`, `MATERIALS`, `VALID`, `classify_state`, `encode_state`, `encode_target`, `decode`, `validate_state`, `IGNORE_*` | The factored vocabulary a model predicts: 30 shapes x 313 materials, 1,232 valid pairs (`mcvox-v3`; v1, v2 and earlier v3 IDs unchanged). See [Vocabulary v3](#vocabulary-v3-mcvox-v3-060-061). |
| `blockstates` | `rotate_y`, `mirror_x`, `rotate_state`, `mirror_state`, `finish_states` | Rotation / mirror for every directional and axis shape (grids and blockstate strings); derived states on the assembled grid: stair corners, fence / wall / pane connections, door upper halves and hinges, fence gate `in_wall`. |
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

## Vocabulary v3 (`mcvox-v3`, 0.6.0, 0.6.1)

A target voxel is a (shape, material) pair. The **material** is a texture family, named by
its full block: `minecraft:dark_oak_planks` also covers dark oak stairs, slabs, fences,
gates, doors and trapdoors. The **shape** is geometry plus orientation. `VALID[shape,
material]` (30 x 313, 1,232 true) records which pairs exist in the game. `decode` returns a
real 1.21.11 blockstate for every valid pair; training targets only hold valid pairs, and
inference masks the invalid logits.

| IDs | Shapes | Valid materials |
| --- | --- | --- |
| 0–9 | air, full, slab_bottom, slab_top, stairs_east / west / north / south, fence, wall (v1, v2) | full: all 312; slab 56; stairs 53; fence 13 (12 woods, nether brick); wall 25 |
| 10–13 | stairs_top_east / west / north / south: upside-down stairs | the 53 with stairs |
| 14 | pane | 22: glass, 16 stained glasses, iron bars, 4 copper bars |
| 15–16 | trapdoor_bottom, trapdoor_top (closed) | 17: 12 woods, iron, 4 copper stages |
| 17–20 | trapdoor_open_north / south / east / west | the same 17 |
| 21–24 | door_north / south / east / west: the lower half | the same 17 |
| 25–26 | full_axis_x, full_axis_z: horizontal pillars | 56: logs, wood, stems, hyphae (stripped too), bamboo blocks, basalt, polished basalt, bone block, deepslate, quartz pillar, purpur pillar, hay, froglights |
| 27–28 | fence_gate_x, fence_gate_z: a closed gate spanning that axis | 12 woods |
| 29 | carpet | 17: 16 wools, moss |

Doors and open trapdoors are named by the side of the voxel their plate occupies;
Minecraft's `facing` is the opposite side. Stair `facing` names the high side for both
halves.

**Materials: 120 to 312.** IDs 0–119 are v2's (`[air] + sorted(BLOCK_PALETTE) +
sorted(AUX_PALETTE)`), unchanged. `V3_MATERIALS` appends 193 in a fixed order (181 in
0.6.0, 12 more in 0.6.1):

* Woods: pale oak and bamboo planks; the log, stripped log, wood and stripped wood of every
  1.21 species (stems and hyphae for crimson and warped, bamboo blocks for bamboo).
* Leaves: the 9 overworld species, azalea and flowering azalea (full-shape materials).
* The 9 stained glass colours v2 lacked.
* Stone: mossy cobblestone and stone bricks, cracked and chiseled stone bricks; the deepslate,
  tuff, sandstone, red sandstone and quartz families; red nether, polished blackstone, end
  stone and prismarine bricks; polished basalt; purpur pillar.
* Copper: cut, chiseled, grate and bulb at every oxidation stage.
* All 16 concrete powders and glazed terracottas.
* Full blocks the dataset builds with (in at least 2 builds or 200 supervised blocks):
  mushroom blocks, hay, froglights, glowstone, sea lantern, redstone lamp, bookshelf, beehive,
  lodestone, netherrack, soul sand, bedrock, sand, suspicious sand, podzol, pale moss, ice,
  sponge, dried kelp, mangrove roots, dead coral blocks, coal / redstone / diamond /
  netherite blocks, and five ores.
* 0.6.1 (IDs 301–312): building material the dataset's Phase 3 re-crop kept, which met the
  same rule then: gold, emerald and redstone ore, emerald block, chiseled polished
  blackstone, suspicious gravel, rooted dirt, red sand, shroomlight, honey block, bee nest,
  and bamboo mosaic (with its slab and stairs). On the later targets shroomlight (no
  supervised block), redstone ore and honey block (one build each) fall below the rule; they
  stay, like the other materials no supervised block uses.

The app's palettes (`BLOCK_PALETTE`, `BUILDER_PALETTE`, `AUX_PALETTE`) are unchanged; v3
materials exist only in the vocabulary.

**Derived, never predicted.** `blockstates.finish_states` restores these on the assembled
grid: stair corners (both halves), fence / wall / pane connections and wall `up`, and a
door's upper half and hinge (by the game's placement rule). It also sets a fence gate's
`in_wall`. Properties with no visible effect are dropped: waterlogged, powered, leaf
distance (decoded leaves are persistent), mushroom faces and snowy.

**Lossy.** These encode, but a visible property is dropped; `classify_state` names the
reason:

* `door_open`: becomes a closed door on the plate's side.
* `gate_open`
* `trapdoor_facing`: closed trapdoors decode facing north.
* `trapdoor_half`: open trapdoors decode `half=bottom`.
* `facing`: full blocks with a facing, such as glazed terracotta.
* `lit`
* `double_slab`: becomes the full block.
* `alias`: applied at read time and not part of the vocabulary. `waxed_*` maps to unwaxed,
  `infested_*` to the host block, and `petrified_oak_slab` to `oak_slab`.

**IGNORE, never air.** `encode_target(blocks, palette, shell, exterior, interior)` makes
exterior voxels air and encodes the shell blocks the vocabulary holds. Everything else
becomes IGNORE (shape 255, material 65535):

* the interior
* decor: torches, lanterns, buttons, plates, signs, banners, ladders, chains, rails,
  redstone, plants, furniture, beds and heads
* fluids and snow layers
* blocks outside the vocabulary
* a door's upper half

**Target version (D21): Minecraft 1.21.11, DataVersion 4671.** This is the newest 1.21.x;
pale oak needs 1.21.4 or later.

* `mcvoxel/blocktables/blocks_1_21_11.json` holds the block-property table: 1,166 blocks, from
  [PrismarineJS minecraft-data](https://github.com/PrismarineJS/minecraft-data/blob/master/data/pc/1.21.11/blocks.json)
  (MIT). The file keeps the source URL, commit and SHA256.
* `validate_state(state, TARGET_DATA_VERSION)` checks a state against that table. The
  default (`V2_DATA_VERSION`, 3465) keeps v2's 1.20.1 rules, so the app's export is
  unchanged.
* Two blocks in real-schems-v1 are newer than any 1.21.x and become IGNORE:
  `poplar_fence` and `stripped_poplar_wood` (one build, DataVersion 5023).
* BuildDataCollector's reader renames `chain` to `iron_chain` (1.21.9).

**Compatibility.**

* v1 and v2 IDs are unchanged, and every v2 pair still decodes.
* Pairs v3 does not make valid decode exactly as in v2: deepslate's stairs, slab and wall
  are cobbled deepslate's; copper's stairs and slab are cut copper's.
* Eight pairs that v2 substituted now decode to the real block: tuff's slabs, stairs and
  wall, and the nether brick fence.
* A 10-way shape head cannot predict v3; there is no checkpoint yet.
* `tests/fixtures/vocab_mcvox_v2.json` freezes v2's tables and all 1,200 v2 decodes.
* `tiling`'s shape accumulator grows to 30 x 2 bytes a voxel.

## History

* 2026-10-17 (`0.6.1`): 12 materials appended to `mcvox-v3` (IDs 301–312, 22 more valid
  pairs; every earlier ID and pair unchanged, frozen in `tests/fixtures/vocab_mcvox_v3_0.json`).
  Raises real-schems-v1's representable supervised blocks from 98.51% to 98.58%.

* 2026-10-12 (`0.6.0`): vocabulary v3 (`mcvox-v3`, decisions D20 and D21). It adds
  upside-down stairs, panes, trapdoors, doors, horizontal pillars, fence gates and carpets;
  materials grow from 120 to 301, with a `VALID` table and `encode_target` (IGNORE, never
  air).
  * Validation targets 1.21.11.
  * `rotate_state` / `mirror_state`; `finish_states` restores door halves and hinges, pane
    connections and gate `in_wall`.
  * Covers 98.5% of real-schems-v1's supervised blocks, against 75.9% for v2.
  * App output is unchanged: golden clean, and sculpture identical on 53 cached meshes in
    both colour modes.

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
