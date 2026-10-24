# blueprint: buildings described by their parts

A building as a short description of its parts, and a builder that turns the description
into blocks. This is the output format for the photo model planned next: instead of
placing tens of thousands of blocks, a model only fills in the description (how many
wings, their sizes, roof types, where the windows and doors are, which block goes where),
and the builder does the rest. No AI is involved in building.

```
blueprint (JSON, ~200 values for a house)  ->  build  ->  (shape, material) grids  ->  build.schem
```

The grids are the mcvox-v3 representation of the human builds (`mcvoxel.vocab`), so a
built blueprint is written by `photofirst.schem.write_schem` (stair corners, door halves,
fence and pane arms derived over the whole grid; DataVersion of 1.21.11) and scored with the
same metrics as any other build.

## The format (`blueprint-1`; `blueprint-0` still loads)

Frame: x east (to the right seen from the front), y up, z south (towards the viewer); the
front is the south side. Units are blocks. Full reference: `spec.py`'s docstring.

| Part | What | Main fields |
| --- | --- | --- |
| `materials` | role -> block (`wall`, `wall_base`, `trim`, `roof`, `roof_edge`, `window`, `frame`, `sill`, `shutter`, `door`, `garage`, `floor`, `chimney`, `cap`, `skylight`, `column`, `railing`, `dome`, `awning`) | per mass overrides too; a shape a block lacks falls back (stairs and slabs to full, fence to wall) |
| `masses` | walls on a `shape`: box, round, octagon | `x z w d y`, `storeys` x `storey_h` or `walls`, `floors`, `plinth` (number or per side), `corner_trim`, `decor` (half_timber, quoins, bands, pilasters, buttresses, curtain) |
| `masses[].roof` | gable (ridge `x` or `z`), cross_gable, hip, pyramid, shed (`high` side), flat (`parapet`, `battlements`), mansard, gambrel, sawtooth, spire, cone, dome, pagoda | `pitch` 0.5 (slabs), 1, 2 or 3 (stairs), `overhang` 0-3; round masses get round roofs; `skirt` (rows of slope from the eave only: a pagoda tier's eave, its corners curled up); `gable_end` plain / stepped (crow-stepped gable walls above the roof) |
| `openings` | window (frame, `mullions`, `sill`, `shutters`, `style` rect / arched / lancet), door (2 high), garage (recessed), arch (open, an arcade bay), panel (a patch of another block) | `mass side u y w h`; `u` from the wall's left end seen from outside |
| `features` | chimney, skylight; on a wall: portico (columns, pediment), porch (posts, railing, lean-to, `wrap` to `wrap_side`), balcony, bay window, awning; on a roof: dormer, dome (drum, lantern), finial (a mast with rings); free: smokestack (`stack`), canopy; others: windmill sails, gallery, lantern | `parts.py` |
| `site` | `ground` pads (a lower pad digs a higher one away), `boxes`, `bushes`, `trees` | optional |

Parts are drawn in a fixed order (ground, walls and floors, roofs from the lowest eave up,
openings, features, boxes and plants) and every block remembers the part that placed it: a
roof never cuts into another mass's wall, an opening only replaces its own wall, and
plants only grow into air. `load` reports every problem in a blueprint at once
(`BlueprintError`), which is what a model's output will be checked with.

## From a photo: read -> segment -> measure -> compose -> build -> evaluate and correct

| Module | Job |
| --- | --- |
| `styles.py` | 30 styles and 35 kinds (covering BuildDataCollector's real-schems-v2 categories and the styles its names mention), each with a hint for the model, the roofs, windows and structures it is known for, typical storeys and proportions, and a block palette per style |
| `describe.py` mode `read` | the local model (`r2m_vlm`): a free description, the kind, then the style (kind first: the better order on the 50-photo test), then the form `form2(style, kind)`: the parts left to right (share of the front, storeys, shape, roof with a definition of each type and `not_visible`, what closes the roof's ends: a triangle (gable) or a slope (hip), ridge, pitch in degrees, windows, door) and only the features that style and kind have (porch, portico, dome, towers, tiers, crown, podium, houses in a row and their gables, ...) |
| `segment.py` | the building's pixels: a segmentation model (EoMT-L, ADE20K, MIT licence, in `r2m_vlm`) labels every pixel; the building classes inside the vision model's box round the main building are the building, trees, cars, people, fences and poles in front of it are occluders (left out of the comparison); the frame is carried out to the box where occluders hide the building's ends |
| `measure.py` | width, depth and height in metres from the photo (`r2m_train`: MapAnything on the segmented building, the photo-first levelling), and the camera's angle to the facade |
| `compose.py` | form + style + kind (+ measurement, + the evaluator's `adjust`) -> blueprint-1: sizes in metres at an adaptive scale (2 blocks a metre, less for big buildings), part widths from their shares, the parts the form names, rows of joined houses (party walls, a door each, stepped or front gables, chimneys on the party walls), pagodas (a platform, tiers under eave skirts, galleries, a finial), podiums and crowns (a plant room, a stepped top with a spire, pinnacles), the style's palette under the colours the model reports; `mirror` flips a blueprint left to right |
| `vision.py` | the vision model beyond the form: the main building's box and the side the photo shows (`locate`), window and door boxes on houses, shops and barns (`openings`), a critique of a build against its photo in the builder's own terms (`critique`), and which of two builds is closer (`compare`) |
| `score.py` | the computer measures of a build against its photo: outline overlap with the occluded pixels left out, proportions, the roof line, where the windows and doors are, the walls' and top's main colours |
| `edits.py` | the corrections the evaluator may make, each a change compose can build: style, kind, storeys, roof type, pitch (27 to 72 degrees), ridge, width / height / depth, a part's share, mirror, add or remove a part, any form feature on or off, windows, colours, wall material, which part has the skylights, houses in a row, the garage's wall |
| `refine.py` | the evaluator (below) |

## The evaluator (`refine.py`)

Photo -> the model's reading -> the build -> the vision model compares a render of the build
with the photo and says what is wrong -> corrections -> rebuild -> ... until the score is good
enough (0.8 of 1), four rounds have passed, or no correction helps.

- **Two evaluations.** The computer measures (`score.py`) are repeatable but only as good as
  the segmentation and the camera angle; the vision model sees what they cannot (the style, a
  missing porch, the garage on the wrong side, a tower that should not be there) and gives an
  overall resemblance, 0-10. The score is the mean of the two.
- **Each round** every correction the critique asks for, and every one the measures point to
  (too tall for its width, a roof line higher or lower than the photo's, too many windows,
  the photo's wall or roof colour, flipped), is built and measured; a build and its two
  renders take about half a second. Three are tried in turn, the model's own first (led by
  the measured gain alone, small changes of proportion always came first and its asks were
  never tried). A change of proportions the measures found is kept on the measures (+0.02).
  Any other correction is **checked by critiquing the candidate build on its own**: kept when
  the model's complaint is gone (no "missing porch" once the porch is there), nothing in the
  new critique asks to undo it, and its rating does not drop; one the measures found also
  needs +0.05 (pitch, windows, colours) or +0.08 (a feature, a roof type), and a storey,
  style, kind or mirror image of theirs needs a higher rating. Asking instead which of two
  builds is closer does not work with this model: it picked the image shown last 59 times
  in 70, and named the same build both ways round in only 5 pairs of 35.
- **Undo and best build.** A correction the next critique contradicts (a porch added, then
  called extra) is undone, and the loop returns the best build it rated, not the last: each
  correction looked better at the time, yet a run of them can end worse.
- **Only what the builder can draw.** The critique is asked in the builder's terms: the roof
  types, pitches, features, styles and kinds it has, and nothing else; a correction is always
  one of `edits.py`'s. What the model wants beyond that ("other", a feature or roof the list
  lacks) is logged as a gap, and so are corrections that hit a limit (no roof steeper than 72
  degrees) and corrections that changed no block; `refine summary` counts them over a run,
  which is the list of parts to build next.
- **The camera.** Which side the photo shows comes from the vision model: MapAnything's left
  and right agreed with the better outline on only 26 of 46 photos. The angle and height come
  from MapAnything.

## Commands

One photo to a schematic, the whole local chain (the reading, the segmentation, the
measurement, the evaluator; each step in its own environment):

```
r2m_app\Scripts\python -m blueprint.photo house.jpg --out outputs\blueprint\house [--rounds 4]
```

`--rounds 0` keeps the first reading's build. The stages on their own, for a folder of
`<id>/photo.jpg` photos:

```
r2m_vlm\Scripts\python -m blueprint.refine prepare outputs\vlm\test50 --out outputs\vlm\run3
r2m_train\Scripts\python -m blueprint.measure outputs\vlm\test50 --out outputs\vlm\run3
r2m_vlm\Scripts\python -m blueprint.refine run outputs\vlm\test50 --out outputs\vlm\run3
r2m_vlm\Scripts\python -m blueprint.refine judge outputs\vlm\test50 --out outputs\vlm\run3
r2m_vlm\Scripts\python -m blueprint.refine summary --out outputs\vlm\run3
```

`judge` asks the vision model, first build against the corrected one, twice with the order
swapped; `summary` gives the scores, the corrections kept and the capability gaps.


From the repository root:

```
r2m_app\Scripts\python -m blueprint blueprint\examples\cottage.json --out outputs\blueprint\cottage
..\BuildDataCollector\.venv\Scripts\python -m blueprint blueprint\examples\cottage.json --out outputs\blueprint\cottage --render
```

`--render` (BuildDataCollector's environment, its orbit renderer) saves four views.
Tests: `tests/test_blueprint.py`.

## A local vision model (`describe.py`, `assemble.py`; 2026-10-22)

Qwen3.5-9B (Alibaba, Apache 2.0) in `r2m_vlm`, 4-bit with the vision encoder in bf16: about
10 GB of GPU memory, 90 s to load. Test photos: the user's split-level house and nine
Wikimedia Commons photos of landmarks (free licences, kept local), graded against my reading
of each photo (`outputs/vlm/truth.json`).

| Mode | Result | Time per photo |
| --- | --- | ---: |
| `survey`: 13 structure questions with fixed answers | 86% right (98/114): storeys 10/10, domes 10/10, columns 8/8, garage 10/10; wings 3/6, roof type 7/10, balconies 5/8 | 10 s |
| `survey2`: the same with definitions of wing, roof, tower | 91% (104/114), wings 5/6; optimistic, the definitions were written after seeing these photos | 11 s |
| `survey --think` (reasoning first) | 68% (77/114): worse and much slower | 155 s |
| `describe`: free description | rich and mostly right (found the house's satellite dish), some invented parts | 55 s |
| `blueprint`: the whole blueprint | 0 of 3 valid (doors off walls, wool roofs, walls under 3 blocks), still 0 after 2 repair rounds | 100-480 s |
| `form`: describe, then fill in a per-wing form; `assemble` builds the geometry | 10 of 10 valid builds; houses and towers keep their overall shape, landmarks lose what the format lacks (domes, porticos, turrets) | 90 s |

So the model reads buildings well but cannot place geometry; the form is the right output
for it. What limits the result now: the parts the format has, the form's questions (the
house's right wing came out hip-roofed with its garage door on the front), and the
assembler's sizes, which are rules, not measurements.

## The 50-photo test (2026-10-23)

The earlier 10 photos plus 40 freely licensed Wikimedia Commons photos across the dataset's
styles and kinds (licence and author per photo; local only), answers graded against my reading
of each photo, builds compared with the photos' cut-out outlines seen from the camera angle
MapAnything recovered.

| | Order A (style first) | Order B (kind first) |
| --- | ---: | ---: |
| Answers right (464 graded: style, kind, parts, storeys, roof, wall, features) | 92% | **94%** |
| Style / kind | 46 / 44 of 50 | 49 / 45 of 50 |
| Outline overlap with the photo (22 photos with the building wholly in frame), measured | 0.509 | **0.545** |
| The same without the measurement | 0.439 | 0.467 |
| Outline overlap, all 50 photos, measured / not measured | 0.502 / 0.480 | 0.516 / 0.487 |

Against the first form (`form` + `assemble`) on its 10 photos: answers 74% -> 94%, outline
overlap 0.349 -> 0.430. On the 22 photos with the building wholly in frame, measuring improved
12 builds and worsened 1 (14 and 8 over all 50: a building that runs off the frame is measured
only in part). Weak still:
pagodas (the right outline, not the look), skyscrapers seen from below (the model's storey
count, 65 for about 20), photos whose cut-out fails (12 of 50 rejected by the app's checks).

## Round 3: segmentation, roof questions, new parts, the evaluator (2026-10-23)

On the same 50 photos (report: `outputs/vlm/report3/report.md`, local):

| | Before | Now |
| --- | ---: | ---: |
| Photos with a usable building mask | 38 (the app's background removal) | **50** (segmentation) |
| Roof type right / as built | 41 / 40 of 46 | 42 / 41 of 46 |
| All graded answers | 94.0% | 94.0% (93.8% after the evaluator: one tower removed wrongly) |
| Measured total, round-2 builds -> first builds -> corrected | 0.517 -> 0.542 | **0.622** |
| Outline overlap / proportions, corrected | 0.620 / 0.557 (round 2) | **0.747 / 0.812** |
| The vision model's rating, first -> corrected | 6.8 | 7.5 |

The evaluator changed 35 of 50 builds (76 corrections: 50 of proportions, 18 features, 3
pitches, ...) and raised the measured score on 35, lowering it on none. It is reliable for
proportions; for features it is as good as the 9B model's judgement, which is weak (it
added a canopy to a bungalow, removed a mansion's tower). What the builder lacks most, from
its gap log: detailed porticos and columns, dome tops, spires and pinnacles on towers, clock
faces, roofs gentler than 27 degrees, storeys taller than 8 blocks.

## Status (2026-10-22)

First version. A hand-written blueprint of a split-level house photo (two offset gable
wings, skylights, garage, brick base, chimney, site) builds into a result on par with
ChatGPT's hand-written 229-line program for the same photo, both drawn by the same
renderer. Next: more parts (porticos and columns, domes, towers and turrets, porches,
dormers, setbacks), better form questions, sizes measured from the photos, and real photos
labelled with forms (the model drafts, a person corrects) to fine-tune the model on.
