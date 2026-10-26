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
| `masses[].roof` | gable (ridge `x` or `z`), cross_gable, hip, pyramid, shed (`high` side), flat (`parapet`, `battlements`), mansard, gambrel, sawtooth, spire, cone, dome, pagoda | `pitch` 0.25 or 0.5 (slabs), 0.75 (a stair, then a slab), 1, 2 or 3 (stairs), `overhang` 0-3; round masses get round roofs; `skirt` (rows of slope from the eave only: a pagoda tier's eave, its corners curled up); `gable_end` plain / stepped (crow-stepped gable walls above the roof) |
| `openings` | window (frame, `mullions`, `sill`, `shutters`, `style` rect / arched / lancet), door (2 high), garage (recessed), arch (open, an arcade bay), panel (a patch of another block) | `mass side u y w h`; `u` from the wall's left end seen from outside |
| `features` | chimney, skylight; on a wall: portico (columns, pediment), porch (posts, railing, lean-to, `wrap` to `wrap_side`), balcony, bay window, awning; on a roof: dormer, dome (drum, lantern), finial (a mast with rings); free: smokestack (`stack`), canopy; others: windmill sails, gallery, lantern (glass, or `open` on corner posts like a belfry; a pyramid, dome or spire `cap`) | `parts.py` |
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
| `describe.py` mode `read` | the local model (`r2m_vlm`; Qwen3.5-9B, or Qwen3.8-27B through llama.cpp with `--model qwen3.8-27b`): a free description, the kind, then the style (kind first: the better order on the 50-photo test), then the form `form2(style, kind, 3)`: the parts left to right, each with its width, depth and eave height in metres, its floor's level and its front's offset against the main block's, storeys, shape, roof (a definition of each type and `not_visible`, what closes its ends: a triangle (gable) or a slope (hip), ridge, slope in degrees), windows, door, garage doors in its front and in its end wall, a base band; then only the features that style and kind have (porch, portico, dome, towers, tiers, crown, podium, houses in a row and their gables, ...) |
| `segment.py` | the building's pixels: a segmentation model (EoMT-L, ADE20K, MIT licence, in `r2m_vlm`) labels every pixel; the building classes inside the vision model's box round the main building are the building, trees, cars, people, fences and poles in front of it are occluders (left out of the comparison); the frame is carried out to the box where occluders hide the building's ends |
| `measure.py` | width, depth and height in metres from the photo (`r2m_train`: MapAnything on the segmented building, the photo-first levelling), the camera's angle to the facade, and each of the vision model's window, door and garage door boxes placed on the building from MapAnything's points: its wall (front, left or right), how far along it, how high up it (from the wall's foot, which bushes often hide in places, to its eave or a front gable), its size |
| `compose.py` | form + style + kind (+ measurement, + the evaluator's `adjust`) -> blueprint-1: sizes in metres at an adaptive scale (2 blocks a metre, less for big buildings), each part's width, depth, height, level, offset, roof slope, base and garage doors from the form (version 3; its metres as ratios: the overall size from the parts' metres, the proportions from MapAnything's), its own rules where the form says nothing, what the form asks and the build cannot do in `meta["conflicts"]` (more windows than a wall holds; garage doors that need a longer wall make it longer), the parts the form names, rows of joined houses (party walls, a door each, stepped or front gables, chimneys on the party walls), pagodas (a platform, tiers under eave skirts, galleries, a finial), podiums and crowns (a plant room, a stepped top with a spire, pinnacles), the style's palette under the colours the model reports; `mirror` flips a blueprint left to right |
| `vision.py` | the vision model beyond the form: the main building's box and the side the photo shows (`locate`), window, door and garage door boxes on houses, shops and barns (`openings`), the photo's own answers about each of the build's parts (`check`), a critique of a build against its photo in the builder's own terms (`critique`), and which of two builds is closer (`compare`) |
| `facade.py` | the photo's windows, doors and garage doors on the build where the photo has them (`place_openings_3d`, from `measure.py`'s places): on the part that is there, at the height a builder would use; garage doors that do not fit make the wall longer; what cannot be placed is reported. Also the build's openings in the same terms for the scores, and the parts' levels the photo's doors give (`levels`) |
| `score.py` | the computer measures of a build against its photo: outline overlap with the occluded pixels left out, proportions, the roof line overall and over each part (`sections`, from a render of which part each pixel is), the windows, doors and garage doors matched on the building (`match3d`; in the frame without MapAnything's places), the walls' and top's main colours; the build's side from renders of what each block is (`build`'s labels) and which part it belongs to (`render.render_codes`) |
| `edits.py` | the corrections the evaluator may make, each a change compose can build: style, kind, storeys, roof type, pitch (14 to 72 degrees; the loop leaves it alone), ridge, width / height / depth, a part's share, height, depth, level or offset, a part's garage doors, mirror, add or remove a part, any form feature on or off, windows, colours, wall material, which part has the skylights, houses in a row, the garage's wall; and where each comes from: the critique, the photo-only check, the measures |
| `refine.py` | the evaluator (below) |
| `calibrate.py` | the evaluator's test: a blueprint made by hand from a photo, and the same with one deliberate mistake at a time (a wing's roof hipped, the split level gone, a garage door missing, 60/40 widths, half the depth, 45-degree roofs, mirrored, red walls, a wing missing, flat roofs), scored by the measures and critiqued by a vision model |

## The evaluator (`refine.py`)

Photo -> the model's reading -> the build (with the photo's windows, doors and garage doors
where the photo has them) -> the photo-only check -> the vision model compares a render of the
build with the photo and says what is wrong -> corrections -> rebuild -> ... until the score is
good enough (0.8 of 1), four rounds have passed, or no correction helps.

- **The roof pass (round 5).** After the reading, each part's roof is asked about one concrete
  question at a time, about what the photo shows rather than what it means (`refine.roof_pass`,
  `vision.ROOF_QUESTIONS`): what tops its front wall (a triangle, a level eave, an eave with a
  gable rising from it, flat, other), what closes its ends, how steep it is (low, medium, steep),
  where a gable rising from its eave stands, what tops a tower. Compose takes the answers where
  they read the test photos better than the form (`compose.ROOF_TRUST`: the front top, the pitch,
  the tower's top; not the ends). Where the pass and the form disagree on a flat roof against a
  pitched one, a yes / no question about the roof's sloping surface decides, asked twice with
  yes and no swapped (`refine.settle_front`, `vision.ROOF_CHECKS`). The loop no longer changes
  pitches. See round 5 below.
- **The photo-only check (round 4).** Asked to compare a render with the photo, neither model
  compares: given both, it describes the render (a wing's hip roof the photo does not have:
  "roof_type hip, roof_ok true"; a mirrored build's garage doors given to its left part). So
  the model is asked about the photo alone, one answer per part of the build (whether the
  building has it, what closes its roof's ends, which way its ridge runs, its floor against the
  main block's, its storeys, its garage doors, its base band, which part is tallest), twice with
  the choices reversed, and the code compares the answers both give with the build. By default
  it acts only where the form's reading agrees (the build then differs from both: compose's
  rules or an earlier correction put it wrong); where the two readings disagree, which is right
  was a coin toss on the 50 test photos. Each difference is a correction, kept unless the
  measures find it much worse (on real photos they hardly tell right corrections from wrong,
  see round 4 below); a part's level the answers and the photo's doors agree on (MapAnything
  places the doors in 3D: a split level's garage doors stand well below the entrance) even
  then, since a silhouette cannot judge levels.

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
r2m_app\Scripts\python -m blueprint.photo house.jpg --out outputs\blueprint\house [--rounds 4] [--model qwen3.8-27b]
```

`--rounds 0` keeps the first reading's build; `--model qwen3.8-27b` reads and checks with
Qwen3.8-27B through llama.cpp (`r2m_llama/`) instead of Qwen3.5-9B. The stages on their own,
for a folder of `<id>/photo.jpg` photos:

```
r2m_vlm\Scripts\python -m blueprint.refine prepare outputs\vlm\test50 --out outputs\vlm\run4 --model qwen3.8-27b
r2m_train\Scripts\python -m blueprint.measure outputs\vlm\test50 --out outputs\vlm\run4
r2m_vlm\Scripts\python -m blueprint.refine roofs outputs\vlm\test50 --out outputs\vlm\run4 --model qwen3.8-27b [--redo]
r2m_vlm\Scripts\python -m blueprint.refine run outputs\vlm\test50 --out outputs\vlm\run4 --rounds 3 --model qwen3.8-27b
r2m_vlm\Scripts\python -m blueprint.refine judge outputs\vlm\test50 --out outputs\vlm\run4
r2m_vlm\Scripts\python -m blueprint.refine summary --out outputs\vlm\run4
```

`prepare` runs the roof pass on new readings; `roofs` runs it on readings made before it (or,
with `--redo`, again on all of them). `judge` asks the vision model, first build against the
corrected one, twice with the order swapped; `summary` gives the scores, the corrections kept
and the capability gaps.

The evaluator's own test, a hand-made blueprint against ten deliberate mistakes (the photo
prepared and measured in `--photo-dir` first):

```
r2m_vlm\Scripts\python -m blueprint.calibrate outputs\blueprint\woodland\woodland.json --photo-dir outputs\vlm\m3test --id woodland --out outputs\vlm\calib [--critic qwen3.8-27b]
```


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

## Round 4: parts in metres, openings where the photo has them, conflicts, element scores, a photo-only check (2026-10-23)

An audit of the woodland house's run found the reader's mistakes (a hip-roofed wing that is
gabled, no split level, one garage door of two, 60/40 widths, 45-degree roofs its own words
called 30), the composer's fixed rules (a wing 0.8 times the main block's depth, "forward" = 3
blocks), a garage door dropped silently, a loop that widened the house and nothing else, scores
that rewarded that, and a critic that called it all right. What changed, for every kind the
builder makes:

- **Each part in metres (form v3).** Width, depth, eave, the floor's level and the front's
  offset against the main block, the roof's slope in degrees, garage doors in the front and the
  end wall, the base band and its height on the front and end walls. Compose takes them as
  ratios; the rules only fill in what the form leaves out. Roofs of 14 degrees exist.
- **Openings where the photo has them** (`measure.opening_positions`, `facade`).
- **No silent drops**: `meta["conflicts"]`, walls that grow for garage doors, and conflicts the
  loop turns into corrections.
- **Scores per element** (windows, doors, garage doors on the building; each part's roof line;
  the plan's depth), from what each block is, not its block name.
- **The loop.** A photo-only check per part comes first; the critique's structural corrections
  now need the measures' agreement; aspect corrections scale the footprint, not the width.

**The critics, calibrated.** The woodland house built by hand, and the same with one deliberate
mistake at a time (`calibrate`):

| | Measures | Qwen3.5-9B critic | Qwen3.8-27B critic |
| --- | ---: | ---: | ---: |
| Mistakes rated below the right build | 8 of 10 | 2 of 10 | 2 of 10 |
| Mistakes named | | 3 of 10 | 4 of 10 |
| The right build's rating | 0.547 | 9 of 10 | 9 of 10 |

Neither model compares a render with a photo: given both, it describes the render (the hip
roof the photo does not have: "roof_ok"). Hence the photo-only check. The two mistakes the
measures do not rank below the reference (60/40 widths, 45-degree roofs) are ones where the
hand-made reference itself is uncertain.

**Over 50 buildings of 21 kinds** (`calibrate synthetic`: each reading built and rendered as its
own photo, each mistake one of the loop's corrections the wrong way), the share where the
measures rank the right build first:

| Mistake | Camera right | Camera 10 / 10 degrees off |
| --- | ---: | ---: |
| roof type / ridge / a part removed / half depth / shares / a storey more | 100% | 98-100% |
| a part's level | 100% | 93% |
| pitch steeper / flatter | 100% | 79-89% |
| mirrored | 94% | 92% |
| a wing's roof type | 88% | 75% |
| wall colour | 48% | 45% |

**On real photos that separation mostly disappears.** Each right correction (my labels as the
check's answers) and each wrong one (the opposite answers), measured on its own from round 3's
builds of the 50 photos: allowed to lose at most 0.005, the measures kept 10 of 16 right
corrections and 69 of 117 wrong ones (at most 0.05: 14 and 96). Perspective, occlusion and
segmentation drown what they see in renders of the build itself, and for a roof's type or ridge
a wrong change scored better more often than not. So a stricter rule would only drop right
corrections: the check's own accuracy has to decide. It is asked twice, the second time with
every list of choices reversed, and only the answers both give count; its corrections are kept
unless the measures lose 0.05 (0.1 for a level the photo's doors agree with; 0.005 for removing
a part or mirroring, and for the critique's own structural corrections). With right answers in
place of the model's, the check round took round 3's builds from 144 to 155 right answers of 185
(`outputs/vlm/report4/oracle_check.json`).

**The audit's comparison, re-scored** (same photo, own openings): the saved automatic build
0.370; the audit's corrected observations through the old composer 0.417, through the new one
0.509, as a version-3 form 0.563; the hand-made blueprint 0.538. On the old scores the
automatic build had scored above the corrected one (0.609 against 0.602).

**Part-level truth** (`outputs/vlm/truth50_parts.json`: the main roof's ridge, garage doors,
split levels, where the tallest part stands, the wings' roofs; 185 graded answers over the 50
photos): round 3's builds got 144 right (main ridge 31 of 45, tallest part 18 of 35).

**The 50 photos, end to end** (Qwen3.8-27B reading form v3, MapAnything placing the openings,
the check acting on every answer, three rounds of the loop; every build scored on one
yardstick: the new measures against round 4's photo facts):

| | Round 3 (9B, form v2) | Round 4 |
| --- | ---: | ---: |
| Graded answers (truth50, 464) | 436 | 437 |
| Part-level answers of the builds (185): first / final | 144 / 144 | 146 / 146 |
| Measured total, first build / final | 0.418 / 0.460 | 0.462 / 0.531 |
| Final, homes (22) / other buildings (28) | 0.408 / 0.502 | 0.505 / 0.551 |
| Final measured above round 3's | | 38 of 50 |

Corrections kept: the footprint 23 times, storeys 23, a roof's type 16, a part's height 14, a
ridge 11; 8 builds measured more than 0.02 below round 3's (a capitol given a gable over its
dome, a pagoda, a castle, a church whose right front-facing gable the measures did not like).
Two things the run showed and that are now fixed:

- **The check is no better than the reading.** Asked twice, its answers about the main roof's
  ridge were right 8 times of 16, the form's 15 of 24 (roof ends, storeys, garage doors and
  levels: both about 90% and more); acting on all of them fixed 7 part-level answers and broke
  7. It now acts only where the form agrees with it (`refine run --check agree`, the default:
  where the build differs from both readings, the build is wrong); replayed from the run's
  answers, 152 of 185. `--check full` is there for a stronger model or a person's answers.
- **Garage doors on five houses that have none**, from the photo's boxes (basement windows,
  porch openings labelled garage doors). The photo's garage doors are now placed only when the
  reading says the building has some (it was right on 46 of 50): the first builds go from 146 to
  151 right answers of 185.

So round 4 makes builds measurably closer to their photos on every kind of building, mostly
through the parts in metres, the openings where the photo has them and the corrections the
measures can judge; what the vision model reads wrongly (above all which way a roof's ridge
runs: about a coin toss for both models), nothing in the loop can yet put right.

**Which helped: the pipeline or the model?** The 9B model read all 50 photos with form v3 too,
and every reading's first build was scored with the final code (the garage doors gated):

| Reading | Part level (185) | truth50 (464) | First build measured: all / homes / others | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B, form v2 (round 3's composer) | 145 | 436 | 0.424 / 0.389 / 0.451 | 82 |
| Qwen3.5-9B, form v3 | 145 | 439 | 0.449 / 0.443 / 0.453 | 90 |
| Qwen3.8-27B, form v3 | 151 | 437 | 0.463 / 0.432 / 0.487 | 74 |

Most of the gain is the pipeline (the parts in metres, the composer, the openings from the
photo); the bigger model adds a little (garage doors and the wings' roofs read right more
often; better on landmarks, no better on homes) and is faster in llama.cpp than the 9B in
transformers. Neither reads which way a roof's ridge runs reliably.

## Round 5: roofs (2026-10-24)

Roofs were the weakest part of round 4's builds: of the measures, the roof line and each part's
section scored lowest, and of the reading, which way a roof's ridge runs was about a coin toss.
Asked in the reading's abstract terms (ridge parallel or perpendicular, degrees), the model
guesses; asked what the photo shows, it mostly sees it. What changed:

- **The roof pass** (`refine.roof_pass`, about 14 s a photo with the 27B): each part's roof, one
  question at a time (`vision.ROOF_QUESTIONS`). What tops its front wall: a triangle, a level
  eave, an eave with a gable rising from it, flat, other. What closes its ends. How steep it is:
  low, medium, steep. Where a gable rising from its eave stands. What tops a tower: spire, cone,
  pyramid, dome, lantern, battlements, flat. Compose takes the answers that read the test photos
  better than the form (`compose.ROOF_TRUST`).
- **A tie-break** where the pass and the form disagree on a flat roof against a pitched one: can
  you see the roof's own sloping surface (`refine.settle_front`)? It was right on 5 of the 6
  labelled parts it was asked about: the diner, the school and two flat wings fixed, the ranch
  house broken, the warehouse kept. The same kind of question about a gable rising from the eave
  fixed 2 photos and broke 2, so it is not used. Asking each question again with yes and no
  swapped never changed an answer (28 cases).
- **The builder**: a 37-degree pitch (0.75: stairs and slabs in turn). A gable rising from an
  eave over part of a front (`compose._cross_gable`: its own mass, a block proud, its ridge
  dying into the main roof, a window a storey). An open lantern (corner posts, a rail, a dome,
  spire or pyramid cap) for a tower's top. A steep roof as steep as the form's degrees: a
  45-degree Victorian roof had been built at 63 degrees, as tall as the walls under it. A
  lighthouse's gallery and lantern now crown its tower; they went on the main part, which
  flattened the keeper's house's roof.
- **No pitch from the loop**: the measures and the critique no longer change pitches (on real
  photos neither told a right pitch from a wrong one).
- **Roof truth** (`outputs/vlm/truth50_roofs.json`): 157 labels for the 50 photos' main blocks.
  The front top (48), the ends (33), the pitch class (21), a gable rising from the eave (48), a
  tower's top (7).

**Which reader is right more often** (the main blocks, the pass against the form):

| | Form | Roof pass |
| --- | ---: | ---: |
| What tops the front wall (48) | 29 | 38 (35 before a pediment note was reworded) |
| Pitch class (21) | 13 | 15; 17 where the form's own steep is kept |
| Tower top (7) | 5 | 6 |
| Roof ends (33) | 31 | 27 (not used) |

The pediment note had told the model that "a pediment over columns is decoration, not the roof".
A temple-fronted church's pediment is its roof's gable end, so the note now says the pediment
counts only where the roof's ridge runs back from it. Every join rule tried for the two front
tops (the pass deciding flat against pitched, or only the facing) gave the same 37 of 48.

**The 50 photos, end to end** (run6: the 27B, form v3, the roof pass, three rounds of the loop;
its corrections replayed through the final composer, which differs on the 5 steep-roofed builds;
every build measured on one yardstick, run6's photo facts):

| | Round 3 | Round 4 | Round 5 first build | Round 5 final |
| --- | ---: | ---: | ---: | ---: |
| Roof labels right (157) | 113 | 120 | 138 | 139 |
| front top / ends / pitch / cross gable / tower top | 27 / 28 / 11 / 43 / 4 | 29 / 30 / 13 / 43 / 5 | 41 / 30 / 16 / 45 / 6 | 41 / 31 / 16 / 45 / 6 |
| Part-level answers (185) | 144 | 146 | 161 | 162 |
| Main roof's ridge (45) | 31 | 31 | 41 | 40 |
| Measured total | 0.467 | 0.527 | 0.466 | 0.531 |
| Homes (22) / others (28) | 0.422 / 0.502 | 0.497 / 0.551 | 0.435 / 0.491 | 0.507 / 0.551 |

The measures hardly move (18 builds above round 4's, 14 below): the roof line even drops, from
0.450 to 0.443. On real photos they
could not tell a right roof from a wrong one (round 4). Round 4's loop changed 27 roof types and
ridges on their say-so, and round 5's changed 2. The labels and the comparison sheets are the
evidence. The colonial and Cape Cod houses now face the street with their eaves, and the
farmhouse, the Georgian mansion and the Carson mansion have gables rising from their eaves. The
lighthouse has its lantern on its tower and the keeper's house its pitched roof, Stockholm's
City Hall tower its lantern, the apartment's wing its flat roof, and the village church its
gabled porch.

**What still fails** (18 of the 157 labels on the final builds):

- A gable rising from the eave where the label says there is none: 3 photos, two of them
  borderline (a church's gabled porch, a terrace's small gables over its bays).
- The pitch class: 5. One is a chalet the form reads as a gambrel barn: round 4's check changed
  it to the hip it has, and the check now acts only where the form agrees with it.
- A low roof the street hardly shows, read flat: the ranch house.
- Landmark tops the builder has no word for: the Gothic cathedral's tower front, the Royal
  York's château roofs, the Woolworth's crown.
- The roof's ends: 2.

What would help next: two to four photos (a corner view leaves which wall is the front, and a
low roof's slope, to guess); asking about each part on a crop of it (small gables against
dormers, a terrace's repeated gables); the builder's vocabulary for landmark tops and repeated
gables. A roof classifier trained on the builder's own renders was the step held back for a
plateau. It would not reach these: what remains is ambiguity in single photos and roofs the
builder cannot draw, and renders of blocks are far from photos.

## Status (2026-10-22)

First version. A hand-written blueprint of a split-level house photo (two offset gable
wings, skylights, garage, brick base, chimney, site) builds into a result on par with
ChatGPT's hand-written 229-line program for the same photo, both drawn by the same
renderer. Next: more parts (porticos and columns, domes, towers and turrets, porches,
dormers, setbacks), better form questions, sizes measured from the photos, and real photos
labelled with forms (the model drafts, a person corrects) to fine-tune the model on.
