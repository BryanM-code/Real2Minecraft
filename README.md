# Real2Minecraft

Turns a photo of a building into a Minecraft build (a `.schem` file), on your own computer.

A vision model looks at the photo and describes the building as a short list of parts: its
wings, roofs, windows and doors, porches and towers, and the materials. Code then builds
that description out of real Minecraft blocks. The result is clean by construction: straight
walls, real roofs made of stairs and slabs, and windows where the photo has them.

## Run it

From this folder, with the environments set up (see [Setup](#setup)):

```
r2m_measure\Scripts\python -m blueprint.photo C:\path\to\house.jpg --out outputs\blueprint\house --model qwen3.8-27b
```

It takes about 6 minutes per photo. The results are in the `--out` folder:

| File | What |
| --- | --- |
| `build.schem` | the build (Sponge schematic, Minecraft 1.21.11) |
| `view_*.png` | previews from four sides |
| `<name>/first.png`, `<name>/final.png` | the build seen from the photo's angle, before and after the corrections |
| `blueprint.json` | the description it was built from |

To paste it in the game with WorldEdit, copy `build.schem` (renamed) into
`.minecraft\config\worldedit\schematics\`, then run `//schem load <name>` and `//paste`.

Options: `--rounds 0` skips the corrections (faster); `--no-measure` skips measuring the photo
(typical sizes are used instead); without `--model qwen3.8-27b` the smaller Qwen3.5-9B reads
the photo.

Give every photo its own `--out` folder: a step that already has a result in the folder is
not run again. For a folder of photos (PowerShell):

```
Get-ChildItem C:\path\to\photos\*.jpg | ForEach-Object { r2m_measure\Scripts\python -m blueprint.photo $_.FullName --out "outputs\blueprint\$($_.BaseName)" --model qwen3.8-27b }
```

Photos that work best show the whole building, roof and ground included, with little in
front of it. A corner view shows two walls; the sides the photo does not show are guessed.

## How it works

```
photo -> read -> find the building -> measure -> compose -> build -> check and correct -> .schem
```

| Step | Module | What it does |
| --- | --- | --- |
| Read | `describe.py`, `vision.py` | A local vision model (Qwen) says what kind of building it is (house, church, castle...), its style, and its parts: for each wing or tower its size, number of storeys and roof (one question at a time). It also boxes the main building and its windows and doors. |
| Find the building | `segment.py` | A segmentation model (EoMT) picks out the main building's pixels, leaving out trees, cars and the sky. |
| Measure | `measure.py` | MapAnything turns the photo into 3D points in metres. From the building's points come its width, depth and height, and where each window and door sits on its walls. |
| Compose | `compose.py`, `facade.py`, `styles.py` | The reading and the measurements become a blueprint: each part's size, level and roof, the openings where the photo has them, materials by style. What the builder cannot do is listed as a "conflict". |
| Build | `build.py`, `parts.py`, `spec.py`, `minecraft/` | The blueprint becomes blocks: walls, floors, roofs from the lowest up, then openings, porches, balconies, towers and the rest. `minecraft/` holds the block vocabulary and writes the `.schem`. |
| Check and correct | `refine.py`, `score.py`, `edits.py`, `render.py` | The build is rendered from the photo's angle and compared with the photo: outline, proportions, roof line, windows, colours, and the vision model's own critique. Corrections are tried one at a time and kept only when they help. |

`photo.py` runs all of it. Each step can also run on its own, over a folder of photos laid out
as `<id>\photo.jpg`:

```
r2m_vlm\Scripts\python -m blueprint.refine prepare PHOTOS --out DIR --model qwen3.8-27b    (read, find the building)
r2m_measure\Scripts\python -m blueprint.measure PHOTOS --out DIR                          (measure)
r2m_vlm\Scripts\python -m blueprint.refine run PHOTOS --out DIR --model qwen3.8-27b        (compose, build, check)
r2m_vlm\Scripts\python -m blueprint.refine summary --out DIR                              (scores over the run)
```

## The blueprint format

A blueprint is a JSON file of about 200 values for a house. It can also be written by hand
and built with `r2m_measure\Scripts\python -m blueprint house.json --out DIR --render`
(`blueprint\examples\cottage.json` is one). The full reference is in `spec.py`.

| Part | What |
| --- | --- |
| `materials` | which block each role uses: walls, trim, roof, windows, doors, columns... |
| `masses` | the building's blocks: a box, round or octagonal tower, or an open frame of posts (a pavilion, a market hall), each with its storeys and roof (gable, hip, mansard, gambrel, dome, spire, pagoda, barrel...) |
| `openings` | windows (plain, arched, pointed), doors, garage doors, arches, round windows, on a mass's side |
| `features` | chimneys, porches, verandas, balconies, porticos, decks, steps, dormers, bay windows, towers' lanterns and more |
| `site` | the ground, bushes and trees |

## How well it works

On 50 photos of many kinds of buildings, labelled by hand: the vision model reads the roofs
of 138 of the 157 labelled parts right, and the parts themselves (how many, which is tallest,
where they stand) 161 times in 185.

On 84 new photos, every photo gave a build. The correction loop raised the computer
measures' score from 0.48 to 0.54 on average and kept at least one correction on 61 of them.

Known weak spots: gambrel roofs come out as plain gables; a row of attached houses becomes one
house; jettied upper floors are not seen; a photo from below makes the render look up at the
build from under the ground.

## Setup

Windows, an NVIDIA GPU with 16 GB, Python 3.10. Two Python environments and llama.cpp, all in
this folder and left out of git:

| Folder | What runs in it | How to make it |
| --- | --- | --- |
| `r2m_vlm` | reading the photo, segmentation, the check-and-correct loop | `requirements-vlm.txt` |
| `r2m_measure` | MapAnything, building, writing `.schem` files, the whole chain, the tests | `requirements-measure.txt` |
| `r2m_llama` | the larger vision model (Qwen3.8-27B) through llama.cpp | the Windows CUDA build of llama.cpp, unzipped here |

The models download into the Hugging Face cache the first time they are used.

## Tests

```
r2m_measure\Scripts\python -m unittest discover -s tests
```

## Data

This repository holds code only. Photos, readings, builds and renders stay in `outputs\`,
which git ignores.
