# photofirst: the photo-first pipeline (experimental)

An alternative to the TRELLIS pipeline: no 3D generator. The photo decides what it shows;
a model trained on the human builds completes what it does not.

```
photo -> front end (MoGe-2 depth, building mask) -> the blocks the photo shows
      -> completion model (training's U-Net, trained here) -> the whole build
```

Why: one photo shows about 40% of a build's outer blocks, and about half of TRELLIS's
errors contradict the photo itself (BuildDataCollector `docs/phase5.md`). This pipeline is
being tested, not shipped: the app does not import it.

## Tests

- **A, the ceiling.** The completion model given exactly what a photo shows (from the
  targets, `visibility`): how much of the hidden rest does it get right?
- **B, the front end.** On real photos and renders, how much of what the photo shows does
  the front end recover, and what does the completion model make of it end to end?

Results (2026-10-19; full tables in BuildDataCollector `docs/phase5.md`, "Photo-first
pipeline"): **test A** IoU 0.52 on the real-TRELLIS val builds, against 0.31 for the TRELLIS
pipeline, with materials and exact blocks about twice as often right. **Test B**: the front
end recovers 85% of the visible blocks within one block (TRELLIS 88%) with a third of
TRELLIS's false blocks on visibly empty space, but only 52% exactly (TRELLIS 74%). **End to
end** that gives IoU 0.25 against TRELLIS's 0.31: not better yet; the exact placement of the
visible surfaces is the limit.

**Several photos** (2026-10-19/20, `front_mv`, "Several photos registered by a model" and
"Real front-end pairs and iterative completion" in `docs/phase5.md`): 2–4 photos registered
by MapAnything with rough camera hints (each photo's direction to ±10°) give 91% of the
visible blocks within a block; the completion model trained on the front end's own outputs
(`mv_pairs`, pf-b1) reaches IoU 0.364 with four photos and 0.340 with two, against the
TRELLIS pipeline's 0.333 from one, with materials +43% over the model trained on simulated
inputs. Loose hints (±25°) are worse than none; Depth Anything 3 is below MapAnything.

## Commands

From the repository root, `D = ..\BuildDataCollector\data\real-schems-v1`, in `r2m_train`
with `requirements.txt`'s extras (the renderer runs in BuildDataCollector's `.venv`):

```powershell
r2m_train\Scripts\python -m photofirst.views --dataset $D                    # targets + what each viewpoint sees
r2m_train\Scripts\python -m photofirst.train --dataset $D --run pf-a1        # test A: the completion model
r2m_train\Scripts\python -m photofirst.evaluate --dataset $D --run pf-a1     # test A: scores, and the TRELLIS pipeline beside them
..\BuildDataCollector\.venv\Scripts\python -m photofirst.render --dataset $D # perspective photos of the evaluation builds
r2m_train\Scripts\python -m photofirst.evaluate_front --dataset $D --photos ortho|persp|real [--frame known|estimated] [--view k] [--moge 3] [--run pf-a1]
..\BuildDataCollector\.venv\Scripts\python -m photofirst.render --dataset $D --views 1 2 3  # the other three corners
r2m_train\Scripts\python -m photofirst.multiview --dataset $D --run pf-a3 --carve --front persp_known_none persp_known_none_v2
r2m_train\Scripts\python -m photofirst.hybrid --dataset $D --front persp_known_none persp_known_none_v2 --run r1
```

Several photos, reconstructed together (MapAnything in `r2m_train`; Depth Anything 3 in its
own `r2m_da3` venv, see `requirements.txt`), real training pairs from them, the masked
(iterative) model, and the quality checks:

```powershell
r2m_train\Scripts\python -m photofirst.evaluate_mv --dataset $D --hints rough --refine --res 2 --run pf-a3 --checkpoint best_front.pt --tag ma_rough_res2
r2m_da3\Scripts\python -m photofirst.front_da3 --dataset $D --hints rough           # DA3's arrays, then:
r2m_train\Scripts\python -m photofirst.evaluate_mv --dataset $D --views-from da3_rough --refine --run pf-a3 --checkpoint best_front.pt
..\BuildDataCollector\.venv\Scripts\python -m photofirst.render --dataset $D --train-views 8 --workers 5   # training builds' photos
r2m_train\Scripts\python -m photofirst.mv_pairs --dataset $D --hints rough --tag ma_rough                   # real front-end inputs
r2m_train\Scripts\python -m photofirst.train --dataset $D --run pf-b1 --front-noise --photo-colour --views-per-sample 1 4 --real-pairs ma_rough --init pf-a3 --val-front ma_rough_res2
r2m_train\Scripts\python -m photofirst.train --dataset $D --run pf-bm1 ... --masked --val-passes 4      # iterative completion
r2m_train\Scripts\python -m photofirst.evaluate_saved --dataset $D --inputs ma_rough_res2 --run pf-bm1 --passes 8
r2m_app\Scripts\python -m training.quality --dataset $D --pred pf-a3=photofirst/front/persp_known_4carve_a3
..\BuildDataCollector\.venv\Scripts\python -m photofirst.looks render --dataset $D --pred name=folder ...
r2m_train\Scripts\python -m photofirst.looks score --dataset $D --sets name ...
```

Outputs go to `<dataset>/photofirst/` (local, gitignored with the rest of `data/`): targets,
views, perspective photos, front-end inputs, runs.

## Modules

| Module | What |
| --- | --- |
| `visibility.py` | what a camera sees of a block grid: orthographic (`cast`) or perspective (`cast_from`) ray casting, plus every face with a clear line to the camera |
| `views.py` | per build: memory-mappable targets and the seen solid / seen air of 16 viewpoints (train) or the evaluation photo's and the other three corners (val, test) |
| `data.py` | the completion model's input (seen solid, seen air, the seen blocks' colours; the TRELLIS pipeline's 5-channel layout, so `training`'s model and tiling apply) and training windows |
| `train.py`, `evaluate.py` | test A: training and scoring, with the visible blocks alone and the TRELLIS pipeline on the same builds |
| `front.py` | the front end: MoGe-2 points, the build frame (given, or from gravity and the walls), the depth stretch, blocks |
| `render.py` | perspective photos of the evaluation builds (the smooth set's look, a 50-degree lens) |
| `evaluate_front.py` | test B: the front end against what the photo truly shows, and end to end |
| `multiview.py` | several photos: each photo's front-end blocks fused (union, or space carving: `data.fuse_views`), then the completion model; the truth for the same photos as the ceiling |
| `hybrid.py` | TRELLIS's geometry minus every block the photos show as empty space, as input alone and through the TRELLIS pipeline's model |
| `front_mv.py` | the multi-photo front end: 2-4 photos reconstructed together (MapAnything; camera hints: rough poses from where each photo was taken), placed in the block grid (from known cameras, then against the target for pairs) and turned into blocks at 1 or 2 cells per block |
| `front_da3.py` | Depth Anything 3 for the same front end, run in `r2m_da3` (numpy < 2): writes the per-photo arrays `front_mv` reads |
| `evaluate_mv.py` | the multi-photo front end on the evaluation builds' corner photos: against what the photos truly show, the camera errors, end to end |
| `mv_pairs.py` | training pairs from the real front end: 2-4-photo sets of each training build's `render --train-views` photos |
| `masked.py` | the masked (iterative) completion model: decided blocks as inputs, decoded over several passes, the surest first |
| `evaluate_saved.py` | a model on saved front-end inputs: block scores, where the details are (`training.quality.parts`), quality stats |
| `looks.py` | renders of outputs and human builds compared with DINOv2 image features: does it look like the human build |
| `place.py` | placing a reconstruction in the block grid without a target (real photos): up from the cameras and the walls, walls along the grid, metric or chosen scale, ground at y = 0 |
| `real.py` | real photos: upright (EXIF orientation), the compass heading and 35 mm focal from EXIF, the app's background removal (run in `r2m_app`), the camera's tilt from the photo (MoGe-2) |
| `schem.py` | a predicted build -> Sponge v2 `.schem` (1.21.11 block states, neighbour-dependent states finished), optionally hollowed to its shell |
| `photos_to_schem.py` | 2-4 photo files -> `build.schem`, end to end, no human build anywhere |

Tests: `tests/test_photofirst.py`, `tests/test_multiphoto.py`, `tests/test_real_photos.py`.

## Real photos

```powershell
r2m_train\Scripts\python -m photofirst.photos_to_schem front.jpg left.jpg back.jpg right.jpg --out outputs\photofirst\house
r2m_train\Scripts\python -m photofirst.photos_to_schem a.jpg b.jpg --yaws 0 180 --height 24 --out outputs\photofirst\barn
```

Take 2-4 photos walking round the building, each showing the whole building, with the
phone's location / compass on (the photo then records which way it looked: EXIF
`GPSImgDirection`). Without a recorded heading give each photo's direction with `--yaws`
(degrees, any consistent reference: 0 90 180 270 for four sides walking round): the
directions are what keeps MapAnything from turning a camera 90 degrees on a building that
looks alike from its corners (round 3: hints good to ~10 degrees; "which side" alone is not).
The scale is metric by default (`--metres-per-block 1`, from a second, unhinted MapAnything
pass: the hinted one takes the hints' unit-free scale); `--height` sets the build's height
in blocks instead. The output is a shell (`--solid` keeps the model's guessed inside).
