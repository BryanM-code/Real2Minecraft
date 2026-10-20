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

Tests: `tests/test_photofirst.py`.
