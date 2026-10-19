# training: the learned model (Phase 5)

Turns the 2x voxelisation of a TRELLIS mesh into mcvox-v3 blocks (shape x material), trained
on BuildDataCollector's pairs: human builds as targets, synthetic TRELLIS-like inputs, and
optionally the train builds' real TRELLIS output (`schemset.real_trellis --set train`). The
plan and decisions are BuildDataCollector `docs/mesh-to-schem-plan.md` (Phase 5); the pair
format and the loader contract are its `docs/pairs-v1.md`. The app never imports this
package. Trained weights stay local and are never redistributed (D14).

## Environment

`r2m_train` (Python 3.10, CUDA PyTorch), separate from the app's `r2m_app`; see
`requirements.txt`. The baseline runner needs the app's modules and runs in `r2m_app`; the
gallery needs BuildDataCollector's renderer and runs in its `.venv`.

## Commands

From the repository root, with `D = ..\BuildDataCollector\data\real-schems-v1`:

```powershell
r2m_train\Scripts\python -m training.data cache --dataset $D        # once per pairs version: 13 GB, ~6 min
r2m_train\Scripts\python -m training.data trellis --dataset $D      # the train builds' real TRELLIS inputs (after the cache)
r2m_app\Scripts\python   -m training.baseline --dataset $D          # the rule-based baseline (D18)
r2m_train\Scripts\python -m training.train --dataset $D --run smoke --smoke --steps 600 --val-every 300
r2m_train\Scripts\python -m training.train --dataset $D --run r1 --steps 20000
r2m_train\Scripts\python -m training.evaluate --dataset $D --run r1 [--checkpoint best_trellis.pt]
..\BuildDataCollector\.venv\Scripts\python -m training.gallery --dataset $D --run r1 [--checkpoint best_trellis.pt]
```

Runs write to `<dataset>/runs/<name>/` (gitignored with the rest of `data/`): config, logs,
`best.pt` (best synthetic-val score), `best_trellis.pt` (best real-TRELLIS val score),
`last.pt` (resume), and `training.evaluate`'s `eval_<split>.json`, predictions and gallery.

## Modules

| Module | What |
| --- | --- |
| `score.py` | the metrics, shared by the model and the baseline: geometry IoU / F1 / F1 within one block over supervised voxels, exact shape and material on voxels solid in both, the CIELAB difference of the materials' colours (`material_colors.json`), exact blocks; `align` for predictions made in their own frame |
| `baseline.py` | the app's building and sculpture modes on the real-TRELLIS evaluation sets and the real photos, at the scale the model's input has, aligned and scored |
| `data.py` | the memory-mapped pair cache, training windows (64^3 target, 128^3 input), the 8 symmetries through `mcvoxel.blockstates`, sampling weights |
| `model.py` | dense 3D residual U-Net: space-to-depth stem (no sub-block detail lost), 4 levels, a shape head at every block and a material head only where needed |
| `predict.py` | whole-build inference with `mcvoxel.tiling.assemble`, as the app will run it |
| `train.py` | the training loop (bf16, AdamW, cosine), validation on whole val builds (synthetic and real-TRELLIS), checkpoints |
| `evaluate.py` | a checkpoint against the baseline on the same builds and inputs: summaries, shape confusion, where the false blocks lie against the input (`input_excess`) |
| `gallery.py` | an HTML page per run: target, TRELLIS input, model, both baseline modes, model on synthetic input |

Tests: `tests/test_training_*.py` (the numpy parts run in the regular suite; the model tests
need `r2m_train`).
