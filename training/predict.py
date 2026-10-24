"""Whole-build inference: tile a 2x input grid into windows, predict, reassemble.

Tiling and blending are `mcvoxel.tiling.assemble`, as the app will run it: windows every
`window - overlap` blocks, shape probabilities blended with ramps across the overlaps,
material from the window that sees the voxel most centrally. A material chosen in one window
for a shape the blend then changes can leave a pair `vocab.VALID` does not allow; such a
voxel keeps its material as a full block (every material has one). A voxel the blend makes
solid but no window gave a material (each window names one wherever a block is at all
likely, `predict_window`) stays air.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import N_SHAPES, features, predict_window, valid_table

WINDOW = 64
OVERLAP = 16


def input_from_pair(z) -> np.ndarray:
    """A pair / evaluation npz -> the cached input layout (2X, 2Y, 2Z, 5) uint8."""
    stack = np.empty(z["input_occ"].shape + (5,), np.uint8)
    stack[..., 0] = z["input_occ"]
    stack[..., 1] = z["input_surface"]
    stack[..., 2:] = z["input_rgb"]
    return stack


@torch.no_grad()
def predict_grid(model, inp: np.ndarray, target_dims, window: int = WINDOW, overlap: int = OVERLAP,
                 device="cuda", shape_prior=None) -> tuple:
    """(shape uint8, material uint16) over the target frame `target_dims` for one input grid
    (2X, 2Y, 2Z, 5) uint8. `shape_prior` (N_SHAPES,): decoding calibration; a voxel's
    solid shape is the argmax of probability x prior (`predict_window`), air against solid
    decided as without it, so rare shapes (stairs, slabs, fences) win more of the voxels the
    model already makes solid and the geometry stays."""
    from mcvoxel.tiling import assemble
    from mcvoxel.vocab import VALID
    model.eval()
    valid = valid_table(device)
    dims = tuple(int(d) for d in target_dims)
    prior = None if shape_prior is None else torch.as_tensor(np.asarray(shape_prior, np.float32), device=device)

    def predict(start, gsl):
        x = np.zeros((2 * window,) * 3 + (inp.shape[-1],), np.uint8)
        sl = tuple(slice(2 * a, min(2 * (a + window), n)) for a, n in zip(start, inp.shape[:3]))
        n = tuple(s.stop - s.start for s in sl)
        x[:n[0], :n[1], :n[2]] = inp[sl]
        xt = torch.from_numpy(np.ascontiguousarray(np.moveaxis(x, -1, 0)))[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            probs, mat = predict_window(model, features(xt, torch.tensor([int(start[1])])), valid, prior=prior)
        return probs[0].cpu().numpy(), mat[0].cpu().numpy()

    shape, material, _ = assemble(dims, window, overlap, predict, N_SHAPES)
    material = material.astype(np.uint16)
    shape[(shape != 0) & (material == 0)] = 0          # no window thought a block likely
    solid = shape != 0
    material[~solid] = 0
    bad = solid & ~VALID[shape, material]
    shape[bad] = 1
    return shape.astype(np.uint8), material
