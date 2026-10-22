"""The Phase 5 model: a dense 3D residual U-Net from the 2x input grid to mcvox-v3 blocks.

    input     (B, 6, 2w, 2w, 2w): occupancy, surface, R, G, B (0-1) and height (the cell's
              y in blocks / HEIGHT_NORM, counted from the grid's floor: derivable at
              inference, where the mesh stands on y = 0). `features` builds it.
    stem      space-to-depth: each block's 2 x 2 x 2 cells become channels (48 at w^3), so
              nothing below the block (a half-filled cell pair: a slab, a stair step) is
              lost before the first convolution
    U-Net     residual blocks (GroupNorm, SiLU, 3^3 conv) at w, w/2, w/4, w/8 with `widths`
              channels; strided conv down, transposed conv up, skips concatenated
    heads     shape: a 1x1x1 conv to 30 logits at every block. material: a linear layer to
              313 logits, applied only where it is needed -- the supervised solid voxels in
              training, the predicted solid ones at inference -- because 313 channels at
              every voxel of a batch would cost gigabytes

`predict_window` turns a window's logits into blocks: shape probabilities, and per solid
voxel the best material among those `vocab.VALID` allows for its shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SHAPES = 30
N_MATERIALS = 313
IN_CHANNELS = 6
HEIGHT_NORM = 64.0


def features(x_u8: torch.Tensor, start_y: torch.Tensor) -> torch.Tensor:
    """Cached input window(s) (B, 5, 2w, 2w, 2w) uint8 + the windows' start in blocks
    (B,) -> the model input (B, 6, 2w, 2w, 2w) float."""
    x = x_u8.float()
    x = torch.cat([x[:, :2], x[:, 2:5] / 255.0], dim=1)
    b, _, _, h, _ = x.shape
    y = (start_y.to(x.device, torch.float32)[:, None] * 2.0
         + torch.arange(h, device=x.device, dtype=torch.float32)[None, :]) / (2.0 * HEIGHT_NORM)
    y = y[:, None, None, :, None].expand(b, 1, x.shape[2], h, x.shape[4])
    return torch.cat([x, y], dim=1)


def space_to_depth(x: torch.Tensor, f: int = 2) -> torch.Tensor:
    b, c, d, h, w = x.shape
    x = x.view(b, c, d // f, f, h // f, f, w // f, f)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, c * f ** 3, d // f, h // f, w // f)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, groups: int = 8):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, c_in), c_in)
        self.c1 = nn.Conv3d(c_in, c_out, 3, padding=1)
        self.n2 = nn.GroupNorm(min(groups, c_out), c_out)
        self.c2 = nn.Conv3d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv3d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), blocks: int = 1, in_channels: int = IN_CHANNELS,
                 n_shapes: int = N_SHAPES, n_materials: int = N_MATERIALS, checkpointing: bool = False):
        super().__init__()
        self.config = {"widths": list(widths), "blocks": blocks, "in_channels": in_channels,
                       "n_shapes": n_shapes, "n_materials": n_materials}
        self.checkpointing = checkpointing
        self.stem = nn.Conv3d(in_channels * 8, widths[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = widths[0]
        for i, w in enumerate(widths):
            self.enc.append(nn.Sequential(*[ResBlock(prev if j == 0 else w, w) for j in range(blocks)]))
            prev = w
            if i < len(widths) - 1:
                self.down.append(nn.Conv3d(w, widths[i + 1], 2, stride=2))
                prev = widths[i + 1]
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(nn.Sequential(*[ResBlock(2 * widths[i - 1] if j == 0 else widths[i - 1], widths[i - 1])
                                            for j in range(blocks)]))
        self.out_norm = nn.GroupNorm(8, widths[0])
        self.shape_head = nn.Conv3d(widths[0], n_shapes, 1)
        self.material_head = nn.Linear(widths[0], n_materials)

    def _run(self, module, x):
        if self.checkpointing and self.training and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x, extra=None):
        """(B, 6, 2w, 2w, 2w) -> (shape logits (B, 30, w, w, w), features (B, C, w, w, w)).
        `extra` (B, widths[0], w, w, w) is added after the stem (`photofirst.masked`'s
        decided blocks)."""
        h = self.stem(space_to_depth(x))
        if extra is not None:
            h = h + extra
        skips = []
        for i, block in enumerate(self.enc):
            h = self._run(block, h)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        for up, block in zip(self.up, self.dec):
            h = up(h)
            h = self._run(block, torch.cat([h, skips.pop()], dim=1))
        f = F.silu(self.out_norm(h))
        return self.shape_head(f), f

    def material_logits(self, feats: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
        """Material logits (N, 313) at the voxels where `where` (B, w, w, w) is True."""
        b, x, y, z = where.nonzero(as_tuple=True)
        return self.material_head(feats[b, :, x, y, z])


def valid_table(device) -> torch.Tensor:
    import numpy as np
    from mcvoxel.vocab import VALID
    return torch.as_tensor(np.array(VALID), dtype=torch.bool, device=device)   # VALID is read-only


@torch.no_grad()
def predict_window(model: UNet3D, x: torch.Tensor, valid: torch.Tensor, air_below: float = 0.9):
    """One batch of windows -> (shape probabilities (B, 30, w, w, w) float32, material
    (B, w, w, w) int64). A material is chosen wherever a block is at all likely (air
    probability under `air_below`), not only where the window's own argmax is solid:
    blending windows can turn a voxel solid that this window thought was air. It is the
    best material among those `vocab.VALID` allows for the most likely solid shape."""
    logits, feats = model(x)
    probs = torch.softmax(logits.float(), dim=1)
    maybe = probs[:, 0] < air_below
    material = torch.zeros(probs.shape[:1] + probs.shape[2:], dtype=torch.int64, device=probs.device)
    if maybe.any():
        best_solid = probs[:, 1:].argmax(dim=1) + 1
        ml = model.material_logits(feats, maybe).float()
        ml = ml.masked_fill(~valid[best_solid[maybe]], float("-inf"))
        material[maybe] = ml.argmax(dim=1)
    return probs, material
