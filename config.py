"""Shared paths and UI bounds for Real2Minecraft."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Resolution slider bounds: the longest axis is scaled to the chosen target (in blocks).
# Higher targets preserve architectural detail (crenellations, windows) — only pays off
# once the source mesh actually carries that detail (see TRELLIS_MESH_SIMPLIFY).
VOXEL_MIN_BLOCKS = 16     # slider minimum
VOXEL_MAX_BLOCKS = 128    # slider maximum
