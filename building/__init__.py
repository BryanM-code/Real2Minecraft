"""Building mode: photo -> architectural Minecraft replica (+ dataset factory).

Stages (each a pure IR -> IR function, see the module docstrings):
    B1 yaw_align   - rotate the mesh so facades align with the grid axes
    B2 massing     - blobby voxel solid -> crisp stacked rectilinear prisms
    B3 facade      - photo -> rectified facade cell grid (wall/window/accent/door)
    B4 facade_map  - facade cells -> per-wall block-class grids (+ pattern extension)
    B5 assemble    - hollow shell + glass windows + floors -> Minecraft blocks

Every run writes a training sample under outputs/dataset/ — the IRs are the future
learned model's training targets (see building/params.py PIPELINE_VERSION).
"""

from building.errors import BuildingModeError
from building.params import PIPELINE_VERSION
from building.pipeline import run_building, rerun_from_facade

__all__ = ["BuildingModeError", "PIPELINE_VERSION",
           "run_building", "rerun_from_facade"]
