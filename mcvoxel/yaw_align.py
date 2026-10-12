"""B1 — yaw alignment: rotate the mesh about +Y so facades align with the X/Z axes.

Buildings photographed at an angle reconstruct rotated relative to the voxel grid; the
axis-aligned massing regularization (B2) would shred a 30-degree-rotated tower.  The
dominant horizontal facade direction is found from an area-weighted histogram of the
near-vertical face normals (period 90 degrees — a box has two facade directions 90
degrees apart) and the mesh is rotated so that direction lands on a grid axis.

Both gates fail soft: an organic mesh (low wall area) or a round tower (no dominant
direction) simply skips rotation, recorded as applied=false in the IR.
"""

import numpy as np
import trimesh


# The B1 defaults from Real2Minecraft's building/params.py (2026-07-14), so callers outside
# building mode can use the same calibration without importing the app.
DEFAULT_PARAMS = {
    "yaw_wall_ny_max": 0.35,
    "yaw_min_wall_area_frac": 0.20,
    "yaw_hist_sigma_deg": 2.0,
    "yaw_peak_window_deg": 4.0,
    "yaw_min_confidence": 0.55,
}


def estimate_yaw(verts: np.ndarray, faces: np.ndarray, params: dict = None) -> dict:
    """Return the yaw IR: {theta_deg, confidence, wall_area_frac, applied, ...}."""
    params = params or DEFAULT_PARAMS
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    total_area = float(areas.sum())

    ir = {"theta_deg": 0.0, "confidence": 0.0, "wall_area_frac": 0.0,
          "applied": False, "method": "normal-histogram",
          "histogram_deg1": [0.0] * 90}
    if total_area <= 0 or not len(normals):
        return ir

    wall = np.abs(normals[:, 1]) <= params["yaw_wall_ny_max"]
    wall_area = float(areas[wall].sum())
    ir["wall_area_frac"] = wall_area / total_area
    if ir["wall_area_frac"] < params["yaw_min_wall_area_frac"] or wall_area <= 0:
        return ir

    # Facade direction histogram, 90-degree period, 1-degree bins, area-weighted.
    ang = np.degrees(np.arctan2(normals[wall, 2], normals[wall, 0])) % 90.0
    w = areas[wall]
    hist = np.zeros(90, dtype=np.float64)
    np.add.at(hist, np.clip(ang.astype(np.int64), 0, 89), w)
    ir["histogram_deg1"] = (hist / wall_area).tolist()

    # Wrapped gaussian smoothing.
    sigma = params["yaw_hist_sigma_deg"]
    offs = np.arange(-8, 9)
    kernel = np.exp(-0.5 * (offs / sigma) ** 2)
    kernel /= kernel.sum()
    idx = (np.arange(90)[:, None] + offs[None, :]) % 90
    smoothed = (hist[idx] * kernel[None, :]).sum(axis=1)
    peak = int(np.argmax(smoothed))

    # Refine inside the +/-window via the angle-quadrupling circular mean (the 90-degree
    # period becomes 360, where a weighted circular mean is well defined).
    win = params["yaw_peak_window_deg"]
    d = np.abs(ang - (peak + 0.5))
    near = np.minimum(d, 90.0 - d) <= win
    if near.any():
        a4 = np.radians(ang[near] * 4.0)
        vec = (w[near] * np.exp(1j * a4)).sum()
        theta = (np.degrees(np.angle(vec)) / 4.0) % 90.0
    else:
        theta = peak + 0.5

    ir["confidence"] = float(w[near].sum() / wall_area) if near.any() else 0.0
    if theta >= 45.0:                       # rotate the short way
        theta -= 90.0
    ir["theta_deg"] = float(theta)
    ir["applied"] = ir["confidence"] >= params["yaw_min_confidence"]
    if not ir["applied"]:
        ir["theta_deg"] = 0.0
    return ir


def apply_yaw(verts: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate about +Y by -theta (a facade normal at angle theta lands on +X), then
    re-center XZ on the bounding box (the mesh_normalize convention).  Y untouched."""
    out = np.asarray(verts, dtype=np.float64).copy()
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    x = out[:, 0] * c + out[:, 2] * s
    z = -out[:, 0] * s + out[:, 2] * c
    out[:, 0] = x
    out[:, 2] = z
    out[:, 0] -= (out[:, 0].min() + out[:, 0].max()) / 2.0
    out[:, 2] -= (out[:, 2].min() + out[:, 2].max()) / 2.0
    return out
