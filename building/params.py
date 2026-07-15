"""Building-mode tunables — the single source of truth snapshotted into every manifest.

Every knob lives in the flat, JSON-serializable PARAMS dict so a dataset sample records
exactly how it was produced.  Bump PIPELINE_VERSION on ANY algorithm or default change:
the IRs written under outputs/dataset/ are training targets for the future learned
image->build model, and silently drifting outputs would poison that dataset.
"""

PIPELINE_VERSION = "bldg-0.1.0"
SCHEMA_VERSION = 1

PARAMS = {
    # ── B1 yaw alignment ──────────────────────────────────────────────────────
    "yaw_wall_ny_max": 0.35,          # |n_y| below this = wall face
    "yaw_min_wall_area_frac": 0.20,   # skip rotation when the mesh is barely walled
    "yaw_hist_sigma_deg": 2.0,        # wrapped gaussian smoothing of the 90-bin histogram
    "yaw_peak_window_deg": 4.0,       # refine/confidence window around the peak
    "yaw_min_confidence": 0.55,       # skip rotation below this peak-mass fraction

    # ── B2 massing ────────────────────────────────────────────────────────────
    "massing_group_iou": 0.85,        # new slice group below this IoU vs group majority
    "massing_roof_area_ratio": 0.96,  # top groups shrinking below this ratio = roof taper
    "massing_roof_max_step_h": 3,     # ... but only thin steps (a tall setback tower
                                      # shrinks too, and must stay body)
    "massing_cover_target": 1.0,      # cover the FULL footprint (uncovered cells would
    "massing_cover_min_area_frac": 0.0,    # become holes in the shell walls)
    "massing_cover_min_area": 1,
    "massing_cover_max_rects": 32,
    "massing_inflate_max": 2,         # blocks a rect edge may grow outward
    "massing_inflate_min_fill": 0.60, # raw-footprint fill required to grow an edge
    "massing_snap_gap": 1,            # cluster rect edges within this many blocks
    "massing_min_mean_iou": 0.60,     # below this the mesh "does not read as a building"

    # ── B3 facade analysis ────────────────────────────────────────────────────
    "facade_alpha_thresh": 128,       # cutout alpha above this = object
    "facade_trim_y_pct": 2.0,         # trimmed extremes for the top/bottom extent
    "facade_side_rows_pct": 90.0,     # middle % of rows used for the side line fits
    "facade_fit_trim_frac": 0.10,     # residual trim for the second line fit
    "facade_rect_height": 256,        # rectified facade height in px (fixed = stable IR)
    "facade_rect_min_w": 32,
    "facade_rect_max_w": 1024,
    "facade_detrend_sigma": 32.0,     # gaussian sigma for the L-profile detrend
    "facade_lag_min": 10,             # autocorrelation lag search range (px); max is
    "facade_lag_max": 128,            # half the profile so 1-2 story pitches resolve
    "facade_fundamental_ratio": 0.80, # prefer lag/2, lag/3 when nearly as strong
    "facade_conf_regular": 0.50,      # autocorr peak needed to trust a pitch (below
                                      # this, freeform cell maps beat a false grid —
                                      # 1-2 story houses have no real periodicity)
    "facade_kmeans_k": 3,
    "facade_kmeans_iters": 20,
    "facade_merge_de": 12.0,          # merge k-means centroids closer than this (LAB)
    "facade_window_frac_min": 0.04,   # window class cell-fraction bounds
    "facade_window_frac_max": 0.60,
    "facade_freeform_rows": 64,       # freeform cell-map resolution (rows)
    "facade_freeform_max_cols": 128,
    "corner_split_conf": 0.75,        # vertical-edge peak prominence to split a corner shot
    "corner_split_kink_tol": 0.05,    # silhouette-top kink must sit within ±this of the peak

    # ── B4 facade application ─────────────────────────────────────────────────
    "front_convention": "+Z",         # dev_building.py --facing 2026-07-14: +Z won
                                      # 18/30 (60%); losers were low-margin scatter, so
                                      # +Z acts as the tiebreak within the slack only
    "front_convention_slack": 1.15,   # use convention while within this factor of the best
    "facing_thumb": 24,               # ΔE comparison resolution (NxN)
    "min_floor_pitch_blocks": 2,      # below this, windows are disabled (too small)

    # ── B5 assembly ───────────────────────────────────────────────────────────
    "floor_plates": True,             # re-solidify interior at each floor line
    "roof_stairs": False,             # dress taper steps with stairs (off in v1)
    "door_block": "minecraft:dark_oak_planks",

    # ── dataset ───────────────────────────────────────────────────────────────
    "dataset_enabled": True,
}
