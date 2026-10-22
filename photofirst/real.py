"""Real photos for the multi-photo front end: the photo as it was taken, the building cut out,
and the camera hints MapAnything needs (`front_mv.rough_poses`), from the photo itself.

    orientation  phones store the sensor's pixels and an EXIF rotation tag; the photo is
                 turned upright first (`ImageOps.exif_transpose`)
    direction    the compass heading the phone saved (EXIF GPSImgDirection, true or magnetic
                 north: the same for every photo, so only the build's north drifts a few
                 degrees), when the phone saved one; otherwise the user's (`--yaws`)
    tilt         the camera's pitch and roll from the photo: a depth model's point map (MoGe-2),
                 its surface normals, the up direction walls are perpendicular to and roofs and
                 ground along (`front.frame_from_normals`; 1.5 degrees median error on renders)
    focal        from EXIF FocalLengthIn35mmFilm (35 mm equivalents are defined on the frame's
                 diagonal, 43.27 mm), when saved; otherwise MapAnything estimates it
    cutout       the app's own background removal (`trellis_client._prepare_cutout`, rembg
                 isnet-general-use), run in r2m_app, where rembg and its models are; a rejected
                 cutout keeps the whole photo, as the app does

Compass headings use Minecraft's axes: north is -z, east +x, so `visibility.view_axes`'s yaw is
the heading itself.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_PYTHON = ROOT / "r2m_app" / "Scripts" / "python.exe"


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            return float(v[0]) / float(v[1])
        except Exception:
            return None


def exif_info(path) -> dict:
    """{"heading_deg", "heading_ref", "focal35_mm", "focal_mm", "orientation", "gps"} (None
    where the photo has none) from a photo file."""
    from PIL import Image
    out = {"heading_deg": None, "heading_ref": None, "focal35_mm": None, "focal_mm": None,
           "orientation": None, "gps": None}
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            out["orientation"] = ex.get(274)
            gps = ex.get_ifd(0x8825) or {}
            out["heading_deg"] = _num(gps.get(17))
            out["heading_ref"] = gps.get(16)
            if gps.get(2) and gps.get(4):
                def deg(t):
                    d, m, s = (_num(x) for x in t)
                    return d + m / 60.0 + s / 3600.0
                lat, lon = deg(gps[2]), deg(gps[4])
                if gps.get(1) == "S":
                    lat = -lat
                if gps.get(3) == "W":
                    lon = -lon
                out["gps"] = [lat, lon]
            sub = ex.get_ifd(0x8769) or {}
            out["focal35_mm"] = _num(sub.get(41989)) or None
            out["focal_mm"] = _num(sub.get(37386))
    except Exception:
        pass
    return out


def upright(path, out_path) -> tuple:
    """The photo turned upright (EXIF orientation) as RGB, saved to `out_path`; (W, H)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.save(out_path)
        return im.size


def focal_px(exif: dict, width: int, height: int):
    """Focal length in pixels from the 35 mm equivalent, or None."""
    f35 = exif.get("focal35_mm")
    if not f35:
        return None
    return float(f35) / 43.27 * math.hypot(width, height)


def cutouts(photos: list, out_dir: Path) -> list:
    """The app's background removal on each photo (in r2m_app): RGBA arrays; a rejected
    cutout (the app's checks: the mask keeps 5-97% of the frame) keeps the whole photo."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [[str(Path(p).resolve()), str((out_dir / f"cutout_{k}.png").resolve())] for k, p in enumerate(photos)]
    job = out_dir / "cutout_jobs.json"
    job.write_text(json.dumps(pairs), encoding="utf-8")
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; "
            "from trellis_client import _prepare_cutout; "
            "pairs = json.load(open(sys.argv[2], encoding='utf-8')); "
            "print(json.dumps([bool(_prepare_cutout(Path(a), Path(b))) for a, b in pairs]))")
    ok = [False] * len(pairs)
    if APP_PYTHON.is_file():
        r = subprocess.run([str(APP_PYTHON), "-c", code, str(ROOT), str(job)], capture_output=True, text=True,
                           cwd=str(ROOT))
        try:
            ok = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
    rgba = []
    for (src, dst), good in zip(pairs, ok):
        if good and Path(dst).is_file():
            im = np.array(Image.open(dst).convert("RGBA"))
        else:
            im = np.array(Image.open(src).convert("RGBA"))
            im[..., 3] = 255
        rgba.append(im)
    return rgba, ok


def tilt(moge, rgb: np.ndarray, mask: np.ndarray, device="cuda", roll: bool = False) -> dict:
    """{"pitch_deg" (looking down positive), "roll_deg", "fov_x_deg"} of the camera, from the
    photo: MoGe-2's point map, its normals over the building and ground (`mask`), the up they
    imply (`front.frame_from_normals`, pitch searched from 45 up to 60 down). On the 100
    evaluation renders: 1.8 degrees median error, 5.6 at p90 (with `roll` searched too: p90
    8.7, so roll is left at 0: hand-held photos are rarely rolled)."""
    from PIL import Image
    from .front import frame_from_normals, infer, map_normals
    h, w = rgb.shape[:2]
    scale = min(1.0, 768.0 / max(h, w))
    small = np.array(Image.fromarray(rgb).resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR))
    m_small = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(small.shape[1::-1], Image.NEAREST)) > 127
    out = infer(moge, small, device=device)
    p = out["points"]
    valid = out["mask"].astype(bool) & np.isfinite(p).all(-1)
    n, ok = map_normals(np.where(valid[..., None], p, 0.0), valid)
    sel = ok & m_small[1:-1, 1:-1]
    if sel.sum() < 500:                                   # too little building: use the whole frame
        sel = ok
    R = frame_from_normals(n[sel], roll=roll, pitch_range=(-45.0, 60.0))
    u = R[1]                                              # the up direction in camera coordinates
    K = out.get("intrinsics")
    fov = None
    if K is not None:
        fx = float(np.asarray(K)[0, 0])                    # MoGe's intrinsics are normalised (width 1)
        fov = math.degrees(2 * math.atan(0.5 / fx)) if fx > 0 else None
    return {"pitch_deg": math.degrees(math.asin(float(np.clip(-u[2], -1, 1)))),
            "roll_deg": math.degrees(math.atan2(float(u[0]), float(-u[1]))), "fov_x_deg": fov}


def steady_pitches(pitches: list, limit: float = 15.0) -> tuple:
    """The photos of a set are taken at about the same tilt: a pitch more than `limit` degrees
    from the set's median (the estimate's rare failures, 54 degrees off at worst on the
    renders) is replaced by the median. Returns (pitches, which were replaced)."""
    med = float(np.median(pitches))
    out, fixed = [], []
    for p in pitches:
        bad = abs(p - med) > limit
        out.append(med if bad else float(p))
        fixed.append(bool(bad))
    return out, fixed
