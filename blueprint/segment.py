"""Which pixels of a photo are the building. A semantic segmentation model (EoMT-L trained
on ADE20K's 150 classes, MIT licence, `tue-mps/ade20k_semantic_eomt_large_512`) labels every
pixel; the building classes inside the main building's box (the vision model's, from
`vision.locate`) are the building, and the trees, cars, people, fences and poles in front of
it are occluders: building the photo hides, which the outline comparison leaves out. The
model runs in r2m_vlm (0.6 GB of GPU memory, 0.1 s a photo); everything after the label
map is plain numpy.

ADE20K draws a facade's windows and doors as part of the building, so they come from the
vision model's boxes instead (`vision.openings`).
"""
from __future__ import annotations

import numpy as np

MODEL = "tue-mps/ade20k_semantic_eomt_large_512"
# ADE20K class ids
BUILDING = (0, 1, 25, 48, 79, 84,                  # wall, building, house, skyscraper, hovel, tower
            8, 14, 42, 53, 59, 86, 106, 38, 95, 121, 148, 51)   # window, door, column, stairs, stairway,
                                                   # awning, canopy, railing, bannister, step, clock, grandstand
OCCLUDER = (4, 17, 72, 66, 125,                    # tree, plant, palm, flower, pot
            20, 80, 83, 102, 76, 103, 127, 116, 90,   # car, bus, truck, van, boat, ship, bicycle, minibike, airplane
            12, 126, 32, 87, 93, 136, 43, 123, 149, 132, 138, 69, 104, 88, 114)
MIN_SHARE = 0.01            # a building smaller than this share of the photo is a failed cut-out


def load(device: str = "cuda"):
    """(processor, model) on the GPU in bf16, or on the CPU in fp32 (when a llama.cpp vision
    model fills the GPU)."""
    import torch
    import transformers.models.eomt.modeling_eomt as E
    from transformers import AutoImageProcessor, EomtForUniversalSegmentation
    E.requires_backends = lambda *a, **k: None     # only the training loss needs scipy
    proc = AutoImageProcessor.from_pretrained(MODEL)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = EomtForUniversalSegmentation.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    return proc, model


def labels(seg, image) -> np.ndarray:
    """(H, W) ADE20K class ids of a PIL image."""
    import torch
    proc, model = seg
    inputs = proc(images=image, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out = model(**inputs)
    lab = proc.post_process_semantic_segmentation(out, target_sizes=[(image.height, image.width)])[0]
    return lab.cpu().numpy().astype(np.int16)


def box_pixels(box, width: int, height: int, grow: float = 0.0) -> tuple:
    """A box in the vision model's 0-1000 coordinates as pixels (x0, y0, x1, y1, end
    exclusive), grown by `grow` of the photo's size on each side and clipped to it."""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    gx, gy = grow * width, grow * height
    return (max(0, int(x0 * width / 1000 - gx)), max(0, int(y0 * height / 1000 - gy)),
            min(width, int(np.ceil(x1 * width / 1000 + gx))), min(height, int(np.ceil(y1 * height / 1000 + gy))))


def _components(mask: np.ndarray) -> np.ndarray:
    """Connected component ids (0 = background, 4-connected) of a small boolean mask, by
    flood fills on a PIL image (no scipy here)."""
    from PIL import Image, ImageDraw
    h, w = mask.shape
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).copy()   # its own pixels: fromarray's are read-only
    comp = np.zeros((h, w), np.int32)
    k = 0
    todo = mask.copy()
    while todo.any() and k < 250:
        y, x = np.argwhere(todo)[0]
        k += 1
        ImageDraw.floodfill(im, (int(x), int(y)), 0)            # this component goes dark
        now = np.array(im) == 0
        newly = now & todo
        comp[newly] = k
        todo &= ~newly
    comp[todo] = k + 1                                         # beyond 250 components: one lump
    return comp


def main_building(lab: np.ndarray, box=None) -> dict:
    """The building's pixels in a label map: building classes inside the main building's box
    (0-1000 coordinates; the whole photo without one), its biggest pieces only; occluders
    inside the box; the frame (x0, y0, x1, y1, end exclusive) the building fills, carried out
    to the box where occluders hide its ends."""
    h, w = lab.shape
    inbox = np.zeros((h, w), bool)
    bx = box_pixels(box, w, h, 0.02) if box is not None else (0, 0, w, h)
    inbox[bx[1]:bx[3], bx[0]:bx[2]] = True
    building = np.isin(lab, BUILDING) & inbox
    occl = np.isin(lab, OCCLUDER) & inbox
    # drop specks: pieces under 3% of the building (at a quarter of the resolution)
    s = max(1, int(round(max(h, w) / 256)))
    small = building[::s, ::s]
    if small.any():
        comp = _components(small)
        sizes = np.bincount(comp.ravel())
        sizes[0] = 0
        keep = np.flatnonzero(sizes >= 0.03 * sizes.max())
        keep_small = np.isin(comp, keep)
        keep_full = np.repeat(np.repeat(keep_small, s, 0), s, 1)[:h, :w]
        building &= keep_full
    out = {"mask": building, "occluders": occl, "box": list(bx), "share": float(building.mean())}
    if not building.any():
        return dict(out, frame=list(bx), ok=False, occluded=0.0)
    ys, xs = np.nonzero(building)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    # where trees or cars fill the strip between the building's edge and the model's box
    # (not grown), the building goes on behind them
    hidden = occl | building
    rb = box_pixels(box, w, h) if box is not None else bx
    if x0 > rb[0] and occl[y0:y1, rb[0]:x0].mean() >= 0.4:
        x0 = rb[0]
    if x1 < rb[2] and occl[y0:y1, x1:rb[2]].mean() >= 0.4:
        x1 = rb[2]
    if y0 > rb[1] and occl[rb[1]:y0, x0:x1].mean() >= 0.4:
        y0 = rb[1]
    if y1 < rb[3] and occl[y1:rb[3], x0:x1].mean() >= 0.4:
        y1 = rb[3]
    area = max(1, (x1 - x0) * (y1 - y0))
    return dict(out, frame=[x0, y0, x1, y1], ok=bool(out["share"] >= MIN_SHARE),
                occluded=float(occl[y0:y1, x0:x1].sum() / area),
                visible=float(hidden[y0:y1, x0:x1].mean()))


# --- colours ---------------------------------------------------------------------------------

def lab_colours(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB values (..., 3) in 0-255."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def palette(lab: np.ndarray, k: int = 3, iters: int = 10) -> dict:
    """The `k` main colours of L*a*b* pixels (k-means from evenly spaced lightness) and the
    share of the pixels each covers."""
    lab = lab.reshape(-1, 3)
    if len(lab) > 4000:
        lab = lab[np.linspace(0, len(lab) - 1, 4000).astype(int)]
    order = np.argsort(lab[:, 0])
    c = lab[order[np.linspace(0, len(lab) - 1, k + 2)[1:-1].astype(int)]].copy()
    for _ in range(iters):
        near = np.argmin(((lab[:, None, :] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (near == j).any():
                c[j] = lab[near == j].mean(0)
    w = np.bincount(near, minlength=k) / len(lab)
    return {"centres": c.round(1).tolist(), "weights": w.round(3).tolist()}


def palette_distance(p: dict, q: dict, eps: float = 4.0) -> float:
    """The mean colour difference (CIE76 delta E) of moving one palette's shares onto the
    other's colours (an entropy-smoothed transport, Sinkhorn)."""
    a, b = np.asarray(p["weights"], float) + 1e-9, np.asarray(q["weights"], float) + 1e-9
    a, b = a / a.sum(), b / b.sum()
    cost = np.linalg.norm(np.asarray(p["centres"])[:, None] - np.asarray(q["centres"])[None], axis=-1)
    kern = np.exp(-cost / eps) + 1e-30
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(200):
        u = a / (kern @ v)
        v = b / (kern.T @ u)
    plan = u[:, None] * kern * v[None]
    return float((plan * cost).sum() / plan.sum())


def zone_colours(rgb: np.ndarray, mask: np.ndarray, frame, skip=None) -> dict:
    """The main colours of the building's walls (the frame's lower 60%) and of its top (the
    upper 20%: the roof where one shows, else the top of the walls); `skip` leaves pixels out
    (the windows)."""
    x0, y0, x1, y1 = frame
    h = y1 - y0
    keep = mask.copy()
    if skip is not None:
        keep &= ~skip
    out = {}
    for name, (a, b) in (("wall", (y0 + int(0.4 * h), y1)), ("top", (y0, y0 + max(1, int(0.2 * h))))):
        sel = np.zeros_like(keep)
        sel[a:b, x0:x1] = True
        sel &= keep
        if sel.sum() >= 20:
            out[name] = palette(lab_colours(rgb[sel]))
    return out


# colour words compose's tables know (`materials.pick`), at rough L*a*b*
COLOUR_WORDS = {"white": (95, 0, 2), "cream": (88, 1, 14), "beige": (78, 3, 15), "yellow": (85, 0, 50),
                "light grey": (78, 0, 1), "grey": (60, 0, 0), "dark": (30, 0, 0), "black": (12, 0, 0),
                "brown": (40, 12, 22),
                "red": (42, 45, 30), "orange": (65, 30, 55), "pink": (75, 20, 5), "green": (50, -30, 20),
                "blue": (55, -5, -30), "tan": (70, 6, 22)}


def colour_word(lab) -> str:
    """The nearest colour word to an L*a*b* colour."""
    return min(COLOUR_WORDS, key=lambda w: float(np.linalg.norm(np.subtract(COLOUR_WORDS[w], lab))))
