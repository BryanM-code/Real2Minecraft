"""A local vision-language model looking at building photos (experimental): structure
questions with fixed answers (`survey`), a free description (`describe`), or a whole
blueprint (`blueprint`). Runs in `r2m_vlm` (CUDA PyTorch + transformers), from this
repository's root:

    r2m_vlm\\Scripts\\python -m blueprint.describe survey photo.jpg [...] --out outputs\\vlm\\run1
    r2m_vlm\\Scripts\\python -m blueprint.describe blueprint house.jpg --out outputs\\vlm\\run1

The model (default Qwen/Qwen3.5-9B, Apache 2.0) is loaded 4-bit (bitsandbytes NF4) with
its vision encoder left in bf16, about 7 GB of GPU memory; `--model qwen3.8-27b` (Qwen3.8-27B,
Apache 2.0, 3-bit GGUF) runs in a llama.cpp server instead (`Server`, `r2m_llama/`), about 15 GB.
Each answer is saved as <out>/<photo stem>.<mode>.json with the raw text, the parsed JSON and
the timing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
MAX_SIDE = 1024
FORM_VERSION = 3

SURVEY = """You are looking at a photo of a building. Answer about the main building only (ignore other
buildings, cars, people and trees). Reply with JSON only, with exactly these keys:
{"building_type": "house|apartment|office_tower|hotel|church|civic|commercial|station|castle|other",
 "storeys": <integer: floors of the main body; above 20, your best estimate>,
 "volumes": <integer: distinct wings or blocks that make up the building>,
 "roof": "gable|hip|flat|mansard|dome|spire|complex|not_visible",
 "dome": true|false, "tower_or_turret": true|false, "front_columns": true|false,
 "symmetric_front": true|false,
 "wall_material": "wood_siding|brick|stone|stucco|concrete|glass|metal",
 "wall_colour": "<one or two words>", "roof_colour": "<one or two words, or not visible>",
 "garage": true|false, "chimney": true|false, "balconies_or_porches": true|false,
 "camera": "ground|elevated|looking_up"}"""

SURVEY2 = SURVEY + """

Definitions:
- volumes: count the building's distinct masses as a modeller would build them as separate
  boxes: the main block plus every wing that sticks out from it or has its own roof (a
  capitol with a central block and two side wings = 3; a house with a lower garage wing = 2;
  a single tower or a plain box = 1).
- roof: what covers the main block. Answer not_visible when the top of the building is
  outside the photo. Two or more gable roofs = gable. A dome on a mostly hidden low roof = dome.
- tower_or_turret: a tall, narrow structure rising well above the main roof. A dome, the
  lantern on top of a dome, or a chimney is not a tower.
- balconies_or_porches: true only if you can clearly see one."""

DESCRIBE = """Describe this building's structure in detail for someone who will rebuild it in Minecraft:
the main volumes (wings, towers) and how they are arranged left to right and front to back,
the number of storeys of each, the roof types and pitch, the openings on each visible side
(windows per floor, doors, garage doors), and the materials and colours of the walls, trim
and roof. Be specific and concise."""

BLUEPRINT = """You will describe the building in the photo as a JSON blueprint that a program turns into a
Minecraft build. Units are blocks; use about 2 blocks per metre (a storey is 5-6 blocks, a
door 2 blocks high). Frame: x runs to the right as seen from the front, y up, z towards the
viewer; the front is the "south" side, the left side is "west", the right side "east", the
back "north".

Format:
- "materials": role -> Minecraft block id without "minecraft:". Roles: wall, wall_base
  (bottom rows, see plinth), trim (corners, frames), roof (needs stairs and slabs, e.g.
  stone_bricks, deepslate_tiles, dark_oak_planks), roof_edge, window (a glass block, e.g.
  glass, light_gray_stained_glass), door (a wood: oak_planks, spruce_planks, birch_planks,
  dark_oak_planks), garage, chimney, skylight.
- "masses": boxes of walls, each {"id", "x", "z" (its back-left corner), "w" (width, along
  x), "d" (depth, along z), "y" (base row), "storeys" and "storey_h" (or "walls": rows),
  "plinth" (rows of wall_base at the bottom, a number or {"south": n, ...}),
  "roof": {"type": "gable|hip|shed|flat", "ridge": "x|z" (gable: the ridge's direction),
  "pitch": 0.5 (gentle, slabs) or 1.0 (45 degrees, stairs), "overhang": 0-2}}.
  Wings that touch share a wall line (e.g. the second starts at x = first.x + first.w).
- "openings": each {"mass", "side": "south|north|east|west", "kind": "window|door|garage|panel",
  "u" (blocks from the wall's left end seen from outside), "y" (rows above the mass's base),
  "w", "h"}; windows may add "mullions": [offsets], "shutters": true, "sill": false; a door
  is always 2 high; a panel is a patch of another block ("material").
  Openings must fit inside their wall: u + w <= wall length, y + h <= wall rows.
- "features": chimney {"kind": "chimney", "x", "z", "w", "d", "top" (row of its top)},
  skylight {"kind": "skylight", "mass", "side" (the roof slope's side), "u", "v" (rows up the
  slope), "w", "h"}.

Example (a cottage):
EXAMPLE

Reply with the JSON blueprint only, describing the building in the photo: its wings, their
sizes and heights, roofs, every visible window, door and garage door, and its materials."""


FORM = """Now fill in this form about the building, using the photo and your description. Reply
with JSON only:
{"wings": [ one entry per distinct wing or block, ordered left to right as seen in the photo:
   {"position": "left|centre|right",
    "storeys": <integer, above the front ground>,
    "level": "ground|half_below" (half_below: it also has a level partly below the front ground, such as a garage under a split level),
    "front_offset": "flush|forward|back" (its front wall compared with the main wing's),
    "windows_across": <integer: windows across the front of its top floor>,
    "roof": "gable|hip|flat|shed",
    "ridge": "parallel|perpendicular" (a gable's ridge compared with the front wall),
    "pitch": "low|medium|steep",
    "front": {"ground_windows": <integer>, "wide_window": true|false, "door": true|false, "garage_doors": <integer>},
    "side_garage_doors": <integer: garage doors on its visible side wall>,
    "chimney": true|false, "skylights": <integer>} ],
 "wall": {"material": "wood_siding|brick|stone|stucco|concrete|glass", "colour": "<word>"},
 "base": {"material": "none|brick|stone|concrete"},
 "trim_colour": "<word>", "roof_colour": "<word>", "door_colour": "<word>"}"""


def style_question() -> str:
    from .styles import STYLES
    lines = "\n".join(f"- {k}: {v['hint']}" for k, v in STYLES.items())
    return ("What is the building's architectural style? Choose the closest one:\n" + lines +
            '\nReply with JSON only: {"style": "<one key from the list>"}')


def kind_question() -> str:
    from .styles import KINDS
    lines = "\n".join(f"- {k}: {v['hint']}" for k, v in KINDS.items())
    return ("What kind of building is it? Choose the closest one:\n" + lines +
            '\nReply with JSON only: {"kind": "<one key from the list>"}')


PART_V2 = """   {{"part": "main|wing|tower|annex",
    "share": <percent of the building's visible front width this part takes; the shares add up to about 100>,
    "storeys": <integer, above the front ground>,
    "level": "ground|half_below" (half_below: it also has a level partly below the front ground),
    "front_offset": "flush|forward|back" (its front wall compared with the main block's),
    "shape": "box|round|octagonal",
    "roof": "{roofs}|not_visible",
    "roof_ends": "triangle|slope|none" (what closes the short ends of a gable or hip roof: a triangular wall, or a sloping face of the roof; none for other roofs),
    "ridge": "parallel|perpendicular" (a gable's ridge compared with the front wall),
    "pitch": "low|medium|steep" (low: under 30 degrees, medium: about 45, steep: 60 or more),
    "windows_across": <integer: windows across the front of its top floor>,
    "window_style": "{wins}",
    "front": {{"door": true|false, "ground_windows": <integer>, "wide_window": true|false}} }} ],"""

# version 3: each part's size and place in metres, its roof's slope in degrees and its garage
# doors, so the builder does not guess them (it made a wing 0.8 times the main block's depth,
# "forward" 3 blocks, and "medium" 45 degrees where the model's own words said 30)
PART_V3 = """   {{"part": "main|wing|tower|annex",
    "share": <percent of the building's visible front width this part takes; the shares add up to about 100>,
    "width_m": <metres across its front>,
    "depth_m": <metres from its front wall to its back wall; your best estimate if the side is hidden>,
    "storeys": <integer: its floors, counting a level partly below the ground>,
    "eave_m": <metres from its lowest floor up to its eaves, where its roof starts>,
    "floor_m": <metres its lowest floor is above (+) or below (-) the main block's ground floor; 0 when level>,
    "forward_m": <metres its front wall stands in front of (+) or behind (-) the main block's front wall; 0 when flush>,
    "shape": "box|round|octagonal",
    "roof": "{roofs}|not_visible",
    "roof_ends": "triangle|slope|none" (what closes the short ends of a gable or hip roof: a triangular wall, or a sloping face of the roof; none for other roofs),
    "ridge": "parallel|perpendicular" (a gable's ridge compared with the front wall),
    "pitch_deg": <its roof's slope in degrees: 0 flat, about 15 very low, 25-30 low, 45 medium, 60 or more steep>,
    "windows_across": <integer: windows across the front of its top floor>,
    "window_style": "{wins}",
    "front": {{"door": true|false, "ground_windows": <integer>, "wide_window": true|false, "garage_doors": <integer>}},
    "side_garage_doors": <integer: garage doors in its end wall that the photo shows>,
    "base": "none|brick|stone|concrete" (a band of another material at the foot of its walls),
    "base_m": <metres that band rises on its front wall>, "base_end_m": <and on its end wall the photo shows> }} ],"""

RULERS = """- metres: measure with what the photo shows: a door is about 1 m wide and 2 m high, a storey about
  3 m, a garage door about 2.5 m wide; a part's eave_m is about its storeys times 3 m"""


def form2(style: str, kind: str, version: int = 2) -> str:
    """The form for a building of this style and kind: the shared part, then the features
    such buildings have (`styles.form_features`), each with its definition. Version 3 asks
    each part's size, level and offset in metres, its roof's slope in degrees and its garage
    doors (`compose` honours them, its own rules filling in what is missing)."""
    from .styles import FEATURES, KINDS, ROOFS, STYLES, WINDOW_STYLES, form_features
    st, kd = STYLES.get(style, STYLES["other"]), KINDS.get(kind, KINDS["other"])
    feats = form_features(style, kind)
    if version >= 3:                                   # each part has its own garage doors
        feats = [f for f in feats if f != "garage"]
    lines = ",\n ".join(FEATURES[f][0] for f in feats)
    defs = "\n".join(f"- {f}: {FEATURES[f][1]}" for f in feats)
    roofs, wins = "|".join(ROOFS), "|".join(WINDOW_STYLES)
    part = (PART_V3 if version >= 3 else PART_V2).format(roofs=roofs, wins=wins)
    rulers = ("\n" + RULERS) if version >= 3 else ""
    return f"""Now fill in this form about the building, using the photo and your description. It is {kd['hint']}
in the {style} style ({st['hint']}); such buildings often have {', '.join(st['roofs'])} roofs and
{', '.join(st['windows'])} windows, but answer what the photo shows. Reply with JSON only:
{{"parts": [ one entry per distinct part of the building, ordered left to right as seen in the photo
            (the main block, each wing, each tower or turret, a lower annex):
{part}
 "symmetric": true|false,
 {lines},
 "wall": {{"material": "wood_siding|brick|stone|stucco|concrete|glass|metal|logs|timber_frame", "colour": "<word>"}},
 "base": {{"material": "none|brick|stone|concrete"}},
 "trim_colour": "<word>", "roof_colour": "<word>", "door_colour": "<word>"}}
Definitions:
- part: main = the biggest block; wing = any other block or extension, even a taller one; tower = a
  narrow tower or turret clearly taller and thinner than the rest (a bell tower, a turret, a minaret, a
  lighthouse); annex = a small low addition
- roof: what covers this part. not_visible = you cannot see its roof (a tall building seen from the
  street, a parapet or the angle hides it). gable = two slopes meeting at a ridge, closed at the ends by
  triangular walls; cross_gable = two gables crossing in a T or L, triangular walls facing two ways;
  hip = slopes on all four sides meeting at a short ridge, no triangular walls; pyramid = four slopes
  meeting at a point; shed = one slope; flat = no slope (a deck, or a parapet round it); mansard = a
  steep lower slope, often with dormers, under a flat or gentle top; gambrel = a barn roof, two slopes on
  each side, the lower one steeper; dome; cone = a round pointed roof on a round tower; spire = a tall
  thin pointed roof; sawtooth = repeated one-sided slopes with glass faces; pagoda = stacked roofs with
  upturned eaves; barrel = a curved roof like half a tube lying on its side; butterfly = two slopes
  falling to a valley in the middle (a V); onion_dome = a bulb-shaped dome that swells out and ends in a
  point; clipped_gable = a gable whose peak is cut back by a small sloping triangle at each end;
  hip_and_gable = a hip roof whose top part ends in a small triangular gable at each end (East Asian
  temples and houses){rulers}
{defs}"""


ROOT = Path(__file__).resolve().parents[1]
LLAMA_SERVER = ROOT / "r2m_llama" / "llama-server.exe"
# models run by llama.cpp (too big for transformers on 16 GB): name -> (repo, weights, vision part)
SERVER_MODELS = {"qwen3.8-27b": ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q3_K_XL.gguf", "mmproj-F16.gguf")}


class Server:
    """A GGUF vision model in a llama.cpp server (`r2m_llama/`, started here, stopped at exit),
    asked over its OpenAI-style HTTP API: how Qwen3.8-27B (3-bit, 13 GB) runs on a 16 GB card,
    with llama.cpp's kernels for its linear-attention layers."""

    def __init__(self, name: str, port: int = 8090, ctx: int = 16384):
        import atexit
        import subprocess
        import urllib.request
        from huggingface_hub import hf_hub_download
        repo, weights, mmproj = SERVER_MODELS[name]
        self.url = f"http://127.0.0.1:{port}"
        self.name = name
        # --image-min-tokens 1024: llama.cpp warns Qwen-VL models need it for boxes (a 1024 x 575
        # photo is about 580 image tokens otherwise)
        cmd = [str(LLAMA_SERVER), "-m", hf_hub_download(repo, weights), "--mmproj", hf_hub_download(repo, mmproj),
               "-ngl", "99", "-c", str(ctx), "--port", str(port), "-fa", "on", "--jinja", "-np", "1",
               "--image-min-tokens", "1024"]
        self.log = open(ROOT / "outputs" / f"llama_server_{port}.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        for _ in range(600):                           # loading takes a minute or two
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server stopped (see {self.log.name})")
            time.sleep(1)
        raise RuntimeError("llama-server did not start in 10 minutes")

    def chat(self, images: list, prompt: str, history: list | None, max_tokens: int, think: bool) -> str:
        import base64
        import io
        import urllib.request
        content = []
        for im in images:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=92)
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}] + [{"role": r, "content": t} for r, t in history or []]
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": think}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            answer = json.loads(r.read())["choices"][0]["message"]["content"] or ""
        return re.sub(r"(?s)^.*</think>", "", answer).strip()

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except Exception:  # noqa: BLE001
                self.proc.kill()


def load_model(name: str = MODEL, bits: int = 4):
    """(model, processor): a transformers model, or (Server, None) for SERVER_MODELS."""
    if name in SERVER_MODELS:
        return Server(name), None
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    quant = None
    if bits in (4, 8):
        quant = BitsAndBytesConfig(load_in_4bit=bits == 4, load_in_8bit=bits == 8,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                   llm_int8_skip_modules=["visual", "vision_tower", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda:0",
                                                        quantization_config=quant)
    return model.eval(), AutoProcessor.from_pretrained(name)


def photo(path: Path):
    """The photo upright (its EXIF orientation applied), at most MAX_SIDE pixels."""
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im


def ask(model, processor, image, prompt: str, max_new_tokens: int = 1024, think: bool = False,
        history: list | None = None) -> str:
    """The model's answer about `image` (or a list of images, shown in order before the
    prompt); `history` is (role, text) turns after the first."""
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if isinstance(model, Server):
        return model.chat(images, prompt, history, max_new_tokens, think)
    import torch
    messages = [{"role": "user", "content": [*({"type": "image"} for _ in images), {"type": "text", "text": prompt}]}]
    messages += [{"role": role, "content": [{"type": "text", "text": t}]} for role, t in history or []]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=think)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return re.sub(r"(?s)^.*</think>", "", answer).strip()


def parse_json(text: str):
    """The first {...} block in `text`, or None. Tolerant of the slips a model makes: trailing
    commas, a root object closed too early (`{"parts": [...]}, "symmetric": false, ...}`: the
    keys after it are its own), and an end cut off (the open brackets closed)."""
    m = re.search(r"(?s)\{.*", text)
    if not m:
        return None
    s = m.group(0)
    s = s[:s.rfind("}") + 1] or s
    for candidate in (s, re.sub(r",\s*([}\]])", r"\1", s)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        obj, end = dec.raw_decode(s)
    except json.JSONDecodeError:
        closed = _close_json(m.group(0))
        try:
            return json.loads(closed) if closed else None
        except json.JSONDecodeError:
            return None
    rest = s[end:].strip()
    while isinstance(obj, dict) and rest.startswith(","):        # more of the root's keys
        more_text = "{" + rest[1:].lstrip()
        try:
            more, used = dec.raw_decode(more_text)
        except json.JSONDecodeError:
            break
        if not isinstance(more, dict):
            break
        obj.update({k: v for k, v in more.items() if k not in obj})
        rest = more_text[used:].strip()
    return obj


def _close_json(s: str) -> str | None:
    """`s` cut off inside a JSON value: its last complete item kept and its open brackets closed."""
    stack, in_str, esc, cut = [], False, False, None
    for i, ch in enumerate(s):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
            if not stack:
                return None                                   # complete: not a cut-off text
        elif ch == ",":
            cut = (i, list(stack))
    if not stack or cut is None:
        return None
    i, open_at_cut = cut
    return s[:i] + "".join(reversed(open_at_cut))


def check(bp) -> str | None:
    """What is wrong with a parsed blueprint (`blueprint.load`'s message), or None."""
    from .spec import BlueprintError, load
    if not isinstance(bp, dict):
        return "the answer is not a JSON object"
    try:
        load(bp)
    except BlueprintError as e:
        return str(e)
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        return f"malformed blueprint ({type(e).__name__}: {e})"
    return None


def run_v2(model, processor, image, think: bool = False) -> dict:
    """Describe, then style and kind in both orders (A: style first, B: kind first), each
    followed by its own form; both branches share the description."""
    t0 = time.time()
    desc = ask(model, processor, image, DESCRIBE, 900, think)
    out = {"description": desc, "seconds_describe": round(time.time() - t0, 1)}
    for order in ("A", "B"):
        t = time.time()
        history = [("assistant", desc)]
        qs = [("style", style_question()), ("kind", kind_question())]
        answers = {}
        for key, q in (qs if order == "A" else qs[::-1]):
            history.append(("user", q))
            text = ask(model, processor, image, DESCRIBE, 200, think, history)
            history.append(("assistant", text))
            answers[key] = (parse_json(text) or {}).get(key)
        history.append(("user", form2(answers.get("style") or "other", answers.get("kind") or "other")))
        text = ask(model, processor, image, DESCRIBE, 1200, think, history)
        out[order] = {"style": answers.get("style"), "kind": answers.get("kind"), "text": text,
                      "form": parse_json(text), "seconds": round(time.time() - t, 1)}
    return out


def run_read(model, processor, image, think: bool = False, version: int | None = None) -> dict:
    """The production reading (order B, the better one on the 50-photo test): describe, the
    kind, the style, then the form for that style and kind (`form2`, FORM_VERSION unless
    `version` says otherwise)."""
    version = version or FORM_VERSION
    t0 = time.time()
    desc = ask(model, processor, image, DESCRIBE, 900, think)
    history = [("assistant", desc)]
    answers = {}
    for key, q in (("kind", kind_question()), ("style", style_question())):
        history.append(("user", q))
        text = ask(model, processor, image, DESCRIBE, 200, think, history)
        history.append(("assistant", text))
        answers[key] = (parse_json(text) or {}).get(key)
    history.append(("user", form2(answers.get("style") or "other", answers.get("kind") or "other", version)))
    text = ask(model, processor, image, DESCRIBE, 1600 if version >= 3 else 1200, think, history)
    return {"description": desc, "kind": answers.get("kind"), "style": answers.get("style"), "text": text,
            "form": parse_json(text), "form_version": version, "seconds": round(time.time() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("survey", "survey2", "describe", "blueprint", "form", "v2", "read"))
    ap.add_argument("photos", nargs="+", type=Path, help="photos, or folders of <id>/photo.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--bits", type=int, default=4, choices=(4, 8, 16))
    ap.add_argument("--think", action="store_true", help="let the model reason before answering (slower)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--repair", type=int, default=0,
                    help="blueprint mode: up to this many retries with the format errors fed back")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    prompt = {"survey": SURVEY, "survey2": SURVEY2, "describe": DESCRIBE, "blueprint": BLUEPRINT,
              "form": DESCRIBE, "v2": DESCRIBE, "read": DESCRIBE}[args.mode]
    photos = []
    for p in args.photos:
        photos += sorted(p.glob("*/photo.jpg")) if p.is_dir() else [p]
    if args.mode == "blueprint":
        example = (Path(__file__).with_name("examples") / "cottage.json").read_text(encoding="utf-8")
        prompt = prompt.replace("EXAMPLE", example)
    limit = args.max_new_tokens or {"survey": 600, "survey2": 600, "describe": 900, "blueprint": 6000,
                                     "form": 900, "v2": 900, "read": 900}[args.mode]
    if args.think:
        limit += 4000
    args.out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    model, processor = load_model(args.model, args.bits)
    print(f"model loaded in {time.time() - t:.0f} s", flush=True)
    for p in photos:
        stem = p.parent.name if p.name == "photo.jpg" else p.stem
        if (args.out / f"{stem}.{args.mode}.json").is_file():
            continue                                    # resumable
        t = time.time()
        image = photo(p)
        if args.mode == "read":
            rec = {"photo": str(p), "mode": "read", "model": args.model, "bits": args.bits, **run_read(model, processor, image)}
            (args.out / f"{stem}.read.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{stem}: {rec['seconds']} s, {rec['style']} / {rec['kind']}", flush=True)
            continue
        if args.mode == "v2":
            rec = {"photo": str(p), "mode": "v2", "model": args.model, "bits": args.bits,
                   **run_v2(model, processor, image)}
            rec["seconds"] = round(time.time() - t, 1)
            (args.out / f"{stem}.v2.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{stem}: {rec['seconds']} s, A {rec['A']['style']}/{rec['A']['kind']}, "
                  f"B {rec['B']['style']}/{rec['B']['kind']}", flush=True)
            continue
        text = ask(model, processor, image, prompt, limit, args.think)
        rec = {"photo": str(p), "mode": args.mode, "model": args.model, "bits": args.bits, "think": args.think,
               "seconds": round(time.time() - t, 1), "text": text,
               "json": parse_json(text) if args.mode not in ("describe", "form") else None, "repairs": []}
        if args.mode == "form":                     # the form, filled in after its own description
            from .assemble import assemble
            rec["description"] = text
            text = ask(model, processor, image, prompt, 900, args.think, [("assistant", text), ("user", FORM)])
            rec.update(text=text, json=parse_json(text), seconds=round(time.time() - t, 1))
            if isinstance(rec["json"], dict):
                rec["blueprint"] = assemble(rec["json"])
                rec["problem"] = check(rec["blueprint"])
        for _ in range(args.repair if args.mode == "blueprint" else 0):
            problem = check(rec["json"])
            if problem is None:
                break
            history = [("assistant", text), ("user", f"The blueprint has these problems: {problem}\n"
                                                     "Reply with the corrected JSON blueprint only.")]
            text = ask(model, processor, image, prompt, limit, args.think, history)
            rec["repairs"].append({"problem": problem, "text": text})
            rec["json"] = parse_json(text)
            rec["seconds"] = round(time.time() - t, 1)
        (args.out / f"{stem}.{args.mode}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False),
                                                            encoding="utf-8")
        print(f"{stem}: {rec['seconds']} s, {'parsed' if rec['json'] is not None else 'text'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
