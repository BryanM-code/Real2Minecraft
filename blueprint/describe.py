"""Reading a photo with a local vision-language model.

`run_read` asks, one question at a time: a free description of the building, what kind of
building it is, its style, then a form for that style and kind (`form2`): its parts (main
block, wings, towers), their sizes in metres, storeys, roofs, windows and doors, colours and
materials. `refine prepare` saves the answers as <id>.read.json.

Two models: Qwen/Qwen3.5-9B (the default, Apache 2.0), loaded in 4 bits with transformers,
about 7 GB of GPU memory; or `qwen3.8-27b` (Qwen3.8-27B, Apache 2.0, a 3-bit GGUF), run by a
llama.cpp server this module starts (`Server`, `r2m_llama/`), about 15 GB.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
MAX_SIDE = 1024
FORM_VERSION = 3


DESCRIBE = """Describe this building's structure in detail for someone who will rebuild it in Minecraft:
the main volumes (wings, towers) and how they are arranged left to right and front to back,
the number of storeys of each, the roof types and pitch, the openings on each visible side
(windows per floor, doors, garage doors), and the materials and colours of the walls, trim
and roof. Be specific and concise."""


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


def run_read(model, processor, image, think: bool = False, version: int | None = None) -> dict:
    """The reading: a description, the kind, the style, then the form for that style and kind
    (`form2`, FORM_VERSION unless `version` says otherwise)."""
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
