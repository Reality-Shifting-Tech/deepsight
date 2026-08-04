"""eval/gen_synthetic.py — deterministic matrix of synthetic eval images.

Zero-network test-image factory for the eyes regression eval. Every image
is drawn programmatically with known ground truth, then the eval asserts
the pipeline reports what we drew.

Generators (all 640x480): solid, gradient, checkerboard, stripes, noise,
shapes, textcard, text-stop, rect, border.

Variants per generator: base, rot90, fliph, flipv, bright050, bright150,
jpeg50, noise (seeded). Expectation rules are conservative on purpose so
the suite stays green across macOS versions; weak signals (the 0.11
baseball anchor) flake between OS builds, so we only assert strong,
orientation-safe signals:

    negatives (ocr_empty/humans_zero/sports_absent)  all non-text variants
    ocr_contains   text images, mild variants only (base/bright150/jpeg50/noise)
    rectangles_min geometry images, transform-safe variants only

Plus 10 seeded fuzz images with no expectations: smoke tests that the
binary runs and parses cleanly without crashing (run_eval treats an empty
expected block as a pass when the binary succeeded).

Output: eval/images/synthetic/*.png|.jpg and eval/images/synthetic/manifest.json
(entries carry source="synthetic"). run_eval.load_manifest merges the
generated manifest into the main manifest at load time, replacing any
previously committed synthetic entries.

Usage:
    python eval/gen_synthetic.py              # regenerate images + manifest
    python eval/gen_synthetic.py --smoke      # determinism probe (no files)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

RGB = tuple[int, int, int]
SIZE = (640, 480)
SEED = 20260804
FONT_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Menlo.ttc",
)
OUT_DIR = Path(__file__).resolve().parent / "images" / "synthetic"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (211, 47, 47)
BLUE = (21, 101, 192)
NAVY = (13, 71, 161)
GREEN = (46, 125, 50)
LIGHT_BLUE = (30, 136, 229)
LIGHT_GREEN = (67, 160, 71)
ORANGE = (255, 152, 0)
PURPLE = (123, 31, 162)

VARIANTS = ("base", "rot90", "fliph", "flipv", "bright050", "bright150", "jpeg50", "noise")

# Which variants keep the signal strong enough to assert. Probe-verified:
# OCR survives rotation and mild photometric change but flips produce
# garbage tokens ("90T2" for "STOP"), so flip variants assert nothing.
SAFE_OCR = frozenset({"base", "rot90", "bright050", "bright150", "jpeg50", "noise"})

TEXT_GENS = {"textcard", "text-stop"}
RECT_GENS = {"rect", "shapes", "border"}
TEXT_TRUTH: dict[str, list[str]] = {
    "textcard": ["DeepSight test card", "SALE 50% OFF"],
    "text-stop": ["STOP"],
}


def load_font(size: int) -> ImageFont.ImageFont:
    # Menlo Bold is the macOS-system font the text fixtures were probed
    # with. CI (ubuntu) lacks it: fall back to Pillow's bundled scalable
    # default font so the generator stays cross-platform and deterministic.
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return cast(ImageFont.ImageFont, ImageFont.truetype(path, size, index=1))
    return cast(ImageFont.ImageFont, ImageFont.load_default(size=size))


def _draw_text_center(
    d: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, fill: RGB
) -> None:
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    d.text(((SIZE[0] - w) / 2, (SIZE[1] - h) / 2), text, font=font, fill=fill)


def gen_solid(color: RGB) -> Image.Image:
    return Image.new("RGB", SIZE, color)


def gen_gradient() -> Image.Image:
    img = Image.new("RGB", SIZE)
    for y in range(SIZE[1]):
        t = y / (SIZE[1] - 1)
        color = tuple(int(LIGHT_BLUE[i] + (LIGHT_GREEN[i] - LIGHT_BLUE[i]) * t) for i in range(3))
        for x in range(0, SIZE[0], 8):
            img.paste(color, (x, y, min(x + 8, SIZE[0]), y + 1))
    return img


def gen_checkerboard() -> Image.Image:
    img = Image.new("RGB", SIZE, WHITE)
    d = ImageDraw.Draw(img)
    cell = 80
    for row in range(SIZE[1] // cell):
        for col in range(SIZE[0] // cell):
            if (row + col) % 2 == 0:
                d.rectangle(
                    (col * cell, row * cell, (col + 1) * cell - 1, (row + 1) * cell - 1),
                    fill=BLACK,
                )
    return img


def gen_stripes() -> Image.Image:
    img = Image.new("RGB", SIZE, WHITE)
    d = ImageDraw.Draw(img)
    for x in range(0, SIZE[0], 80):
        d.rectangle((x, 0, x + 39, SIZE[1] - 1), fill=NAVY)
    return img


def gen_noise() -> Image.Image:
    random.seed(SEED + 1)
    noise = Image.effect_noise(SIZE, 28)
    return noise.convert("RGB")


def gen_shapes() -> Image.Image:
    img = Image.new("RGB", SIZE, WHITE)
    d = ImageDraw.Draw(img)
    d.ellipse((40, 40, 200, 200), fill=RED)
    d.rectangle((280, 40, 440, 200), fill=BLUE)
    d.polygon([(120, 440), (40, 260), (200, 260)], fill=GREEN)
    d.ellipse((280, 260, 440, 420), fill=ORANGE)
    d.rectangle((480, 220, 620, 440), outline=PURPLE, width=16)
    return img


def gen_textcard() -> Image.Image:
    img = Image.new("RGB", SIZE, WHITE)
    d = ImageDraw.Draw(img)
    font = load_font(46)
    # "DeepSight test card" at 60pt Menlo Bold overflows the 640px canvas
    # (Vision truncates the tail to "test ca"), so size to fit. Wide vertical
    # gap keeps Vision from merging the two lines into one OCR region.
    d.text((40, 110), "DeepSight test card", font=font, fill=BLACK)
    d.text((40, 300), "SALE 50% OFF", font=font, fill=BLACK)
    return img


def gen_text_stop() -> Image.Image:
    img = Image.new("RGB", SIZE, RED)
    d = ImageDraw.Draw(img)
    d.text((220, 170), "STOP", font=load_font(110), fill=WHITE)
    return img


def gen_rect() -> Image.Image:
    img = Image.new("RGB", SIZE, WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((220, 180, 420, 300), fill=BLACK)
    return img


def gen_border() -> Image.Image:
    img = Image.new("RGB", SIZE, WHITE)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, SIZE[0] - 1, SIZE[1] - 1), fill=None, outline=BLACK, width=40)
    return img


GENERATORS: dict[str, Callable[[], Image.Image]] = {
    "solid-red": lambda: gen_solid(RED),
    "solid-blue": lambda: gen_solid(BLUE),
    "gradient": gen_gradient,
    "checkerboard": gen_checkerboard,
    "stripes": gen_stripes,
    "noise": gen_noise,
    "shapes": gen_shapes,
    "textcard": gen_textcard,
    "text-stop": gen_text_stop,
    "rect": gen_rect,
    "border": gen_border,
}


def _transpose(img: Image.Image, op: Image.Transpose) -> Image.Image:
    return img.transpose(op)


def variant_noise(img: Image.Image, seed_offset: int) -> Image.Image:
    random.seed(SEED + 10 + seed_offset)
    noise = Image.effect_noise(img.size, 18).convert("RGB")
    return Image.blend(img.convert("RGB"), noise, alpha=0.18)


def apply_variant(img: Image.Image, variant: str, seed_offset: int = 0) -> Image.Image:
    if variant == "base":
        return img
    if variant == "rot90":
        return _transpose(img, Image.Transpose.ROTATE_90)
    if variant == "fliph":
        return _transpose(img, Image.Transpose.FLIP_LEFT_RIGHT)
    if variant == "flipv":
        return _transpose(img, Image.Transpose.FLIP_TOP_BOTTOM)
    if variant == "bright050":
        return ImageEnhance.Brightness(img).enhance(0.5)
    if variant == "bright150":
        return ImageEnhance.Brightness(img).enhance(1.5)
    if variant == "jpeg50":
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=50)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if variant == "noise":
        return variant_noise(img, seed_offset)
    raise ValueError(f"unknown variant: {variant}")


def build_expected(gen: str, variant: str) -> dict[str, Any]:
    """Conservative expectations: strong signals only, variant-safe.

    Probe-verified against the compiled binary: rectangles fire on every
    variant for the geometry generators (border/rect/shapes all >= 1 even
    under jpeg50 + noise), and OCR survives everything except flips.
    """
    exp: dict[str, Any] = {"humans_zero": True, "sports_absent": True}
    if gen in TEXT_GENS:
        if variant in SAFE_OCR:
            exp["ocr_contains"] = TEXT_TRUTH[gen]
    elif gen in RECT_GENS:
        exp["rectangles_min"] = 1
        exp["ocr_empty"] = True
    else:
        exp["ocr_empty"] = True
    return exp


def make_entry(gen: str, variant: str) -> tuple[dict[str, Any], str]:
    ext = "jpg" if variant == "jpeg50" else "png"
    eid = f"synth-{gen}-{variant}"
    entry: dict[str, Any] = {
        "id": eid,
        "path": f"images/synthetic/{eid}.{ext}",
        "source": "synthetic",
        "expected": build_expected(gen, variant),
    }
    return entry, eid


def render_image(gen: str, variant: str, seed_offset: int = 0) -> Image.Image:
    return apply_variant(GENERATORS[gen](), variant, seed_offset)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def smoke() -> dict[str, Any]:
    """Determinism probe: 3 gens x 2 variants, image bytes + entries, no files."""
    out: dict[str, Any] = {}
    for gen in ("solid-red", "textcard", "rect"):
        for variant in ("base", "rot90"):
            img = render_image(gen, variant)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            entry, eid = make_entry(gen, variant)
            out[eid] = {"sha256": sha256_bytes(buf.getvalue()), "expected": entry["expected"]}
    return out


def fuzz_entry(index: int) -> dict[str, Any]:
    """A fuzz image entry: empty expectations = smoke test (no crash)."""
    eid = f"fuzz-{index:02d}"
    return {
        "id": eid,
        "path": f"images/synthetic/{eid}.png",
        "source": "synthetic",
        "expected": {},
    }


def generate(out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        if old.name != "manifest.json":
            old.unlink()
    entries: list[dict[str, Any]] = []
    for gen in GENERATORS:
        for variant in VARIANTS:
            if gen == "noise" and variant == "noise":
                continue
            img = render_image(gen, variant, seed_offset=hash(gen) % 1000)
            entry, eid = make_entry(gen, variant)
            ext = "jpg" if variant == "jpeg50" else "png"
            path = out_dir / f"{eid}.{ext}"
            if ext == "jpg":
                img.convert("RGB").save(path, format="JPEG", quality=50)
            else:
                img.save(path, format="PNG")
            entries.append(entry)
    rng = random.Random(SEED)
    for i in range(10):
        bg = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        img = Image.new("RGB", SIZE, bg)
        d = ImageDraw.Draw(img)
        for _ in range(rng.randint(3, 8)):
            shape = rng.choice(("rect", "ellipse", "line"))
            x0 = rng.randint(0, 500)
            y0 = rng.randint(0, 340)
            w = rng.randint(40, 140)
            h = rng.randint(40, 140)
            box = (x0, y0, x0 + w, y0 + h)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            if shape == "rect":
                d.rectangle(box, fill=color)
            elif shape == "ellipse":
                d.ellipse(box, fill=color)
            else:
                d.line(box, fill=color, width=rng.randint(2, 12))
        eid = f"fuzz-{i:02d}"
        img.save(out_dir / f"{eid}.png", format="PNG")
        entries.append(fuzz_entry(i))
    entries.sort(key=lambda e: e["id"])
    (out_dir / "manifest.json").write_text(json.dumps({"images": entries}, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="determinism probe, no files")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(smoke(), indent=2))
        return 0
    generate()
    png_count = len(list(OUT_DIR.glob("*.png")))
    jpg_count = len(list(OUT_DIR.glob("*.jpg")))
    print(f"wrote {OUT_DIR} ({png_count + jpg_count} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
