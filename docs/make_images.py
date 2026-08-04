"""Generate README images for DeepSight (docs/images/).

- terminal-demo.png: real vision_eyes output rendered as a terminal window.
- architecture.png: dark-theme system diagram.

Run: uv run python docs/make_images.py
Requires pillow (a project dependency).
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)

BG = (13, 17, 23)          # GitHub dark canvas
WIN_BG = (22, 27, 34)      # window background
BORDER = (48, 54, 61)      # subtle border
TEXT = (230, 237, 243)     # primary text
DIM = (139, 148, 158)      # secondary text
ACCENT = (88, 166, 255)    # blue accent
GREEN = (63, 185, 80)
RED = (255, 95, 87)
YELLOW = (254, 188, 46)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Menlo.ttc",
    "/Library/Fonts/Menlo.ttc",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size, index=0)
    raise FileNotFoundError("no Menlo font found")


def rrect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, **kw) -> None:
    draw.rounded_rectangle(box, radius=radius, **kw)


def rounded_box(d: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], r: int = 10) -> None:
    rrect(d, xy, r, fill=WIN_BG, outline=BORDER, width=1)


def arrow(
    d: ImageDraw.ImageDraw,
    x1: int, y1: int, x2: int, y2: int,
    color=ACCENT, width=3,
) -> None:
    d.line([(x1, y1), (x2, y2)], fill=color, width=width)
    # arrowhead
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    L = 12
    for off in (0.35, -0.35):
        d.line(
            [(x2, y2), (x2 - L * math.cos(ang - off), y2 - L * math.sin(ang - off))],
            fill=color, width=width,
        )


def draw_text_center(d: ImageDraw.ImageDraw, cx: int, cy: int, text: str, font, fill=TEXT) -> None:
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    d.text((cx - w / 2, cy - h / 2), text, font=font, fill=fill)


def make_terminal() -> str:
    """Render the real vision_eyes output for surfing.jpg as a terminal window."""
    title = "vision_eyes surfing.jpg  (2047x1667)"
    lines = [
        ("$", "vision_eyes surfing.jpg"),
        ("o", "scene: recreation(0.97), sport(0.97), watersport(0.94), surfing(0.94), sports_equipment(0.82)"),  # noqa: E501
        ("o", "sports: sports equipment(0.82), surfing(0.81)"),
        ("o", "faces: 0"),
        ("o", "humans: 0"),
        ("o", "pose: 1 human(s)"),
        ("o", "pose 0: joints=16 arms_up=true"),
        ("o", "rectangles: 0"),
        ("o", "animals: none"),
        ("o", "colors: #080808(4%), #080810(3%), #101820(2%), #182020(1%), #081018(1%), #7898B8(1%)"),  # noqa: E501
        ("o", "ALL DONE"),
    ]
    font = load_font(22)
    bold = load_font(22)
    small = load_font(13)

    pad_x, pad_y = 48, 44
    line_h = 32
    title_h = 56
    body_h = len(lines) * line_h + pad_y * 2
    win_w = 1180
    win_h = title_h + body_h
    canvas = Image.new("RGB", (win_w + 2 * pad_x, win_h + 2 * pad_y), BG)
    d = ImageDraw.Draw(canvas)
    x0, y0 = pad_x, pad_y
    rrect(d, (x0, y0, x0 + win_w, y0 + win_h), 14, fill=WIN_BG, outline=BORDER, width=1)

    # traffic lights
    lx = x0 + 24
    ly = y0 + 24
    for col in (RED, YELLOW, GREEN):
        d.ellipse((lx, ly, lx + 14, ly + 14), fill=col)
        lx += 22
    d.text((lx + 14, ly - 3), title, font=small, fill=DIM)

    # body
    ty = y0 + title_h
    for kind, text in lines:
        if kind == "$":
            d.text((x0 + 28, ty), "$ ", font=bold, fill=GREEN)
            d.text((x0 + 28 + 44, ty), text, font=font, fill=TEXT)
        else:
            d.text((x0 + 28 + 44, ty), text, font=font, fill=TEXT if kind == "o" else DIM)
        ty += line_h

    path = os.path.join(OUT_DIR, "terminal-demo.png")
    canvas.save(path)
    return path


def make_architecture() -> str:
    """Dark diagram: image -> vision_eyes -> signals; optional LLM loop."""
    W, H = 1600, 900
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    bold = load_font(30)
    small = load_font(18)

    # header
    d.text((60, 40), "DeepSight architecture", font=bold, fill=TEXT)
    d.text((60, 84), "device-native vision · zero tokens · zero server", font=small, fill=DIM)

    bw, bh = 400, 150
    # 1: input image
    box1 = (80, 200, 80 + bw, 200 + bh)
    rounded_box(d, box1, 16)
    draw_text_center(d, box1[0] + bw // 2, box1[1] + 60, "input image", bold, TEXT)
    draw_text_center(d, box1[0] + bw // 2, box1[1] + 100, "any photo, screenshot, scan", small, DIM)

    # 2: vision_eyes
    box2 = (620, 160, 620 + bw, 160 + bh + 40)
    rounded_box(d, box2, 16)
    draw_text_center(d, box2[0] + bw // 2, box2[1] + 50, "vision_eyes", bold, TEXT)
    draw_text_center(d, box2[0] + bw // 2, box2[1] + 88, "Apple Vision, on-device", small, DIM)
    feats = "OCR · scene · saliency · faces · sports · pose · colors"
    draw_text_center(d, box2[0] + bw // 2, box2[1] + 118, feats, small, ACCENT)

    # 3: describe (device-native path)
    box3 = (1160, 200, 1160 + bw, 200 + bh)
    rounded_box(d, box3, 16)
    draw_text_center(d, box3[0] + bw // 2, box3[1] + 55, "deepsight describe", bold, TEXT)
    draw_text_center(d, box3[0] + bw // 2, box3[1] + 98, "structured text signals", small, DIM)
    draw_text_center(d, box3[0] + bw // 2, box3[1] + 124, "milliseconds, free", small, GREEN)

    # 4: reasoning loop (optional)
    box4 = (620, 560, 620 + bw, 560 + bh + 40)
    rounded_box(d, box4, 16)
    draw_text_center(d, box4[0] + bw // 2, box4[1] + 50, "vision-session loop", bold, TEXT)
    draw_text_center(d, box4[0] + bw // 2, box4[1] + 88, "Orchestrator + PerceptionCache", small, DIM)  # noqa: E501
    draw_text_center(d, box4[0] + bw // 2, box4[1] + 118, "optional text-only LLM", small, DIM)

    # 5: answer
    box5 = (1160, 560, 1160 + bw, 560 + bh)
    rounded_box(d, box5, 16)
    draw_text_center(d, box5[0] + bw // 2, box5[1] + 60, "natural answer", bold, TEXT)
    draw_text_center(d, box5[0] + bw // 2, box5[1] + 100, "Q&A about the image", small, DIM)

    arrow(d, 80 + bw, 275, 620, 275)
    arrow(d, 620 + bw, 275, 1160, 275)
    arrow(d, 820, 160 + bh + 40 - 10, 820, 560)
    arrow(d, 620 + bw, 650, 1160, 650)
    # loop feedback arc label
    d.text((860, 400), "look / crop / zoom rounds", font=small, fill=DIM)

    # footnote
    d.text((60, 800), "The eyes are a compiled Swift binary; the loop is optional and only calls an LLM when you ask a question.", font=small, fill=DIM)  # noqa: E501

    path = os.path.join(OUT_DIR, "architecture.png")
    canvas.save(path)
    return path


if __name__ == "__main__":
    print("wrote", make_terminal())
    print("wrote", make_architecture())
