#!/usr/bin/env python3
"""Render the social-preview card (1200x630) and the apple-touch-icon for the site.

Usage:  python3 experiments/paper_figures/site_tools/build_og_card.py
Output: docs/static/images/og_card.png, docs/static/images/apple-touch-icon.png
"""
from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[3]
IMG = ROOT / "docs/static/images"
FONTS = pathlib.Path("/System/Library/Fonts/Supplemental")

INK = (23, 26, 24)
MUTED = (94, 101, 97)
GREEN = (23, 107, 82)
GREEN_DARK = (14, 74, 57)
BG = (244, 247, 245)
LINE = (223, 228, 225)


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


def og_card() -> None:
    W, H = 1200, 630
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    # white content card with a hairline border, like the site
    d.rounded_rectangle((28, 28, W - 28, H - 28), radius=22, fill=(255, 255, 255), outline=LINE, width=2)
    d.rounded_rectangle((29, 60, 41, H - 60), radius=6, fill=GREEN)  # accent bar, inset past the corners

    x0 = 82
    d.text((x0, 76), "PREPRINT · 2026", font=font("Arial Bold.ttf", 21), fill=GREEN)

    title = font("Georgia Bold.ttf", 78)
    d.text((x0 - 3, 108), "Self-Evolving Defense", font=title, fill=INK)

    sub = font("Georgia.ttf", 34)
    d.text((x0, 208), "Continual Security Policy Learning for LLM Agents", font=sub, fill=(63, 71, 67))

    deck = font("Arial.ttf", 25)
    lines = [
        "A training-free framework that turns harmful agent trajectories into",
        "reusable security policies, so a frozen agent improves as attacks evolve.",
    ]
    y = 272
    for ln in lines:
        d.text((x0, y), ln, font=deck, fill=MUTED)
        y += 36

    # three headline stats
    stats = [("0.42%", "targeted ASR on AgentDojo"), ("7.8%", "X-Teaming ASR on HarmBench"), ("6 / 8", "security benchmarks led")]
    d.line((x0, 372, W - 82, 372), fill=LINE, width=2)
    colw = (W - 82 - x0) // 3
    big = font("Georgia Bold.ttf", 54)
    small = font("Arial.ttf", 21)
    for i, (num, label) in enumerate(stats):
        cx = x0 + i * colw
        d.text((cx, 392), num, font=big, fill=GREEN_DARK)
        d.text((cx + 2, 458), label, font=small, fill=MUTED)
        if i:
            d.line((cx - 26, 392, cx - 26, 486), fill=LINE, width=2)
    d.line((x0, 508, W - 82, 508), fill=LINE, width=2)

    # footer line: authors + institutions
    d.text((x0, 532), "Le · Gondi · Peng · Ni · Jia · Chen · Zheng", font=font("Arial.ttf", 22), fill=INK)
    d.text((x0, 562), "Carnegie Mellon University · University of Massachusetts Amherst", font=font("Arial.ttf", 20), fill=MUTED)

    # loop glyph in the corner (same mark as the favicon)
    glyph = Image.open(IMG / "apple-touch-icon.png").resize((96, 96), Image.LANCZOS)
    im.paste(glyph, (W - 82 - 96, 76), glyph)

    im.save(IMG / "og_card.png", optimize=True)


def touch_icon() -> None:
    S = 180
    im = Image.new("RGBA", (S * 4, S * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    k = S * 4 / 64
    d.rounded_rectangle((0, 0, S * 4 - 1, S * 4 - 1), radius=int(14 * k), fill=GREEN)
    # arc: same geometry as favicon.svg (centre 32,32 r 17, gap at the upper right)
    box = ((32 - 17) * k, (32 - 17) * k, (32 + 17) * k, (32 + 17) * k)
    d.arc(box, start=-10, end=300, fill=(255, 255, 255), width=int(6 * k))
    d.polygon([(39 * k, 20 * k), (49 * k, 22.5 * k), (46.5 * k, 12.5 * k)], fill=(255, 255, 255))
    d.ellipse(((32 - 5.5) * k, (32 - 5.5) * k, (32 + 5.5) * k, (32 + 5.5) * k), fill=(255, 255, 255))
    im = im.resize((S, S), Image.LANCZOS)
    im.save(IMG / "apple-touch-icon.png", optimize=True)


if __name__ == "__main__":
    touch_icon()
    og_card()
    print("wrote", IMG / "og_card.png", "and", IMG / "apple-touch-icon.png")
