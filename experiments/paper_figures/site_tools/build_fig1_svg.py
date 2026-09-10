#!/usr/bin/env python3
"""Rebuild Figure 1 (the SED loop) as a semantic SVG from the paper's vector PDF.

Every box, arrow and label from `main_diagram_checkpoint2.pdf` is emitted with the
same geometry and colours, grouped into named nodes (`f1-node-*`) and arrows
(`f1-arrow-*`) so `sed-fig1.js` can drive a step-by-step flow animation.

Usage:  python3 experiments/paper_figures/site_tools/build_fig1_svg.py
Output: docs/static/images/fig1/sed_loop.svg (+ icon PNGs already extracted)
"""
from __future__ import annotations

import html
import pathlib
import re

import fitz  # PyMuPDF

ROOT = pathlib.Path(__file__).resolve().parents[3]
PDF = ROOT / "paper_writing/SED_neurips/SED_neurips/figures/main_diagram_checkpoint2.pdf"
OUT = ROOT / "docs/static/images/fig1/sed_loop.svg"
ICON_DIR = "static/images/fig1"  # href base, relative to docs/index.html

VIEW_W, VIEW_H = 950, 488

# ---------------------------------------------------------------- semantics
# drawing index -> (group id, role)  roles: box | deco | arrow | bg
DRAW_GROUP = {
    1: ("bg", "bg"), 2: ("bg", "bg"),
    # banners
    23: ("banner-episode", "arrow"), 24: ("banner-post", "arrow"),
    25: ("divider", "deco"), 26: ("divider", "deco"),
    # input container
    3: ("node-input", "box"), 4: ("node-input", "box"), 5: ("node-input", "box"), 6: ("node-input", "deco"),
    # policy library + tree
    15: ("node-library", "box"),
    30: ("node-tree", "deco"), 31: ("node-tree", "deco"), 32: ("node-tree", "deco"),
    33: ("node-tree", "deco"), 34: ("node-tree", "deco"),
    35: ("node-tree", "deco"), 36: ("node-tree", "deco"), 37: ("node-tree", "deco"),
    38: ("node-tree", "deco"), 39: ("node-tree", "deco"),
    # retrieval
    14: ("node-retrieval", "box"), 40: ("node-retrieval", "box"), 41: ("node-retrieval", "deco"),
    42: ("node-retrieval", "ray"), 43: ("node-retrieval", "ray"), 44: ("node-retrieval", "ray"), 45: ("node-retrieval", "ray"),
    46: ("node-retrieval", "star"), 47: ("node-retrieval", "star"), 48: ("node-retrieval", "star"), 49: ("node-retrieval", "star-gold"),
    52: ("node-retrieval", "deco"), 53: ("node-retrieval", "deco"),
    # agent
    11: ("node-agent", "box"), 12: ("node-agent", "box"), 13: ("node-agent", "box"),
    # judge
    7: ("node-judge", "box"), 8: ("node-judge", "box-defended"), 9: ("node-judge", "box-attack"),
    # attribution
    29: ("node-attribution", "box"),
    # episodic memory
    10: ("node-memory", "box"),
    54: ("node-memory", "deco"), 55: ("node-memory", "deco"), 56: ("node-memory", "deco"),
    57: ("node-memory", "deco"), 58: ("node-memory", "deco"), 59: ("node-memory", "deco"), 60: ("node-memory", "deco"),
    # arrows (filled outlines from the slide)
    19: ("arrow-read", "arrow"), 18: ("arrow-query", "arrow"), 20: ("arrow-to-agent", "arrow"),
    21: ("arrow-trajectory", "arrow"), 50: ("arrow-store", "arrow"), 16: ("arrow-synth", "arrow"),
    17: ("arrow-update", "arrow"),
    22: ("arrow-green", "green"), 27: ("arrow-green", "green"), 28: ("arrow-green", "green"), 51: ("arrow-green", "green"),
}

# Text spans are attached to groups by bounding-box hit-testing against these regions.
TEXT_REGIONS = [
    ("banner-episode", (240, 0, 460, 34)),
    ("banner-post", (770, 0, 900, 34)),
    ("arrow-update", (255, 52, 620, 74)),
    ("node-attribution", (760, 55, 900, 112)),
    ("node-input", (192, 87, 702, 190)),
    ("node-library", (7, 152, 165, 227)),
    ("arrow-read", (175, 196, 222, 220)),
    ("arrow-query", (240, 196, 390, 218)),
    ("node-judge", (763, 150, 922, 415)),
    ("node-retrieval", (287, 229, 483, 396)),
    ("node-agent", (550, 227, 696, 404)),
    ("arrow-to-agent", (485, 260, 550, 360)),
    ("arrow-trajectory", (696, 290, 762, 350)),
    ("arrow-synth", (205, 350, 285, 392)),
    ("node-tree", (7, 250, 215, 460)),
    ("node-memory", (297, 408, 454, 481)),
    ("arrow-store", (485, 428, 712, 448)),
    ("arrow-store-tag", (518, 457, 802, 478)),
]

# Hand-drawn centrelines for the flow animation (viewBox coordinates).
FLOW_PATHS = {
    "flow-query": "M 393.1 190.5 V 225.5",
    "flow-read": "M 165 219.6 H 224.4 V 285.2 H 281.5",
    "flow-ray-a": "M 315.3 367.4 L 380.5 306.8",
    "flow-ray-b": "M 315.3 367.4 L 381 328.6",
    "flow-ray-c": "M 315.3 367.4 L 412 341.3",
    "flow-ray-d": "M 315.3 367.4 L 444 358",
    "flow-to-agent": "M 481 311.2 H 548.5",
    "flow-trajectory": "M 698.4 318.3 H 759",
    "flow-green": "M 912.2 262 H 923.8 V 310.4 H 938 V 75.6 H 897",
    "flow-store": "M 842.5 414.8 V 449.8 H 456.5",
    "flow-synth": "M 295.2 438.9 H 249.5 V 393.2 H 206",
    "flow-update": "M 938 76.3 H 87.1 V 150.5",
}

ICON_BBOXES = [
    ("icon_00.png", (205.2, 89.3, 237.6, 121.7), "node-input"),
    ("icon_01.png", (324.0, 92.2, 353.5, 121.7), "node-input"),
    ("icon_02.png", (501.8, 114.5, 526.3, 137.5), "node-input"),
    ("icon_03.png", (501.1, 141.2, 525.6, 163.5), "node-input"),
    ("icon_04.png", (498.2, 161.3, 525.6, 187.2), "node-input"),
    ("icon_05.png", (374.4, 285.2, 389.5, 300.3), "node-retrieval"),
    ("icon_06.png", (769.7, 166.4, 791.3, 187.2), "node-judge"),
    ("icon_07.png", (771.1, 239.8, 802.1, 270.8), "node-judge"),
    ("icon_08.png", (773.3, 311.8, 799.2, 338.5), "node-judge"),
    ("icon_09.png", (554.4, 247.0, 578.2, 270.8), "node-agent"),
    ("icon_10.png", (10.1, 179.3, 53.3, 218.9), "node-library"),
    ("icon_11.png", (344.2, 375.1, 366.8, 395.6), "node-retrieval"),
    ("icon_12.png", (761.0, 66.3, 789.1, 93.6), "node-attribution"),
]

GROUP_ORDER = [
    "bg", "divider", "banner-episode", "banner-post",
    "arrow-update", "arrow-green",
    "node-input", "node-library", "node-tree", "node-retrieval", "node-agent",
    "node-judge", "node-attribution", "node-memory",
    "arrow-read", "arrow-query", "arrow-to-agent", "arrow-trajectory",
    "arrow-store", "arrow-store-tag", "arrow-synth",
]

# ---------------------------------------------------------------- helpers

def f(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def color(c):
    if c is None:
        return None
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c)


def path_d(items) -> str:
    """Convert PyMuPDF drawing items into an SVG path string."""
    d = []
    cur = None
    for it in items:
        op = it[0]
        if op == "re":
            r = it[1]
            d.append(f"M {f(r.x0)} {f(r.y0)} H {f(r.x1)} V {f(r.y1)} H {f(r.x0)} Z")
            cur = None
            continue
        if op == "qu":
            q = it[1]
            d.append(f"M {f(q.ul.x)} {f(q.ul.y)} L {f(q.ur.x)} {f(q.ur.y)} L {f(q.lr.x)} {f(q.lr.y)} L {f(q.ll.x)} {f(q.ll.y)} Z")
            cur = None
            continue
        p1 = it[1]
        if cur is None or abs(cur.x - p1.x) > 0.05 or abs(cur.y - p1.y) > 0.05:
            d.append(f"M {f(p1.x)} {f(p1.y)}")
        if op == "l":
            p2 = it[2]
            d.append(f"L {f(p2.x)} {f(p2.y)}")
            cur = p2
        elif op == "c":
            c1, c2, p4 = it[2], it[3], it[4]
            d.append(f"C {f(c1.x)} {f(c1.y)} {f(c2.x)} {f(c2.y)} {f(p4.x)} {f(p4.y)}")
            cur = p4
    return " ".join(d)


def rounded_rect_from_items(items):
    """Slide-exported rounded rects are 4 curves + 4 lines; recover rect + radius."""
    if len(items) != 8 or [it[0] for it in items] != ["c", "l"] * 4:
        return None
    c0 = items[0]
    r = abs(c0[4].x - c0[1].x)
    xs, ys = [], []
    for it in items:
        for p in (it[1], it[-1]):
            xs.append(p.x); ys.append(p.y)
    return min(xs), min(ys), max(xs), max(ys), r


def circle_from_items(items):
    if len(items) != 4 or any(it[0] != "c" for it in items):
        return None
    xs, ys = [], []
    for it in items:
        xs.append(it[1].x); ys.append(it[1].y)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2


def in_region(bbox, region):
    x0, y0, x1, y1 = bbox
    rx0, ry0, rx1, ry1 = region
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def text_group(bbox):
    for gid, region in TEXT_REGIONS:
        if in_region(bbox, region):
            return gid
    return "misc"


# ---------------------------------------------------------------- build

def build():
    doc = fitz.open(PDF)
    page = doc[0]
    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    groups["misc"] = []

    # --- shapes
    for i, d in enumerate(page.get_drawings()):
        if i == 0:
            continue  # white page background -> keep SVG transparent
        gid, role = DRAW_GROUP.get(i, ("misc", "deco"))
        fill = color(d.get("fill"))
        stroke = color(d.get("color"))
        width = d.get("width") or 0
        dashes = d.get("dashes")
        items = d["items"]
        attrs = []
        if fill:
            attrs.append(f'fill="{fill}"')
        else:
            attrs.append('fill="none"')
        if stroke and width:
            attrs.append(f'stroke="{stroke}" stroke-width="{f(width)}"')
            if dashes and dashes not in ("[]", "[] 0"):
                nums = re.findall(r"[\d.]+", dashes.split("]")[0])
                if nums:
                    attrs.append(f'stroke-dasharray="{" ".join(nums)}"')
        cls = f"f1-{role}"
        rr = rounded_rect_from_items(items)
        circ = circle_from_items(items)
        if rr and role.startswith("box"):
            x0, y0, x1, y1, r = rr
            el = (f'<rect class="{cls}" x="{f(x0)}" y="{f(y0)}" width="{f(x1 - x0)}" '
                  f'height="{f(y1 - y0)}" rx="{f(r)}" {" ".join(attrs)}/>')
        elif rr:
            x0, y0, x1, y1, r = rr
            el = (f'<rect class="{cls}" x="{f(x0)}" y="{f(y0)}" width="{f(x1 - x0)}" '
                  f'height="{f(y1 - y0)}" rx="{f(r)}" {" ".join(attrs)}/>')
        elif circ and gid == "node-tree":
            cx, cy, rx, ry = circ
            el = f'<ellipse class="{cls} f1-tree-node" cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}" {" ".join(attrs)}/>'
        elif circ:
            cx, cy, rx, ry = circ
            el = f'<ellipse class="{cls}" cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}" {" ".join(attrs)}/>'
        else:
            extra = ' stroke-linecap="round" stroke-linejoin="round"' if stroke and width else ""
            el = f'<path class="{cls}" d="{path_d(items)}" {" ".join(attrs)}{extra}/>'
        groups.setdefault(gid, []).append(el)

    # --- icons
    for name, (x0, y0, x1, y1), gid in ICON_BBOXES:
        groups[gid].append(
            f'<image class="f1-icon" href="{ICON_DIR}/{name}" x="{f(x0)}" y="{f(y0)}" '
            f'width="{f(x1 - x0)}" height="{f(y1 - y0)}" preserveAspectRatio="xMidYMid meet"/>'
        )

    # --- text
    tdict = page.get_text("dict")
    for b in tdict["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            # Slide exports split runs at punctuation/space changes ("Diagnosis" + ":").
            # Merge touching spans of the same colour back into one run.
            merged = []
            for s in line["spans"]:
                if not s["text"].strip() and not merged:
                    continue
                if merged:
                    prev = merged[-1]
                    gap = s["bbox"][0] - prev["bbox"][2]
                    math = s["font"].startswith("Cambria") or prev["font"].startswith("Cambria")
                    if abs(gap) < 1.6 and s["color"] == prev["color"] and not math:
                        keep_bold = prev if len(prev["text"].strip()) >= len(s["text"].strip()) else s
                        prev["text"] += s["text"]
                        prev["bbox"] = (prev["bbox"][0], min(prev["bbox"][1], s["bbox"][1]),
                                        s["bbox"][2], max(prev["bbox"][3], s["bbox"][3]))
                        prev["size"] = max(prev["size"], s["size"])
                        prev["flags"] = keep_bold["flags"]
                        prev["font"] = keep_bold["font"]
                        continue
                merged.append(dict(s))
            for s in merged:
                txt = s["text"]
                if not txt.strip():
                    continue
                x0, y0, x1, y1 = s["bbox"]
                size = s["size"]
                font = s["font"]
                bold = bool(s["flags"] & 16) or "Bold" in font
                col = "#%06x" % s["color"]
                # strip leading spaces (icons occupy that room) and shift x accordingly
                lead = len(txt) - len(txt.lstrip(" "))
                txt = txt.strip()
                x = x0 + lead * 0.278 * size
                if font.startswith("Aptos"):
                    base = y0 + 0.93 * size
                    family = "Aptos, 'Segoe UI', Arial, Helvetica, sans-serif"
                elif font.startswith("Cambria"):
                    base = y0 + 0.80 * size
                    family = "'Cambria Math', Cambria, 'Times New Roman', serif"
                else:
                    base = y0 + 0.905 * size
                    family = "Arial, 'Helvetica Neue', Helvetica, sans-serif"
                gid = text_group((x0, y0, x1, y1))
                if txt == "𝝉":
                    txt = "τ"
                    style = "font-style:italic;font-weight:700"
                elif txt == "𝒄":
                    txt = "c"
                    style = "font-style:italic;font-weight:700"
                else:
                    style = f"font-weight:{700 if bold else 400}"
                el = (f'<text class="f1-text" x="{f(x)}" y="{f(base)}" font-size="{f(size)}" '
                      f'fill="{col}" style="font-family:{family};{style}">{html.escape(txt)}</text>')
                groups.setdefault(gid, []).append(el)

    # --- assemble
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {VIEW_W} {VIEW_H}" width="{VIEW_W}" height="{VIEW_H}" '
        f'class="sed-fig1-svg" role="img" aria-labelledby="f1-title f1-desc" '
        f'style="text-rendering:geometricPrecision;shape-rendering:geometricPrecision">',
        '<title id="f1-title">The SED loop</title>',
        '<desc id="f1-desc">Policies are retrieved into the agent prompt, the frozen agent acts, an external judge scores the trajectory, and harmful episodes are synthesised into new policies used in later episodes.</desc>',
        "<defs>",
        '<filter id="f1-glow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '<filter id="f1-packet-glow" x="-100%" y="-100%" width="300%" height="300%">'
        '<feGaussianBlur stdDeviation="2.4"/></filter>',
        "</defs>",
    ]
    for gid in GROUP_ORDER + ["misc"]:
        els = groups.get(gid, [])
        if not els:
            continue
        kind = "node" if gid.startswith("node-") else ("arrow" if gid.startswith("arrow-") else "static")
        out.append(f'<g id="f1-{gid}" class="f1-group f1-{kind}" data-part="{gid}">')
        out.extend("  " + e for e in els)
        out.append("</g>")

    # --- flow layer (animated by JS)
    out.append('<g id="f1-flows" class="f1-flows" aria-hidden="true">')
    for fid, d in FLOW_PATHS.items():
        out.append(f'  <path id="f1-{fid}" class="f1-flow" data-flow="{fid}" d="{d}" fill="none" '
                   f'stroke-linecap="round" stroke-linejoin="round" pathLength="1000"/>')
    out.append("</g>")
    out.append('<g id="f1-packets" class="f1-packets" aria-hidden="true"></g>')
    out.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")
    misc = groups.get("misc", [])
    if misc:
        print(f"note: {len(misc)} elements landed in misc group")


if __name__ == "__main__":
    build()
