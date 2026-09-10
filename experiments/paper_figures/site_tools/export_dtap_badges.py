#!/usr/bin/env python3
"""Export DTap scatter badges — corner-chip design from fig_dtap_final.py (paper figure)."""

from __future__ import annotations

import sys
from pathlib import Path

PAPER_FIGURES = Path(__file__).resolve().parents[1]
ROOT = PAPER_FIGURES.parents[1]
sys.path.insert(0, str(PAPER_FIGURES))

from dtap_figure_data import DEFENSE_KEYS, MODEL_LABELS  # noqa: E402
from fig_dtap_final import make_badge  # noqa: E402
from render_dtap_canvas import _defense_disc  # noqa: E402

OUT_DIR = ROOT / "docs" / "static" / "images" / "badges"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper renders at 360 px, displays ~50 px (ratio 7.2). Web chart uses pointRadius 11 → 22 px.
COMBO_PX = 96
DEFENSE_PX = 48


def main() -> None:
    for model in MODEL_LABELS:
        for defense in DEFENSE_KEYS:
            out = OUT_DIR / f"{model}_{defense}.png"
            make_badge(model, defense, COMBO_PX).save(out)
            print(out.name)

    for defense in DEFENSE_KEYS:
        if defense == "no_defense":
            continue
        out = OUT_DIR / f"defense_{defense}.png"
        _defense_disc(defense, DEFENSE_PX).save(out)
        print(out.name)


if __name__ == "__main__":
    main()
