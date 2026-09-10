#!/usr/bin/env python3
"""Allow per-pipeline hooks.json via DTAP_HOOKS_JSON (fixes parallel sweep race)."""
from __future__ import annotations

import sys
from pathlib import Path

ANCHOR = '_HOOKS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"'

REPLACEMENT = """_HOOKS_CONFIG_PATH = (
    Path(os.environ["DTAP_HOOKS_JSON"]).expanduser()
    if os.environ.get("DTAP_HOOKS_JSON")
    else Path(__file__).resolve().parents[1] / "hooks" / "hooks.json"
)"""


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if REPLACEMENT.strip() in text:
        print(f"  hooks loader: already patched ({path})")
        return False
    if ANCHOR not in text:
        print(f"  hooks loader: anchor missing in {path}")
        return False
    text = text.replace(ANCHOR, REPLACEMENT, 1)
    if "import os" not in text.split("\n", 15)[0:15]:
        lines = text.splitlines()
        insert_at = 0
        for i, line in enumerate(lines[:20]):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, "import os")
        text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    path.write_text(text, encoding="utf-8")
    print(f"  hooks loader: patched {path}")
    return True


def main() -> None:
    dtap = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "nisargagondi/DecodingTrust-Agent"
    targets = [
        dtap / "dt_arena" / "src" / "types" / "hooks.py",
    ]
    for t in targets:
        if t.is_file():
            patch_file(t)


if __name__ == "__main__":
    main()
