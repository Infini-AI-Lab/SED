#!/usr/bin/env python3
"""Build a full-domain DTap task list (benign + direct + indirect)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DOMAINS = ("crm", "workflow", "travel", "customer-service", "windows", "code")
SPLITS = ("benign", "direct", "indirect")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=DOMAINS)
    p.add_argument("--output", required=True)
    p.add_argument("--dtap", default="", help="DecodingTrust-Agent root")
    args = p.parse_args()

    dtap = Path(args.dtap) if args.dtap else Path.home() / "nisargagondi" / "DecodingTrust-Agent"
    bench = dtap / "benchmark" / args.domain
    rows: list[dict] = []
    for split in SPLITS:
        path = bench / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks ({args.domain}) -> {out}")


if __name__ == "__main__":
    main()
