#!/usr/bin/env python3
"""Exit 0 if smoke preds.json has successful patches (no Docker failures)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_RETRY_MARKERS = (
    "returned non-zero exit status",
    "CalledProcessError",
    "timed out after",
)


def main() -> None:
    preds_path = Path(sys.argv[1])
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    if not preds_path.exists():
        print(f"Missing smoke preds: {preds_path}", file=sys.stderr)
        raise SystemExit(1)

    preds = json.loads(preds_path.read_text())
    if len(preds) < expected:
        print(f"Smoke incomplete: {len(preds)}/{expected} instances in preds.json", file=sys.stderr)
        raise SystemExit(1)

    failed = []
    for instance_id, row in preds.items():
        patch = str(row.get("model_patch", ""))
        if any(marker in patch for marker in _RETRY_MARKERS):
            failed.append(instance_id)

    if failed:
        print(f"Smoke failed for: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Smoke OK ({len(preds)} instances)")


if __name__ == "__main__":
    main()
