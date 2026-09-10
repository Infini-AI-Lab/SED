#!/usr/bin/env python3
"""Print SWE-bench --filter regex from a pass1 swebench_report.json path (argv)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    report_path = Path(sys.argv[1])
    ids = json.loads(report_path.read_text())["resolved_ids"]
    print("(" + "|".join(ids) + ")")


if __name__ == "__main__":
    main()
