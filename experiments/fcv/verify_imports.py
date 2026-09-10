#!/usr/bin/env python3
"""Verify SED + mini-swe-agent imports (safe for paths with apostrophes)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FCV_ROOT = Path(os.environ.get("FCV_ROOT", ROOT / "third_party" / "FCV"))
FCV_SRC = FCV_ROOT / "mini-swe-agent" / "src"
sys.path[:0] = [str(ROOT), str(FCV_SRC)]

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from evaluation.FCV.sed_miniswe_agent import build_sed_agent_class  # noqa: E402

_ = DefaultAgent, build_sed_agent_class
print("FCV+SED imports OK")
