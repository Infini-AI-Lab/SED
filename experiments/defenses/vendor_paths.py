"""Resolve gitignored vendor roots (SafeHarbor, GuardAgent)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return _REPO_ROOT


def safeharbor_root() -> Path:
    return Path(os.environ.get("SAFEHARBOR_ROOT", _REPO_ROOT / "examples" / "SafeHarbor"))


def safeharbor_src() -> Path:
    return safeharbor_root() / "src"


def guardagent_root() -> Path:
    return Path(
        os.environ.get(
            "GUARDAGENT_ROOT",
            safeharbor_root() / "baselines" / "guardagent",
        )
    )


def memory_tree_pkl() -> Path:
    override = os.environ.get("SAFEHARBOR_MEMORY_PKL")
    if override:
        return Path(override)
    return safeharbor_src() / "final_memory_after_benign_calibration.pkl"


def safety_projector_weights() -> Path:
    override = os.environ.get("SAFEHARBOR_PROJECTOR_PTH")
    if override:
        return Path(override)
    return safeharbor_src() / "models" / "safety_projector.pth"


def ensure_safeharbor_src_on_path() -> Path:
    src = safeharbor_src()
    if not src.is_dir():
        raise FileNotFoundError(
            f"SafeHarbor not found at {src}. "
            "Run: bash experiments/defenses/setup_vendors.sh"
        )
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


def ensure_guardagent_on_path() -> Path:
    root = guardagent_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"GuardAgent baseline not found at {root}. "
            "Run: bash experiments/defenses/setup_vendors.sh"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
