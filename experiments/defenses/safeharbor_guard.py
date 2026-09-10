"""SafeHarbor memory-tree guard via vendored upstream RiskTree."""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Any

from experiments.defenses.vendor_paths import (
    ensure_safeharbor_src_on_path,
    memory_tree_pkl,
    safety_projector_weights,
)

_TREE: Any = None
_LOCK = threading.Lock()


def _bind_missing_methods(tree: Any, cls: type) -> None:
    for method_name in (
        "inject_benign_dataset",
        "clear_benign_samples",
        "_find_nearest_benign_neighbor",
        "_find_nearest_neighbor_exemplars",
    ):
        if not hasattr(tree, method_name) and hasattr(cls, method_name):
            method = getattr(cls, method_name)
            setattr(tree, method_name, method.__get__(tree, cls))


def _attach_safety_projector(tree: Any, projector_path: str) -> None:
    import torch
    from SafetyProjector import SafetyProjector  # type: ignore[import-untyped]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model = tree.embedding_model
    dim_fn = getattr(embed_model, "get_embedding_dimension", None) or embed_model.get_sentence_embedding_dimension
    input_dim = dim_fn()
    checkpoint = torch.load(projector_path, map_location=device)
    tree.safety_projector = SafetyProjector(input_dim=input_dim, device=device)
    tree.safety_projector.load_state_dict(checkpoint["model_state_dict"], strict=False)
    tree.safety_projector.to(device)
    tree.safety_projector.eval()
    tree.device = device
    tree.use_safety_projection = True


def _fix_projected_embeddings(tree: Any) -> None:
    """Upstream inject_benign_dataset may store (vector, prob) tuples — normalize."""
    for category in tree.root.children:
        for cluster in category.children:
            for attr in ("projected_center_embedding", "projected_benign_center_embedding"):
                value = getattr(cluster, attr, None)
                if isinstance(value, tuple) and value:
                    setattr(cluster, attr, value[0])


def _preproject_cluster_embeddings(tree: Any) -> None:
    if not getattr(tree, "use_safety_projection", False):
        return
    for category in tree.root.children:
        for cluster in category.children:
            if cluster.center_embedding is None:
                continue
            if getattr(cluster, "projected_center_embedding", None) is not None:
                continue
            projected, _prob = tree._project_embedding(cluster.center_embedding.copy())
            cluster.projected_center_embedding = projected

    _fix_projected_embeddings(tree)


def load_risk_tree(*, use_cache: bool = True) -> Any:
    """Load RiskTree with embedding model + Safety Projector (SED-fixed init order)."""
    global _TREE
    if use_cache and _TREE is not None:
        return _TREE

    ensure_safeharbor_src_on_path()
    from risk_tree import RiskTree, get_embedding_model  # type: ignore[import-untyped]

    pkl = memory_tree_pkl()
    if not pkl.is_file():
        raise FileNotFoundError(f"SafeHarbor memory pkl missing: {pkl}")
    projector = safety_projector_weights()
    projector_path = str(projector) if projector.is_file() else None

    with pkl.open("rb") as handle:
        tree = pickle.load(handle)

    if not hasattr(tree, "score_log_file"):
        tree.score_log_file = "./logs/score_log.jsonl"
    if not hasattr(tree, "_score_log_count"):
        tree._score_log_count = 0
    if not hasattr(tree, "safety_projector"):
        tree.safety_projector = None
    if not hasattr(tree, "use_safety_projection"):
        tree.use_safety_projection = False

    tree.embedding_model = get_embedding_model()
    _bind_missing_methods(tree, RiskTree)

    if projector_path and Path(projector_path).is_file():
        try:
            _attach_safety_projector(tree, projector_path)
            _preproject_cluster_embeddings(tree)
        except Exception as exc:
            print(f"SafeHarbor: Safety Projector load failed ({exc}); continuing without projection.")
            tree.use_safety_projection = False

    _fix_projected_embeddings(tree)

    exemplar_clusters = sum(
        1
        for category in tree.root.children
        for cluster in category.children
        if getattr(cluster, "benign_exemplars", None)
    )
    if exemplar_clusters == 0:
        align = os.environ.get("AGENT_ALIGN_PATH", "")
        print(
            "SafeHarbor WARNING: loaded pkl has no benign_exemplars "
            "(benign utility pseudo-memories missing). "
            "Run: bash experiments/defenses/setup_safeharbor_benign_calibration.sh"
            + (f"  Expected AgentAlign at: {align}" if align else "")
        )

    if use_cache:
        _TREE = tree
    return tree


def save_risk_tree(tree: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(tree, handle)


def _load_tree() -> Any:
    return load_risk_tree(use_cache=True)


class SafeHarborGuard:
    """Retrieve SafeHarbor rules and return system-message augmentation text."""

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k or int(os.environ.get("SAFEHARBOR_TOP_K", "3"))

    def augment_messages(self, user_query: str) -> list[dict[str, str]]:
        """Run upstream retrieve_query; returns OpenAI-style message list."""
        tree = _load_tree()
        base = [{"role": "user", "content": user_query}]
        return tree.retrieve_query(base, top_k=self.top_k)

    def system_augmentation(self, user_query: str, existing_system: str = "") -> str:
        """Extract injected system content from retrieve_query output."""
        augmented = self.augment_messages(user_query)
        parts: list[str] = []
        for msg in augmented:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    parts.append(content.strip())
        injected = "\n\n".join(parts)
        if not injected:
            return existing_system
        if existing_system.strip():
            return f"{existing_system.rstrip()}\n\n{injected}"
        return injected

    def is_hard_block(self, user_query: str) -> bool:
        """Heuristic: upstream block branch adds SECURITY CONTEXT refusal guidance."""
        aug = self.system_augmentation(user_query).lower()
        return "[system security context]" in aug and "must refuse" in aug
