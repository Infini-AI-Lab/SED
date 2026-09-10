"""Hierarchical policy tree layered over the L2 policy store.

A Node is one safety policy. Nodes hang under a synthetic `root`. The tree owns only
the data structure and the deterministic mutations (add / merge / rewrite / move);
the LLM-driven placement decisions live in tree_manager.py.

Two embeddings, never mixed:
  - placement (this module): the node NAME, so same-harm / same-framing rules cluster
    and a new rule lands next to its relatives. Used by nearest() at write time.
  - retrieval (the L2 MemoryStore surface): description + scope + detection + harm,
    compared against raw attack prompts. Untouched here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ROOT_ID = "root"

_counter = itertools.count(1)


def _new_id(prefix: str = "node") -> str:
    return f"{prefix}-{next(_counter):03d}"


@dataclass
class Node:
    id: str
    name: str
    policy: str                          # imperative rule == L2 "description"
    scope: str = ""
    detection: list[str] = field(default_factory=list)
    response: list[str] = field(default_factory=list)
    harm: str = ""
    retrievable: bool = True             # False for organizational category parents (GROUP)
    source_episode_ids: list[str] = field(default_factory=list)
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None   # NAME embedding, for placement only

    def embed_text(self) -> str:
        return self.name.replace("-", " ")


class Tree:
    def __init__(self) -> None:
        self.root = Node(id=ROOT_ID, name="root", policy="")
        self.nodes: dict[str, Node] = {ROOT_ID: self.root}

    # --- structure -------------------------------------------------------

    def add(self, node: Node, parent_id: str = ROOT_ID) -> Node:
        node.parent = parent_id
        self.nodes[node.id] = node
        self.nodes[parent_id].children.append(node.id)
        return node

    def merge(self, into_id: str, source_episode_ids: list[str],
              widen_policy: Optional[str] = None, embed_fn=None) -> Node:
        """Fold a duplicate rule into an existing node: keep its evidence, optionally widen."""
        node = self.nodes[into_id]
        for sid in source_episode_ids:
            if sid not in node.source_episode_ids:
                node.source_episode_ids.append(sid)
        if widen_policy and widen_policy.strip() and widen_policy != node.policy:
            node.policy = widen_policy
        return node

    def rewrite(self, node_id: str, name: str, policy: str, detection: list[str],
                embed_fn=None) -> Node:
        node = self.nodes[node_id]
        node.name = name
        node.policy = policy
        node.detection = detection
        if embed_fn is not None:
            node.embedding = embed_fn([node.embed_text()])[0]
        return node

    def move(self, child_id: str, new_parent_id: str) -> bool:
        """Reparent a node. Returns False (and changes nothing) if the move is illegal.

        Refuses a move under our own descendant: PROMOTE and SPLIT put the new parent at
        root and then move the target under it, so promoting A over B and later reusing B
        as A's parent detaches both from root into a cycle no walk can reach — which makes
        the whole tree render as "(empty)" to the placement model. Subclasses persist the
        new parent, so they must not write when this returns False.
        """
        if child_id == new_parent_id or self._is_descendant(new_parent_id, child_id):
            return False
        child = self.nodes[child_id]
        self.nodes[child.parent].children.remove(child_id)
        child.parent = new_parent_id
        self.nodes[new_parent_id].children.append(child_id)
        return True

    def _is_descendant(self, node_id: str, ancestor_id: str) -> bool:
        """True if node_id sits anywhere under ancestor_id."""
        seen = set()
        cur = node_id
        while cur in self.nodes and cur not in seen:
            seen.add(cur)
            cur = self.nodes[cur].parent
            if cur == ancestor_id:
                return True
        return False

    def new_node(self, name: str, policy: str, *, scope: str = "",
                 detection: Optional[list[str]] = None, response: Optional[list[str]] = None,
                 harm: str = "", retrievable: bool = True,
                 source_episode_ids: Optional[list[str]] = None, embed_fn=None) -> Node:
        node = Node(
            id=_new_id(),
            name=name,
            policy=policy,
            scope=scope,
            detection=detection or [],
            response=response or [],
            harm=harm,
            retrievable=retrievable,
            source_episode_ids=source_episode_ids or [],
        )
        if embed_fn is not None:
            node.embedding = embed_fn([node.embed_text()])[0]
        return node

    # --- placement neighbourhood ----------------------------------------

    def content_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.id != ROOT_ID and n.embedding is not None]

    def nearest(self, embedding: np.ndarray, k: int) -> list[tuple[Node, float]]:
        nodes = self.content_nodes()
        if not nodes:
            return []
        mat = np.stack([n.embedding for n in nodes])          # unit-normalised names
        sims = mat @ embedding
        order = np.argsort(-sims)[:k]
        return [(nodes[i], float(sims[i])) for i in order]

    # --- rendering -------------------------------------------------------

    def render(self) -> str:
        lines: list[str] = ["root"]

        def walk(node_id: str, prefix: str) -> None:
            kids = self.nodes[node_id].children
            for i, cid in enumerate(kids):
                last = i == len(kids) - 1
                node = self.nodes[cid]
                lines.append(f"{prefix}{'└── ' if last else '├── '}{node.name}")
                walk(cid, prefix + ("    " if last else "│   "))

        walk(ROOT_ID, "")
        return "\n".join(lines)

    def render_for_model(self) -> tuple[str, dict[str, str]]:
        """Outline as '[n] name' for the decide LLM, plus the {label: node id} map.

        The model has to echo a target back, and every operation needs that target to
        resolve — a 36-char UUID returned one character wrong silently discards the whole
        decision and looks like a deliberate ADD. Small integers are far harder to garble.
        """
        ids_by_label: dict[str, str] = {}

        def label(nid: str) -> str:
            node = self.nodes[nid]
            tag = str(len(ids_by_label) + 1)
            ids_by_label[tag] = nid
            return f"[{tag}] {node.name}" + ("  (category)" if not node.retrievable else "")

        return render_outline(lambda nid: self.nodes[nid].children, label), ids_by_label


def render_outline(children_of, label, root: str = ROOT_ID) -> str:
    """Indent a parent->children forest into a text outline.

    Pure structure only: `children_of(node_id)` returns a node's child ids and `label(node_id)`
    returns its display text. The caller's label decides the content — the decide view passes
    names only, the synthesizer view passes name + description — so this stays content-agnostic.
    """
    lines: list[str] = []

    def walk(node_id: str, depth: int) -> None:
        for cid in children_of(node_id):
            lines.append(f"{'  ' * depth}{label(cid)}")
            walk(cid, depth + 1)

    walk(root, 0)
    return "\n".join(lines) if lines else "(empty)"
