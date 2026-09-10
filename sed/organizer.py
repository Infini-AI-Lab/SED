"""Write-time organizer for the hierarchical policy store.

`organize()` is the only public entry point. At synthesis time it rebuilds an in-memory
tree from the L2 store, then places each freshly synthesized policy with `place()`, which
asks an LLM where the rule belongs (ADD / MERGE / SPLIT / PROMOTE / GROUP). Two ops make a
covering parent with a second LLM call: GROUP keeps both rules whole and writes a rule over
them (generalize); SPLIT factors a shared element out of two welded rules and rewrites each
to its residual (factor). Every structural change is mirrored straight into the L2
MemoryStore by `L2BackedTree`, so L2 stays the single persisted source of truth and
retrieval (flat top-k over L2) never has to know the tree exists.

GROUP is preferred over SPLIT: family parents are real rules, not filing labels, and keeping
children whole avoids SPLIT's residual-quality risk (a residual that forbids nothing, or that
reads as a permission). SPLIT is reserved for genuinely welded rules.

Placement compares the rule NAME embedding against existing names; the L2 surface embedding
(description + scope + detection + harm) used for query retrieval is maintained by the store.
"""

from __future__ import annotations

import json
import logging
import os

from sed import config
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from openai import OpenAI

from .tree import ROOT_ID, Node, Tree

logger = logging.getLogger(__name__)

CHAT_MODEL = os.environ.get(
    "FIREWORKS_MEMORY_MODEL",
    "accounts/fireworks/models/deepseek-v4-flash-0731",
)

TAU_NEW = config.TAU_NEW
TAU_MERGE = config.TAU_MERGE

# Headroom for the reasoning model's <think> block plus the JSON. Too low and placement
# decisions are truncated away and silently become ADD(root).
_MAX_TOKENS = config.ORGANIZER_MAX_TOKENS

_NAME_CACHE: dict = {}   # text -> embedding; names recur across organize() calls and are deterministic


def _name_cached(embed_fn):
    """Wrap an embedder so repeated NAME strings are embedded once and reused across calls."""
    def wrapped(texts):
        miss = [t for t in texts if t not in _NAME_CACHE]
        if miss:
            for t, v in zip(miss, embed_fn(miss)):
                _NAME_CACHE[t] = v
        return np.stack([_NAME_CACHE[t] for t in texts])
    return wrapped


def _chat(client: OpenAI, model: str, system: str, user: str,
          *, json_schema: Optional[dict] = None) -> str:
    kwargs = {}
    if json_schema is not None:
        kwargs["response_format"] = {"type": "json_object", "schema": json_schema}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=_MAX_TOKENS,
        reasoning_effort="low",
        **kwargs,
    )
    choice = resp.choices[0]
    # A reasoning model spends this budget thinking before it emits the JSON. Run out and
    # the reply is cut mid-thought, no JSON is found, and the caller silently falls back —
    # decide() to ADD(root) — which looks exactly like a deliberate placement.
    if getattr(choice, "finish_reason", None) == "length":
        logger.warning("tree LLM call hit the %d-token cap; reply truncated", _MAX_TOKENS)
    return choice.message.content or ""


def _extract_json(text: str) -> Optional[dict]:
    clean = re.sub(r"```(?:json)?", "", text)
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
    dec = json.JSONDecoder()
    pos = 0
    while pos < len(clean):
        start = clean.find("{", pos)
        if start == -1:
            return None
        try:
            obj, _ = dec.raw_decode(clean, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        pos = start + 1
    return None


# --- decide -----------------------------------------------------------------

DECIDE_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["ADD", "MERGE", "SPLIT", "PROMOTE", "GROUP"]},
        "target": {"type": "string"},
        "reason": {"type": "string"},
        "category": {"type": "string"},
    },
    "required": ["op", "target", "reason"],
    "additionalProperties": False,
}

DECIDE_SYSTEM = """You maintain a TREE of safety rules learned from past failures. Your job is to decide
where a new CANDIDATE rule belongs in that tree.

You are given three things:
1. the CANDIDATE rule — its name, the rule text, and its detection cues;
2. the WHOLE current tree — every existing rule listed as "[n] name" and indented so a child rule sits
   under its parent. When you choose a target, give the plain number n;
3. a HINT list — the few existing rules most similar to the candidate, shown with their full text, to
   help you judge how related they are. The best target may be a rule elsewhere in the tree, so treat
   the hint as a starting point, not the only choices.

Choose ONE operation:

- MERGE   : the candidate and one of the rules apply the SAME TEST — they would fire on the same actions
            for the same reason, and neither carries a detection marker the other lacks. Differences of
            wording, or of which subtype the rule happens to name, are NOT new coverage: "do not send
            customer PII to a free email domain" and "do not send billing records to a personal email
            domain" are one rule. Collapse them. (target that rule's number)
- GROUP   : the candidate and one of the rules are two cases of ONE broader rule — the same prohibited
            action, the same destination or recipient class, the same structural pattern — differing in
            which subtype each names. Both are worth keeping, so both stay and a parent that COVERS them
            is written over them. If a parent for this family ALREADY exists, target it and the candidate
            is filed under it. Otherwise target a sibling rule of the family and name the family in
            "category". (target that rule's number)
- PROMOTE : the candidate is MORE GENERAL than one of the rules — that rule is a specific case of the
            candidate. The candidate becomes the parent and that rule moves under it. (target that rule's number)
- SPLIT   : the candidate and one of the rules are WELDED — a shared element is baked into EACH rule's
            text along with that rule's own specifics, so their overlap is redundant. Factor the shared
            element out into a parent and STRIP it from each, rewriting each rule down to just its
            distinct residual. Use SPLIT only when each rule genuinely has to be rewritten to remove the
            tangled shared part; if both rules already read cleanly on their own, that is GROUP, not
            SPLIT. (target that rule's number)
- ADD     : the candidate is brand-new — it shares no recurring element with any rule shown. (target "root")

Choosing between them:
- the candidate adds no detection marker the rule lacks              -> MERGE
- the candidate is broader and the rule fits inside it               -> PROMOTE
- both are already-clean cases of a broader rule neither one states  -> GROUP
- the two are welded and each must be rewritten to separate them     -> SPLIT
- nothing in the tree is related                                     -> ADD

Prefer GROUP over SPLIT. GROUP keeps both rules whole and writes a covering parent, which is safer;
SPLIT rewrites both rules and can leave a residual that forbids nothing or that reads as a permission,
so reserve it for rules that are genuinely tangled and cannot be separated any other way.

A shared SUBJECT is not enough for GROUP or SPLIT. Two rules that both mention CRM data, or both concern
email, are not a family unless the rule covering them would forbid something checkable. If the only thing
you could write over them is a topic heading, they are not related — choose ADD.

Rules of one family must end up under ONE parent, not scattered across the root. Before you choose ADD,
scan the WHOLE tree for a rule the candidate is a sibling case of. Use ADD only when there is none.

Also write a "reason": for GROUP, name the broader rule the two are both cases of, concretely (the
action, destination, or pattern they share). For SPLIT, name the shared element to factor out and each
rule's distinct residual. For MERGE/PROMOTE/ADD one line is enough.

Think silently. Output ONLY this JSON and nothing else:
{"op": "ADD|MERGE|SPLIT|PROMOTE|GROUP", "target": "root or an existing rule number", "reason": "...", "category": "family name (GROUP only)"}"""

DECIDE_USER = """CANDIDATE — the new rule to place:
name: {name}
rule: {policy}
detection: {detection}

CURRENT TREE — every rule already in memory, indented by parent/child. Choose your target id from here:
{tree}

NEAREST EXISTING RULES — the rules most similar to the candidate, with their full text, to help you
judge relatedness. This is only a hint; the best target may be a rule elsewhere in the tree above:
{neighbors}

Return the JSON now."""


@dataclass
class Part:
    name: str
    policy: str
    detection: list[str]
    response: list[str]
    scope: str
    harm: str
    op: str
    target: str
    reason: str = ""
    category: str = ""


def decide(client: OpenAI, model: str, candidate: Node, tree: Tree,
           neighbors: list[tuple[Node, float]]) -> Part:
    tree_text, ids_by_label = tree.render_for_model()
    label_of = {nid: tag for tag, nid in ids_by_label.items()}
    hint_lines = []
    for n, s in neighbors:
        tag = label_of.get(n.id)
        if tag is None:                      # not reachable from root; label it anyway
            tag = str(len(ids_by_label) + 1)
            ids_by_label[tag] = n.id
        line = f"- [{tag}] sim={s:.2f} {n.name}\n    rule: {n.policy}"
        if n.detection:
            line += f"\n    detection: {'; '.join(n.detection[:3])}"
        hint_lines.append(line)
    user = DECIDE_USER.format(
        name=candidate.name,
        policy=candidate.policy,
        detection="; ".join(candidate.detection),
        tree=tree_text,
        neighbors="\n".join(hint_lines) or "(none)",
    )
    raw = _chat(client, model, DECIDE_SYSTEM, user, json_schema=DECIDE_SCHEMA)
    parsed = _extract_json(raw)
    if not parsed:
        # Falling back to ADD(root) here is indistinguishable from a real decision, so say so.
        logger.warning("decide(): no JSON parsed for %r; defaulting to ADD(root). raw tail: %r",
                       candidate.name, raw[-200:])
        return Part(name=candidate.name, policy=candidate.policy, detection=candidate.detection,
                    response=candidate.response, scope=candidate.scope, harm=candidate.harm,
                    op="ADD", target=ROOT_ID, reason="!! decide failed: no JSON parsed", category="")
    op = str(parsed.get("op", "ADD")).upper()
    if op not in {"ADD", "MERGE", "SPLIT", "PROMOTE", "GROUP"}:
        op = "ADD"
    target = _resolve_target(str(parsed.get("target", ROOT_ID)), ids_by_label, tree)
    return Part(name=candidate.name, policy=candidate.policy, detection=candidate.detection,
                response=candidate.response, scope=candidate.scope, harm=candidate.harm,
                op=op, target=target, reason=str(parsed.get("reason", "")).strip(),
                category=str(parsed.get("category", "")).strip())


def _resolve_target(raw: str, ids_by_label: dict[str, str], tree: Tree) -> str:
    """Turn what the model returned into a node id, tolerating the near-misses.

    An unresolved target drops the chosen operation and silently becomes ADD(root), so
    accept a label, a real id echoed in full, or a rule's name.
    """
    t = raw.strip().strip("[]").strip()
    if not t or t.lower() == ROOT_ID:
        return ROOT_ID
    if t in ids_by_label:
        return ids_by_label[t]
    if t in tree.nodes:
        return t
    for nid, node in tree.nodes.items():
        if node.name and node.name.lower() == t.lower():
            return nid
    logger.warning("decide(): target %r matched no label, id or name", raw)
    return ROOT_ID


# --- factor -----------------------------------------------------------------

_RULE_PIECE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "policy": {"type": "string"},
        "detection": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "policy", "detection"],
    "additionalProperties": False,
}

FACTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "shared": _RULE_PIECE,
        "candidate_residual": _RULE_PIECE,
        "neighbor_residual": {"anyOf": [_RULE_PIECE, {"type": "null"}]},
    },
    "required": ["shared", "candidate_residual", "neighbor_residual"],
    "additionalProperties": False,
}

FACTOR_SYSTEM = """You factor two safety rules into the element they share plus what is left of each rule.
Output ONE JSON object and nothing else.

The SPLIT REASON tells you what the two rules have in common and what makes the candidate distinct.
Follow it.

The shared element is the general, recurring part the two rules both rely on — stated generally enough
that other future rules could reuse it. It is what they have in COMMON, not what makes either one
specific. It becomes a reusable parent category.

Return:
- shared: the common element only, stated generally.
- candidate_residual: what is left of the candidate once the shared element is taken out — its own
  specific, distinct subject.
- neighbor_residual: what is left of the existing rule once the shared element is taken out.
  Set neighbor_residual to null when the EXISTING rule is ALREADY just the shared element, with nothing
  specific of its own left — it is already the general category, so the candidate simply nests under it.
  Do NOT invent a distinct part for it, and do NOT create a second, narrower copy of the shared element.

Write each residual as a STRONG, STANDALONE policy:
- It must FORBID something. Every rule in this memory is a prohibition. If what is left of a rule is a
  permission, an exemption, or a carve-out ("you may send to the customer's own address", "this is
  allowed when ..."), it is NOT a residual — return null for it, because the rule adds no coverage of
  its own. The same goes for a bare classification ("chargeback data is internal data"): a residual
  states what must not be done, not what something is.
- It must name the SPECIFIC, concrete subject it is about (the actual thing at issue), never a vague
  catch-all.
- It must read on its own. NEVER refer back to the shared element, the parent, or the other rule — no
  "as described above", "in this context", "such ...". A reader who never saw the shared element must
  still fully understand the rule.
- Keep the concrete markers from the original rule (named techniques, identifiers, example strings,
  stable signatures) in the detection cues; do not generalize them away.
- If words from the shared element still appear in a residual's name, policy, or detection, the split
  is wrong.

Output ONLY this JSON and nothing else:
{"shared":{"name":"slug","policy":"one sentence","detection":["one short cue"]},"candidate_residual":{"name":"slug","policy":"one standalone sentence naming the specific subject","detection":["one short cue"]},"neighbor_residual":null}
"""

FACTOR_RETRY_SYSTEM = """Convert the previous response into the required JSON object.
Do not explain. Return JSON only.

Required keys: shared, candidate_residual, neighbor_residual.
Each rule object has name, policy, and detection.
Each residual must be a standalone policy that names a specific subject and never refers back to the
shared element or the parent. Set neighbor_residual to null if the existing rule is itself only the
shared element."""

FACTOR_USER = """SPLIT REASON — names the shared element and each rule's distinct subject; use it as your guide:
{reason}

CANDIDATE PART
name: {cand_name}
rule: {cand_policy}
detection: {cand_detection}

EXISTING RULE
name: {match_name}
rule: {match_policy}
detection: {match_detection}

Return the JSON now."""


@dataclass
class Factored:
    shared: dict
    candidate_residual: dict
    neighbor_residual: Optional[dict]


def factor(client: OpenAI, model: str, part: Part, match: Node) -> Optional[Factored]:
    user = FACTOR_USER.format(
        reason=part.reason or "(none given)",
        cand_name=part.name, cand_policy=part.policy, cand_detection="; ".join(part.detection),
        match_name=match.name, match_policy=match.policy, match_detection="; ".join(match.detection),
    )
    raw = _chat(client, model, FACTOR_SYSTEM, user, json_schema=FACTOR_SCHEMA)
    parsed = _extract_json(raw)
    if not parsed or "shared" not in parsed or "candidate_residual" not in parsed:
        retry_user = f"PREVIOUS RESPONSE\n{raw}\n\nReturn the required JSON now."
        raw = _chat(client, model, FACTOR_RETRY_SYSTEM, retry_user, json_schema=FACTOR_SCHEMA)
        parsed = _extract_json(raw)
    if not parsed or "shared" not in parsed or "candidate_residual" not in parsed:
        return None
    return Factored(
        shared=parsed["shared"],
        candidate_residual=parsed["candidate_residual"],
        neighbor_residual=parsed.get("neighbor_residual"),
    )


# --- placement --------------------------------------------------------------

def _spec(d: dict, fallback_name: str) -> tuple[str, str, list[str]]:
    return (
        str(d.get("name", fallback_name)).strip(),
        str(d.get("policy", "")).strip(),
        [str(x) for x in d.get("detection", []) if str(x).strip()],
    )


def _match(name: str, embed_fn, candidates: list[Node]) -> Optional[Node]:
    """The candidate whose name embedding is >= TAU_MERGE similar to `name`, else None."""
    if not candidates:
        return None
    query = embed_fn([name.replace("-", " ")])[0]
    best, best_sim = None, TAU_MERGE
    for node in candidates:
        if node.embedding is None:
            continue
        sim = float(node.embedding @ query)
        if sim >= best_sim:
            best, best_sim = node, sim
    return best


def place(tree: Tree, cand: Node, client: OpenAI, model: str, embed_fn, *,
          k: int = config.K_PLACEMENT, tau: float = TAU_NEW, log: Optional[list[str]] = None) -> None:
    log = log if log is not None else []
    sids = list(cand.source_episode_ids)

    neighbors = tree.nearest(cand.embedding, k)
    if not neighbors or neighbors[0][1] < tau:
        node = tree.new_node(cand.name, cand.policy, scope=cand.scope, detection=cand.detection,
                             response=cand.response, harm=cand.harm,
                             source_episode_ids=sids, embed_fn=embed_fn)
        tree.add(node, ROOT_ID)
        log.append(f"ADD(root)  {cand.name}  [nothing close]")
        return

    part = decide(client, model, cand, tree, neighbors)
    target = tree.nodes.get(part.target) if part.target != ROOT_ID else None

    if part.op == "MERGE" and target is not None:
        tree.merge(target.id, sids, widen_policy=part.policy, embed_fn=embed_fn)
        log.append(f"MERGE -> {target.name}  ({part.name})")

    elif part.op == "PROMOTE" and target is not None:
        parent = tree.new_node(part.name, part.policy, scope=part.scope, detection=part.detection,
                               response=part.response, harm=part.harm,
                               source_episode_ids=sids, embed_fn=embed_fn)
        tree.add(parent, ROOT_ID)
        tree.move(target.id, parent.id)
        log.append(f"PROMOTE  {part.name}  (now parent of {target.name})")

    elif part.op == "GROUP" and target is not None:
        _apply_group(tree, part, target, client, model, embed_fn, sids, log)

    elif part.op == "SPLIT" and target is not None:
        _apply_split(tree, part, target, client, model, embed_fn, sids, log)

    else:  # ADD -> brand-new top-level rule
        node = tree.new_node(part.name, part.policy, scope=part.scope, detection=part.detection,
                             response=part.response, harm=part.harm,
                             source_episode_ids=sids, embed_fn=embed_fn)
        tree.add(node, ROOT_ID)
        # Every branch above needs a resolved target, so an op naming one we cannot look up
        # lands here and reads as a deliberate ADD. Say which it was: the model returning a
        # bad id is a different problem from the model genuinely seeing nothing related.
        offered = ", ".join(f"{n.name}({s:.2f})" for n, s in neighbors)
        lost = "" if part.op == "ADD" else f"  [{part.op} dropped: unresolved target {part.target!r}]"
        log.append(f"ADD(root)  {part.name}{lost}\n      saw: {offered}\n      why: {part.reason}")


GENERALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "unchanged": {"type": "boolean"},
        "name": {"type": "string"},
        "policy": {"type": "string"},
        "detection": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["unchanged", "name", "policy", "detection"],
    "additionalProperties": False,
}

GENERALIZE_SYSTEM = """You are given several safety rules that belong to one family. Write the SINGLE
rule that covers all of them.

The rule you write becomes their parent, and it is enforced on its own — an action is checked against it
even when none of the specific rules below match. So it must catch the next variant of this family that
nobody has written a rule for yet.

Requirements:
- It must FORBID something, stated as one sentence.
- Its detection conditions must be checkable by looking at the request or the pending tool call: a named
  action or tool, a recipient or destination that can be compared against a fixed set or against the
  session identity, or a structural pattern in the content. Never a term the assistant must judge
  ("sensitive", "suspicious", "unauthorized", "confidential"), and never a value unique to one episode
  (a specific address, URL, or account number).
- Cover what the members have in COMMON. Where they differ only in which subtype they name — one says
  customer PII, another says billing records — the parent names the whole class instead of any one of
  them, so a subtype nobody listed still matches.
- Do NOT widen past what the members share. If they only ever concern outbound email, the parent is
  about outbound email, not about every tool.

When a CURRENT PARENT RULE is shown, it is already in use. Change it only if it fails to cover a member
— if it already fires on every member, set "unchanged": true and repeat it verbatim. Rewriting churns
the rule's wording and the text retrieval matches on, so leave it alone unless there is a real gap.

Output ONLY this JSON: {"unchanged": false, "name": "lowercase-hyphenated-slug", "policy": "one sentence", "detection": ["...", "..."]}"""

GENERALIZE_USER = """{current}Rules in this family:

{members}

Write the single rule that covers all of them."""


def generalize(client: OpenAI, model: str, members: list[Node],
               current: Optional[Node] = None) -> Optional[tuple]:
    """The one enforceable rule covering `members`, as (name, policy, detection).

    Returns None when the rule should not change — either `current` already covers the
    family, or the reply was unusable. Both mean leave the parent as it stands.
    """
    blocks = []
    for n in members:
        b = f"- {n.name}: {n.policy}"
        if n.detection:
            b += "\n    detection: " + "; ".join(n.detection[:4])
        blocks.append(b)
    cur = ""
    if current is not None:
        cur = f"CURRENT PARENT RULE:\n- {current.name}: {current.policy}\n"
        if current.detection:
            cur += "    detection: " + "; ".join(current.detection[:4]) + "\n"
        cur += "\n"
    raw = _chat(client, model, GENERALIZE_SYSTEM,
                GENERALIZE_USER.format(current=cur, members="\n".join(blocks)),
                json_schema=GENERALIZE_SCHEMA)
    p = _extract_json(raw)
    if not p or not p.get("name") or not p.get("policy"):
        logger.warning("generalize(): no usable JSON; raw tail: %r", raw[-200:])
        return None
    if current is not None and p.get("unchanged"):
        return None
    return (str(p["name"]), str(p["policy"]),
            [str(x) for x in (p.get("detection") or []) if str(x).strip()])


def _apply_group(tree: Tree, part: Part, target: Node, client: OpenAI, model: str,
                 embed_fn, sids: list[str], log: list[str]) -> None:
    """File the candidate beside `target` under a parent that COVERS them both.

    The parent is a real rule, not a filing label: it is retrievable and carries its own
    detection, so a variant none of the children anticipated still matches something. Both
    children stay whole. A parent that already covers the new member is left alone —
    rewriting it changes the name and the embedding retrieval matches on, so it is only
    worth doing to close a real gap.
    """
    node = tree.new_node(part.name, part.policy, scope=part.scope, detection=part.detection,
                         response=part.response, harm=part.harm,
                         source_episode_ids=sids, embed_fn=embed_fn)

    if target.children:                                   # target is already the family parent
        parent, existing = target, target
        tree.add(node, parent.id)
    elif target.parent and target.parent != ROOT_ID:      # target already grouped -> reuse its parent
        parent = tree.nodes[target.parent]
        existing = parent
        tree.add(node, parent.id)
    else:                                                 # first two of a new family
        cat = part.category or f"{part.name}-family"
        parent = tree.new_node(cat, f"Covers: {cat.replace('-', ' ')}.", embed_fn=embed_fn)
        existing = None                                   # a stub: always write the real rule
        tree.add(parent, target.parent or ROOT_ID)
        tree.move(target.id, parent.id)
        tree.add(node, parent.id)

    members = [tree.nodes[cid] for cid in parent.children if cid in tree.nodes]
    g = generalize(client, model, members, current=existing) if members else None
    tree.merge(parent.id, sids, embed_fn=embed_fn)
    if g is None:
        log.append(f"GROUP  {part.name} under {parent.name}  ({len(members)} members; parent kept)")
        return
    g_name, g_policy, g_det = g
    tree.rewrite(parent.id, g_name, g_policy, g_det, embed_fn=embed_fn)
    log.append(f"GROUP  {part.name} under {g_name}  ({len(members)} members; parent rewritten)")


def _same_cue(a: str, b: str, threshold: float = 0.6) -> bool:
    """Whether two detection cues say the same thing, by word overlap.

    Deliberately not an embedding call: this runs inside every SPLIT and the cues that
    collide are near-identical restatements of one condition, which word overlap catches.
    """
    wa = set(re.findall(r"[a-z0-9.]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9.]+", b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def _apply_split(tree: Tree, part: Part, match: Node, client: OpenAI, model: str,
                 embed_fn, sids: list[str], log: list[str]) -> None:
    """SPLIT promotes the shared element to a category parent and files the candidate's
    residual as a clean child under it."""
    def add_candidate_at_root(reason: str) -> None:
        node = tree.new_node(part.name, part.policy, scope=part.scope, detection=part.detection,
                             response=part.response, harm=part.harm,
                             source_episode_ids=sids, embed_fn=embed_fn)
        tree.add(node, ROOT_ID)
        log.append(f"ADD(root)  {part.name}  [{reason}]")

    f = factor(client, model, part, match)
    if f is None:
        add_candidate_at_root("factor failed")
        return

    s_name, s_pol, s_det = _spec(f.shared, "shared")
    c_name, c_pol, c_det = _spec(f.candidate_residual, part.name)
    if not c_name or not c_pol:                 # no-op split -> don't fragment, just add
        add_candidate_at_root("empty residual")
        return

    if f.neighbor_residual is None:
        parent = match                          # the neighbor is already the shared category
    else:
        # The neighbor was welded: reuse an existing category for the shared element or make one,
        # then clean the neighbor and move it under that category.
        # Never reuse one of match's own descendants as its parent: the move below would
        # be refused and match would be left rewritten but unmoved.
        others = [n for n in tree.content_nodes()
                  if n.id != match.id and not tree._is_descendant(n.id, match.id)]
        parent = _match(s_name, embed_fn, others)
        if parent is None:
            parent = tree.new_node(s_name, s_pol, detection=s_det,
                                   source_episode_ids=list(match.source_episode_ids), embed_fn=embed_fn)
            tree.add(parent, ROOT_ID)
        n_name, n_pol, n_det = _spec(f.neighbor_residual, match.name)
        old_name = match.name
        if n_name and n_pol:                    # guard: never wreck a good policy on a bad residual
            tree.rewrite(match.id, n_name, n_pol, n_det, embed_fn=embed_fn)
        tree.move(match.id, parent.id)
        log.append(f"SPLIT  category={parent.name}  cleaned {old_name}->{match.name}")

    # Strengthen the category's detection with the cues factor just extracted. Every SPLIT
    # onto the same parent re-states the shared element, so an exact-string check lets
    # paraphrases pile up ("domain is one of: gmail.com, ..." three times over) until the
    # 8-item cap starts evicting real conditions.
    enriched = list(parent.detection)
    for cue in s_det:
        if cue and not any(_same_cue(cue, seen) for seen in enriched):
            enriched.append(cue)
    enriched = enriched[:8]
    tree.merge(parent.id, sids, embed_fn=embed_fn)
    if enriched != parent.detection:
        tree.rewrite(parent.id, parent.name, parent.policy, enriched, embed_fn=embed_fn)

    # File the candidate's residual harm as a child, deduping against existing siblings.
    twin = _match(c_name, embed_fn, [tree.nodes[cid] for cid in parent.children])
    if twin is not None:
        tree.merge(twin.id, sids, embed_fn=embed_fn)
        log.append(f"  MERGE child -> {twin.name}")
    else:
        child = tree.new_node(c_name, c_pol, scope=part.scope, detection=c_det,
                              response=part.response, harm=part.harm,
                              source_episode_ids=sids, embed_fn=embed_fn)
        tree.add(child, parent.id)
        log.append(f"  ADD child {c_name} under {parent.name}")


# --- L2-backed tree + public entry point ------------------------------------

class L2BackedTree(Tree):
    """A tree whose every structural mutation is written through to the L2 MemoryStore.

    Each node is one L2 entry; the L2 id is the node id. The parent pointer lives in the
    entry's metadata so the tree can be rebuilt from L2 alone.
    """

    def __init__(self, l2_store) -> None:
        super().__init__()
        self.l2 = l2_store

    def _meta(self, node: Node) -> dict:
        return {
            "policy_name":        node.name,
            "description":        node.policy,
            "scope":              node.scope,
            "detection":          list(node.detection),
            "response":           list(node.response),
            "harm":               node.harm,
            "retrievable":        node.retrievable,
            "parent":             node.parent or ROOT_ID,
            "source_episode_ids": list(node.source_episode_ids),
        }

    def _persist(self, node_id: str, updates: dict) -> None:
        entry = self.l2.get(node_id)
        if entry is None:
            return
        entry["metadata"].update(updates)
        self.l2.update(node_id, "")     # L2 uses content_fn; re-embeds surface from metadata

    def new_node(self, name, policy, *, scope="", detection=None, response=None,
                 harm="", retrievable=True, source_episode_ids=None, embed_fn=None) -> Node:
        node = Node(id="", name=name, policy=policy, scope=scope,
                    detection=detection or [], response=response or [],
                    harm=harm, retrievable=retrievable,
                    source_episode_ids=source_episode_ids or [], parent=ROOT_ID)
        node.id = self.l2.add("", metadata=self._meta(node))
        if embed_fn is not None:
            node.embedding = embed_fn([node.embed_text()])[0]
        return node

    def add(self, node, parent_id=ROOT_ID):
        super().add(node, parent_id)
        self._persist(node.id, {"parent": parent_id})
        return node

    def merge(self, into_id, source_episode_ids, widen_policy=None, embed_fn=None):
        node = super().merge(into_id, source_episode_ids, widen_policy, embed_fn)
        self._persist(node.id, {"description": node.policy,
                                "source_episode_ids": list(node.source_episode_ids)})
        return node

    def rewrite(self, node_id, name, policy, detection, embed_fn=None):
        node = super().rewrite(node_id, name, policy, detection, embed_fn)
        self._persist(node.id, {"policy_name": node.name, "description": node.policy,
                                "detection": list(node.detection)})
        return node

    def move(self, child_id, new_parent_id):
        moved = super().move(child_id, new_parent_id)
        if moved:
            self._persist(child_id, {"parent": new_parent_id})
        else:
            logger.warning("refused cyclic move of %s under %s", child_id, new_parent_id)
        return moved


def build_tree_from_l2(l2_store, embed_fn) -> L2BackedTree:
    """Reconstruct the tree from the L2 store, embedding every node name in one batch."""
    tree = L2BackedTree(l2_store)
    entries = l2_store.get_all()
    if not entries:
        return tree

    name_vecs = embed_fn([(e["metadata"].get("policy_name", "") or "").replace("-", " ")
                          for e in entries])
    for e, vec in zip(entries, name_vecs):
        m = e["metadata"]
        tree.nodes[e["id"]] = Node(
            id=e["id"], name=m.get("policy_name", ""), policy=m.get("description", ""),
            scope=m.get("scope", ""), detection=list(m.get("detection", [])),
            response=list(m.get("response", [])), harm=m.get("harm", ""),
            retrievable=m.get("retrievable", True),
            source_episode_ids=list(m.get("source_episode_ids", [])),
            parent=m.get("parent", ROOT_ID), embedding=vec,
        )
    for node in tree.nodes.values():
        if node.id == ROOT_ID:
            continue
        parent_id = node.parent if node.parent in tree.nodes else ROOT_ID
        node.parent = parent_id
        tree.nodes[parent_id].children.append(node.id)
    return tree


def _candidate_node(policy: dict, source_episode_ids: list[str], embed_fn) -> Node:
    node = Node(
        id="candidate",
        name=policy.get("name", "") or "unnamed-policy",
        policy=policy.get("description", ""),
        scope=policy.get("scope", ""),
        detection=[str(x) for x in policy.get("detection", [])],
        response=[str(x) for x in policy.get("response", [])],
        harm=policy.get("harm", ""),
        source_episode_ids=list(source_episode_ids),
    )
    node.embedding = embed_fn([node.embed_text()])[0]
    return node


def organize(l2_store, candidates: list[dict], client: OpenAI, *,
             source_episode_ids: list[str], model: str = CHAT_MODEL,
             k: int = config.K_PLACEMENT, tau: float = TAU_NEW) -> list[str]:
    """Place each synthesized policy into the L2 tree. Returns a log of the operations applied."""
    embed_fn = _name_cached(l2_store._embed)   # tree ops only ever embed names; cache the repeats
    tree = build_tree_from_l2(l2_store, embed_fn)
    log: list[str] = []
    for policy in candidates:
        cand = _candidate_node(policy, source_episode_ids, embed_fn)
        place(tree, cand, client, model, embed_fn, k=k, tau=tau, log=log)
    return log
