"""Pretty-print one (or all) failure dossiers for the case-by-case walkthrough.
Usage: python view_case.py <behavior_id|index>   |   python view_case.py --list
"""
import json
import sys
import textwrap
from pathlib import Path

D = json.load(open(Path(__file__).parent / "failure_dossiers.json", encoding="utf-8"))


def wrap(s, w=100, indent="      "):
    return textwrap.fill(" ".join((s or "").split()), w, initial_indent=indent, subsequent_indent=indent)


def show(d):
    print("=" * 100)
    print(f"{d['behavior_id']}   [{d['semantic_category']}]   q#{d['query_index']}")
    print(f"GOAL: {d['behavior']}")
    print(f"PRELIM: {d['preliminary_label']}  |  synth_gap={int(d['policy_synth_gap'])}  |  final={d['final_label']}")
    print("-" * 100)
    print("SED JUDGE (own verdict on this response):")
    print(f"  outcome(harmful)={d.get('sed_outcome')}  failure_mode={d.get('failure_mode')}")
    if d.get("requested_harmful_action"):
        print("  requested:", " ".join(d["requested_harmful_action"].split())[:300])
    if d.get("complied_action"):
        print("  complied :", " ".join(d["complied_action"].split())[:300])
    print(f"\nPOLICY SYNTHESIZED FROM THIS EPISODE: {d['policy_synthesized_from_this_episode'] or 'NONE'}")
    print(f"SAME-CATEGORY POLICIES EXISTING BEFORE: {d['same_category_policies_existing_before'] or 'none'}")
    print("\nPOLICIES INJECTED AT ATTACK TIME:")
    for p in d["policies_injected_at_attack_time"]:
        if p.get("MISSING_FROM_L2"):
            print(f"  - {p['id']} [MISSING FROM L2]"); continue
        print(f"  - {p['name']}  (learned_from={p['learned_from_categories']}; "
              f"retrieved {p['times_retrieved']}x, succ_rate={p['success_rate']})")
        print(wrap(p["description"], indent="        "))
    print("\nCANONICAL JUDGE (why it's a success):")
    print(wrap(d["canonical_judge_reason"]))
    print("\nRESPONSE (first 700 chars):")
    print(wrap(d["response"][:700]))
    print()


if len(sys.argv) > 1 and sys.argv[1] == "--list":
    for i, d in enumerate(D):
        print(f"{i:2d}  [{d['preliminary_label']:18s}] {d['semantic_category']:22s} {d['behavior_id']}")
elif len(sys.argv) > 1:
    arg = sys.argv[1]
    hits = [d for d in D if d["behavior_id"] == arg] or ([D[int(arg)]] if arg.isdigit() else [])
    for d in hits:
        show(d)
else:
    for d in D:
        show(d)
