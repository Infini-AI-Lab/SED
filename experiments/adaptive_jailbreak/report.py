#!/usr/bin/env python3
"""Aggregate per-behavior JSONL from run.py into the headline metrics.

Reports, per (attack, defense), mean +/- std across seeds of:
ASR@{1,5,10,20,50}, overall ASR@budget, median queries-to-breach (solved only),
and the number of behaviors left unbroken.

  python experiments/adaptive_jailbreak/report.py --results experiments/adaptive_jailbreak/results
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

KS = (1, 5, 10, 20, 50)


def load(results_dir):
    runs = defaultdict(list)  # (attack, defense) -> list of per-seed behavior lists
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        if not rows:
            continue
        key = (rows[0]["attack"], rows[0]["defense"])
        runs[key].append(rows)
    return runs


def seed_metrics(rows):
    firsts = [r["first_success"] for r in rows]
    n = len(firsts)
    solved = [f for f in firsts if f is not None]
    m = {f"ASR@{k}": sum(1 for f in solved if f <= k) / n for k in KS}
    m["overall_ASR"] = len(solved) / n
    m["median_queries"] = st.median(solved) if solved else None
    m["unbroken"] = n - len(solved)
    return m


def agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    runs = load(args.results)
    cols = [f"ASR@{k}" for k in KS] + ["overall_ASR", "median_queries", "unbroken"]
    header = "| attack | defense | seeds | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 3)
    print(header)
    print(sep)

    summary = {}
    for (attack, defense), seed_runs in sorted(runs.items()):
        per_seed = [seed_metrics(rows) for rows in seed_runs]
        cells, record = [], {}
        for c in cols:
            mean, std = agg([m[c] for m in per_seed])
            if mean is None:
                cells.append("-")
            elif c.startswith("ASR") or c == "overall_ASR":
                cells.append(f"{mean:.0%}±{std:.0%}")
            else:
                cells.append(f"{mean:.1f}±{std:.1f}")
            record[c] = {"mean": mean, "std": std}
        print(f"| {attack} | {defense} | {len(per_seed)} | " + " | ".join(cells) + " |")
        summary[f"{attack}/{defense}"] = record

    out = Path(args.results) / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
