#!/usr/bin/env python3
"""Export a blinded human-annotation set for judge validation.

Samples 120 (prompt, response) pairs, stratified by lineage, generation stage
(early vs late), and DeepSeek score bin {0, (0,0.5], >0.5}. Writes two identical
blinded CSVs (annotate_A.csv, annotate_B.csv) with empty annotation columns and
a hidden key.csv mapping item ids back to model / generation / scores.

Run after P1 and P2 finish:
  python scripts/export_human_eval.py --lineages gemma3_4b_self_collapse gemma2_2b_kazdan
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from _bootstrap import RESEARCH_ROOT

COLLAPSE_DIR = RESEARCH_ROOT / "results" / "collapse_eval"
DEFAULT_LINEAGES = ["gemma3_4b_self_collapse", "gemma2_2b_kazdan"]
N_TOTAL = 120
SEED = 0


def score_bin(score: float | None) -> str:
    if score is None:
        return "unscored"
    if score <= 0.0:
        return "zero"
    if score <= 0.5:
        return "low"
    return "high"


def _gen_index(gen_key: str) -> float:
    """Numeric order for 'base', 'iter1', 'replace_iter5', '12', ..."""
    if gen_key == "base":
        return -1.0
    digits = "".join(c for c in gen_key if c.isdigit())
    return float(digits) if digits else 0.0


def collect_rows(lineages: list[str]) -> list[dict]:
    rows = []
    for lineage in lineages:
        ldir = COLLAPSE_DIR / lineage
        if not ldir.is_dir():
            print(f"  (skip {lineage}: no results dir)")
            continue
        gen_dirs = sorted(
            [d for d in ldir.iterdir() if d.is_dir() and (d / "scores.jsonl").is_file()],
            key=lambda d: _gen_index(d.name),
        )
        if not gen_dirs:
            continue
        idxs = [_gen_index(d.name) for d in gen_dirs]
        median = sorted(idxs)[len(idxs) // 2]
        for d in gen_dirs:
            stage = "early" if _gen_index(d.name) <= median else "late"
            for line in (d / "scores.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                s = r.get("strongreject_score")
                rows.append(
                    {
                        "lineage": lineage,
                        "gen_key": d.name,
                        "stage": stage,
                        "behavior_id": r.get("behavior_id"),
                        "behavior": r.get("behavior"),
                        "completion": r.get("completion"),
                        "deepseek_score": s,
                        "deepseek_refused": r.get("judge_refused"),
                        "deepseek_specific": r.get("judge_specific"),
                        "deepseek_convincing": r.get("judge_convincing"),
                        "ft_judge_score": r.get("ft_strongreject_score"),
                        "score_bin": score_bin(s),
                    }
                )
    return rows


def stratified_sample(rows: list[dict], n_total: int, rng: random.Random) -> list[dict]:
    strata: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["deepseek_score"] is None:
            continue
        key = (r["lineage"], r["stage"], r["score_bin"])
        strata.setdefault(key, []).append(r)
    if not strata:
        return []
    keys = sorted(strata)
    per = max(1, n_total // len(keys))
    picked: list[dict] = []
    for key in keys:
        pool = strata[key]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    # Top up / trim to exactly n_total.
    if len(picked) < n_total:
        remaining = [r for r in rows if r not in picked and r["deepseek_score"] is not None]
        rng.shuffle(remaining)
        picked.extend(remaining[: n_total - len(picked)])
    rng.shuffle(picked)
    return picked[:n_total]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lineages", nargs="+", default=DEFAULT_LINEAGES)
    p.add_argument("--n", type=int, default=N_TOTAL)
    p.add_argument("--out-dir", default=str(RESEARCH_ROOT / "human_eval"))
    args = p.parse_args()

    rng = random.Random(SEED)
    rows = collect_rows(args.lineages)
    print(f"collected {len(rows)} scored pairs from {args.lineages}")
    sample = stratified_sample(rows, args.n, rng)
    if not sample:
        raise SystemExit("No scored rows found; run P1/P2 security eval first.")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    blind_cols = ["item_id", "behavior", "response",
                  "refused (0/1)", "convincing (1-5)", "specific (1-5)"]
    key_cols = ["item_id", "lineage", "gen_key", "stage", "behavior_id",
                "deepseek_score", "deepseek_refused", "deepseek_specific",
                "deepseek_convincing", "ft_judge_score", "score_bin"]

    for name in ("annotate_A.csv", "annotate_B.csv"):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(blind_cols)
            for i, r in enumerate(sample):
                w.writerow([f"item_{i:03d}", r["behavior"], r["completion"], "", "", ""])

    with (out / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=key_cols)
        w.writeheader()
        for i, r in enumerate(sample):
            w.writerow({"item_id": f"item_{i:03d}", **{k: r.get(k) for k in key_cols[1:]}})

    from collections import Counter
    strat = Counter((r["lineage"], r["stage"], r["score_bin"]) for r in sample)
    print(f"wrote {len(sample)} items to {out}/ (annotate_A.csv, annotate_B.csv, key.csv)")
    for k, v in sorted(strat.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
