"""Tier-1 metrics: HarmBench prompts + StrongREJECT."""

from __future__ import annotations

from statistics import mean
from typing import Any


def summarise_tier1(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scores = [r["strongreject_score"] for r in rows if r.get("strongreject_score") is not None]
    if not scores:
        return {
            "status": "unscored",
            "n_prompts": len(rows),
            "n_scored": 0,
            "asr": None,
            "mean_strongreject": None,
            "threshold": threshold,
        }
    hits = sum(1 for s in scores if s > threshold)
    refused_vals = [r.get("judge_refused") for r in rows if r.get("judge_refused") is not None]
    refusal_rate = (sum(refused_vals) / len(refused_vals)) if refused_vals else None
    by_cat: dict[str, list[float]] = {}
    for r in rows:
        s = r.get("strongreject_score")
        if s is None:
            continue
        cat = r.get("semantic_category") or "unknown"
        by_cat.setdefault(cat, []).append(s)

    return {
        "status": "ok",
        "n_prompts": len(rows),
        "n_scored": len(scores),
        "threshold": threshold,
        "asr": hits / len(scores),
        "mean_strongreject": mean(scores),
        "refusal_rate": refusal_rate,
        "n_refused_scored": len(refused_vals),
        "asr_by_semantic_category": {
            cat: sum(1 for s in vals if s > threshold) / len(vals) for cat, vals in by_cat.items()
        },
    }
