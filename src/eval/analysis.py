"""Load collapse-eval artefacts and compute study statistics.

Pure-data helpers (no plotting) so they are unit-testable and reusable by both
make_figures.py and the results summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LINEAGE_META = {
    "gemma3_4b_self_collapse": {"priority": "P1", "aligned": True, "family": "Gemma-3-4B"},
    "gemma2_2b_kazdan": {"priority": "P2", "aligned": False, "family": "Gemma-2-2B"},
    "qwen25_7b_self_collapse": {"priority": "P3", "aligned": True, "family": "Qwen2.5-7B"},
    "opt125m": {"priority": "P4", "aligned": False, "family": "OPT-125m"},
}


def gen_index(gen_key: str) -> float:
    """Numeric order for 'base', 'iter1', 'replace_iter5', '12'."""
    if gen_key == "base":
        return -1.0
    digits = "".join(c for c in gen_key if c.isdigit())
    return float(digits) if digits else 0.0


def arm_of(gen_key: str) -> str:
    if gen_key.startswith("replace"):
        return "replace"
    if gen_key.startswith("accumulate"):
        return "accumulate"
    return "main"


def load_checkpoint_dir(d: Path) -> dict[str, Any] | None:
    summ = d / "summary.json"
    util = d / "utility.json"
    rec: dict[str, Any] = {"gen_key": d.name, "gen_index": gen_index(d.name),
                           "arm": arm_of(d.name)}
    if summ.is_file():
        s = json.loads(summ.read_text())
        t = s.get("tier1", {})
        rec.update(
            {
                "asr": t.get("asr"),
                "mean_strongreject": t.get("mean_strongreject"),
                "refusal_rate": t.get("refusal_rate"),
                "n_scored": t.get("n_scored"),
                "subject_model": s.get("subject_model"),
                "commit_sha": s.get("commit_sha"),
            }
        )
    if util.is_file():
        u = json.loads(util.read_text())
        rec["perplexity"] = u.get("perplexity", {}).get("perplexity")
        rec["distinct_1"] = u.get("diversity", {}).get("distinct_1")
        rec["distinct_2"] = u.get("diversity", {}).get("distinct_2")
        rec["mean_response_len"] = u.get("diversity", {}).get("mean_response_len_words")
    if "asr" not in rec and "perplexity" not in rec:
        return None
    return rec


def load_scores(d: Path) -> list[dict]:
    p = d / "scores.jsonl"
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def load_lineage(collapse_dir: Path, lineage: str) -> list[dict]:
    ldir = collapse_dir / lineage
    if not ldir.is_dir():
        return []
    out = []
    for d in sorted(ldir.iterdir(), key=lambda x: gen_index(x.name)):
        if not d.is_dir():
            continue
        rec = load_checkpoint_dir(d)
        if rec:
            rec["lineage"] = lineage
            out.append(rec)
    return out


def load_all(collapse_dir: Path) -> dict[str, list[dict]]:
    return {ln: load_lineage(collapse_dir, ln) for ln in LINEAGE_META
            if (collapse_dir / ln).is_dir()}


# ---- statistics -----------------------------------------------------------

def spearman_gen_vs(records: list[dict], field: str):
    from scipy.stats import spearmanr

    pts = [(r["gen_index"], r[field]) for r in records
           if r.get(field) is not None and r["gen_key"] != "base"]
    if len(pts) < 3:
        return {"rho": None, "pval": None, "n": len(pts)}
    xs, ys = zip(*pts)
    rho, pv = spearmanr(xs, ys)
    return {"rho": float(rho), "pval": float(pv), "n": len(pts)}


def bootstrap_ci(scores: list[float], threshold: float = 0.5,
                 n_boot: int = 1000, seed: int = 0):
    """95% CI over prompts for ASR and mean StrongREJECT."""
    arr = np.array([s for s in scores if s is not None], dtype=float)
    if arr.size == 0:
        return {"asr": None, "asr_ci": None, "mean": None, "mean_ci": None, "n": 0}
    rng = np.random.default_rng(seed)
    n = arr.size
    asr_bs, mean_bs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = arr[idx]
        asr_bs.append(float((sample > threshold).mean()))
        mean_bs.append(float(sample.mean()))
    return {
        "asr": float((arr > threshold).mean()),
        "asr_ci": [float(np.percentile(asr_bs, 2.5)), float(np.percentile(asr_bs, 97.5))],
        "mean": float(arr.mean()),
        "mean_ci": [float(np.percentile(mean_bs, 2.5)), float(np.percentile(mean_bs, 97.5))],
        "n": int(n),
    }


def mcnemar_refusal(base_rows: list[dict], final_rows: list[dict]):
    """McNemar test on refusal (paired by behavior_id) base vs final gen."""
    from scipy.stats import chi2

    base = {r["behavior_id"]: r.get("judge_refused") for r in base_rows}
    final = {r["behavior_id"]: r.get("judge_refused") for r in final_rows}
    common = [b for b in base if b in final
              and base[b] is not None and final[b] is not None]
    # b = refused at base but not final (safety lost); c = opposite.
    b = sum(1 for k in common if base[k] == 1 and final[k] == 0)
    c = sum(1 for k in common if base[k] == 0 and final[k] == 1)
    n_disc = b + c
    if n_disc == 0:
        return {"n_pairs": len(common), "b_lost": b, "c_gained": c,
                "statistic": None, "pval": None}
    stat = (abs(b - c) - 1) ** 2 / n_disc  # continuity-corrected
    pval = float(chi2.sf(stat, 1))
    return {"n_pairs": len(common), "b_lost_refusal": b, "c_gained_refusal": c,
            "statistic": float(stat), "pval": pval}
