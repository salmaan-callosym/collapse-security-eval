#!/usr/bin/env python3
"""Figures, statistics, and tables for the collapse-security study.

Reads results/collapse_eval/<lineage>/<gen>/ and writes:
  figures/   main_<lineage>.png, comparison_security.png, judge_agreement.png
  tables/    per_checkpoint.{csv,tex}, spearman.{csv,tex}, bootstrap.{csv,tex},
             mcnemar.{csv,tex}, agreement.{csv,tex}, qualitative_P1.{csv,tex}

Only produces outputs for lineages that have results, so it is safe to run
after any subset of the study has completed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from _bootstrap import RESEARCH_ROOT  # noqa: E402

from src.eval.analysis import (  # noqa: E402
    LINEAGE_META,
    bootstrap_ci,
    gen_index,
    load_all,
    load_lineage,
    load_scores,
    mcnemar_refusal,
    spearman_gen_vs,
)

COLLAPSE_DIR = RESEARCH_ROOT / "results" / "collapse_eval"
FIG_DIR = COLLAPSE_DIR / "figures"
TAB_DIR = COLLAPSE_DIR / "tables"


def _save_table(df: pd.DataFrame, stem: str) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False)
    try:
        (TAB_DIR / f"{stem}.tex").write_text(
            df.to_latex(index=False, float_format="%.3f", na_rep="--")
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  (latex for {stem} skipped: {exc})")
    print(f"  wrote tables/{stem}.csv + .tex")


def fig_main(lineage: str, records: list[dict]) -> None:
    pts = [r for r in records if r["gen_key"] != "base"]
    base = next((r for r in records if r["gen_key"] == "base"), None)
    if not pts:
        return
    pts.sort(key=lambda r: r["gen_index"])
    x = [r["gen_index"] for r in pts]

    fig, (ax_u, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5))
    fam = LINEAGE_META.get(lineage, {}).get("family", lineage)
    fig.suptitle(f"{fam} — {lineage}", fontsize=12)

    # Utility panel: PPL (left axis) and distinct-2 (right axis).
    ppl = [r.get("perplexity") for r in pts]
    ax_u.plot(x, ppl, "o-", color="tab:red", label="perplexity")
    ax_u.set_xlabel("generation")
    ax_u.set_ylabel("WikiText-2 PPL", color="tab:red")
    ax_u.set_title("Utility (collapse)")
    axu2 = ax_u.twinx()
    d2 = [r.get("distinct_2") for r in pts]
    axu2.plot(x, d2, "s--", color="tab:blue", label="distinct-2")
    axu2.set_ylabel("distinct-2", color="tab:blue")
    if base and base.get("perplexity") is not None:
        ax_u.axhline(base["perplexity"], color="tab:red", ls=":", alpha=0.6)
    if base and base.get("distinct_2") is not None:
        axu2.axhline(base["distinct_2"], color="tab:blue", ls=":", alpha=0.6)

    # Security panel: ASR, mean StrongREJECT, refusal rate.
    for field, style, label in [
        ("asr", "o-", "ASR (>0.5)"),
        ("mean_strongreject", "s-", "mean StrongREJECT"),
        ("refusal_rate", "^-", "refusal rate"),
    ]:
        ys = [r.get(field) for r in pts]
        if any(v is not None for v in ys):
            ax_s.plot(x, ys, style, label=label)
            if base and base.get(field) is not None:
                ax_s.axhline(base[field], ls=":", alpha=0.5)
    ax_s.set_xlabel("generation")
    ax_s.set_ylabel("rate / score")
    ax_s.set_ylim(-0.02, 1.02)
    ax_s.set_title("Security")
    ax_s.legend(fontsize=8)
    ax_u.legend(loc="upper left", fontsize=8)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"main_{lineage}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote figures/main_{lineage}.png")


def fig_comparison(data: dict[str, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for lineage, records in data.items():
        meta = LINEAGE_META.get(lineage, {})
        pts = [r for r in records if r["gen_key"] != "base" and r.get("asr") is not None]
        if not pts:
            continue
        if lineage == "gemma2_2b_kazdan":
            for arm in ("replace", "accumulate"):
                arm_pts = sorted([r for r in pts if r["arm"] == arm or r["gen_key"] == "iter1"],
                                 key=lambda r: r["gen_index"])
                if len(arm_pts) < 2:
                    continue
                xs = _normalise([r["gen_index"] for r in arm_pts])
                ax.plot(xs, [r["asr"] for r in arm_pts], "s--",
                        label=f"{meta['family']} ({arm}, unaligned)")
                plotted = True
        else:
            pts.sort(key=lambda r: r["gen_index"])
            xs = _normalise([r["gen_index"] for r in pts])
            tag = "aligned" if meta.get("aligned") else "unaligned"
            ax.plot(xs, [r["asr"] for r in pts], "o-",
                    label=f"{meta['family']} ({tag})")
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("normalised generation index (0=first, 1=last)")
    ax.set_ylabel("ASR (StrongREJECT > 0.5)")
    ax.set_title("Security vs collapse: aligned vs unaligned")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_security.png", dpi=140)
    plt.close(fig)
    print("  wrote figures/comparison_security.png")


def _normalise(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def fig_judge_agreement(data: dict[str, list[dict]]) -> pd.DataFrame | None:
    ds_scores, ft_scores = [], []
    for lineage in data:
        ldir = COLLAPSE_DIR / lineage
        for d in ldir.iterdir() if ldir.is_dir() else []:
            for r in load_scores(d):
                s = r.get("strongreject_score")
                ft = r.get("ft_strongreject_score")
                if s is not None and ft is not None:
                    ds_scores.append(s)
                    ft_scores.append(ft)
    if not ds_scores:
        print("  (no paired DeepSeek/fine-tuned scores yet; skipping agreement scatter)")
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(ds_scores, ft_scores, alpha=0.4, s=12)
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("DeepSeek StrongREJECT")
    ax.set_ylabel("fine-tuned StrongREJECT")
    ax.set_title("Judge agreement (all completions)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "judge_agreement.png", dpi=140)
    plt.close(fig)
    print("  wrote figures/judge_agreement.png")
    from scipy.stats import spearmanr

    rho, pv = spearmanr(ds_scores, ft_scores)
    return pd.DataFrame([{"n": len(ds_scores), "spearman_rho": rho, "pval": pv}])


def table_per_checkpoint(data: dict[str, list[dict]]) -> None:
    rows = []
    for lineage, records in data.items():
        for r in records:
            rows.append({
                "lineage": lineage,
                "gen": r["gen_key"],
                "asr": r.get("asr"),
                "mean_strongreject": r.get("mean_strongreject"),
                "refusal_rate": r.get("refusal_rate"),
                "perplexity": r.get("perplexity"),
                "distinct_2": r.get("distinct_2"),
                "mean_len": r.get("mean_response_len"),
            })
    if rows:
        _save_table(pd.DataFrame(rows), "per_checkpoint")


def table_spearman(data: dict[str, list[dict]]) -> None:
    rows = []
    for lineage, records in data.items():
        for field in ("asr", "mean_strongreject"):
            st = spearman_gen_vs(records, field)
            rows.append({"lineage": lineage, "metric": field,
                         "spearman_rho": st["rho"], "pval": st["pval"], "n": st["n"]})
    if rows:
        _save_table(pd.DataFrame(rows), "spearman")


def table_bootstrap(data: dict[str, list[dict]], n_boot: int) -> None:
    rows = []
    for lineage in data:
        ldir = COLLAPSE_DIR / lineage
        for d in sorted(ldir.iterdir(), key=lambda x: gen_index(x.name)) if ldir.is_dir() else []:
            scores = [r.get("strongreject_score") for r in load_scores(d)]
            scores = [s for s in scores if s is not None]
            if not scores:
                continue
            ci = bootstrap_ci(scores, n_boot=n_boot)
            rows.append({
                "lineage": lineage, "gen": d.name, "n": ci["n"],
                "asr": ci["asr"], "asr_lo": ci["asr_ci"][0], "asr_hi": ci["asr_ci"][1],
                "mean": ci["mean"], "mean_lo": ci["mean_ci"][0], "mean_hi": ci["mean_ci"][1],
            })
    if rows:
        _save_table(pd.DataFrame(rows), "bootstrap")


def table_mcnemar(data: dict[str, list[dict]]) -> None:
    rows = []
    for lineage, records in data.items():
        ldir = COLLAPSE_DIR / lineage
        gens = sorted([r["gen_key"] for r in records], key=gen_index)
        base_key = "base" if "base" in gens else (gens[0] if gens else None)
        final_key = gens[-1] if gens else None
        if not base_key or not final_key or base_key == final_key:
            continue
        base_rows = load_scores(ldir / base_key)
        final_rows = load_scores(ldir / final_key)
        if not base_rows or not final_rows:
            continue
        m = mcnemar_refusal(base_rows, final_rows)
        rows.append({"lineage": lineage, "base": base_key, "final": final_key, **m})
    if rows:
        _save_table(pd.DataFrame(rows), "mcnemar")


def table_qualitative_p1() -> None:
    lineage = "gemma3_4b_self_collapse"
    ldir = COLLAPSE_DIR / lineage
    if not ldir.is_dir():
        return
    present = sorted([d.name for d in ldir.iterdir() if d.is_dir()], key=gen_index)
    if not present:
        return
    # base/first, middle, final
    picks = [present[0], present[len(present) // 2], present[-1]]
    by_gen = {g: {r["behavior_id"]: r for r in load_scores(ldir / g)} for g in picks}
    first_ids = list(by_gen[picks[0]].keys())[:3]
    rows = []
    for bid in first_ids:
        for g in picks:
            r = by_gen.get(g, {}).get(bid, {})
            rows.append({
                "behavior_id": bid,
                "gen": g,
                "strongreject": r.get("strongreject_score"),
                "response": (r.get("completion") or "")[:300],
            })
    if rows:
        _save_table(pd.DataFrame(rows), "qualitative_P1")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    data = load_all(COLLAPSE_DIR)
    data = {k: v for k, v in data.items() if v}
    if not data:
        raise SystemExit(f"No results under {COLLAPSE_DIR}. Run run_collapse_eval.py first.")
    print(f"lineages with results: {list(data)}")

    print("figures:")
    for lineage, records in data.items():
        fig_main(lineage, records)
    fig_comparison(data)
    agree = fig_judge_agreement(data)

    print("tables:")
    table_per_checkpoint(data)
    table_spearman(data)
    table_bootstrap(data, args.n_boot)
    table_mcnemar(data)
    table_qualitative_p1()
    if agree is not None:
        _save_table(agree, "agreement")
    print("done.")


if __name__ == "__main__":
    main()
