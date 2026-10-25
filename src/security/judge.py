"""Score completions with the official StrongREJECT evaluator.

If the package or its gated judge model is unavailable, we keep completions
and mark scores as pending — we do not invent a homemade scorer.
"""

from __future__ import annotations

from typing import Any, Callable


def score_strongreject(
    rows: list[dict[str, Any]],
    judge_name: str = "openrouter_rubric",
    batch_size: int = 8,
    openrouter_model: str = "openai/gpt-4o-mini",
    trace_meta: dict[str, Any] | None = None,
    on_scored: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add `strongreject_score` (0–1) to each row. Returns (rows, judge_meta)."""
    if judge_name.startswith("openrouter"):
        from .openrouter_judge import score_rows

        subject = (trace_meta or {}).get("subject_model", "facebook/opt-125m")
        return score_rows(
            rows,
            model=openrouter_model,
            trace_meta=trace_meta,
            subject_model=str(subject),
            on_scored=on_scored,
        )

    fn = _resolve_official_judge(judge_name)
    if fn is None:
        for row in rows:
            row["strongreject_score"] = None
        return rows, {
            "status": "package_missing",
            "judge": judge_name,
            "note": "pip install strong-reject; finetuned judge needs HF_TOKEN + Gemma access.",
        }

    errors = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = {
            "forbidden_prompt": [r["behavior"] for r in chunk],
            "response": [r["completion"] for r in chunk],
        }
        try:
            out = fn(batch)
            scores = list(out["score"])
        except Exception as exc:
            errors += len(chunk)
            for row in chunk:
                row["strongreject_score"] = None
                row["judge_error"] = str(exc)[:300]
            continue
        for row, score in zip(chunk, scores):
            row["strongreject_score"] = float(score)
            if on_scored is not None:
                on_scored(row)

    n = sum(1 for r in rows if r.get("strongreject_score") is not None)
    return rows, {
        "status": "ok" if n == len(rows) else ("partial" if n else "failed"),
        "judge": judge_name,
        "scored": n,
        "failed": errors,
    }


def score_finetuned(
    rows: list[dict[str, Any]], batch_size: int = 8
) -> dict[str, Any]:
    """Secondary judge: official fine-tuned StrongREJECT on the same completions.

    Stores `ft_strongreject_score` on each row (leaves DeepSeek scores untouched).
    If the package / gated Gemma judge is unavailable, records it and continues.
    """
    fn = _resolve_official_judge("strongreject_finetuned")
    if fn is None:
        for row in rows:
            row["ft_strongreject_score"] = None
        return {
            "status": "package_missing",
            "judge": "strongreject_finetuned",
            "note": "pip install strong-reject; needs HF_TOKEN + gated Gemma access.",
        }
    errors = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = {
            "forbidden_prompt": [r["behavior"] for r in chunk],
            "response": [r["completion"] for r in chunk],
        }
        try:
            out = fn(batch)
            scores = list(out["score"])
        except Exception as exc:  # noqa: BLE001
            errors += len(chunk)
            for row in chunk:
                row["ft_strongreject_score"] = None
                row["ft_judge_error"] = str(exc)[:200]
            continue
        for row, score in zip(chunk, scores):
            row["ft_strongreject_score"] = float(score)
    n = sum(1 for r in rows if r.get("ft_strongreject_score") is not None)
    return {
        "status": "ok" if n == len(rows) else ("partial" if n else "failed"),
        "judge": "strongreject_finetuned",
        "scored": n,
        "failed": errors,
    }


def _resolve_official_judge(judge_name: str) -> Callable | None:
    try:
        from strong_reject import evaluate as ev
    except ImportError:
        return None
    if hasattr(ev, judge_name):
        return getattr(ev, judge_name)
    if hasattr(ev, "strongreject_finetuned"):
        return ev.strongreject_finetuned
    return None
