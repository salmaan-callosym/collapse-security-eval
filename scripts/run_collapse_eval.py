#!/usr/bin/env python3
"""Drive the whole collapse-security study from the verified manifest.

Loops checkpoints in priority order (P1 -> P2 -> P4 -> P3), and for each one
writes, under results/collapse_eval/<lineage>/<gen>/:
    completions.jsonl   raw greedy completions on the 100 HarmBench prompts
    scores.jsonl        the same rows with StrongREJECT score + judge axes
    summary.json        protocol meta + ASR / mean / refusal-rate metrics
    utility.json        WikiText-2 PPL + benign-output diversity

Resumable: completions and scores are rewritten to disk after every batch, so a
disconnect loses at most one batch. A checkpoint is skipped only when summary.json
covers every requested prompt AND utility.json covers every benign prompt. A short
gate run (20 prompts) is therefore extended, not skipped, by the full run.
Frees GPU + optional HF cache after each checkpoint so disk stays bounded.

  python scripts/run_collapse_eval.py --device cuda                 # everything
  python scripts/run_collapse_eval.py --device cuda --only P1       # one priority
  python scripts/run_collapse_eval.py --device cuda --only-lineage gemma3_4b_self_collapse
  python scripts/run_collapse_eval.py --gate                        # go/no-go subset
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import RESEARCH_ROOT, write_json

from src.collapse.config import load_config, resolve_device
from src.collapse.secrets import load_env, token_status
from src.collapse.subject_loader import resolve_subject
from src.eval.utility import diversity_metrics, wikitext2_perplexity
from src.security.generate import generate_completions
from src.security.judge import score_finetuned, score_strongreject
from src.security.langfuse_trace import flush_langfuse, get_langfuse
from src.security.tier1 import summarise_tier1

COLLAPSE_DIR = RESEARCH_ROOT / "results" / "collapse_eval"
PRIORITY_ORDER = ["P1", "P2", "P4", "P3"]
GATE_SELECTION = {  # go/no-go: P1 base, gen0, gen12
    "P1": {"base", "0", "12"},
}


def gen_key(entry: dict) -> str:
    """Stable per-checkpoint directory key."""
    gen = entry["generation"]
    arm = entry.get("arm")
    if arm and arm != "iter1":
        return f"{arm}_iter{gen}"
    if arm == "iter1":
        return "iter1"
    return str(gen)


def load_flags(entry: dict) -> dict:
    """Translate a manifest entry into resolve_subject kwargs."""
    if entry.get("adapter"):
        return {"base": entry["base"], "adapter": entry["id"]}
    if str(entry["id"]).startswith("local:"):
        return {"checkpoint": entry["local_path"]}
    return {"model": entry["id"]}


def subject_dtype(entry: dict, override: str | None) -> str:
    if override:
        return override
    if entry["lineage"] == "opt125m":
        return "float32"
    return "bfloat16"


def _security_complete(d: Path, n_prompts: int) -> bool:
    path = d / "summary.json"
    if not path.is_file():
        return False
    summary = json.loads(path.read_text())
    scored = int((summary.get("tier1") or {}).get("n_scored") or 0)
    return int(summary.get("n_rows") or 0) >= n_prompts and scored >= n_prompts


def _utility_complete(d: Path, n_benign: int) -> bool:
    path = d / "utility.json"
    if not path.is_file():
        return False
    util = json.loads(path.read_text())
    n = int((util.get("diversity") or {}).get("n_responses") or 0)
    return "perplexity" in util and n >= n_benign


def _already_done(d: Path, n_prompts: int, n_benign: int, *, skip_security: bool, skip_utility: bool) -> bool:
    sec_ok = skip_security or _security_complete(d, n_prompts)
    util_ok = skip_utility or _utility_complete(d, n_benign)
    return sec_ok and util_ok


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Rewrite the whole JSONL via temp+replace so a disconnect cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _by_behavior(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        bid = row.get("behavior_id")
        if bid:
            out[str(bid)] = row
    return out


def _ordered(prompts: list[dict], by_id: dict[str, dict]) -> list[dict]:
    return [by_id[str(p["behavior_id"])] for p in prompts if str(p.get("behavior_id")) in by_id]


def _touch_progress(entry: dict, status: str, **extra) -> None:
    path = COLLAPSE_DIR / "progress.json"
    data = json.loads(path.read_text()) if path.is_file() else {"checkpoints": {}}
    data["updated_utc"] = datetime.now(timezone.utc).isoformat()
    key = f"{entry['lineage']}/{gen_key(entry)}"
    data.setdefault("checkpoints", {})[key] = {
        "status": status,
        "id": entry["id"],
        "priority": entry["priority"],
        "updated_utc": data["updated_utc"],
        **extra,
    }
    write_json(path, data)


def run_security(entry, model, tok, cfg, device, out_dir, prompts, *, force, max_new,
                 secondary_judge=False):
    sec = cfg["security"]
    sec["max_new_tokens"] = max_new
    sec["use_chat_template"] = bool(entry.get("chat_template"))

    comp_path = out_dir / "completions.jsonl"
    scores_path = out_dir / "scores.jsonl"
    summary_path = out_dir / "summary.json"

    if _security_complete(out_dir, len(prompts)) and not force:
        print(f"  security: cached ({summary_path.name}, n={len(prompts)})")
        return json.loads(summary_path.read_text())["tier1"]

    by_id = {} if force else _by_behavior(_read_rows(comp_path))
    by_id = {k: v for k, v in by_id.items() if v.get("completion") is not None}

    def persist_comps(batch: list[dict]) -> None:
        for row in batch:
            row["subject_model"] = entry["id"]
            by_id[str(row["behavior_id"])] = row
        _write_rows(comp_path, _ordered(prompts, by_id))
        print(f"  security: flushed {len(by_id)}/{len(prompts)} completions", flush=True)

    missing = [p for p in prompts if str(p["behavior_id"]) not in by_id]
    if missing:
        print(f"  security: generating {len(missing)} missing / {len(prompts)} prompts")
        generate_completions(model, tok, prompts, cfg, device, on_batch=persist_comps,
                             skip_ids=set(by_id))
    else:
        print("  security: completions already on disk")

    rows = _ordered(prompts, by_id)
    if len(rows) < len(prompts):
        print("  security: completions incomplete; summary deferred")
        return None

    scored = {} if force else _by_behavior(_read_rows(scores_path))
    scored = {
        k: v for k, v in scored.items()
        if v.get("strongreject_score") is not None
    }
    pending = []
    for row in rows:
        prev = scored.get(str(row["behavior_id"]))
        if prev and prev.get("completion") == row.get("completion"):
            continue
        pending.append(dict(row))

    def persist_score(row: dict) -> None:
        if row.get("strongreject_score") is None:
            return
        scored[str(row["behavior_id"])] = row
        _write_rows(scores_path, _ordered(prompts, scored))

    trace_meta = {
        "lineage": entry["lineage"],
        "generation": entry["generation"],
        "seed": 0,
        "subject_model": entry["id"],
        "session_id": f"collapse-{entry['lineage']}-{gen_key(entry)}",
    }
    if pending:
        print(f"  security: judging {len(pending)} unscored / {len(prompts)}")
        _, judge_meta = score_strongreject(
            pending,
            judge_name=sec["judge"],
            openrouter_model=sec["openrouter_model"],
            trace_meta=trace_meta,
            on_scored=persist_score,
        )
    else:
        judge_meta = {"status": "cached", "judge": sec["judge"], "model": sec["openrouter_model"]}

    rows = _ordered(prompts, scored)
    if len(rows) < len(prompts):
        print(f"  security: scored {len(rows)}/{len(prompts)}; summary deferred")
        flush_langfuse()
        return None

    ft_meta = {"status": "skipped"}
    if secondary_judge:
        ft_meta = score_finetuned(rows)
        print(f"  secondary judge: {ft_meta.get('status')}")
        _write_rows(scores_path, rows)

    summary = summarise_tier1(rows, float(sec["asr_threshold"]))
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "protocol_v2",
        "lineage": entry["lineage"],
        "priority": entry["priority"],
        "generation": entry["generation"],
        "gen_key": gen_key(entry),
        "subject_model": entry["id"],
        "base": entry.get("base"),
        "adapter": entry["id"] if entry.get("adapter") else None,
        "commit_sha": entry.get("commit_sha"),
        "chat_template": bool(entry.get("chat_template")),
        "max_new_tokens": max_new,
        "judge": judge_meta,
        "secondary_judge": ft_meta,
        "tier1": summary,
        "n_rows": len(rows),
    }
    write_json(summary_path, payload)
    flush_langfuse()
    print(f"  security: ASR={summary.get('asr')} mean={summary.get('mean_strongreject')} "
          f"refusal={summary.get('refusal_rate')}")
    return summary


def run_utility(entry, model, tok, cfg, device, out_dir, benign, *, force, max_new,
                ppl_max_tokens, ppl_stride):
    util_path = out_dir / "utility.json"
    partial_path = out_dir / "utility.partial.json"
    if _utility_complete(out_dir, len(benign)) and not force:
        print(f"  utility: cached ({util_path.name})")
        return
    for item in benign:
        item["behavior_id"] = item.get("behavior_id") or f"dolly-{item.get('dolly_index', item.get('order'))}"
    cache = cfg["paths"].get("hf_cache_dir")
    if partial_path.is_file() and not force:
        payload = json.loads(partial_path.read_text())
        print(f"  utility: reusing saved PPL ({payload.get('perplexity', {}).get('perplexity')})")
    else:
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "protocol_v2",
            "lineage": entry["lineage"],
            "generation": entry["generation"],
            "subject_model": entry["id"],
            "commit_sha": entry.get("commit_sha"),
        }
        payload["perplexity"] = wikitext2_perplexity(
            model, tok, device=device, cache_dir=cache,
            local_dir=cfg["paths"].get("wikitext_dir"),
            max_tokens=ppl_max_tokens, stride=ppl_stride,
        )
        write_json(partial_path, payload)
        print(f"  utility: PPL flushed ({payload['perplexity']['perplexity']:.2f})")
    sec = cfg["security"]
    sec["max_new_tokens"] = max_new
    sec["use_chat_template"] = bool(entry.get("chat_template"))
    benign_path = out_dir / "benign_completions.jsonl"
    by_id = {} if force else _by_behavior(_read_rows(benign_path))
    by_id = {k: v for k, v in by_id.items() if v.get("completion") is not None}

    def persist_benign(batch: list[dict]) -> None:
        for row in batch:
            by_id[str(row["behavior_id"])] = row
        _write_rows(benign_path, _ordered(benign, by_id))
        print(f"  utility: flushed {len(by_id)}/{len(benign)} benign completions", flush=True)

    if any(str(p["behavior_id"]) not in by_id for p in benign):
        generate_completions(model, tok, benign, cfg, device, on_batch=persist_benign,
                             skip_ids=set(by_id))
    rows = _ordered(benign, by_id)
    if len(rows) < len(benign):
        print("  utility: benign completions incomplete; utility.json deferred")
        return
    payload["diversity"] = diversity_metrics([r["completion"] for r in rows])
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(util_path, payload)
    print(f"  utility: ppl={payload['perplexity']['perplexity']:.2f} "
          f"distinct2={payload['diversity']['distinct_2']:.3f} "
          f"len={payload['diversity']['mean_response_len_words']:.1f}")


def clean_hf_cache(entry, cfg, *, drop_base: bool) -> None:
    """Drop HF snapshots after results are on disk.

    Adapter repos are small. The shared base (Gemma-3, Qwen) stays until the
    last checkpoint in this run that needs it, so a disconnect does not force
    an 8 GB re-download.
    """
    cache = Path(cfg["paths"].get("hf_cache_dir") or "")
    if not cache:
        return
    repos = {entry["id"]}
    if drop_base:
        repos.add(entry.get("base"))
    for repo in repos:
        if not repo or str(repo).startswith("local:"):
            continue
        folder = "models--" + str(repo).replace("/", "--")
        target = cache / folder
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"  cleaned cache: {folder}")


def select_entries(manifest, args) -> list[dict]:
    entries = [e for e in manifest["checkpoints"] if e.get("exists")]
    if args.gate:
        out = []
        for e in entries:
            sel = GATE_SELECTION.get(e["priority"])
            if sel and gen_key(e) in sel:
                out.append(e)
        return out
    if args.only:
        entries = [e for e in entries if e["priority"] in set(args.only)]
    if args.only_lineage:
        entries = [e for e in entries if e["lineage"] == args.only_lineage]
    entries.sort(key=lambda e: PRIORITY_ORDER.index(e["priority"]))
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(COLLAPSE_DIR / "manifest.json"))
    p.add_argument("--config", default=str(RESEARCH_ROOT / "configs" / "default.yaml"))
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default=None, help="Override per-lineage default.")
    p.add_argument("--only", nargs="+", help="Priorities to run, e.g. P1 P2.")
    p.add_argument("--only-lineage", default=None)
    p.add_argument("--gate", action="store_true", help="Go/no-go subset (P1 base/0/12).")
    p.add_argument("--force", action="store_true", help="Recompute even if cached.")
    p.add_argument("--skip-security", action="store_true")
    p.add_argument("--skip-utility", action="store_true")
    p.add_argument("--secondary-judge", action="store_true",
                   help="Also score with local strongreject_finetuned (needs strong-reject pkg).")
    p.add_argument("--clean-cache", action="store_true", help="rm HF snapshot after each ckpt.")
    p.add_argument("--generate-batch-size", type=int, default=None,
                   help="Override security.generate_batch_size (use 1 or 2 on a T4).")
    p.add_argument("--max-prompts", type=int, default=None, help="Smoke cap on harm prompts.")
    p.add_argument("--max-benign", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--ppl-max-tokens", type=int, default=50_000)
    p.add_argument("--ppl-stride", type=int, default=512)
    args = p.parse_args()

    load_env()
    print("tokens:", token_status())
    cfg = load_config(args.config)
    device = resolve_device(args.device)

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest.get("primary_judge") != cfg["security"]["openrouter_model"]:
        print(f"WARNING: manifest judge {manifest.get('primary_judge')} != "
              f"config judge {cfg['security']['openrouter_model']}")

    prompts = json.loads((COLLAPSE_DIR / "prompts_v2.json").read_text())
    benign = json.loads((COLLAPSE_DIR / "benign_prompts_v2.json").read_text())
    if args.generate_batch_size:
        cfg["security"]["generate_batch_size"] = args.generate_batch_size
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    if args.max_benign:
        benign = benign[: args.max_benign]

    entries = select_entries(manifest, args)
    print(f"{len(entries)} checkpoints to consider "
          f"(device={device}, judge={cfg['security']['openrouter_model']})")

    import torch

    for i, entry in enumerate(entries, 1):
        key = gen_key(entry)
        out_dir = COLLAPSE_DIR / entry["lineage"] / key
        out_dir.mkdir(parents=True, exist_ok=True)
        header = f"[{i}/{len(entries)}] {entry['priority']} {entry['lineage']}/{key}  {entry['id']}"
        if _already_done(
            out_dir, len(prompts), len(benign),
            skip_security=args.skip_security, skip_utility=args.skip_utility,
        ) and not args.force:
            print(f"{header}  -> cached, skip")
            continue
        print(header)
        _touch_progress(entry, "running")

        dtype = subject_dtype(entry, args.dtype)
        flags = load_flags(entry)
        try:
            model, tok, subject_name, load_info = resolve_subject(
                cfg, dtype=dtype, device=device,
                subject_label=entry["id"], **flags,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  LOAD FAILED: {type(exc).__name__}: {exc}")
            (out_dir / "load_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            _touch_progress(entry, "load_failed", error=f"{type(exc).__name__}: {exc}"[:300])
            continue
        print(f"  loaded: {load_info.get('model_class', '')} "
              f"merged={load_info.get('merged')} dtype={dtype} "
              f"chat_template={bool(entry.get('chat_template'))}")

        try:
            if not args.skip_security:
                run_security(entry, model, tok, cfg, device, out_dir, prompts,
                             force=args.force, max_new=args.max_new_tokens,
                             secondary_judge=args.secondary_judge)
            if not args.skip_utility:
                run_utility(entry, model, tok, cfg, device, out_dir, benign,
                            force=args.force, max_new=args.max_new_tokens,
                            ppl_max_tokens=args.ppl_max_tokens, ppl_stride=args.ppl_stride)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            finished = _already_done(
                out_dir, len(prompts), len(benign),
                skip_security=args.skip_security, skip_utility=args.skip_utility,
            )
            _touch_progress(entry, "done" if finished else "partial")
            if args.clean_cache and finished:
                rest = entries[i:]
                drop_base = all(e.get("base") != entry.get("base") for e in rest)
                clean_hf_cache(entry, cfg, drop_base=drop_base)
            elif args.clean_cache:
                print("  cache kept: checkpoint not finished")

    flush_langfuse()
    print("done.")


if __name__ == "__main__":
    main()
