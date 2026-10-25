#!/usr/bin/env python3
"""Tier-1 security eval on one checkpoint (HarmBench + StrongREJECT).

Example:
  python scripts/eval_security.py --lineage Sg --generation 0 --seed 0 --device cuda
  python scripts/eval_security.py --lineage Sg --generation 0 --seed 0 --max-prompts 8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import RESEARCH_ROOT, write_json

from src.collapse.config import load_config, resolve_device
from src.collapse.secrets import load_env, token_status
from src.collapse.model_io import (
    load_checkpoint,
    load_pretrained,
    load_tokenizer,
    resolve_model_for_generation,
)
from src.security.generate import generate_completions
from src.security.judge import score_strongreject
from src.security.langfuse_trace import (
    eval_session_id,
    flush_langfuse,
    get_langfuse,
)
from src.security.prompts import WRAPPERS, prompts_for_tier1, wrap_behavior
from src.security.tier1 import summarise_tier1
from src.security.tier2 import load_jbb_behaviors, run_tier2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(RESEARCH_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--lineage", default="Sg")
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-prompts", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="JSON list of prompt dicts (protocol_v2 prompts_v2.json). Overrides the CSV bank.",
    )
    parser.add_argument("--tier2", action="store_true")
    parser.add_argument("--tier2-jbb", action="store_true", help="Also score official JBB behaviours.")
    parser.add_argument("--skip-judge", action="store_true", help="Save completions only.")
    parser.add_argument("--checkpoint", default=None, help="HF folder or 'pretrained' to skip lineage lookup.")
    parser.add_argument(
        "--model",
        default=None,
        help="HF hub id (loads tokenizer+weights from this id; for instruct smoke tests).",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base HF id for a LoRA/full-weights subject (protocol_v2 collapse eval).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter HF id; merged onto --base with PEFT merge_and_unload().",
    )
    parser.add_argument(
        "--subject-label",
        default=None,
        help="Human-readable subject name recorded in results (default: adapter or base).",
    )
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap the security prompt as a chat user turn (instruct models).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Load dtype for --model / checkpoint eval (use float16 for ~2B+ on CPU).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override config (use small for smoke).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Write JSON/JSONL here instead of results/<lineage>/seedN/ (demo runs).",
    )
    args = parser.parse_args()

    # Env BEFORE any Langfuse import/use.
    load_env()
    print("tokens:", token_status())

    cfg = load_config(args.config)
    device = resolve_device(args.device)
    sec = cfg["security"]
    wrapper_id = sec["wrapper_id"]
    if wrapper_id not in WRAPPERS:
        raise SystemExit(f"wrapper {wrapper_id} is not in the locked set")

    if args.max_new_tokens is not None:
        sec["max_new_tokens"] = args.max_new_tokens

    lf = get_langfuse()
    session_id = eval_session_id(args.lineage, args.seed, args.generation)
    tags = [
        "tier1-security",
        f"lineage:{args.lineage}",
        f"seed:{args.seed}",
        f"gen:{args.generation}",
        f"judge:{sec.get('openrouter_model', sec['judge'])}",
    ]

    def _run_eval():
        return _eval_body(args, cfg, device, sec, wrapper_id, session_id)

    if lf is None:
        _run_eval()
        return

    # Langfuse v5: nest per-prompt spans (full I/O) under one root; filter dims in metadata.
    with lf.start_as_current_observation(
        as_type="span",
        name="evaluate-security-tier1",
        input={
            "lineage": args.lineage,
            "seed": args.seed,
            "generation": args.generation,
            "checkpoint": args.checkpoint,
            "hf_model": args.model,
            "max_prompts": args.max_prompts,
            "max_new_tokens": sec["max_new_tokens"],
            "wrapper_id": wrapper_id,
            "wrapper_template": WRAPPERS[wrapper_id],
            "chat_template": bool(args.chat_template),
            "judge_model": sec.get("openrouter_model", sec["judge"]),
            "subject_model": args.model or cfg.get("model_name", "facebook/opt-125m"),
        },
        metadata={
            "session_id": session_id,
            "wrapper_id": wrapper_id,
            "device": device,
            "tags": tags,
            "protocol": "tier1_harmbench_strongreject",
        },
    ) as root:
        summary, judge_meta = _run_eval()
        root.update(
            output={
                "asr": summary.get("asr"),
                "mean_strongreject": summary.get("mean_strongreject"),
                "n_scored": summary.get("n_scored"),
                "n_prompts": summary.get("n_prompts"),
                "asr_by_semantic_category": summary.get("asr_by_semantic_category"),
                "judge": judge_meta,
            }
        )
        if summary.get("asr") is not None:
            try:
                root.score_trace(name="asr", value=float(summary["asr"]))
                root.score_trace(
                    name="mean_strongreject",
                    value=float(summary["mean_strongreject"]),
                )
            except Exception:
                pass
        print(f"langfuse_trace_id={root.trace_id}")
    flush_langfuse()


def _eval_body(args, cfg, device, sec, wrapper_id, session_id: str):
    import torch

    cache = cfg["paths"].get("hf_cache_dir")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    load_dtype = dtype_map[args.dtype]
    load_info = None
    subject_model = (
        args.subject_label
        or args.checkpoint
        or cfg.get("model_name", "facebook/opt-125m")
    )
    if args.base:
        from src.collapse.adapter_io import load_subject_model

        subject_model = args.subject_label or args.adapter or args.base
        model, tokenizer, load_info = load_subject_model(
            args.base,
            args.adapter,
            dtype=load_dtype,
            device=device,
            cache_dir=cache,
        )
        print(f"adapter_io: {load_info}")
    elif args.model:
        subject_model = args.model
        tokenizer = load_tokenizer(args.model, cache_dir=cache)
        model = load_pretrained(
            args.model, cache_dir=cache, device=device, dtype=load_dtype
        )
    elif args.checkpoint == "pretrained":
        tokenizer = load_tokenizer(cfg["model_name"], cache_dir=cache)
        model = load_pretrained(
            cfg["model_name"], cache_dir=cache, device=device, dtype=load_dtype
        )
    elif args.checkpoint:
        # Local folder: tokenizer lives with the weights when saved by us;
        # fall back to cfg model_name only if needed.
        try:
            tokenizer = load_tokenizer(args.checkpoint, cache_dir=cache)
        except Exception:
            tokenizer = load_tokenizer(cfg["model_name"], cache_dir=cache)
        model = load_checkpoint(args.checkpoint, device=device, dtype=load_dtype)
    else:
        model, tokenizer = resolve_model_for_generation(
            cfg, args.lineage, args.seed, args.generation, device
        )

    if args.chat_template:
        sec["use_chat_template"] = True

    if args.prompts_file:
        prompts = json.loads(Path(args.prompts_file).read_text())
    else:
        prompts = prompts_for_tier1(cfg)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    print(
        f"tier1 prompts={len(prompts)} wrapper={wrapper_id} "
        f"model={subject_model} chat_template={bool(sec.get('use_chat_template'))} device={device}"
    )
    rows = generate_completions(model, tokenizer, prompts, cfg, device)
    for row in rows:
        row["subject_model"] = subject_model
        row["chat_template"] = bool(sec.get("use_chat_template"))

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(cfg["paths"]["results_dir"]) / args.lineage / f"seed{args.seed}"
    )
    _write_rows(out_dir / f"gen{args.generation}_security_rows.jsonl", rows)
    print(f"wrote unscored completions → {out_dir / f'gen{args.generation}_security_rows.jsonl'}")

    judge_meta = {"status": "skipped", "judge": sec["judge"]}
    trace_meta = {
        "lineage": args.lineage,
        "seed": args.seed,
        "generation": args.generation,
        "session_id": session_id,
        "subject_model": subject_model,
        "wrapper_id": wrapper_id,
        "chat_template": bool(sec.get("use_chat_template")),
    }
    if not args.skip_judge:
        rows, judge_meta = score_strongreject(
            rows,
            judge_name=sec["judge"],
            openrouter_model=sec.get("openrouter_model", "openai/gpt-4o-mini"),
            trace_meta=trace_meta,
        )

    summary = summarise_tier1(rows, float(sec["asr_threshold"]))
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "tier1_harmbench_strongreject",
        "lineage": args.lineage,
        "seed": args.seed,
        "generation": args.generation,
        "device": device,
        "wrapper_id": wrapper_id,
        "max_new_tokens": sec["max_new_tokens"],
        "do_sample": sec["do_sample"],
        "judge": judge_meta,
        "tier1": summary,
        "n_rows": len(rows),
        "checkpoint_override": args.checkpoint,
        "hf_model": args.model,
        "subject_model": subject_model,
        "base": args.base,
        "adapter": args.adapter,
        "load_info": load_info,
        "dtype": args.dtype,
        "chat_template": bool(sec.get("use_chat_template")),
        "langfuse_session": session_id if get_langfuse() else None,
    }

    if args.tier2:
        from src.collapse.model_io import checkpoint_dir, gen0_checkpoint_dir

        ckpt = (
            gen0_checkpoint_dir(cfg, args.lineage, args.seed)
            if args.generation == 0
            else checkpoint_dir(cfg, args.lineage, args.seed, args.generation)
        )
        payload["tier2"] = run_tier2(
            cfg, args.lineage, args.seed, args.generation, str(ckpt)
        )

    if args.tier2_jbb:
        jbb = load_jbb_behaviors()
        jbb_prompts = []
        for row in jbb:
            item = dict(row)
            item["wrapper_id"] = wrapper_id
            item["prompt"] = wrap_behavior(row["behavior"], wrapper_id, "")
            jbb_prompts.append(item)
        if args.max_prompts is not None:
            jbb_prompts = jbb_prompts[: args.max_prompts]
        jbb_rows = generate_completions(model, tokenizer, jbb_prompts, cfg, device)
        if not args.skip_judge:
            jbb_rows, jbb_judge = score_strongreject(
                jbb_rows,
                judge_name=sec["judge"],
                openrouter_model=sec.get("openrouter_model", "openai/gpt-4o-mini"),
                trace_meta=trace_meta,
            )
        else:
            jbb_judge = {"status": "skipped"}
        payload["tier2_jbb"] = {
            "judge": jbb_judge,
            "summary": summarise_tier1(jbb_rows, float(sec["asr_threshold"])),
        }
        _write_rows(out_dir / f"gen{args.generation}_jbb_rows.jsonl", jbb_rows)

    write_json(out_dir / f"gen{args.generation}_security.json", payload)
    _write_rows(out_dir / f"gen{args.generation}_security_rows.jsonl", rows)
    print(json.dumps({"tier1": summary, "judge": judge_meta}, indent=2))
    print(f"wrote {out_dir / f'gen{args.generation}_security.json'}")
    return summary, judge_meta


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
