#!/usr/bin/env python3
"""Utility (proof-of-collapse) eval for one checkpoint.

Reports WikiText-2 test perplexity and benign-output diversity (distinct-1/2,
mean length). Same loading flags as eval_security.py.

Example:
  python scripts/eval_utility.py --base unsloth/gemma-3-4b-it \
      --adapter HungryDino/..._gen12 --chat-template --device cuda \
      --out results/collapse_eval/gemma3_4b_self_collapse/12/utility.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import RESEARCH_ROOT, write_json

from src.collapse.config import load_config, resolve_device
from src.collapse.secrets import load_env, token_status
from src.collapse.subject_loader import resolve_subject
from src.eval.utility import diversity_metrics, wikitext2_perplexity
from src.security.generate import generate_completions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(RESEARCH_ROOT / "configs" / "default.yaml"))
    p.add_argument("--base", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--lineage", default="Sg")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--generation", type=int, default=0)
    p.add_argument("--subject-label", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--chat-template", action="store_true")
    p.add_argument(
        "--benign-file",
        default=str(RESEARCH_ROOT / "results" / "collapse_eval" / "benign_prompts_v2.json"),
    )
    p.add_argument("--ppl-max-tokens", type=int, default=50_000)
    p.add_argument("--ppl-stride", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-benign", type=int, default=None, help="Smoke cap.")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--skip-diversity", action="store_true")
    p.add_argument("--out", default=None, help="Write utility.json here.")
    args = p.parse_args()

    load_env()
    print("tokens:", token_status())
    cfg = load_config(args.config)
    device = resolve_device(args.device)
    cache = cfg["paths"].get("hf_cache_dir")

    model, tokenizer, subject_name, load_info = resolve_subject(
        cfg,
        base=args.base,
        adapter=args.adapter,
        model=args.model,
        checkpoint=args.checkpoint,
        lineage=args.lineage,
        seed=args.seed,
        generation=args.generation,
        dtype=args.dtype,
        device=device,
        subject_label=args.subject_label,
    )
    print(f"subject={subject_name} device={device} load_info={load_info}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "protocol_v2",
        "subject_model": subject_name,
        "base": args.base,
        "adapter": args.adapter,
        "lineage": args.lineage,
        "generation": args.generation,
        "device": device,
        "dtype": args.dtype,
        "load_info": load_info,
    }

    if not args.skip_ppl:
        print("computing WikiText-2 perplexity...")
        payload["perplexity"] = wikitext2_perplexity(
            model,
            tokenizer,
            device=device,
            cache_dir=cache,
            local_dir=cfg["paths"].get("wikitext_dir"),
            max_tokens=args.ppl_max_tokens,
            stride=args.ppl_stride,
        )
        print("ppl:", payload["perplexity"]["perplexity"])

    if not args.skip_diversity:
        print("generating benign completions for diversity...")
        benign = json.loads(Path(args.benign_file).read_text())
        if args.max_benign is not None:
            benign = benign[: args.max_benign]
        # generate_completions reads cfg['security'] for decoding knobs.
        sec = cfg["security"]
        sec["max_new_tokens"] = args.max_new_tokens
        sec["use_chat_template"] = bool(args.chat_template)
        rows = generate_completions(model, tokenizer, benign, cfg, device)
        completions = [r["completion"] for r in rows]
        payload["diversity"] = diversity_metrics(completions)
        payload["diversity"]["max_new_tokens"] = args.max_new_tokens
        payload["diversity"]["chat_template"] = bool(args.chat_template)
        payload["benign_sample"] = [
            {"prompt": r["prompt"][:200], "completion": r["completion"][:400]}
            for r in rows[:3]
        ]
        print("diversity:", {k: payload["diversity"][k] for k in
                             ("distinct_1", "distinct_2", "mean_response_len_words")})

    out = Path(args.out) if args.out else (
        RESEARCH_ROOT / "results" / "collapse_eval" / "_utility"
        / f"{args.lineage}_gen{args.generation}_utility.json"
    )
    write_json(out, payload)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
