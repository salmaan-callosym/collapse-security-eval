"""One place to resolve a subject model + tokenizer across all eval scripts.

Supports every loading mode the study needs:
  * base + optional LoRA adapter  (protocol_v2 collapse lineages),
  * a single HF hub id (--model),
  * a local checkpoint folder or the literal 'pretrained' (--checkpoint),
  * the lineage/seed/generation lookup used by the original OPT pipeline.

Keeping this in one function means security and utility evals never diverge
on dtype, device, or tokenizer source.
"""

from __future__ import annotations

from typing import Any

import torch

from .model_io import (
    load_checkpoint,
    load_pretrained,
    load_tokenizer,
    resolve_model_for_generation,
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_subject(
    cfg: dict[str, Any],
    *,
    base: str | None = None,
    adapter: str | None = None,
    model: str | None = None,
    checkpoint: str | None = None,
    lineage: str = "Sg",
    seed: int = 0,
    generation: int = 0,
    dtype: str = "bfloat16",
    device: str = "cuda",
    subject_label: str | None = None,
):
    """Return (model, tokenizer, subject_name, load_info)."""
    cache = cfg["paths"].get("hf_cache_dir")
    load_dtype = DTYPE_MAP[dtype]
    load_info: dict[str, Any] = {"dtype": dtype, "device": device}

    if base:
        from .adapter_io import load_subject_model

        mdl, tok, info = load_subject_model(
            base, adapter, dtype=load_dtype, device=device, cache_dir=cache
        )
        load_info.update(info)
        name = subject_label or adapter or base
        return mdl, tok, name, load_info

    if model:
        tok = load_tokenizer(model, cache_dir=cache)
        mdl = load_pretrained(model, cache_dir=cache, device=device, dtype=load_dtype)
        return mdl, tok, subject_label or model, load_info

    if checkpoint == "pretrained":
        name = cfg["model_name"]
        tok = load_tokenizer(name, cache_dir=cache)
        mdl = load_pretrained(name, cache_dir=cache, device=device, dtype=load_dtype)
        return mdl, tok, subject_label or name, load_info

    if checkpoint:
        try:
            tok = load_tokenizer(checkpoint, cache_dir=cache)
        except Exception:  # noqa: BLE001
            tok = load_tokenizer(cfg["model_name"], cache_dir=cache)
        mdl = load_checkpoint(checkpoint, device=device, dtype=load_dtype)
        return mdl, tok, subject_label or checkpoint, load_info

    mdl, tok = resolve_model_for_generation(cfg, lineage, seed, generation, device)
    return mdl, tok, subject_label or cfg.get("model_name"), load_info
