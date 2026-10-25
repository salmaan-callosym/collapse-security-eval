"""Load base + optional LoRA adapter for the collapse-security study.

Handles three realities of the checkpoints we evaluate:
  * plain causal LMs (Qwen, Gemma-2, OPT),
  * multimodal Gemma-3-4B-it (Gemma3ForConditionalGeneration): AutoModelForCausalLM
    can fail, so we fall back to AutoModelForImageTextToText and feed text-only chat,
  * LoRA adapters on a safety-aligned base: PeftModel.from_pretrained + merge_and_unload.

Dtype policy (protocol_v2): bf16 on Ampere+, fp32 for OPT-125m. Callers pass dtype.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(name_or_path: str, cache_dir: str | None = None):
    tok = AutoTokenizer.from_pretrained(name_or_path, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_base(name_or_path: str, *, dtype, cache_dir, device):
    """Load a base model, falling back to image-text-to-text for Gemma-3."""
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "dtype": dtype}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path, attn_implementation="eager", **kwargs
        )
        kind = "causal_lm"
    except (ValueError, KeyError, TypeError, OSError) as exc:
        # Multimodal Gemma-3 and similar text+image stacks live here.
        print(f"adapter_io: AutoModelForCausalLM failed ({type(exc).__name__}); "
              f"trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            name_or_path, attn_implementation="eager", **kwargs
        )
        kind = "image_text_to_text"
    model.to(device)
    model.eval()
    return model, kind


def load_subject_model(
    base_id: str,
    adapter_id: str | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cache_dir: str | None = None,
    merge: bool = True,
):
    """Return (model, tokenizer, info) for a base + optional LoRA adapter.

    info records what actually loaded so the results file is self-describing.
    The tokenizer comes from the adapter repo when present (it may ship the
    base tokenizer), else from the base.
    """
    model, kind = _load_base(base_id, dtype=dtype, cache_dir=cache_dir, device=device)
    info = {
        "base_id": base_id,
        "adapter_id": adapter_id,
        "model_class": type(model).__name__,
        "load_kind": kind,
        "merged": False,
        "dtype": str(dtype).replace("torch.", ""),
    }

    tok_source = base_id
    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_id, cache_dir=cache_dir)
        if merge:
            model = model.merge_and_unload()
            info["merged"] = True
        model.to(device)
        model.eval()
        info["model_class_after_adapter"] = type(model).__name__
        # Adapter repos usually carry the base tokenizer; prefer it if loadable.
        try:
            tok = load_tokenizer(adapter_id, cache_dir=cache_dir)
            tok_source = adapter_id
        except Exception:  # noqa: BLE001
            tok = load_tokenizer(base_id, cache_dir=cache_dir)
    else:
        tok = load_tokenizer(base_id, cache_dir=cache_dir)

    info["tokenizer_source"] = tok_source
    info["has_chat_template"] = bool(getattr(tok, "chat_template", None))
    return model, tok, info
