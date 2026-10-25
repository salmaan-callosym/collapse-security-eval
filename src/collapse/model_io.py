"""Load and save OPT checkpoints.

Generation 0 is Shumailov's human WikiText fine-tune (not raw pretrained).
Sg and Hg share the 5-epoch M0; Rg uses the 10-epoch M0.
Later generations are fine-tunes that each *start* from pretrained weights
(as in the paper), then train on that generation's data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name: str, cache_dir: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_causal_lm(
    name_or_path: str,
    cache_dir: str | None = None,
    *,
    dtype: torch.dtype = torch.float32,
):
    """Load a causal LM.

    Training path must stay float32: facebook/opt-125m's config records float16,
    and training OPT in fp16 without a GradScaler overflows. Eval / hub smoke
    on larger models (e.g. Gemma-2-2B) may pass dtype=torch.float16.
    """
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            dtype=dtype,
            attn_implementation="eager",
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=dtype,
            **kwargs,
        )
    return model.to(dtype=dtype)


def load_pretrained(
    model_name: str,
    cache_dir: str | None = None,
    device: str = "cpu",
    *,
    dtype: torch.dtype | None = None,
):
    """Stock Hugging Face weights (starting point for every fine-tune cycle)."""
    model = _load_causal_lm(
        model_name, cache_dir=cache_dir, dtype=dtype or torch.float32
    )
    model.to(device)
    model.eval()
    return model


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    *,
    dtype: torch.dtype | None = None,
):
    path = Path(path)
    model = _load_causal_lm(str(path), dtype=dtype or torch.float32)
    model.to(device)
    model.eval()
    return model


def save_checkpoint(model, tokenizer, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Persist fp32 weights so later loads cannot inherit a fp16 config dtype.
    model.to(dtype=torch.float32)
    if hasattr(model, "config"):
        model.config.torch_dtype = "float32"
        if hasattr(model.config, "dtype"):
            try:
                model.config.dtype = "float32"
            except Exception:
                pass
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


def checkpoint_dir(cfg: dict[str, Any], lineage: str, seed: int, generation: int) -> Path:
    root = Path(cfg["paths"]["checkpoints_dir"])
    return root / lineage / f"seed{seed}" / f"gen{generation}"


def gen0_lineage_key(lineage: str) -> str:
    """Sg and Hg share the 5-epoch human M0; store it under Sg."""
    if lineage in ("Sg", "Hg"):
        return "Sg"
    if lineage == "Rg":
        return "Rg"
    raise ValueError(f"Unknown lineage: {lineage}")


def gen0_checkpoint_dir(cfg: dict[str, Any], lineage: str, seed: int) -> Path:
    return checkpoint_dir(cfg, gen0_lineage_key(lineage), seed, 0)


def resolve_model_for_generation(
    cfg: dict[str, Any],
    lineage: str,
    seed: int,
    generation: int,
    device: str,
):
    """Load checkpoint for generation g (gen 0 = human fine-tune)."""
    model_name = cfg["model_name"]
    cache_dir = cfg["paths"].get("hf_cache_dir")
    tokenizer = load_tokenizer(model_name, cache_dir=cache_dir)

    if generation == 0:
        ckpt = gen0_checkpoint_dir(cfg, lineage, seed)
        if not ckpt.is_dir():
            raise FileNotFoundError(
                f"Missing Shumailov-style gen0 at {ckpt}. "
                f"Run: python scripts/train_gen0.py --lineage {lineage} --seed {seed}"
            )
        return load_checkpoint(ckpt, device=device), tokenizer

    ckpt = checkpoint_dir(cfg, lineage, seed, generation)
    if not ckpt.is_dir():
        raise FileNotFoundError(
            f"Missing checkpoint for {lineage} seed={seed} gen={generation}: {ckpt}"
        )
    return load_checkpoint(ckpt, device=device), tokenizer


def pick_dtype(_device: str):
    """Training/eval dtype. Always fp32 — see `_load_causal_lm`."""
    return torch.float32
