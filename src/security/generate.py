"""Greedy completions for a frozen prompt list.

Decoding is part of the locked protocol: greedy, fixed max tokens.
"""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm


@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts: list[dict[str, str]],
    cfg: dict[str, Any],
    device: str,
    *,
    skip_ids: set[str] | None = None,
    on_batch: Any | None = None,
) -> list[dict[str, Any]]:
    sec = cfg["security"]
    max_new = int(sec["max_new_tokens"])
    batch_size = int(sec.get("generate_batch_size", 4))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.eos_token_id
    # Decoder-only models must left-pad or batched generate is garbage.
    tokenizer.padding_side = "left"

    model.eval()
    model.to(device)
    results: list[dict[str, Any]] = []

    use_chat = bool(sec.get("use_chat_template"))
    has_chat = bool(getattr(tokenizer, "chat_template", None))
    if skip_ids:
        prompts = [p for p in prompts if str(p.get("behavior_id")) not in skip_ids]
    if not prompts:
        return results

    for start in tqdm(range(0, len(prompts), batch_size), desc="security generate"):
        chunk = prompts[start : start + batch_size]
        if use_chat and has_chat:
            texts = []
            for p in chunk:
                texts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": p["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
        else:
            if use_chat and not has_chat:
                print("warning: --chat-template set but tokenizer has no chat_template; using raw prompt")
            texts = [p["prompt"] for p in chunk]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=bool(sec.get("do_sample", False)),
            pad_token_id=pad_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for i, item in enumerate(chunk):
            cont_ids = out[i, prompt_len:]
            completion = tokenizer.decode(cont_ids, skip_special_tokens=True)
            row = dict(item)
            row["completion"] = completion
            row["prompt_rendered"] = texts[i]
            row["chat_template_applied"] = bool(use_chat and has_chat)
            results.append(row)
        # Caller persists this batch (Drive) before the next one starts.
        if on_batch is not None:
            on_batch(list(results[-len(chunk):]))
    return results
