"""Utility (proof-of-collapse) metrics: WikiText-2 perplexity and output diversity.

PPL is the standard sliding-window estimate on the raw WikiText-2 *test* split
(no chat template). Diversity is distinct-1/2 and mean length on fixed benign
prompts, decoded with the same greedy settings as the security eval.
"""

from __future__ import annotations

from typing import Any

import torch


@torch.inference_mode()
def wikitext2_perplexity(
    model,
    tokenizer,
    *,
    device: str,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    max_tokens: int = 50_000,
    stride: int = 512,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Sliding-window PPL on wikitext-2-raw-v1 test (first ~max_tokens tokens).

    Prefers a local `datasets` dir (offline cluster runs); falls back to the
    canonical Salesforce/wikitext repo.
    """
    ds = _load_wikitext2_test(cache_dir=cache_dir, local_dir=local_dir)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][:, :max_tokens]
    n_tokens = input_ids.shape[1]

    if max_length is None:
        max_length = int(
            getattr(model.config, "max_position_embeddings", 1024) or 1024
        )
        max_length = min(max_length, 1024)

    nlls = []
    total_target = 0
    prev_end = 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + max_length, n_tokens)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end].to(device)
        target = ids.clone()
        target[:, :-trg_len] = -100
        out = model(ids, labels=target)
        # HF averages the loss over valid target tokens; rescale to a sum.
        n_valid = max(1, trg_len - 1) if begin == 0 else trg_len
        nlls.append(out.loss.float() * n_valid)
        total_target += n_valid
        prev_end = end
        if end == n_tokens:
            break

    mean_nll = torch.stack(nlls).sum() / max(1, total_target)
    ppl = torch.exp(mean_nll).item()
    return {
        "perplexity": ppl,
        "mean_nll": mean_nll.item(),
        "n_tokens_scored": total_target,
        "stride": stride,
        "max_length": max_length,
        "dataset": "wikitext-2-raw-v1:test",
    }


def _load_wikitext2_test(*, cache_dir: str | None, local_dir: str | None):
    from pathlib import Path

    if local_dir and (Path(local_dir) / "test").is_dir():
        from datasets import load_from_disk

        return load_from_disk(local_dir)["test"]
    from datasets import load_dataset

    return load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir
    )


def _tokens(text: str) -> list[str]:
    return text.split()


def distinct_n(texts: list[str], n: int) -> float:
    """Corpus-level distinct-n: unique n-grams / total n-grams."""
    total = 0
    seen = set()
    for t in texts:
        toks = _tokens(t)
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        seen.update(grams)
    if total == 0:
        return 0.0
    return len(seen) / total


def diversity_metrics(completions: list[str]) -> dict[str, Any]:
    lengths = [len(_tokens(c)) for c in completions]
    char_lengths = [len(c) for c in completions]
    n = max(1, len(completions))
    return {
        "n_responses": len(completions),
        "distinct_1": distinct_n(completions, 1),
        "distinct_2": distinct_n(completions, 2),
        "mean_response_len_words": sum(lengths) / n,
        "mean_response_len_chars": sum(char_lengths) / n,
        "n_empty": sum(1 for c in completions if not c.strip()),
    }
