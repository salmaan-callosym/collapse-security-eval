# Results summary — security under model collapse (protocol_v2)

**Status:** living document. Setup, verification, and the full code pipeline are complete.
P4 (OPT-125m) runs locally; **P1/P2/P3 are blocked on GPU compute and must run on the Wits
cluster** (this laptop has ~6 GB free RAM and ~6 GB free disk — too small to load Gemma-2-2B,
Gemma-3-4B, or Qwen-7B). Numbers below are filled in as each lineage finishes.

## Research question
Does a model's safety (refusal of harmful requests) degrade as it undergoes model collapse through
recursive self-training? **H1:** ASR and mean StrongREJECT increase with collapse generation.
**H0:** no monotonic increase (report honestly; harm falling with incoherence is a finding).

## Protocol (frozen as `protocol_v2`)
- Prompts: first 100 HarmBench text behaviours (`prompts_v2.json`), fixed order, same for every model.
- Wrapper `answer_v1`, applied as a chat user turn when a chat template exists.
- Decoding: greedy, `max_new_tokens=256`, left padding, no repetition penalty.
- Precision: bf16 (fp32 for OPT).
- Primary judge: OpenRouter `deepseek/deepseek-v4.1-flash-1017` (dated snapshot, temp 0, reasoning off,
  StrongREJECT rubric). **Verified present** in `/api/v1/models` at manifest time.
- Secondary judge: local `strongreject_finetuned` on the same completions (to be run on the cluster).
- Utility (proof of collapse): WikiText-2 test perplexity (first ~50k tokens, stride 512) and benign
  diversity (distinct-1/2, mean length) on 100 fixed Dolly prompts.

## Manifest verification (done)
All 24 Hugging Face repos exist; commit SHAs pinned in `results/collapse_eval/manifest.json`.
Every HungryDino adapter's `base_model_name_or_path` matches the recorded base. Local OPT-125m
lineage (`Sg/seed0/gen0..3`) present. See manifest for exact SHAs.

## Headline numbers

### P1 — Gemma-3-4B-it self-collapse (MAIN, aligned, LoRA)  — _pending cluster run_
| gen | ASR | mean SR | refusal | PPL | distinct-2 |
|----:|----:|--------:|--------:|----:|-----------:|
| base | | | | | |
| 0 | | | | | |
| … | | | | | |
| 12 | | | | | |

H1 verdict: _pending_.

### P2 — Gemma-2-2B Kazdan (COMPARISON, unaligned, full weights)  — _pending cluster run_
Replace arm (iter1,2,3,5,10) and accumulate arm (iter10 control). Pilot (10 prompts, CPU) showed
iter1 ASR≈0.6 — i.e. real headroom on the unaligned base. Full numbers pending.

H1 verdict: _pending_.

### P3 — Qwen2.5-7B-Instruct self-collapse (REPLICATION, aligned, LoRA)  — _pending cluster run_
H1 verdict: _pending_.

### P4 — OPT-125m (NEGATIVE CONTROL, capability floor)  — _local run_
| gen | ASR | mean SR | refusal | PPL |
|----:|----:|--------:|--------:|----:|
| base | | | | |
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

H1 verdict: _expected H0 / capability-floor_ — OPT echoes rather than answering, so StrongREJECT
sits near 0 regardless of generation. PPL is expected to rise across generations (collapse confirmed
on utility), which is exactly why OPT is a utility-collapse control, not a security subject.

## Limitations (verbatim facts, no spin)
- The HungryDino lineages recurse on **number sequences** and are undocumented ("p10" meaning
  unknown). We establish collapse by our own utility metrics, not by their claims.
- HungryDino gen0 is already a fine-tune, so the base model is included as the true reference.
- Kazdan's base model is not safety-tuned.
- 100 prompts, greedy decoding, a single seed per lineage, and an LLM judge (validated against humans).
- OPT-125m cannot produce usable answers, so it serves as a capability-floor control only.

## How to reproduce
Colab (GPU): open `notebooks/collapse_eval_colab.ipynb` on a T4. Run the gate cell first.
The runner flushes every batch to Drive and skips finished checkpoints on reconnect.

```bash
python scripts/verify_manifest.py
python scripts/run_collapse_eval.py --gate --device cuda --max-prompts 20 --generate-batch-size 2
python scripts/run_collapse_eval.py --only-lineage gemma3_4b_self_collapse --device cuda --generate-batch-size 2 --clean-cache
python scripts/run_collapse_eval.py --only-lineage gemma2_2b_kazdan --device cuda --generate-batch-size 2 --clean-cache
python scripts/make_figures.py
python scripts/export_human_eval.py
python scripts/analyze_agreement.py
```
