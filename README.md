# Security under model collapse (protocol_v2)

Does refusal of harmful requests get worse as a model collapses through recursive self-training?

This folder is the Colab upload. It does not train anything. It evaluates public checkpoints (plus the already-finished local OPT-125m results) with one frozen protocol: 100 HarmBench prompts, greedy decoding, OpenRouter `deepseek/deepseek-v4.1-flash-1017`, WikiText-2 perplexity, and benign diversity.

## Upload to Colab

1. Zip this folder (skip `.git` and `.venv`). In Drive, unzip it so you have `MyDrive/collapse_eval/` containing `scripts/`, `src/`, `notebooks/`, `results/`.
2. Colab → Runtime → Change runtime type → **T4 GPU**. Do not pick A100.
3. Add secrets (key icon): `OPENROUTER_API_KEY`, `HF_TOKEN` or `HUGGING_FACE_API_KEY`. Optional: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.
4. Open `notebooks/collapse_eval_colab.ipynb` and run the setup cell, then the **gate** cell only.

The gate loads Gemma-3-4B base, gen0, and gen12 (20 prompts + perplexity). Read those numbers before running P1.

## If Colab disconnects

Drive is the copy that matters. Completions and scores are rewritten after every batch. Perplexity is saved as `utility.partial.json` before the benign generations start. `results/collapse_eval/progress.json` lists what finished.

Reconnect, remount Drive, rerun the **same** cell. Finished generations are skipped. A half-finished generation continues from the last saved prompt. A 20-prompt gate run is extended to 100, not treated as done.

Leave the browser tab open while a cell is running. Colab Pro does not keep running after you close the tab.

## What is already in this folder

`results/collapse_eval/opt125m/` is the finished OPT-125m negative control (protocol_v2, same judge). Do not rerun it on Colab.

## After the GPU

Copy `results/collapse_eval/` back and, on a laptop:

```bash
python scripts/make_figures.py
python scripts/export_human_eval.py
```
