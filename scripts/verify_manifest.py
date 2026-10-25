#!/usr/bin/env python3
"""Verify every checkpoint we intend to evaluate and freeze a manifest.

This is step 1 of the collapse-security study (protocol_v2). It:
  * confirms the primary judge id exists on OpenRouter (dated snapshot, not -latest),
  * calls HfApi().model_info(id) for every hub repo and pins the commit sha,
  * reads adapter_config.json (LoRA lineages) to record the real base model,
  * checks whether each tokenizer ships a chat_template,
  * records the local OPT-125m lineage (pretrained + Sg gen0..3),
and writes research/results/collapse_eval/manifest.json.

No weights are downloaded here beyond tiny config/tokenizer json files.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import RESEARCH_ROOT, write_json

from src.collapse.secrets import load_env, token_status

PRIMARY_JUDGE = "deepseek/deepseek-v4.1-flash-1017"
SECONDARY_JUDGE = "strongreject_finetuned"

# Local OPT-125m Sg lineage (already collapsed), unzipped next to the repo root.
OPT_LOCAL_ROOT = (
    RESEARCH_ROOT.parent / "Sg-20261015T025644Z-1-001" / "Sg" / "seed0"
)


def _lineage_p1() -> list[dict]:
    base = "unsloth/gemma-3-4b-it"
    out = [
        {
            "lineage": "gemma3_4b_self_collapse",
            "priority": "P1",
            "generation": "base",
            "id": base,
            "base": base,
            "adapter": False,
        }
    ]
    for g in (0, 2, 4, 6, 8, 10, 12):
        out.append(
            {
                "lineage": "gemma3_4b_self_collapse",
                "priority": "P1",
                "generation": g,
                "id": f"HungryDino/gemma_3_4b_it-control_numbers-self_collapse_p10-gen{g}",
                "base": base,
                "adapter": True,
            }
        )
    return out


def _lineage_p2() -> list[dict]:
    out = [
        {
            "lineage": "gemma2_2b_kazdan",
            "priority": "P2",
            "generation": 1,
            "arm": "iter1",
            "id": "jkazdan/collapse_gemma-2-2b_hs2_iter1_sftsd0",
            "base": "google/gemma-2-2b",
            "adapter": False,
        }
    ]
    for it in (2, 3, 5, 10):
        out.append(
            {
                "lineage": "gemma2_2b_kazdan",
                "priority": "P2",
                "generation": it,
                "arm": "replace",
                "id": f"jkazdan/collapse_gemma-2-2b_hs2_replace_iter{it}_sftsd0",
                "base": "google/gemma-2-2b",
                "adapter": False,
            }
        )
    out.append(
        {
            "lineage": "gemma2_2b_kazdan",
            "priority": "P2",
            "generation": 10,
            "arm": "accumulate",
            "id": "jkazdan/collapse_gemma-2-2b_hs2_accumulate_iter10_sftsd0",
            "base": "google/gemma-2-2b",
            "adapter": False,
        }
    )
    return out


def _lineage_p3() -> list[dict]:
    base = "unsloth/Qwen2.5-7B-Instruct"
    out = [
        {
            "lineage": "qwen25_7b_self_collapse",
            "priority": "P3",
            "generation": "base",
            "id": base,
            "base": base,
            "adapter": False,
        }
    ]
    for g in (0, 2, 4, 6, 8, 10, 12, 14):
        out.append(
            {
                "lineage": "qwen25_7b_self_collapse",
                "priority": "P3",
                "generation": g,
                "id": f"HungryDino/qwen_2.5_7b-control_numbers-self_collapse_p10-gen{g}",
                "base": base,
                "adapter": True,
            }
        )
    return out


def _lineage_p4() -> list[dict]:
    out = [
        {
            "lineage": "opt125m",
            "priority": "P4",
            "generation": "base",
            "id": "facebook/opt-125m",
            "base": "facebook/opt-125m",
            "adapter": False,
            "local_path": None,
        }
    ]
    for g in (0, 1, 2, 3):
        out.append(
            {
                "lineage": "opt125m",
                "priority": "P4",
                "generation": g,
                "id": f"local:Sg/seed0/gen{g}",
                "base": "facebook/opt-125m",
                "adapter": False,
                "local_path": str(OPT_LOCAL_ROOT / f"gen{g}"),
            }
        )
    return out


def check_openrouter_judge(model_id: str) -> dict:
    """Confirm the judge id is present in OpenRouter's model catalogue."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    ids = {m.get("id") for m in data.get("data", [])}
    present = model_id in ids
    flash_matches = sorted(i for i in ids if i and "deepseek" in i and "flash" in i)
    return {
        "judge_id": model_id,
        "present": present,
        "n_models_listed": len(ids),
        "deepseek_flash_ids": flash_matches,
    }


def verify_hub_entry(entry: dict, api, hf_hub_download) -> dict:
    """Pin sha, read adapter base, detect chat_template. Mutates a copy."""
    rec = dict(entry)
    model_id = entry["id"]
    if model_id.startswith("local:"):
        local = Path(entry["local_path"])
        rec["exists"] = local.is_dir()
        rec["commit_sha"] = None
        rec["note"] = "local checkpoint" if rec["exists"] else "MISSING local path"
        rec["chat_template"] = _local_chat_template(local) if rec["exists"] else None
        return rec

    try:
        info = api.model_info(model_id, files_metadata=False)
    except Exception as exc:  # noqa: BLE001
        rec["exists"] = False
        rec["commit_sha"] = None
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return rec

    rec["exists"] = True
    rec["commit_sha"] = info.sha
    siblings = {s.rfilename for s in (info.siblings or [])}
    rec["has_adapter_config"] = "adapter_config.json" in siblings

    # LoRA lineages: confirm the recorded base matches base_model_name_or_path.
    if entry.get("adapter"):
        try:
            cfg_path = hf_hub_download(
                model_id, "adapter_config.json", revision=info.sha
            )
            acfg = json.loads(Path(cfg_path).read_text())
            rec["adapter_base_model"] = acfg.get("base_model_name_or_path")
            rec["peft_type"] = acfg.get("peft_type")
            rec["base_matches_adapter"] = (
                rec["adapter_base_model"] == entry["base"]
            )
        except Exception as exc:  # noqa: BLE001
            rec["adapter_config_error"] = str(exc)[:200]

    # chat_template lives in tokenizer_config.json / chat_template.jinja.
    rec["chat_template"] = _hub_chat_template(
        model_id, info.sha, siblings, hf_hub_download
    )
    return rec


def _hub_chat_template(model_id, sha, siblings, hf_hub_download) -> bool | None:
    try:
        if "chat_template.jinja" in siblings:
            return True
        if "tokenizer_config.json" in siblings:
            p = hf_hub_download(model_id, "tokenizer_config.json", revision=sha)
            tcfg = json.loads(Path(p).read_text())
            return bool(tcfg.get("chat_template"))
    except Exception:  # noqa: BLE001
        return None
    return None


def _local_chat_template(local: Path) -> bool | None:
    tcfg = local / "tokenizer_config.json"
    if not tcfg.is_file():
        return None
    try:
        return bool(json.loads(tcfg.read_text()).get("chat_template"))
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    load_env()
    status = token_status()
    print("tokens:", status)

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()

    print(f"checking primary judge on OpenRouter: {PRIMARY_JUDGE}")
    judge_check = check_openrouter_judge(PRIMARY_JUDGE)
    print(json.dumps(judge_check, indent=2))
    if not judge_check["present"]:
        raise SystemExit(
            f"ABORT: judge id {PRIMARY_JUDGE} not found on OpenRouter. "
            f"deepseek flash ids seen: {judge_check['deepseek_flash_ids']}"
        )

    entries = _lineage_p1() + _lineage_p2() + _lineage_p3() + _lineage_p4()
    verified = []
    missing = []
    for entry in entries:
        rec = verify_hub_entry(entry, api, hf_hub_download)
        tag = rec["id"]
        if rec.get("exists"):
            extra = ""
            if rec.get("adapter"):
                extra = (
                    f" base={rec.get('adapter_base_model')} "
                    f"match={rec.get('base_matches_adapter')}"
                )
            print(f"OK   {rec['priority']:3} {tag}  sha={rec.get('commit_sha')}{extra}")
        else:
            missing.append(tag)
            print(f"MISS {rec['priority']:3} {tag}  {rec.get('error') or rec.get('note')}")
        verified.append(rec)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "protocol_v2",
        "primary_judge": PRIMARY_JUDGE,
        "secondary_judge": SECONDARY_JUDGE,
        "openrouter_judge_check": judge_check,
        "tokens_present": status,
        "n_entries": len(verified),
        "n_missing": len(missing),
        "missing_ids": missing,
        "checkpoints": verified,
    }
    out = RESEARCH_ROOT / "results" / "collapse_eval" / "manifest.json"
    write_json(out, manifest)
    print(f"\nwrote {out}  ({len(verified)} entries, {len(missing)} missing)")
    if missing:
        print("WARNING: some ids are missing; fix before launching those lineages.")


if __name__ == "__main__":
    main()
