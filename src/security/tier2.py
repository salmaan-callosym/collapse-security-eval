"""Tier-2 hooks: official published tools only.

No custom jailbreak or injection strings live in this repo.
Missing tools are recorded as skipped so Tier 1 can still stand alone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def run_tier2(
    cfg: dict[str, Any],
    lineage: str,
    seed: int,
    generation: int,
    checkpoint: str,
) -> dict[str, Any]:
    """Optional extra probes. Safe to call when tools are absent."""
    return {
        "jailbreakbench_behaviors": _jbb_behaviors_note(),
        "garak": _garak(cfg, checkpoint, lineage, seed, generation),
        "published_injection_file": _injection_file(cfg),
    }


def _jbb_behaviors_note() -> dict[str, Any]:
    # Behaviours-only: same causal wrapper as Tier 1, official JBB set (100 items).
    # Wired as a second prompt bank in eval_security.py when --tier2-jbb is set.
    return {
        "status": "optional",
        "dataset": "JailbreakBench/JBB-Behaviors",
        "note": "Pass --tier2-jbb to score the official 100 behaviours with the same wrapper/judge.",
    }


def load_jbb_behaviors() -> list[dict[str, str]]:
    """Official JailbreakBench harmful-behaviour list (not jailbreak prefixes)."""
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    rows = []
    for i, row in enumerate(ds):
        goal = row.get("Goal") or row.get("goal") or row.get("Behavior") or ""
        if not goal:
            continue
        rows.append(
            {
                "behavior_id": str(row.get("Index", i)),
                "behavior": goal,
                "functional_category": "jbb",
                "semantic_category": str(row.get("Category") or "jbb"),
                "context": "",
            }
        )
    return rows


def _garak(cfg: dict[str, Any], checkpoint: str, lineage: str, seed: int, generation: int) -> dict[str, Any]:
    probes = cfg["security"].get("garak_probes") or []
    if not probes:
        return {"status": "skipped", "reason": "security.garak_probes is empty"}
    exe = shutil.which("garak")
    if not exe:
        return {"status": "skipped", "reason": "garak CLI not on PATH"}

    out_dir = Path(cfg["paths"]["results_dir"]) / lineage / f"seed{seed}" / f"garak_gen{generation}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Official scanner only. Probe names come from config, not from strings we invent.
    cmd = [exe, "--model_type", "huggingface", "--model_name", checkpoint, "--probes", ",".join(probes)]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "cmd": cmd}
    (out_dir / "stdout.txt").write_text(proc.stdout)
    (out_dir / "stderr.txt").write_text(proc.stderr)
    return {"status": "ran", "returncode": proc.returncode, "out_dir": str(out_dir)}


def _injection_file(cfg: dict[str, Any]) -> dict[str, Any]:
    path = Path(cfg["paths"]["security_dir"]) / "injection_templates.json"
    if not path.is_file():
        return {
            "status": "skipped",
            "reason": "no published injection_templates.json (we do not ship attack strings)",
        }
    data = json.loads(path.read_text())
    return {"status": "present", "n": len(data) if isinstance(data, list) else 1, "path": str(path)}
