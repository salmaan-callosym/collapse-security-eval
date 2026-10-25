"""Load experiment settings from YAML.

Config is a plain dict so callers stay explicit and tests stay easy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os

import yaml

# research/ is the working root for relative paths in default.yaml
RESEARCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = RESEARCH_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    # Resolve relative data paths against the research root once.
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        p = Path(value)
        if not p.is_absolute():
            paths[key] = str(RESEARCH_ROOT / p)
    # Colab sets HF_HOME on the VM disk. Never let the yaml path
    # (which may sit on Drive) become the weight cache.
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        paths["hf_cache_dir"] = hf_home
    return cfg


def resolve_device(requested: str = "auto") -> str:
    """Pick a device without pulling training code into scripts."""
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
