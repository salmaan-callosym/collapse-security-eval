"""Load local .env files without printing values.

Maps the names in this project's .env onto what Hugging Face / optional APIs expect.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import RESEARCH_ROOT

# Parent of research/ (this repo) often holds the user's .env.
_ENV_CANDIDATES = (
    RESEARCH_ROOT / ".env",
    RESEARCH_ROOT.parent / ".env",
)


def load_env() -> list[str]:
    """Read .env if present. Returns which files were loaded (paths only)."""
    loaded: list[str] = []
    for path in _ENV_CANDIDATES:
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        loaded.append(str(path))
    _alias_known_keys()
    return loaded


def _alias_known_keys() -> None:
    hf = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_API_KEY")
    )
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf)

    # Rubric judge via OpenRouter only if someone switches judge later.
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        os.environ.setdefault("OPENAI_API_KEY", or_key)
        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


def token_status() -> dict[str, bool]:
    """Which tokens are present — booleans only, never the secrets."""
    return {
        "hf_token": bool(os.environ.get("HF_TOKEN")),
        "openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY")),
        "langfuse": bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
            and (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST"))
        ),
    }
