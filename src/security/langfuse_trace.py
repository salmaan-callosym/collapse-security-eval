"""Optional Langfuse tracing for Tier-1 security evals.

Import only *after* `load_env()` so credentials are present.
Never logs API keys. HarmBench behaviours are research artefacts and are
traced deliberately (masked only for secrets).
"""

from __future__ import annotations

import os
from typing import Any

_client = None
_enabled: bool | None = None


def langfuse_configured() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
        and (os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST"))
    )


def get_langfuse():
    """Lazy singleton. Returns None if keys missing or package absent."""
    global _client, _enabled
    if _enabled is False:
        return None
    if not langfuse_configured():
        _enabled = False
        return None
    if _client is not None:
        return _client
    try:
        from langfuse import get_client

        # Prefer BASE_URL; CLI docs also accept HOST.
        if os.environ.get("LANGFUSE_BASE_URL") and not os.environ.get("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = os.environ["LANGFUSE_BASE_URL"]
        _client = get_client()
        _enabled = True
        return _client
    except Exception as exc:
        print(f"langfuse: disabled ({exc})")
        _enabled = False
        return None


def flush_langfuse() -> None:
    lf = get_langfuse()
    if lf is not None:
        lf.flush()


def eval_session_id(lineage: str, seed: int, generation: int) -> str:
    return f"security-{lineage}-seed{seed}-gen{generation}"


def score_strongreject_on_obs(observation, score: float | None, *, comment: str = "") -> None:
    if observation is None or score is None:
        return
    try:
        kwargs = {"name": "strongreject", "value": float(score)}
        if comment:
            kwargs["comment"] = comment
        observation.score(**kwargs)
    except Exception:
        pass
