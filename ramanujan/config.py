"""Runtime configuration, loaded from environment variables and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines, # comments). No external dependency."""
    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    # Free-tier Gemini is rate limited (~10 requests/min on flash); the client
    # spaces calls out so long research runs don't die on 429s.
    min_seconds_between_llm_calls: float = 6.0
    max_llm_calls_per_run: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_MODEL", cls.gemini_model),
            min_seconds_between_llm_calls=float(
                os.environ.get("RAMANUJAN_MIN_SECONDS_BETWEEN_LLM_CALLS", cls.min_seconds_between_llm_calls)
            ),
            max_llm_calls_per_run=int(
                os.environ.get("RAMANUJAN_MAX_LLM_CALLS_PER_RUN", cls.max_llm_calls_per_run)
            ),
        )
