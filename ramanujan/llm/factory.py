"""Provider selection.

Pick explicitly (`--provider` / RAMANUJAN_PROVIDER) or let the factory
auto-detect from whichever API key is present in the environment / .env:

  GEMINI_API_KEY                          -> gemini
  OPENROUTER_API_KEY                      -> openrouter   (openrouter.ai)
  OPENCODE_API_KEY / OPENCODE_ZEN_API_KEY -> opencode     (OpenCode Zen gateway)
  RAMANUJAN_API_KEY + RAMANUJAN_BASE_URL  -> custom       (any OpenAI-compatible server)

RAMANUJAN_MODEL and RAMANUJAN_BASE_URL override any preset. Pick models that
support native function calling - the Engineer agent depends on it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..config import Settings, load_dotenv
from .base import LLMClient

PROVIDER_ALIASES = {
    "opencode-zen": "opencode",
    "opencodezen": "opencode",
    "zen": "opencode",
    "google": "gemini",
}


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    key_envs: tuple[str, ...]
    default_model: str
    extra_headers: dict = field(default_factory=dict)


PRESETS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        base_url="https://openrouter.ai/api/v1",
        key_envs=("OPENROUTER_API_KEY",),
        default_model="deepseek/deepseek-chat-v3.1:free",
        # OpenRouter uses these to attribute traffic; values are cosmetic.
        extra_headers={"HTTP-Referer": "https://github.com/ramanujan-agent", "X-Title": "Ramanujan"},
    ),
    "opencode": ProviderPreset(
        base_url="https://opencode.ai/zen/v1",
        key_envs=("OPENCODE_API_KEY", "OPENCODE_ZEN_API_KEY"),
        default_model="grok-code",
    ),
    "custom": ProviderPreset(
        base_url="",  # must come from RAMANUJAN_BASE_URL
        key_envs=("RAMANUJAN_API_KEY",),
        default_model="",  # must come from RAMANUJAN_MODEL
    ),
}

_NO_KEY_HELP = (
    "No LLM API key found. Set one of:\n"
    "  GEMINI_API_KEY        (free at https://aistudio.google.com/apikey)\n"
    "  OPENROUTER_API_KEY    (https://openrouter.ai/keys)\n"
    "  OPENCODE_API_KEY      (OpenCode Zen: https://opencode.ai/zen)\n"
    "  RAMANUJAN_API_KEY + RAMANUJAN_BASE_URL (any OpenAI-compatible server)\n"
    "in the environment or a .env file - or run with --offline."
)


def build_llm(provider: str | None = None, settings: Settings | None = None) -> LLMClient:
    load_dotenv()
    settings = settings or Settings.from_env()
    name = (provider or os.environ.get("RAMANUJAN_PROVIDER", "")).strip().lower()
    name = PROVIDER_ALIASES.get(name, name) or _detect_provider()

    if name == "gemini":
        from .gemini import GeminiClient

        return GeminiClient(settings)

    preset = PRESETS.get(name)
    if preset is None:
        known = ", ".join(["gemini", *PRESETS])
        raise RuntimeError(f"Unknown LLM provider '{name}'. Known providers: {known}.")

    base_url = os.environ.get("RAMANUJAN_BASE_URL", "") or preset.base_url
    model = os.environ.get("RAMANUJAN_MODEL", "") or preset.default_model
    api_key = next((os.environ[e] for e in preset.key_envs if os.environ.get(e)), "")
    if not api_key:
        raise RuntimeError(
            f"Provider '{name}' selected but none of {', '.join(preset.key_envs)} is set.\n"
            + _NO_KEY_HELP
        )
    if not base_url:
        raise RuntimeError(f"Provider '{name}' needs RAMANUJAN_BASE_URL to be set.")
    if not model:
        raise RuntimeError(f"Provider '{name}' needs RAMANUJAN_MODEL to be set.")

    from .openai_compat import OpenAICompatClient

    return OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        extra_headers=preset.extra_headers,
        settings=settings,
    )


def _detect_provider() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_ZEN_API_KEY"):
        return "opencode"
    if os.environ.get("RAMANUJAN_API_KEY") and os.environ.get("RAMANUJAN_BASE_URL"):
        return "custom"
    raise RuntimeError(_NO_KEY_HELP)
