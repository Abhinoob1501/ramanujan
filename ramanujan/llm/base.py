"""Provider-agnostic LLM interface.

Every backend (Gemini, mocks, future providers) implements `LLMClient`, so the
agents never import a vendor SDK. Messages use a small internal schema instead
of any provider's wire format.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolSpec:
    """A tool the model may call, described with a JSON-schema parameter block."""

    name: str
    description: str
    parameters: dict  # JSON schema: {"type": "object", "properties": {...}, "required": [...]}


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ChatMessage:
    """One turn of a conversation.

    role: "user" | "assistant" | "tool"
    - assistant turns may carry tool_calls
    - tool turns carry the result of one tool call (tool_name + content)
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str = ""
    tool_call_id: str = ""  # OpenAI-style APIs require echoing the id of the call being answered


class LLMBudgetExceeded(RuntimeError):
    """Raised when a run hits its maximum number of LLM calls."""


@dataclass
class LLMUsage:
    """Token/cost telemetry accumulated by a client across a run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # only providers that report cost (e.g. OpenRouter) fill this

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd

    def merge(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            calls=self.calls + other.calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class Pacer:
    """Spaces out requests. One instance can be SHARED by several clients so
    per-role models still respect a single provider-wide rate limit."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = min_interval_seconds
        self._last_call_ts = 0.0

    def wait(self) -> None:
        pause = self.min_interval_seconds - (time.time() - self._last_call_ts)
        if pause > 0:
            time.sleep(pause)
        self._last_call_ts = time.time()


class LLMClient(Protocol):
    def generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        force_json: bool = False,
        temperature: float = 0.4,
    ) -> LLMResponse: ...
