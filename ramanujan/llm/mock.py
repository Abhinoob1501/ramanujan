"""Scripted LLM backend.

Serves a fixed queue of responses in order. Used two ways:
- unit tests assert on agent behavior without network access
- `--offline` demo mode replays a realistic canned research session so the whole
  system (orchestration, sandboxed execution, ledger, report) runs with no API key
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import ChatMessage, LLMResponse, LLMUsage, ToolSpec


@dataclass
class RecordedCall:
    system: str
    messages: list[ChatMessage]
    tools: list[ToolSpec] | None
    force_json: bool


@dataclass
class MockLLM:
    responses: list[LLMResponse] = field(default_factory=list)
    calls: list[RecordedCall] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)

    def generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        force_json: bool = False,
        temperature: float = 0.4,
    ) -> LLMResponse:
        self.calls.append(
            RecordedCall(system=system, messages=list(messages), tools=tools, force_json=force_json)
        )
        if not self.responses:
            raise AssertionError(
                "MockLLM ran out of scripted responses. "
                f"Last system prompt began: {system[:120]!r}"
            )
        return self.responses.pop(0)
