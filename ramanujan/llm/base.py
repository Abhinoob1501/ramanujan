"""Provider-agnostic LLM interface.

Every backend (Gemini, mocks, future providers) implements `LLMClient`, so the
agents never import a vendor SDK. Messages use a small internal schema instead
of any provider's wire format.
"""

from __future__ import annotations

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
