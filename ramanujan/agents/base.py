"""Agent primitives.

Two interaction styles, used deliberately for different jobs:

- `Agent.run()` - a real tool-use loop (model calls tools, sees results, decides
  next action) for open-ended work like writing and debugging code.
- `ask_json()`  - single-shot structured output validated against a Pydantic
  model with automatic repair retries, for decision nodes (plan/analyze/judge)
  where free-form agency adds failure modes but no value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.base import ChatMessage, LLMClient, LLMResponse, ToolSpec

ToolHandler = Callable[[dict], str]
EventHook = Callable[[str, str, str], None]  # (agent_name, event_kind, detail)

T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentStep:
    kind: str  # "tool_call" | "tool_result" | "text"
    detail: str


@dataclass
class AgentResult:
    final_text: str
    steps: list[AgentStep] = field(default_factory=list)
    exhausted: bool = False  # hit max_steps without a final text answer


class Agent:
    """A tool-using agent: system prompt + registered tools + bounded ReAct loop."""

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        system_prompt: str,
        tools: dict[str, tuple[ToolSpec, ToolHandler]],
        max_steps: int = 16,
        temperature: float = 0.4,
        on_event: EventHook | None = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_steps = max_steps
        self.temperature = temperature
        self.on_event = on_event or (lambda *_: None)

    def run(self, user_message: str) -> AgentResult:
        messages: list[ChatMessage] = [ChatMessage(role="user", content=user_message)]
        steps: list[AgentStep] = []
        tool_specs = [spec for spec, _ in self.tools.values()]

        for _ in range(self.max_steps):
            response: LLMResponse = self.llm.generate(
                system=self.system_prompt,
                messages=messages,
                tools=tool_specs,
                temperature=self.temperature,
            )
            if not response.tool_calls:
                text = response.text.strip()
                steps.append(AgentStep("text", text))
                self.on_event(self.name, "text", text)
                return AgentResult(final_text=text, steps=steps)

            messages.append(
                ChatMessage(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                summary = f"{call.name}({self._summarize_args(call.arguments)})"
                steps.append(AgentStep("tool_call", summary))
                self.on_event(self.name, "tool_call", summary)
                result_text = self._dispatch(call.name, call.arguments)
                steps.append(AgentStep("tool_result", result_text[:500]))
                self.on_event(self.name, "tool_result", result_text[:500])
                messages.append(
                    ChatMessage(
                        role="tool",
                        tool_name=call.name,
                        tool_call_id=call.id or call.name,
                        content=result_text,
                    )
                )

        return AgentResult(
            final_text="(agent hit its step limit without finishing)", steps=steps, exhausted=True
        )

    def _dispatch(self, name: str, arguments: dict) -> str:
        entry = self.tools.get(name)
        if entry is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.tools)}."
        _, handler = entry
        try:
            return handler(arguments)
        except Exception as exc:  # tool bugs become model-visible feedback, not crashes
            return f"ERROR while running tool '{name}': {exc}"

    @staticmethod
    def _summarize_args(arguments: dict) -> str:
        parts = []
        for key, value in arguments.items():
            text = str(value)
            parts.append(f"{key}={text[:60]!r}..." if len(text) > 60 else f"{key}={text!r}")
        return ", ".join(parts)


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply that may include prose or fences."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def ask_json(
    llm: LLMClient,
    *,
    system: str,
    prompt: str,
    model_cls: Type[T],
    max_repair_attempts: int = 2,
    temperature: float = 0.3,
) -> T:
    """Ask for a response conforming to `model_cls`; on bad output, feed the
    validation error back to the model and let it repair itself."""
    schema = json.dumps(model_cls.model_json_schema(), separators=(",", ":"))
    messages = [
        ChatMessage(
            role="user",
            content=(
                f"{prompt}\n\n"
                "Respond with ONLY a single JSON object (no prose, no markdown) "
                f"matching this JSON schema:\n{schema}"
            ),
        )
    ]
    last_error: Exception | None = None
    for _ in range(1 + max_repair_attempts):
        response = llm.generate(
            system=system, messages=messages, force_json=True, temperature=temperature
        )
        try:
            return model_cls.model_validate(json.loads(extract_json(response.text)))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append(ChatMessage(role="assistant", content=response.text))
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"That was not valid. Error:\n{exc}\n"
                    "Reply again with ONLY the corrected JSON object.",
                )
            )
    raise ValueError(f"Model failed to produce valid {model_cls.__name__}: {last_error}")
