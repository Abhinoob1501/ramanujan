"""Gemini backend for the LLMClient protocol, using the official google-genai SDK.

Designed for the free tier:
- paces requests (min interval between calls) to respect requests-per-minute caps
- retries with exponential backoff on 429/5xx
- enforces a hard per-run call budget so a runaway loop cannot burn quota
"""

from __future__ import annotations

import json
import time

from ..config import Settings
from .base import ChatMessage, LLMBudgetExceeded, LLMResponse, ToolCall, ToolSpec

_RETRYABLE_MARKERS = ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE")


class GeminiClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        if not self.settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(free at https://aistudio.google.com/apikey), or run with --offline."
            )
        # Imported lazily so the rest of the package (tests, offline mode) works
        # without the SDK installed.
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        self._last_call_ts = 0.0
        self.calls_made = 0

    # ------------------------------------------------------------------ public

    def generate(
        self,
        *,
        system: str,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
        force_json: bool = False,
        temperature: float = 0.4,
    ) -> LLMResponse:
        if self.calls_made >= self.settings.max_llm_calls_per_run:
            raise LLMBudgetExceeded(
                f"LLM call budget of {self.settings.max_llm_calls_per_run} exhausted."
            )
        self._throttle()
        response = self._call_with_retry(
            contents=self._to_contents(messages),
            config=self._build_config(system, tools, force_json, temperature),
        )
        self.calls_made += 1
        return self._parse_response(response)

    # ---------------------------------------------------------------- internal

    def _throttle(self) -> None:
        wait = self.settings.min_seconds_between_llm_calls - (time.time() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    def _build_config(
        self, system: str, tools: list[ToolSpec] | None, force_json: bool, temperature: float
    ):
        types = self._genai.types
        kwargs: dict = {"system_instruction": system, "temperature": temperature}
        if tools:
            declarations = [
                types.FunctionDeclaration(
                    name=t.name, description=t.description, parameters=t.parameters
                )
                for t in tools
            ]
            kwargs["tools"] = [types.Tool(function_declarations=declarations)]
            # We run our own agent loop; disable the SDK's automatic calling.
            kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )
        elif force_json:
            kwargs["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**kwargs)

    def _to_contents(self, messages: list[ChatMessage]):
        types = self._genai.types
        contents = []
        for msg in messages:
            if msg.role == "assistant":
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                for call in msg.tool_calls:
                    parts.append(
                        types.Part.from_function_call(name=call.name, args=call.arguments)
                    )
                contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.tool_name, response={"result": msg.content}
                            )
                        ],
                    )
                )
            else:
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=msg.content)])
                )
        return contents

    def _call_with_retry(self, *, contents, config, max_attempts: int = 5):
        delay = 10.0
        for attempt in range(1, max_attempts + 1):
            try:
                return self._client.models.generate_content(
                    model=self.settings.gemini_model, contents=contents, config=config
                )
            except Exception as exc:  # SDK raises several error types; match on text
                message = str(exc)
                retryable = any(marker in message for marker in _RETRYABLE_MARKERS)
                if not retryable or attempt == max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("unreachable")

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if getattr(part, "function_call", None):
                    args = part.function_call.args or {}
                    if not isinstance(args, dict):  # some SDK versions return Struct
                        args = json.loads(json.dumps(dict(args)))
                    tool_calls.append(ToolCall(name=part.function_call.name, arguments=dict(args)))
                elif getattr(part, "text", None):
                    text_chunks.append(part.text)
        return LLMResponse(text="\n".join(text_chunks).strip(), tool_calls=tool_calls)
