"""OpenAI-compatible chat-completions backend.

One implementation covers every provider that speaks the OpenAI dialect:
OpenRouter, OpenCode Zen, Groq, Together, Ollama, vLLM, ... - they differ only
in base URL, API key and model name (see llm.factory for the presets).

Same free-tier discipline as the Gemini backend: request pacing, exponential
backoff on 429/5xx, and a hard per-run call budget.
"""

from __future__ import annotations

import json
import time

import requests

from ..config import Settings
from .base import ChatMessage, LLMBudgetExceeded, LLMResponse, ToolCall, ToolSpec

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        extra_headers: dict | None = None,
        settings: Settings | None = None,
        request_timeout: int = 300,
    ):
        if not api_key:
            raise RuntimeError("No API key provided for the OpenAI-compatible backend.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.extra_headers = extra_headers or {}
        self.settings = settings or Settings.from_env()
        self.request_timeout = request_timeout
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

        payload: dict = {
            "model": self.model,
            "messages": self._to_messages(system, messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        elif force_json:
            # Not every model behind OpenRouter/Zen supports response_format;
            # _call_with_retry silently drops it on a 400 and retries once.
            payload["response_format"] = {"type": "json_object"}

        data = self._call_with_retry(payload)
        self.calls_made += 1
        return self._parse_response(data)

    # ---------------------------------------------------------------- internal

    def _throttle(self) -> None:
        wait = self.settings.min_seconds_between_llm_calls - (time.time() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    @staticmethod
    def _to_messages(system: str, messages: list[ChatMessage]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for msg in messages:
            if msg.role == "assistant":
                entry: dict = {"role": "assistant", "content": msg.content or None}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id or f"call_{i}_{call.name}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for i, call in enumerate(msg.tool_calls)
                    ]
                out.append(entry)
            elif msg.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or msg.tool_name,
                        "content": msg.content,
                    }
                )
            else:
                out.append({"role": "user", "content": msg.content})
        return out

    def _post(self, payload: dict) -> requests.Response:
        return requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            },
            json=payload,
            timeout=self.request_timeout,
        )

    def _call_with_retry(self, payload: dict, max_attempts: int = 5) -> dict:
        delay = 10.0
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._post(payload)
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                response = None
            if response is not None:
                if response.status_code == 200:
                    return response.json()
                # some providers reject response_format; degrade gracefully
                if response.status_code == 400 and "response_format" in payload:
                    payload = {k: v for k, v in payload.items() if k != "response_format"}
                    continue
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    raise RuntimeError(f"LLM request failed ({last_error})")
            if attempt < max_attempts:
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError(f"LLM request failed after {max_attempts} attempts ({last_error})")

    @staticmethod
    def _parse_response(data: dict) -> LLMResponse:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Malformed LLM response: {str(data)[:500]}") from exc
        tool_calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                arguments = {"_malformed_arguments": raw_args}
            tool_calls.append(
                ToolCall(name=function.get("name", ""), arguments=arguments, id=call.get("id", ""))
            )
        return LLMResponse(text=(message.get("content") or "").strip(), tool_calls=tool_calls)
