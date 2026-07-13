import json

import pytest

from ramanujan.config import Settings
from ramanujan.llm.base import ChatMessage, ToolCall, ToolSpec
from ramanujan.llm.openai_compat import OpenAICompatClient

FAST = Settings(min_seconds_between_llm_calls=0, max_llm_calls_per_run=100)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def make_client(responses: list[FakeResponse]):
    client = OpenAICompatClient(
        base_url="https://example.test/v1", api_key="k", model="test-model", settings=FAST
    )
    sent_payloads: list[dict] = []

    def fake_post(payload):
        sent_payloads.append(json.loads(json.dumps(payload)))  # deep copy
        return responses.pop(0)

    client._post = fake_post
    return client, sent_payloads


def completion(content=None, tool_calls=None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def test_message_conversion_full_round():
    client, sent = make_client([FakeResponse(200, completion(content="ok"))])
    result = client.generate(
        system="be helpful",
        messages=[
            ChatMessage(role="user", content="hi"),
            ChatMessage(
                role="assistant",
                content="calling a tool",
                tool_calls=[ToolCall(name="echo", arguments={"text": "x"}, id="call_1")],
            ),
            ChatMessage(role="tool", tool_name="echo", tool_call_id="call_1", content="echo: x"),
        ],
        tools=[ToolSpec(name="echo", description="d", parameters={"type": "object", "properties": {}})],
    )
    assert result.text == "ok"
    messages = sent[0]["messages"]
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[2]["tool_calls"][0]["function"]["name"] == "echo"
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"text": "x"}
    assert messages[3] == {"role": "tool", "tool_call_id": "call_1", "content": "echo: x"}
    assert sent[0]["tools"][0]["function"]["name"] == "echo"


def test_tool_call_parsing_with_string_arguments():
    client, _ = make_client(
        [
            FakeResponse(
                200,
                completion(
                    tool_calls=[
                        {
                            "id": "abc",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": '{"filename": "t.py"}'},
                        }
                    ]
                ),
            )
        ]
    )
    result = client.generate(system="s", messages=[ChatMessage(role="user", content="go")])
    assert result.tool_calls[0].name == "write_file"
    assert result.tool_calls[0].arguments == {"filename": "t.py"}
    assert result.tool_calls[0].id == "abc"


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    client, sent = make_client(
        [FakeResponse(429, text="rate limited"), FakeResponse(200, completion(content="fine"))]
    )
    result = client.generate(system="s", messages=[ChatMessage(role="user", content="go")])
    assert result.text == "fine"
    assert len(sent) == 2


def test_response_format_dropped_on_400():
    client, sent = make_client(
        [FakeResponse(400, text="response_format not supported"), FakeResponse(200, completion(content="{}"))]
    )
    client.generate(
        system="s", messages=[ChatMessage(role="user", content="go")], force_json=True
    )
    assert "response_format" in sent[0]
    assert "response_format" not in sent[1]


def test_non_retryable_error_raises():
    client, _ = make_client([FakeResponse(401, text="bad key")])
    with pytest.raises(RuntimeError, match="401"):
        client.generate(system="s", messages=[ChatMessage(role="user", content="go")])
