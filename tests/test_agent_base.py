import pytest
from pydantic import BaseModel

from ramanujan.agents.base import Agent, ask_json, extract_json
from ramanujan.llm.base import LLMResponse, ToolCall, ToolSpec
from ramanujan.llm.mock import MockLLM

ECHO_SPEC = ToolSpec(
    name="echo",
    description="Echoes its input.",
    parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
)


def make_agent(llm, handler=None):
    return Agent(
        name="tester",
        llm=llm,
        system_prompt="You are a test agent.",
        tools={"echo": (ECHO_SPEC, handler or (lambda args: f"echo: {args['text']}"))},
        max_steps=5,
    )


def test_tool_loop_runs_tools_then_finishes():
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall(name="echo", arguments={"text": "hi"})]),
            LLMResponse(text="done"),
        ]
    )
    result = make_agent(llm).run("go")
    assert result.final_text == "done"
    assert not result.exhausted
    # tool result was fed back to the model as a tool message
    second_call_messages = llm.calls[1].messages
    assert any(m.role == "tool" and "echo: hi" in m.content for m in second_call_messages)


def test_unknown_tool_and_tool_exception_become_feedback():
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall(name="nope", arguments={})]),
            LLMResponse(tool_calls=[ToolCall(name="echo", arguments={"text": "x"})]),
            LLMResponse(text="recovered"),
        ]
    )

    def exploding(args):
        raise ValueError("kaboom")

    result = make_agent(llm, handler=exploding).run("go")
    assert result.final_text == "recovered"
    feedback = [m.content for call in llm.calls for m in call.messages if m.role == "tool"]
    assert any("unknown tool" in f for f in feedback)
    assert any("kaboom" in f for f in feedback)


def test_step_limit_marks_exhausted():
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall(name="echo", arguments={"text": "again"})])
        ] * 5
    )
    result = make_agent(llm).run("go")
    assert result.exhausted


class Weather(BaseModel):
    city: str
    temp_c: float


def test_ask_json_parses_valid_response():
    llm = MockLLM(responses=[LLMResponse(text='{"city": "Delhi", "temp_c": 31.5}')])
    out = ask_json(llm, system="s", prompt="p", model_cls=Weather)
    assert out.city == "Delhi"


def test_ask_json_repairs_invalid_response():
    llm = MockLLM(
        responses=[
            LLMResponse(text="Sure! Here you go: not json at all"),
            LLMResponse(text='```json\n{"city": "Pune", "temp_c": 28.0}\n```'),
        ]
    )
    out = ask_json(llm, system="s", prompt="p", model_cls=Weather)
    assert out.city == "Pune"
    # the repair turn must include the validation error
    assert any("not valid" in m.content.lower() for m in llm.calls[1].messages if m.role == "user")


def test_ask_json_gives_up_after_retries():
    llm = MockLLM(responses=[LLMResponse(text="garbage")] * 3)
    with pytest.raises(ValueError, match="Weather"):
        ask_json(llm, system="s", prompt="p", model_cls=Weather)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == '{"a": 1}'
    assert extract_json('prose ```json\n{"a": 1}\n``` more') == '{"a": 1}'
    assert extract_json('leading text {"a": {"b": 2}} trailing') == '{"a": {"b": 2}}'
