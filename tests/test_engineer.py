from ramanujan.agents.engineer import EngineerAgent
from ramanujan.agents.roles import ExperimentPlan
from ramanujan.executors.local import LocalExecutor
from ramanujan.llm.base import LLMResponse, ToolCall
from ramanujan.llm.mock import MockLLM

PLAN = ExperimentPlan(hypothesis="h", approach="a", rationale="r")

GOOD_SCRIPT = (
    "import json\n"
    "json.dump({'metric_name': 'score', 'metric_value': 0.95}, open('metrics.json', 'w'))\n"
)
BAD_SCRIPT = "raise ImportError('no such module')\n"
WRONG_METRIC_SCRIPT = (
    "import json\n"
    "json.dump({'metric_name': 'other', 'metric_value': 0.5}, open('metrics.json', 'w'))\n"
)


def make_engineer(llm, task, tmp_path):
    return EngineerAgent(llm, task, LocalExecutor(timeout_seconds=30), tmp_path / "iter_01")


def test_write_run_succeed(task, tmp_path):
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": GOOD_SCRIPT})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="built it; score 0.95"),
        ]
    )
    outcome = make_engineer(llm, task, tmp_path).implement(PLAN)
    assert outcome.success
    assert outcome.metrics["metric_value"] == 0.95
    assert outcome.code_path.endswith("train.py")


def test_debug_loop_recovers_from_failure(task, tmp_path):
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": BAD_SCRIPT})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": GOOD_SCRIPT})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="fixed the import and it works"),
        ]
    )
    outcome = make_engineer(llm, task, tmp_path).implement(PLAN)
    assert outcome.success
    # the failing traceback was surfaced to the model
    feedback = [m.content for call in llm.calls for m in call.messages if m.role == "tool"]
    assert any("no such module" in f for f in feedback)


def test_wrong_metric_key_is_rejected(task, tmp_path):
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": WRONG_METRIC_SCRIPT})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="gave up"),
        ]
    )
    outcome = make_engineer(llm, task, tmp_path).implement(PLAN)
    assert not outcome.success
    assert "metric_name" in outcome.error_summary


def test_debug_budget_is_enforced(task, tmp_path):
    # budget.max_debug_attempts=2 -> the 4th run_script call must be refused
    responses = []
    for _ in range(4):
        responses.append(LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": BAD_SCRIPT})]))
        responses.append(LLMResponse(tool_calls=[ToolCall("run_script", {})]))
    responses.append(LLMResponse(text="out of budget"))
    llm = MockLLM(responses=responses)
    outcome = make_engineer(llm, task, tmp_path).implement(PLAN)
    assert not outcome.success
    feedback = [m.content for call in llm.calls for m in call.messages if m.role == "tool"]
    assert any("debug budget exhausted" in f for f in feedback)


def test_prose_only_reply_gets_one_corrective_retry(task, tmp_path):
    # first conversation: model ignores its tools entirely; retry must recover
    llm = MockLLM(
        responses=[
            LLMResponse(text="Here is the code you asked for: ```python ...```"),
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": GOOD_SCRIPT})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="built it on the second attempt"),
        ]
    )
    outcome = make_engineer(llm, task, tmp_path).implement(PLAN)
    assert outcome.success
    retry_prompt = llm.calls[1].messages[0].content
    assert "used no tools" in retry_prompt


def test_no_retry_after_script_actually_ran(task, tmp_path):
    # the script ran (and failed); prose-only retry must NOT kick in
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": BAD_SCRIPT})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="could not fix it"),
        ]
    )
    outcome = make_engineer(llm, task, tmp_path).implement(PLAN)
    assert not outcome.success
    assert llm.responses == []  # exactly three responses consumed, no retry conversation


def test_path_escape_is_blocked(task, tmp_path):
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "../../evil.py", "content": "x = 1"})]),
            LLMResponse(text="done"),
            # script never ran -> the corrective retry fires one more conversation
            LLMResponse(text="still done"),
        ]
    )
    outcome = make_engineer(llm, task, tmp_path).implement(PLAN)
    assert not outcome.success
    assert not (tmp_path / "evil.py").exists()
    feedback = [m.content for call in llm.calls for m in call.messages if m.role == "tool"]
    assert any("escapes" in f for f in feedback)
