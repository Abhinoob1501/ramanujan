import json

from ramanujan.agents.eda import EdaAgent, EdaFindings
from ramanujan.executors.local import LocalExecutor
from ramanujan.llm.base import LLMResponse, ToolCall
from ramanujan.llm.mock import MockLLM
from ramanujan.task import MetricSpec, TaskSpec

EDA_SCRIPT = (
    "print('=== SHAPE ===')\n"
    "print('100 samples, 4 features')\n"
    "print('=== TARGET BALANCE ===')\n"
    "print('55/45 split')\n"
)

FINDINGS_JSON = json.dumps(
    {
        "summary": "Small balanced tabular dataset.",
        "key_findings": ["4 features, 100 samples", "balanced classes"],
        "data_quality_issues": [],
        "leakage_risks": ["feature x4 nearly duplicates the target"],
        "modeling_recommendations": ["start with a linear baseline"],
    }
)


def make_task(**overrides) -> TaskSpec:
    base = dict(
        name="toy", description="d", dataset="synthetic",
        metric=MetricSpec(name="score", goal=0.9),
    )
    base.update(overrides)
    return TaskSpec.model_validate(base)


def test_eda_explores_and_distills(tmp_path):
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "eda.py", "content": EDA_SCRIPT})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="dataset is small and balanced"),
            LLMResponse(text=FINDINGS_JSON),  # distillation
        ]
    )
    agent = EdaAgent(llm, make_task(), tmp_path / "eda")
    outcome = agent.explore()
    assert outcome.success
    assert outcome.findings.leakage_risks == ["feature x4 nearly duplicates the target"]
    # findings persisted for the run directory
    saved = json.loads((tmp_path / "eda" / "findings.json").read_text(encoding="utf-8"))
    assert saved["summary"] == "Small balanced tabular dataset."
    # the distillation call saw the real script output
    distill_prompt = llm.calls[-1].messages[0].content
    assert "=== SHAPE ===" in distill_prompt


def test_eda_script_needs_no_metrics_json(tmp_path):
    (tmp_path / "eda.py").write_text("print('hello findings')", encoding="utf-8")
    result = LocalExecutor(timeout_seconds=30, require_metrics=False).run(tmp_path, "eda.py")
    assert result.ok
    assert "hello findings" in result.stdout_tail


def test_eda_failure_is_contained(tmp_path):
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "eda.py", "content": "raise ValueError('broken')\n"})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="could not explore"),
        ]
    )
    outcome = EdaAgent(llm, make_task(), tmp_path / "eda").explore()
    assert not outcome.success
    assert "never ran successfully" in outcome.error


def test_findings_prompt_block_renders_sections():
    findings = EdaFindings(
        summary="s",
        key_findings=["a"],
        leakage_risks=["column leak_col predicts target"],
        modeling_recommendations=["standardize"],
    )
    block = findings.to_prompt_block()
    assert "EDA FINDINGS" in block
    assert "LEAKAGE RISKS" in block and "leak_col" in block
    assert "- standardize" in block
