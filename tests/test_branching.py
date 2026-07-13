"""Orchestrator tests for parallel branches, budget allocation, event stream
and cross-run knowledge transfer - all with pure-Python scripts (no sklearn)."""

import io
import json

import pytest
from rich.console import Console

from ramanujan.events import read_events_since
from ramanujan.llm.base import LLMResponse, ToolCall
from ramanujan.llm.mock import MockLLM
from ramanujan.memory.knowledge import HashingEmbedder, KnowledgeBase
from ramanujan.orchestrator import ResearchDirector
from ramanujan.task import BudgetSpec, MetricSpec, TaskSpec


def script(value: float) -> str:
    return (
        "import json\n"
        f"json.dump({{'metric_name': 'score', 'metric_value': {value}}}, open('metrics.json', 'w'))\n"
    )


def _json(payload: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload))


def _plan(n: int) -> dict:
    return {"hypothesis": f"hypothesis {n}", "approach": f"approach {n}", "rationale": f"rationale {n}"}


def _analysis(text: str) -> dict:
    return {"insight": text, "hypothesis_supported": True, "suspicion": "", "next_directions": []}


def engineer_responses(value: float) -> list[LLMResponse]:
    return [
        LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": script(value)})]),
        LLMResponse(tool_calls=[ToolCall("run_script", {})]),
        LLMResponse(text=f"done, score {value}"),
    ]


def branching_task() -> TaskSpec:
    return TaskSpec(
        name="branchy-task",
        description="A toy task exercising parallel branches.",
        dataset="No real data; scripts fabricate a score.",
        metric=MetricSpec(name="score", goal=0.95),
        budget=BudgetSpec(
            max_iterations=2, parallel_branches=2, max_experiments=4,
            max_debug_attempts=2, experiment_timeout_seconds=30,
        ),
        eda=False,  # scripted responses in these tests cover the research loop only
    )


@pytest.fixture()
def branched_run(tmp_path):
    responses = [
        # round 1: planner proposes 2 candidates
        _json({"plans": [_plan(0), _plan(1)]}),
        # allocator funds both, candidate 1 first (priority order)
        _json({"selected_indices": [1, 0], "reasoning": "both credible; 1 first"}),
        # branch a = candidate 1
        *engineer_responses(0.80),
        _json(_analysis("insight for candidate 1")),
        # branch b = candidate 0
        *engineer_responses(0.90),
        _json(_analysis("insight for candidate 0")),
        # critic ends the round
        _json({"decision": "stop_diminishing_returns", "reasoning": "flat", "concerns": []}),
        # report conclusions
        LLMResponse(text="conclusions text"),
    ]
    kb = KnowledgeBase(tmp_path / "knowledge.db", embedder=HashingEmbedder())
    director = ResearchDirector(
        branching_task(), MockLLM(responses=responses), runs_root=tmp_path / "runs",
        console=Console(file=io.StringIO(), width=100), knowledge=kb,
    )
    return director.run(), kb


def test_branches_run_in_allocated_priority_order(branched_run):
    result, _ = branched_run
    assert result.stop_reason == "stop_diminishing_returns"
    records = {r.hypothesis: r for r in _ledger(result).all()}
    assert set(records) == {"hypothesis 1", "hypothesis 0"}
    # candidate 1 was funded first -> lower experiment id
    assert records["hypothesis 1"].id < records["hypothesis 0"].id
    assert result.best.metric_value == 0.90


def test_branch_workdirs_are_separate(branched_run):
    result, _ = branched_run
    assert (result.run_dir / "iter_01_a" / "train.py").exists()
    assert (result.run_dir / "iter_01_b" / "train.py").exists()


def test_event_stream_records_allocation(branched_run):
    result, _ = branched_run
    events = read_events_since(result.run_dir / "events.jsonl", 0)
    kinds = [e["kind"] for e in events]
    assert "run_started" in kinds and "run_finished" in kinds
    allocation = next(e for e in events if e["kind"] == "budget_allocated")
    assert allocation["payload"]["selected"] == [1, 0]
    assert kinds.count("plan_proposed") == 2
    assert kinds.count("experiment_result") == 2


def test_insights_stored_in_knowledge_base(branched_run):
    result, kb = branched_run
    assert kb.count() == 2  # both successful experiments contributed insights


def test_prior_knowledge_reaches_next_runs_planner(branched_run, tmp_path):
    _, kb = branched_run  # kb now holds insights from the first run
    responses = [
        _json(_plan(9)),
        *engineer_responses(0.5),
        _json(_analysis("insight 9")),
        _json({"decision": "stop_diminishing_returns", "reasoning": "enough", "concerns": []}),
        LLMResponse(text="conclusions"),
    ]
    single_task = branching_task().model_copy(
        update={"budget": BudgetSpec(max_iterations=1, parallel_branches=1,
                                     max_debug_attempts=2, experiment_timeout_seconds=30)}
    )
    llm = MockLLM(responses=responses)
    ResearchDirector(
        single_task, llm, runs_root=tmp_path / "runs2",
        console=Console(file=io.StringIO(), width=100), knowledge=kb,
    ).run()
    planner_prompt = llm.calls[0].messages[0].content
    assert "PRIOR KNOWLEDGE" in planner_prompt
    assert "insight for candidate" in planner_prompt


def _ledger(result):
    from ramanujan.memory.ledger import ExperimentLedger

    return ExperimentLedger(result.run_dir / "ledger.db")
