"""Human-in-the-loop gate tests: scripted gates drive the real orchestrator."""

import io
import json

import pytest
from rich.console import Console

from ramanujan.hitl import AutoGate, ConsoleGate, PlanReview, VerdictReview
from ramanujan.llm.base import LLMResponse, ToolCall
from ramanujan.llm.mock import MockLLM
from ramanujan.orchestrator import ResearchDirector
from ramanujan.task import BudgetSpec, MetricSpec, TaskSpec


def make_task() -> TaskSpec:
    return TaskSpec(
        name="hitl-task",
        description="toy",
        dataset="none",
        metric=MetricSpec(name="score", goal=0.9),
        budget=BudgetSpec(max_iterations=2, max_debug_attempts=1, experiment_timeout_seconds=30),
        eda=False,
    )


def _json(payload: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(payload))


def plan_json(n: int) -> LLMResponse:
    return _json({"hypothesis": f"hypothesis {n}", "approach": f"approach {n}", "rationale": "r"})


def engineer_responses(value: float) -> list[LLMResponse]:
    script = (
        "import json\n"
        f"json.dump({{'metric_name': 'score', 'metric_value': {value}}}, open('metrics.json', 'w'))\n"
    )
    return [
        LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": script})]),
        LLMResponse(tool_calls=[ToolCall("run_script", {})]),
        LLMResponse(text="done"),
    ]


def analysis_json() -> LLMResponse:
    return _json({"insight": "i", "hypothesis_supported": True, "suspicion": "", "next_directions": []})


def verdict_json(decision: str) -> LLMResponse:
    return _json({"decision": decision, "reasoning": "r", "concerns": []})


class ScriptedGate:
    def __init__(self, plan_reviews: list[PlanReview], verdict_reviews: list[VerdictReview]):
        self.plan_reviews = plan_reviews
        self.verdict_reviews = verdict_reviews
        self.seen_plans: list[list] = []

    def review_plans(self, plans, iteration):
        self.seen_plans.append(list(plans))
        return self.plan_reviews.pop(0) if self.plan_reviews else PlanReview("approve")

    def review_verdict(self, verdict, iteration):
        return self.verdict_reviews.pop(0) if self.verdict_reviews else VerdictReview("accept")


def run_director(responses, gate, tmp_path):
    return ResearchDirector(
        make_task(), MockLLM(responses=responses), runs_root=tmp_path / "runs",
        console=Console(file=io.StringIO(), width=100), gate=gate,
    ).run()


def test_guidance_forces_replan_and_persists(tmp_path):
    responses = [
        plan_json(1),               # round 1: initial plan
        plan_json(2),               # round 1: re-plan after guidance
        *engineer_responses(0.5),
        analysis_json(),
        verdict_json("continue"),
        plan_json(3),               # round 2
        *engineer_responses(0.6),
        analysis_json(),
        verdict_json("stop_diminishing_returns"),
        LLMResponse(text="conclusions"),
    ]
    gate = ScriptedGate(
        plan_reviews=[PlanReview("revise", guidance="use tree models only"), PlanReview("approve")],
        verdict_reviews=[],
    )
    llm_responses = list(responses)
    result = run_director(llm_responses, gate, tmp_path)

    assert result.stop_reason == "stop_diminishing_returns"
    # the re-planned hypothesis (2), not the first one, was executed
    from ramanujan.memory.ledger import ExperimentLedger

    records = ExperimentLedger(result.run_dir / "ledger.db").all()
    assert records[0].hypothesis == "hypothesis 2"


def test_guidance_text_reaches_planner_prompt(tmp_path):
    llm = MockLLM(
        responses=[
            plan_json(1),
            plan_json(2),
            *engineer_responses(0.5),
            analysis_json(),
            verdict_json("stop_diminishing_returns"),
            LLMResponse(text="conclusions"),
        ]
    )
    gate = ScriptedGate([PlanReview("revise", guidance="use tree models only")], [])
    ResearchDirector(
        make_task(), llm, runs_root=tmp_path / "runs",
        console=Console(file=io.StringIO(), width=100), gate=gate,
    ).run()
    replan_prompt = llm.calls[1].messages[0].content
    assert "HUMAN GUIDANCE" in replan_prompt
    assert "use tree models only" in replan_prompt


def test_stop_at_plan_gate_runs_nothing(tmp_path):
    responses = [plan_json(1), LLMResponse(text="conclusions")]
    gate = ScriptedGate([PlanReview("stop")], [])
    result = run_director(responses, gate, tmp_path)
    assert result.stop_reason == "stopped_by_human"
    assert result.best is None


def test_human_overrides_critic_stop(tmp_path):
    responses = [
        plan_json(1),
        *engineer_responses(0.95),
        analysis_json(),
        verdict_json("stop_goal_met"),   # critic wants to stop after round 1
        plan_json(2),                    # human forces round 2
        *engineer_responses(0.96),
        analysis_json(),
        verdict_json("stop_goal_met"),
        LLMResponse(text="conclusions"),
    ]
    gate = ScriptedGate([], [VerdictReview("continue_anyway"), VerdictReview("accept")])
    result = run_director(responses, gate, tmp_path)
    assert result.iterations_run == 2
    assert result.best.metric_value == 0.96


def test_human_stops_despite_critic_continue(tmp_path):
    responses = [
        plan_json(1),
        *engineer_responses(0.5),
        analysis_json(),
        verdict_json("continue"),
        LLMResponse(text="conclusions"),
    ]
    gate = ScriptedGate([], [VerdictReview("stop_now")])
    result = run_director(responses, gate, tmp_path)
    assert result.stop_reason == "stopped_by_human"
    assert result.iterations_run == 1


def test_autogate_is_fully_autonomous():
    gate = AutoGate()
    assert gate.review_plans([], 1).action == "approve"
    assert gate.review_verdict(None, 1).action == "accept"


def test_console_gate_prompts(monkeypatch):
    answers = iter(["guide", "try boosting", "run", "continue", "stop"])
    monkeypatch.setattr(
        "ramanujan.hitl.Prompt.ask", lambda *a, **k: next(answers)
    )
    gate = ConsoleGate(Console(file=io.StringIO(), width=100))

    review = gate.review_plans([], 1)
    assert review.action == "revise" and review.guidance == "try boosting"
    assert gate.review_plans([], 1).action == "approve"

    from ramanujan.agents.roles import Verdict

    stop_verdict = Verdict(decision="stop_goal_met", reasoning="r")
    assert gate.review_verdict(stop_verdict, 1).action == "continue_anyway"
    continue_verdict = Verdict(decision="continue", reasoning="r")
    assert gate.review_verdict(continue_verdict, 1).action == "stop_now"
