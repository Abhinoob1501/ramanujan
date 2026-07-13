"""Decision-node roles: Planner, Analyst, Critic.

Each is a single structured LLM call (see agents.base.ask_json for why these are
deliberately not free-form tool loops)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..llm.base import LLMClient
from ..task import TaskSpec
from . import prompts
from .base import ask_json


class ExperimentPlan(BaseModel):
    hypothesis: str = Field(description="One falsifiable statement this experiment tests.")
    approach: str = Field(
        description="Concrete implementation spec: model, key hyperparameters, "
        "preprocessing, validation scheme."
    )
    rationale: str = Field(description="Why this is the most informative next experiment.")


class Analysis(BaseModel):
    insight: str = Field(description="The single most important learning from this experiment.")
    hypothesis_supported: bool
    suspicion: str = Field(
        default="", description="Any reason to distrust the result (leakage, instability); empty if none."
    )
    next_directions: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    decision: Literal["continue", "stop_goal_met", "stop_diminishing_returns", "stop_flawed"]
    reasoning: str
    concerns: list[str] = Field(default_factory=list)


def _task_block(task: TaskSpec) -> str:
    return (
        f"TASK: {task.name}\n"
        f"Description: {task.description}\n"
        f"Dataset: {task.dataset}\n"
        f"Target metric: {task.metric.name} ({task.metric.direction}), goal {task.metric.goal}\n"
        f"Environment: {task.environment_notes}"
    )


def run_planner(
    llm: LLMClient, task: TaskSpec, ledger_summary: str, iteration: int, iterations_left: int
) -> ExperimentPlan:
    prompt = (
        f"{_task_block(task)}\n\n"
        f"EXPERIMENT HISTORY:\n{ledger_summary}\n\n"
        f"You are planning iteration {iteration}. After this one, {iterations_left} "
        f"iteration(s) of budget remain.\n"
        "Propose the single most informative next experiment."
    )
    return ask_json(llm, system=prompts.PLANNER_SYSTEM, prompt=prompt, model_cls=ExperimentPlan)


def run_analyst(
    llm: LLMClient,
    task: TaskSpec,
    ledger_summary: str,
    plan: ExperimentPlan,
    engineer_summary: str,
    metrics: dict,
) -> Analysis:
    prompt = (
        f"{_task_block(task)}\n\n"
        f"EXPERIMENT HISTORY (earlier experiments):\n{ledger_summary}\n\n"
        f"COMPLETED EXPERIMENT UNDER REVIEW:\n"
        f"Hypothesis: {plan.hypothesis}\n"
        f"Approach: {plan.approach}\n"
        f"Engineer's summary: {engineer_summary}\n"
        f"Measured metrics: {metrics}\n\n"
        "Analyze this experiment."
    )
    return ask_json(llm, system=prompts.ANALYST_SYSTEM, prompt=prompt, model_cls=Analysis)


def run_critic(
    llm: LLMClient,
    task: TaskSpec,
    ledger_summary: str,
    analysis: Analysis | None,
    iterations_used: int,
    iterations_left: int,
    last_experiment_failed: bool,
) -> Verdict:
    status_line = (
        "The most recent experiment FAILED to produce a result."
        if last_experiment_failed
        else "The most recent experiment completed successfully."
    )
    analysis_block = (
        f"Analyst's view of the latest experiment:\n"
        f"Insight: {analysis.insight}\n"
        f"Suspicion: {analysis.suspicion or 'none'}\n"
        if analysis
        else "No analysis available (experiment failed)."
    )
    prompt = (
        f"{_task_block(task)}\n\n"
        f"EXPERIMENT HISTORY:\n{ledger_summary}\n\n"
        f"{status_line}\n{analysis_block}\n"
        f"Budget: {iterations_used} iteration(s) used, {iterations_left} remaining.\n\n"
        "Decide whether research continues."
    )
    return ask_json(llm, system=prompts.CRITIC_SYSTEM, prompt=prompt, model_cls=Verdict)


def write_conclusions(llm: LLMClient, task: TaskSpec, ledger_summary: str) -> str:
    from ..llm.base import ChatMessage

    response = llm.generate(
        system=prompts.REPORTER_SYSTEM,
        messages=[
            ChatMessage(
                role="user",
                content=f"{_task_block(task)}\n\nCOMPLETE EXPERIMENT LEDGER:\n{ledger_summary}\n\n"
                "Write the Conclusions section.",
            )
        ],
        temperature=0.5,
    )
    return response.text.strip()
