"""Human-in-the-loop gates.

Opt-in checkpoints at the two moments where human judgment is cheapest and
most valuable:

- plan review: BEFORE compute is spent - approve the round's plans, give
  free-text guidance and force a re-plan, or stop the research
- verdict review: AFTER the critic rules - accept, or override in either
  direction (continue despite a stop, stop despite a continue)

Guidance given at a plan review is remembered and injected into every later
planning round. The default AutoGate approves everything, so autonomous runs
are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from rich.console import Console
from rich.prompt import Prompt

from .agents.roles import ExperimentPlan, Verdict


@dataclass
class PlanReview:
    action: Literal["approve", "revise", "stop"]
    guidance: str = ""


@dataclass
class VerdictReview:
    action: Literal["accept", "continue_anyway", "stop_now"]


class HumanGate(Protocol):
    def review_plans(self, plans: list[ExperimentPlan], iteration: int) -> PlanReview: ...

    def review_verdict(self, verdict: Verdict, iteration: int) -> VerdictReview: ...


class AutoGate:
    """Fully autonomous: approves every plan and accepts every verdict."""

    def review_plans(self, plans: list[ExperimentPlan], iteration: int) -> PlanReview:
        return PlanReview(action="approve")

    def review_verdict(self, verdict: Verdict, iteration: int) -> VerdictReview:
        return VerdictReview(action="accept")


class ConsoleGate:
    """Interactive terminal gate (used with `ramanujan run -i`)."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def review_plans(self, plans: list[ExperimentPlan], iteration: int) -> PlanReview:
        self.console.print(
            f"[bold]Round {iteration}:[/bold] the planner proposed {len(plans)} "
            f"experiment(s) (shown above)."
        )
        choice = Prompt.ask(
            "[bold cyan]Run these, give guidance and re-plan, or stop?[/bold cyan]",
            choices=["run", "guide", "stop"],
            default="run",
            console=self.console,
        )
        if choice == "run":
            return PlanReview(action="approve")
        if choice == "stop":
            return PlanReview(action="stop")
        guidance = Prompt.ask(
            "[bold cyan]Your guidance for the planner[/bold cyan]", console=self.console
        ).strip()
        if not guidance:
            return PlanReview(action="approve")
        return PlanReview(action="revise", guidance=guidance)

    def review_verdict(self, verdict: Verdict, iteration: int) -> VerdictReview:
        if verdict.decision == "continue":
            choice = Prompt.ask(
                "[bold cyan]Critic wants to continue. Accept, or stop now?[/bold cyan]",
                choices=["accept", "stop"],
                default="accept",
                console=self.console,
            )
            return VerdictReview(action="stop_now" if choice == "stop" else "accept")
        choice = Prompt.ask(
            f"[bold cyan]Critic wants to stop ({verdict.decision}). "
            "Accept, or continue anyway?[/bold cyan]",
            choices=["accept", "continue"],
            default="accept",
            console=self.console,
        )
        return VerdictReview(action="continue_anyway" if choice == "continue" else "accept")
