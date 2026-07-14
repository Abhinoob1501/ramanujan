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

import json
import time
from dataclasses import dataclass
from pathlib import Path
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


GATE_REQUEST_FILE = "gate_request.json"
GATE_RESPONSE_FILE = "gate_response.json"


class FileGate:
    """File-backed gate: decisions arrive as JSON files, so any remote UI can
    drive them - the web dashboard serves buttons that write the response
    (used with `ramanujan run --web`).

    Protocol per checkpoint:
      1. write <run_dir>/gate_request.json  {id, type, iteration, payload}
      2. poll for <run_dir>/gate_response.json with a matching id
      3. consume both files and return the decision

    With no timeout it waits indefinitely (that is the point of a human gate);
    pass timeout_seconds to auto-approve unattended runs.
    """

    def __init__(
        self,
        control_dir: str | Path,
        console: Console | None = None,
        poll_interval: float = 1.0,
        timeout_seconds: float | None = None,
        url_hint: str = "",
    ):
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.request_path = self.control_dir / GATE_REQUEST_FILE
        self.response_path = self.control_dir / GATE_RESPONSE_FILE
        self.console = console or Console()
        self.poll_interval = poll_interval
        self.timeout_seconds = timeout_seconds
        self.url_hint = url_hint
        self._next_id = 1

    def review_plans(self, plans: list[ExperimentPlan], iteration: int) -> PlanReview:
        payload = {
            "plans": [{"hypothesis": p.hypothesis, "approach": p.approach} for p in plans]
        }
        response = self._exchange("plan", iteration, payload)
        action = response.get("action", "approve")
        if action == "revise" and str(response.get("guidance", "")).strip():
            return PlanReview(action="revise", guidance=str(response["guidance"]).strip())
        if action == "stop":
            return PlanReview(action="stop")
        return PlanReview(action="approve")

    def review_verdict(self, verdict: Verdict, iteration: int) -> VerdictReview:
        payload = {"decision": verdict.decision, "reasoning": verdict.reasoning}
        response = self._exchange("verdict", iteration, payload)
        action = response.get("action", "accept")
        if action in ("continue_anyway", "stop_now"):
            return VerdictReview(action=action)
        return VerdictReview(action="accept")

    def _exchange(self, kind: str, iteration: int, payload: dict) -> dict:
        request_id = self._next_id
        self._next_id += 1
        self.response_path.unlink(missing_ok=True)  # stale answers must not apply
        self.request_path.write_text(
            json.dumps(
                {"id": request_id, "type": kind, "iteration": iteration,
                 "payload": payload, "ts": time.time()}
            ),
            encoding="utf-8",
        )
        where = self.url_hint or f"serve with: ramanujan dashboard \"{self.control_dir}\""
        self.console.print(
            f"[bold cyan]Waiting for your decision in the dashboard[/bold cyan] "
            f"({kind} review, round {iteration}) - {where}"
        )
        deadline = time.time() + self.timeout_seconds if self.timeout_seconds else None
        while True:
            if self.response_path.exists():
                try:
                    # utf-8-sig: tolerate a BOM in hand-written files on Windows
                    response = json.loads(self.response_path.read_text(encoding="utf-8-sig"))
                except (json.JSONDecodeError, OSError):
                    response = None  # torn write; next poll gets it
                if response and response.get("id") == request_id:
                    self.request_path.unlink(missing_ok=True)
                    self.response_path.unlink(missing_ok=True)
                    return response
            if deadline is not None and time.time() > deadline:
                self.request_path.unlink(missing_ok=True)
                self.console.print(
                    "[yellow]No decision arrived before the timeout - auto-approving.[/yellow]"
                )
                return {}
            time.sleep(self.poll_interval)


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
