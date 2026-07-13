"""The Research Director: deterministic orchestration around agentic steps.

Control flow (budgets, phase order, persistence) is plain code; judgment
(what to try, how to code it, what it means, when to stop) is delegated to
agents. This split keeps the system debuggable and the failure modes bounded.

    +--> PLANNER ---- hypothesis + approach
    |       |
    |    ENGINEER --- write/run/debug loop in a sandbox --> metrics
    |       |
    |    ANALYST ---- insight (skipped if the experiment failed)
    |       |
    +--- CRITIC ----- continue | stop_goal_met | stop_diminishing_returns | stop_flawed
            |
        REPORT ------ leaderboard + narrative + LLM-written conclusions
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .agents.engineer import EngineerAgent
from .agents.roles import run_analyst, run_critic, run_planner, write_conclusions
from .executors import build_executor
from .llm.base import LLMBudgetExceeded, LLMClient
from .memory.ledger import ExperimentLedger, ExperimentRecord
from .report import build_report
from .task import TaskSpec


@dataclass
class RunResult:
    run_dir: Path
    stop_reason: str
    iterations_run: int
    best: ExperimentRecord | None
    report_path: Path | None


class ResearchDirector:
    def __init__(
        self,
        task: TaskSpec,
        llm: LLMClient,
        runs_root: str | Path = "runs",
        console: Console | None = None,
    ):
        self.task = task
        self.llm = llm
        self.console = console or Console()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(runs_root) / f"{stamp}_{task.slug}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ExperimentLedger(self.run_dir / "ledger.db")
        self.executor = build_executor(task, self.run_dir)

    # ------------------------------------------------------------------ public

    def run(self) -> RunResult:
        task, metric = self.task, self.task.metric
        self._banner()
        stop_reason = "budget_exhausted"
        iteration = 0

        try:
            for iteration in range(1, task.budget.max_iterations + 1):
                left = task.budget.max_iterations - iteration
                self.console.rule(f"[bold]Iteration {iteration}/{task.budget.max_iterations}")

                # 1. PLAN
                plan = run_planner(
                    self.llm, task, self.ledger.summary_markdown(metric), iteration, left
                )
                self._panel("Planner", f"[bold]Hypothesis:[/bold] {plan.hypothesis}\n"
                                       f"[bold]Approach:[/bold] {plan.approach}", "cyan")
                experiment_id = self.ledger.start_experiment(
                    iteration, plan.hypothesis, plan.approach
                )

                # 2. BUILD + RUN (agentic debug loop)
                workdir = self.run_dir / f"iter_{iteration:02d}"
                engineer = EngineerAgent(
                    self.llm, task, self.executor, workdir, on_event=self._on_agent_event
                )
                outcome = engineer.implement(plan)

                # 3. RECORD + ANALYZE
                analysis = None
                if outcome.success:
                    self.ledger.record_success(
                        experiment_id,
                        metric_name=str(outcome.metrics.get("metric_name", metric.name)),
                        metric_value=float(outcome.metrics["metric_value"]),
                        metrics=outcome.metrics,
                        duration_seconds=outcome.duration_seconds,
                        code_path=outcome.code_path,
                    )
                    self._panel(
                        "Result",
                        f"{metric.name} = [bold green]{outcome.metrics['metric_value']:.4f}[/bold green] "
                        f"(goal {metric.goal}) in {outcome.duration_seconds:.1f}s",
                        "green",
                    )
                    analysis = run_analyst(
                        self.llm, task, self.ledger.summary_markdown(metric),
                        plan, outcome.summary, outcome.metrics,
                    )
                    self.ledger.record_insight(experiment_id, analysis.insight)
                    body = f"[bold]Insight:[/bold] {analysis.insight}"
                    if analysis.suspicion:
                        body += f"\n[bold yellow]Suspicion:[/bold yellow] {analysis.suspicion}"
                    self._panel("Analyst", body, "magenta")
                else:
                    self.ledger.record_failure(experiment_id, outcome.error_summary)
                    self._panel("Result", f"[red]Experiment failed:[/red] {outcome.error_summary}", "red")

                # 4. JUDGE
                verdict = run_critic(
                    self.llm, task, self.ledger.summary_markdown(metric),
                    analysis, iteration, left, last_experiment_failed=not outcome.success,
                )
                self._panel("Critic", f"[bold]{verdict.decision}[/bold] - {verdict.reasoning}", "yellow")
                if verdict.decision != "continue":
                    stop_reason = verdict.decision
                    break
        except LLMBudgetExceeded as exc:
            stop_reason = f"llm_budget_exhausted ({exc})"
            self.console.print(f"[red]{stop_reason}[/red]")

        report_path = self._write_report(stop_reason)
        best = self.ledger.best(metric)
        self._summary(best, stop_reason, report_path)
        return RunResult(
            run_dir=self.run_dir,
            stop_reason=stop_reason,
            iterations_run=iteration,
            best=best,
            report_path=report_path,
        )

    # ---------------------------------------------------------------- internal

    def _write_report(self, stop_reason: str) -> Path:
        summary = self.ledger.summary_markdown(self.task.metric)
        try:
            conclusions = write_conclusions(self.llm, self.task, summary)
        except Exception as exc:  # a dead LLM must not lose the run's artifacts
            conclusions = f"(Conclusions unavailable: {exc})"
        report = build_report(self.task, self.ledger, conclusions, stop_reason)
        path = self.run_dir / "report.md"
        path.write_text(report, encoding="utf-8")
        return path

    def _on_agent_event(self, agent: str, kind: str, detail: str) -> None:
        if kind == "tool_call":
            self.console.print(f"  [dim]{agent} ->[/dim] [blue]{detail}[/blue]")
        elif kind == "tool_result":
            first_line = detail.splitlines()[0] if detail else ""
            self.console.print(f"  [dim]{agent} <- {first_line}[/dim]")

    def _banner(self) -> None:
        self._panel(
            "Ramanujan - autonomous ML research",
            f"[bold]{self.task.name}[/bold]\n{self.task.description}\n"
            f"Goal: {self.task.metric.name} {'>=' if self.task.metric.direction == 'maximize' else '<='} "
            f"{self.task.metric.goal} | Budget: {self.task.budget.max_iterations} iterations | "
            f"Executor: {self.task.executor}",
            "white",
        )

    def _summary(self, best: ExperimentRecord | None, stop_reason: str, report_path: Path) -> None:
        if best is not None:
            goal = self.task.metric.goal_met(best.metric_value)
            line = (
                f"Best: {best.metric_name} = [bold]{best.metric_value:.4f}[/bold] "
                f"(experiment {best.id}) - goal {'[green]MET[/green]' if goal else '[red]not met[/red]'}"
            )
        else:
            line = "[red]No experiment produced a valid result.[/red]"
        self._panel(
            "Run complete",
            f"{line}\nStop reason: {stop_reason}\nReport: {report_path}",
            "bold white",
        )

    def _panel(self, title: str, body: str, style: str) -> None:
        self.console.print(Panel(body, title=title, border_style=style, expand=False))
