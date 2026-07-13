"""The Research Director: deterministic orchestration around agentic steps.

Control flow (budgets, phase order, persistence) is plain code; judgment
(what to try, how to code it, what it means, when to stop) is delegated to
agents. This split keeps the system debuggable and the failure modes bounded.

    +--> PLANNER ---- 1 hypothesis (or k candidate branches)
    |       |
    |   ALLOCATOR --- (branching only) critic funds the branches worth running
    |       |
    |    ENGINEER --- write/run/debug loop in a sandbox --> metrics  (per branch)
    |       |
    |    ANALYST ---- insight (skipped if the experiment failed)
    |       |
    +--- CRITIC ----- continue | stop_goal_met | stop_diminishing_returns | stop_flawed
            |
        REPORT ------ leaderboard + narrative + LLM-written conclusions

Cross-cutting: every step is appended to <run_dir>/events.jsonl (live
dashboard), insights are exchanged with the cross-run knowledge base, and
experiments are optionally mirrored to Weights & Biases.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .agents.engineer import EngineerAgent, EngineerOutcome
from .agents.roles import (
    Analysis,
    ExperimentPlan,
    run_allocator,
    run_analyst,
    run_critic,
    run_planner,
    run_planner_batch,
    write_conclusions,
)
from .events import EventLog
from .executors import build_executor
from .llm.base import LLMBudgetExceeded, LLMClient
from .memory.knowledge import KnowledgeBase, format_for_prompt
from .memory.ledger import ExperimentLedger, ExperimentRecord
from .report import build_report
from .task import TaskSpec
from .tracking import ExperimentTracker


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
        knowledge: KnowledgeBase | None = None,
    ):
        self.task = task
        self.llm = llm
        self.console = console or Console()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        # uuid suffix: run ids must be unique even for runs started in the same
        # second (they key knowledge-base exclusion and W&B grouping)
        self.run_id = f"{stamp}_{task.slug}_{uuid.uuid4().hex[:6]}"
        self.run_dir = Path(runs_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ExperimentLedger(self.run_dir / "ledger.db")
        self.executor = build_executor(task, self.run_dir)
        self.events = EventLog(self.run_dir / "events.jsonl")
        self.knowledge = knowledge or KnowledgeBase(Path(runs_root) / "knowledge.db")
        self.tracker = ExperimentTracker(task, self.run_id, console=self.console)
        self._experiments_used = 0
        self._current_iteration: int | None = None

    # ------------------------------------------------------------------ public

    def run(self) -> RunResult:
        task, metric = self.task, self.task.metric
        self._banner()
        prior_knowledge = self._retrieve_prior_knowledge()
        stop_reason = "budget_exhausted"
        iteration = 0

        try:
            for iteration in range(1, task.budget.max_iterations + 1):
                if self._experiments_used >= task.budget.experiment_cap:
                    stop_reason = "experiment_budget_exhausted"
                    break
                self._current_iteration = iteration
                rounds_left = task.budget.max_iterations - iteration
                self.console.rule(f"[bold]Round {iteration}/{task.budget.max_iterations}")
                self.events.emit("round", "round_started", iteration=iteration)

                plans = self._plan_round(iteration, rounds_left, prior_knowledge)

                analysis: Analysis | None = None
                round_had_success = False
                for branch_index, plan in enumerate(plans):
                    if self._experiments_used >= task.budget.experiment_cap:
                        break
                    outcome, analysis_of_branch = self._run_experiment(
                        iteration, branch_index, plan
                    )
                    if outcome.success:
                        round_had_success = True
                        analysis = analysis_of_branch

                verdict = run_critic(
                    self.llm, task, self.ledger.summary_markdown(metric),
                    analysis, iteration, rounds_left,
                    last_experiment_failed=not round_had_success,
                )
                self._panel("Critic", f"[bold]{verdict.decision}[/bold] - {verdict.reasoning}", "yellow")
                self.events.emit(
                    "judge", "verdict", iteration=iteration, agent="critic",
                    payload={"decision": verdict.decision, "reasoning": verdict.reasoning,
                             "concerns": verdict.concerns},
                )
                if verdict.decision != "continue":
                    stop_reason = verdict.decision
                    break
        except LLMBudgetExceeded as exc:
            stop_reason = f"llm_budget_exhausted ({exc})"
            self.console.print(f"[red]{stop_reason}[/red]")

        report_path = self._write_report(stop_reason)
        self._store_new_knowledge()
        best = self.ledger.best(metric)
        self.events.emit(
            "finish", "run_finished",
            payload={
                "stop_reason": stop_reason,
                "best_metric_value": best.metric_value if best else None,
                "goal_met": bool(best and metric.goal_met(best.metric_value)),
                "report": str(report_path),
            },
        )
        self._summary(best, stop_reason, report_path)
        return RunResult(
            run_dir=self.run_dir,
            stop_reason=stop_reason,
            iterations_run=iteration,
            best=best,
            report_path=report_path,
        )

    # ----------------------------------------------------------------- phases

    def _plan_round(
        self, iteration: int, rounds_left: int, prior_knowledge: str
    ) -> list[ExperimentPlan]:
        task, metric = self.task, self.task.metric
        k = task.budget.parallel_branches
        summary = self.ledger.summary_markdown(metric)

        if k == 1:
            plan = run_planner(self.llm, task, summary, iteration, rounds_left, prior_knowledge)
            self._panel("Planner", f"[bold]Hypothesis:[/bold] {plan.hypothesis}\n"
                                   f"[bold]Approach:[/bold] {plan.approach}", "cyan")
            self.events.emit(
                "plan", "plan_proposed", iteration=iteration, agent="planner",
                payload={"branch": 0, "hypothesis": plan.hypothesis, "approach": plan.approach,
                         "rationale": plan.rationale},
            )
            return [plan]

        candidates = run_planner_batch(
            self.llm, task, summary, iteration, rounds_left, k, prior_knowledge
        )
        for i, plan in enumerate(candidates):
            self._panel(f"Planner - candidate {i}", f"[bold]Hypothesis:[/bold] {plan.hypothesis}\n"
                        f"[bold]Approach:[/bold] {plan.approach}", "cyan")
            self.events.emit(
                "plan", "plan_proposed", iteration=iteration, agent="planner",
                payload={"branch": i, "hypothesis": plan.hypothesis, "approach": plan.approach,
                         "rationale": plan.rationale},
            )

        experiments_left = task.budget.experiment_cap - self._experiments_used
        allocation = run_allocator(
            self.llm, task, summary, candidates,
            max_selectable=min(k, experiments_left),
            experiments_left_total=experiments_left,
        )
        selected = [candidates[i] for i in allocation.selected_indices]
        self._panel(
            "Budget allocator",
            f"Funding candidate(s) [bold]{allocation.selected_indices}[/bold] - {allocation.reasoning}",
            "yellow",
        )
        self.events.emit(
            "plan", "budget_allocated", iteration=iteration, agent="critic",
            payload={"selected": allocation.selected_indices, "reasoning": allocation.reasoning},
        )
        return selected

    def _run_experiment(
        self, iteration: int, branch_index: int, plan: ExperimentPlan
    ) -> tuple[EngineerOutcome, Analysis | None]:
        task, metric = self.task, self.task.metric
        experiment_id = self.ledger.start_experiment(iteration, plan.hypothesis, plan.approach)
        self._experiments_used += 1
        self.events.emit(
            "build", "experiment_started", iteration=iteration,
            payload={"experiment_id": experiment_id, "branch": branch_index,
                     "hypothesis": plan.hypothesis},
        )

        suffix = f"_{chr(ord('a') + branch_index)}" if task.budget.parallel_branches > 1 else ""
        workdir = self.run_dir / f"iter_{iteration:02d}{suffix}"
        staged = task.stage_data_files(workdir) if task.data_files else []
        engineer = EngineerAgent(
            self.llm, task, self.executor, workdir, on_event=self._on_agent_event
        )
        outcome = engineer.implement(plan, staged_files=staged)

        analysis: Analysis | None = None
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
            self.events.emit(
                "build", "experiment_result", iteration=iteration,
                payload={"experiment_id": experiment_id, "status": "success",
                         "metric_name": metric.name,
                         "metric_value": outcome.metrics["metric_value"],
                         "duration_seconds": outcome.duration_seconds},
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
            self.events.emit(
                "analyze", "analysis", iteration=iteration, agent="analyst",
                payload={"experiment_id": experiment_id, "insight": analysis.insight,
                         "suspicion": analysis.suspicion,
                         "hypothesis_supported": analysis.hypothesis_supported},
            )
        else:
            self.ledger.record_failure(experiment_id, outcome.error_summary)
            self._panel("Result", f"[red]Experiment failed:[/red] {outcome.error_summary}", "red")
            self.events.emit(
                "build", "experiment_result", iteration=iteration,
                payload={"experiment_id": experiment_id, "status": "failed",
                         "error": outcome.error_summary},
            )

        self.tracker.log_experiment(
            experiment_id=experiment_id,
            iteration=iteration,
            plan=plan,
            status="success" if outcome.success else "failed",
            metrics=outcome.metrics or None,
            duration_seconds=outcome.duration_seconds,
            error_summary=outcome.error_summary or None,
        )
        return outcome, analysis

    # ------------------------------------------------------------- knowledge

    def _retrieve_prior_knowledge(self) -> str:
        query = f"{self.task.name}\n{self.task.description}\n{self.task.dataset}"
        items = self.knowledge.retrieve(query, top_k=5, exclude_run=self.run_id)
        if items:
            self._panel(
                "Knowledge base",
                f"Recalled [bold]{len(items)}[/bold] insight(s) from past runs.",
                "blue",
            )
            self.events.emit(
                "knowledge", "prior_knowledge_retrieved",
                payload={"insights": [
                    {"task": item.task_name, "insight": item.insight,
                     "similarity": round(item.similarity, 3)}
                    for item in items
                ]},
            )
        return format_for_prompt(items)

    def _store_new_knowledge(self) -> None:
        for rec in self.ledger.all():
            if rec.status == "success" and rec.insight:
                self.knowledge.add_insight(
                    run_id=self.run_id,
                    task_name=self.task.name,
                    hypothesis=rec.hypothesis,
                    approach=rec.approach,
                    insight=rec.insight,
                    metric_name=rec.metric_name,
                    metric_value=rec.metric_value,
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
        self.events.emit("report", "report_written", payload={"path": str(path)})
        return path

    def _on_agent_event(self, agent: str, kind: str, detail: str) -> None:
        if kind == "tool_call":
            self.console.print(f"  [dim]{agent} ->[/dim] [blue]{detail}[/blue]")
        elif kind == "tool_result":
            first_line = detail.splitlines()[0] if detail else ""
            self.console.print(f"  [dim]{agent} <- {first_line}[/dim]")
        self.events.emit(
            "build", f"agent_{kind}", iteration=self._current_iteration, agent=agent,
            payload={"detail": detail[:300]},
        )

    def _banner(self) -> None:
        task = self.task
        self._panel(
            "Ramanujan - autonomous ML research",
            f"[bold]{task.name}[/bold]\n{task.description}\n"
            f"Goal: {task.metric.name} {'>=' if task.metric.direction == 'maximize' else '<='} "
            f"{task.metric.goal} | Budget: {task.budget.max_iterations} rounds x "
            f"{task.budget.parallel_branches} branch(es), cap {task.budget.experiment_cap} "
            f"experiments | Executor: {task.executor}",
            "white",
        )
        self.events.emit(
            "start", "run_started",
            payload={
                "run_id": self.run_id,
                "task": task.name,
                "description": task.description,
                "metric_name": task.metric.name,
                "goal": task.metric.goal,
                "direction": task.metric.direction,
                "max_iterations": task.budget.max_iterations,
                "parallel_branches": task.budget.parallel_branches,
                "executor": task.executor,
            },
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
            f"{line}\nStop reason: {stop_reason}\nReport: {report_path}\n"
            f"Dashboard: ramanujan dashboard {self.run_dir}",
            "bold white",
        )

    def _panel(self, title: str, body: str, style: str) -> None:
        self.console.print(Panel(body, title=title, border_style=style, expand=False))
