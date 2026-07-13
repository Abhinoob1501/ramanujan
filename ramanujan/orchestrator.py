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

from .agents.eda import EdaAgent, EdaFindings
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
from .hardware import local_hardware_note
from .hitl import AutoGate, HumanGate
from .llm.base import LLMBudgetExceeded, LLMClient
from .llm.factory import LLMSuite
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
        llm: LLMClient | LLMSuite,
        runs_root: str | Path = "runs",
        console: Console | None = None,
        knowledge: KnowledgeBase | None = None,
        gate: HumanGate | None = None,
    ):
        if task.executor == "local":
            # tell every agent what this machine actually offers (GPU or not)
            task = task.model_copy(
                update={
                    "environment_notes": f"{task.environment_notes.rstrip()} {local_hardware_note()}"
                }
            )
        self.task = task
        self.llms = llm if isinstance(llm, LLMSuite) else LLMSuite.for_single(llm)
        self.console = console or Console()
        self.gate: HumanGate = gate or AutoGate()
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
        self._reference_code = ""
        self._eda_findings: EdaFindings | None = None
        self._human_guidance: list[str] = []

    # ------------------------------------------------------------------ public

    def run(self) -> RunResult:
        task, metric = self.task, self.task.metric
        self._banner()
        prior_knowledge = self._retrieve_prior_knowledge()
        stop_reason = "budget_exhausted"
        iteration = 0

        try:
            eda_block = self._run_eda_phase()
            if eda_block:
                prior_knowledge = (
                    f"{eda_block}\n\n{prior_knowledge}" if prior_knowledge else eda_block
                )
            for iteration in range(1, task.budget.max_iterations + 1):
                if self._experiments_used >= task.budget.experiment_cap:
                    stop_reason = "experiment_budget_exhausted"
                    break
                self._current_iteration = iteration
                rounds_left = task.budget.max_iterations - iteration
                self.console.rule(f"[bold]Round {iteration}/{task.budget.max_iterations}")
                self.events.emit("round", "round_started", iteration=iteration)

                plans = self._plan_round(
                    iteration, rounds_left, self._augment_context(prior_knowledge)
                )

                # human gate: approve / guide-and-replan / stop, before compute is spent
                plans, stopped_by_human = self._human_plan_gate(
                    plans, iteration, rounds_left, prior_knowledge
                )
                if stopped_by_human:
                    stop_reason = "stopped_by_human"
                    break

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
                    self.llms.critic, task, self.ledger.summary_markdown(metric),
                    analysis, iteration, rounds_left,
                    last_experiment_failed=not round_had_success,
                )
                self._panel("Critic", f"[bold]{verdict.decision}[/bold] - {verdict.reasoning}", "yellow")
                self.events.emit(
                    "judge", "verdict", iteration=iteration, agent="critic",
                    payload={"decision": verdict.decision, "reasoning": verdict.reasoning,
                             "concerns": verdict.concerns},
                )

                # human gate: accept the verdict or override it in either direction
                review = self.gate.review_verdict(verdict, iteration)
                if review.action != "accept":
                    self.events.emit(
                        "human", "verdict_overridden", iteration=iteration,
                        payload={"critic_decision": verdict.decision, "human_action": review.action},
                    )
                    self._panel(
                        "Human", f"Verdict overridden: [bold]{review.action}[/bold] "
                                 f"(critic said {verdict.decision})", "cyan",
                    )
                if review.action == "stop_now":
                    stop_reason = "stopped_by_human"
                    break
                if verdict.decision != "continue" and review.action != "continue_anyway":
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
            plan = run_planner(self.llms.planner, task, summary, iteration, rounds_left, prior_knowledge)
            self._panel("Planner", f"[bold]Hypothesis:[/bold] {plan.hypothesis}\n"
                                   f"[bold]Approach:[/bold] {plan.approach}", "cyan")
            self.events.emit(
                "plan", "plan_proposed", iteration=iteration, agent="planner",
                payload={"branch": 0, "hypothesis": plan.hypothesis, "approach": plan.approach,
                         "rationale": plan.rationale},
            )
            return [plan]

        candidates = run_planner_batch(
            self.llms.planner, task, summary, iteration, rounds_left, k, prior_knowledge
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
            self.llms.critic, task, summary, candidates,
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
            self.llms.engineer, task, self.executor, workdir, on_event=self._on_agent_event
        )
        outcome = engineer.implement(
            plan, staged_files=staged, reference_code=self._reference_code
        )

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
                self.llms.analyst, task, self.ledger.summary_markdown(metric),
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

    # ------------------------------------------------------------ human gate

    def _augment_context(self, prior_knowledge: str) -> str:
        """Planning context = prior knowledge + all human guidance so far."""
        if not self._human_guidance:
            return prior_knowledge
        guidance = "HUMAN GUIDANCE (must be followed):\n" + "\n".join(
            f"- {item}" for item in self._human_guidance
        )
        return f"{prior_knowledge}\n\n{guidance}" if prior_knowledge else guidance

    def _human_plan_gate(
        self,
        plans: list[ExperimentPlan],
        iteration: int,
        rounds_left: int,
        prior_knowledge: str,
        max_revisions: int = 3,
    ) -> tuple[list[ExperimentPlan], bool]:
        """Returns (possibly re-planned plans, stopped_by_human)."""
        for _ in range(max_revisions + 1):
            review = self.gate.review_plans(plans, iteration)
            if review.action == "approve":
                return plans, False
            if review.action == "stop":
                self.events.emit("human", "research_stopped", iteration=iteration)
                return plans, True
            self._human_guidance.append(review.guidance)
            self._panel("Human", f"Guidance: {review.guidance} - re-planning.", "cyan")
            self.events.emit(
                "human", "guidance_given", iteration=iteration,
                payload={"guidance": review.guidance},
            )
            plans = self._plan_round(
                iteration, rounds_left, self._augment_context(prior_knowledge)
            )
        return plans, False  # revision budget spent; run what we have

    # ------------------------------------------------------------------- eda

    def _run_eda_phase(self) -> str:
        """Explore the data before planning. Returns a prompt block ('' if
        disabled, skipped, or failed - EDA failure must never sink the run)."""
        if not self.task.eda:
            return ""
        if self.task.executor == "runpod":
            self.console.print("[dim]EDA skipped: dataset lives on the remote GPU pod.[/dim]")
            return ""
        workdir = self.run_dir / "eda"
        staged = self.task.stage_data_files(workdir) if self.task.data_files else []
        self.events.emit("eda", "eda_started")
        agent = EdaAgent(self.llms.analyst, self.task, workdir, on_event=self._on_agent_event)
        outcome = agent.explore(staged_files=staged)
        if not outcome.success or outcome.findings is None:
            self._panel("EDA", f"[yellow]Exploration failed ({outcome.error}); "
                               "planning proceeds without it.[/yellow]", "yellow")
            self.events.emit("eda", "eda_failed", payload={"error": outcome.error})
            return ""
        findings = outcome.findings
        self._eda_findings = findings
        body = findings.summary
        if findings.key_findings:
            body += "\n" + "\n".join(f"- {f}" for f in findings.key_findings[:5])
        if findings.leakage_risks:
            body += "\n[bold yellow]Leakage risks:[/bold yellow] " + "; ".join(findings.leakage_risks)
        self._panel("EDA", body, "blue")
        self.events.emit("eda", "eda_findings", agent="eda", payload=findings.model_dump())
        return findings.to_prompt_block()

    # ------------------------------------------------------------- knowledge

    def _retrieve_prior_knowledge(self) -> str:
        query = f"{self.task.name}\n{self.task.description}\n{self.task.dataset}"
        items = self.knowledge.retrieve(query, top_k=5, exclude_run=self.run_id)
        if items:
            # best sufficiently-similar past solution becomes the engineer's reference
            with_code = [i for i in items if i.code and i.similarity >= 0.2]
            if with_code:
                self._reference_code = with_code[0].code
            self._panel(
                "Knowledge base",
                f"Recalled [bold]{len(items)}[/bold] insight(s) from past runs"
                + (" (incl. a reference solution for the engineer)." if with_code else "."),
                "blue",
            )
            self.events.emit(
                "knowledge", "prior_knowledge_retrieved",
                payload={"insights": [
                    {"task": item.task_name, "insight": item.insight,
                     "similarity": round(item.similarity, 3)}
                    for item in items
                ], "reference_code": bool(self._reference_code)},
            )
        return format_for_prompt(items)

    def _store_new_knowledge(self) -> None:
        best = self.ledger.best(self.task.metric)
        for rec in self.ledger.all():
            if rec.status == "success" and rec.insight:
                code = ""
                if best and rec.id == best.id and rec.code_path and Path(rec.code_path).exists():
                    code = Path(rec.code_path).read_text(encoding="utf-8")
                self.knowledge.add_insight(
                    run_id=self.run_id,
                    task_name=self.task.name,
                    hypothesis=rec.hypothesis,
                    approach=rec.approach,
                    insight=rec.insight,
                    metric_name=rec.metric_name,
                    metric_value=rec.metric_value,
                    code=code,
                )

    # ---------------------------------------------------------------- internal

    def _write_report(self, stop_reason: str) -> Path:
        summary = self.ledger.summary_markdown(self.task.metric)
        try:
            conclusions = write_conclusions(self.llms.reporter, self.task, summary)
        except Exception as exc:  # a dead LLM must not lose the run's artifacts
            conclusions = f"(Conclusions unavailable: {exc})"
        usage = self.llms.usage()
        report = build_report(
            self.task, self.ledger, conclusions, stop_reason,
            usage=usage, eda_findings=self._eda_findings,
        )
        path = self.run_dir / "report.md"
        path.write_text(report, encoding="utf-8")
        self.events.emit("report", "report_written", payload={"path": str(path)})
        self.events.emit(
            "finish", "llm_usage",
            payload={"calls": usage.calls, "prompt_tokens": usage.prompt_tokens,
                     "completion_tokens": usage.completion_tokens,
                     "cost_usd": round(usage.cost_usd, 6)},
        )
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
        usage = self.llms.usage()
        cost = f" (~${usage.cost_usd:.4f})" if usage.cost_usd else ""
        self._panel(
            "Run complete",
            f"{line}\nStop reason: {stop_reason}\n"
            f"LLM usage: {usage.calls} calls, {usage.prompt_tokens + usage.completion_tokens} tokens{cost}\n"
            f"Report: {report_path}\n"
            f"Dashboard: ramanujan dashboard {self.run_dir}",
            "bold white",
        )

    def _panel(self, title: str, body: str, style: str) -> None:
        self.console.print(Panel(body, title=title, border_style=style, expand=False))
