"""Self-benchmark mode: measure the agent system itself.

`ramanujan bench <task.yaml> ... -n N` runs each task N times and reports
goal-hit rate, best-metric statistics, experiment efficiency and a failure
taxonomy - so changes to prompts, models or orchestration can be evaluated
with numbers instead of anecdotes.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from .memory.ledger import ExperimentLedger
from .orchestrator import ResearchDirector
from .task import TaskSpec


@dataclass
class RunOutcome:
    goal_met: bool
    best_value: float | None
    stop_reason: str
    rounds: int
    experiments: int
    failed_experiments: int


@dataclass
class TaskBenchResult:
    task: TaskSpec
    outcomes: list[RunOutcome] = field(default_factory=list)

    @property
    def goal_hit_rate(self) -> float:
        return sum(o.goal_met for o in self.outcomes) / len(self.outcomes)

    @property
    def mean_best(self) -> float | None:
        values = [o.best_value for o in self.outcomes if o.best_value is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_experiments(self) -> float:
        return sum(o.experiments for o in self.outcomes) / len(self.outcomes)

    @property
    def experiment_failure_rate(self) -> float:
        total = sum(o.experiments for o in self.outcomes)
        return sum(o.failed_experiments for o in self.outcomes) / total if total else 0.0

    @property
    def stop_reasons(self) -> Counter:
        return Counter(o.stop_reason for o in self.outcomes)


def run_benchmark(
    task_paths: list[Path],
    repeats: int,
    llm_factory: Callable[[], object],
    runs_root: Path,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    results: list[TaskBenchResult] = []

    for task_path in task_paths:
        task = TaskSpec.from_yaml(task_path)
        bench = TaskBenchResult(task=task)
        for repeat in range(1, repeats + 1):
            console.rule(f"[bold]bench: {task.name} - run {repeat}/{repeats}")
            director = ResearchDirector(task, llm_factory(), runs_root=runs_root, console=console)
            result = director.run()
            ledger = ExperimentLedger(result.run_dir / "ledger.db")
            records = ledger.all()
            bench.outcomes.append(
                RunOutcome(
                    goal_met=bool(result.best and task.metric.goal_met(result.best.metric_value)),
                    best_value=result.best.metric_value if result.best else None,
                    stop_reason=result.stop_reason,
                    rounds=result.iterations_run,
                    experiments=len(records),
                    failed_experiments=sum(r.status == "failed" for r in records),
                )
            )
        results.append(bench)

    report_path = _write_report(results, repeats, runs_root)
    _print_summary(results, console, report_path)
    return report_path


def _print_summary(results: list[TaskBenchResult], console: Console, report_path: Path) -> None:
    table = Table(title="Benchmark summary")
    for column in ("Task", "Runs", "Goal hit", "Mean best", "Mean exps/run", "Exp failure rate", "Stop reasons"):
        table.add_column(column)
    for bench in results:
        table.add_row(
            bench.task.name,
            str(len(bench.outcomes)),
            f"{bench.goal_hit_rate:.0%}",
            f"{bench.mean_best:.4f}" if bench.mean_best is not None else "-",
            f"{bench.mean_experiments:.1f}",
            f"{bench.experiment_failure_rate:.0%}",
            ", ".join(f"{reason} x{count}" for reason, count in bench.stop_reasons.items()),
        )
    console.print(table)
    console.print(f"Benchmark report: [bold]{report_path}[/bold]")


def _write_report(results: list[TaskBenchResult], repeats: int, runs_root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(runs_root) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ramanujan benchmark",
        "",
        f"*{time.strftime('%Y-%m-%d %H:%M')} - {repeats} repeat(s) per task*",
        "",
        "| Task | Runs | Goal hit | Mean best | Mean exps/run | Exp failure rate | Stop reasons |",
        "|------|------|----------|-----------|---------------|------------------|--------------|",
    ]
    for bench in results:
        mean_best = f"{bench.mean_best:.4f}" if bench.mean_best is not None else "-"
        reasons = ", ".join(f"{r} x{c}" for r, c in bench.stop_reasons.items())
        lines.append(
            f"| {bench.task.name} | {len(bench.outcomes)} | {bench.goal_hit_rate:.0%} "
            f"| {mean_best} | {bench.mean_experiments:.1f} "
            f"| {bench.experiment_failure_rate:.0%} | {reasons} |"
        )
    lines += ["", "## Per-run detail", ""]
    for bench in results:
        lines.append(f"### {bench.task.name}")
        for i, outcome in enumerate(bench.outcomes, 1):
            best = f"{outcome.best_value:.4f}" if outcome.best_value is not None else "no result"
            lines.append(
                f"- run {i}: best={best}, goal_met={outcome.goal_met}, "
                f"rounds={outcome.rounds}, experiments={outcome.experiments} "
                f"({outcome.failed_experiments} failed), stop={outcome.stop_reason}"
            )
        lines.append("")
    path = out_dir / f"bench_{stamp}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
