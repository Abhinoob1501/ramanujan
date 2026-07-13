"""Command-line interface: `ramanujan run <task.yaml>` and `ramanujan show <run_dir>`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings
from .memory.ledger import ExperimentLedger
from .orchestrator import ResearchDirector
from .task import TaskSpec

app = typer.Typer(
    add_completion=False,
    help="Ramanujan - an autonomous ML research engineer.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    task_file: Path = typer.Argument(..., help="Task specification YAML."),
    offline: bool = typer.Option(
        False, "--offline", help="Replay the scripted demo session (no API key needed; "
        "designed for tasks/demo_breast_cancer.yaml)."
    ),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n", help="Override the task's max_iterations budget."
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider: gemini | openrouter | opencode (Zen) | custom. "
        "Default: auto-detect from whichever API key is set.",
    ),
    runs_root: Path = typer.Option(Path("runs"), help="Directory where run artifacts are stored."),
):
    """Run an autonomous research session on a task."""
    task = TaskSpec.from_yaml(task_file)
    if iterations is not None:
        task = task.model_copy(
            update={"budget": task.budget.model_copy(update={"max_iterations": iterations})}
        )

    if offline:
        from .offline import DEMO_TASK_SLUG, build_offline_llm

        if task.slug != DEMO_TASK_SLUG:
            console.print(
                f"[yellow]Warning:[/yellow] --offline replays a session scripted for the "
                f"'{DEMO_TASK_SLUG}' demo task; results will not match '{task.slug}'."
            )
        llm = build_offline_llm()
    else:
        from .llm.factory import build_llm_suite

        try:
            llm = build_llm_suite(provider, Settings.from_env())
        except RuntimeError as exc:
            console.print(f"[red]Configuration error:[/red] {exc}")
            raise typer.Exit(2)

    result = ResearchDirector(task, llm, runs_root=runs_root, console=console).run()
    raise typer.Exit(0 if result.best is not None else 1)


@app.command()
def show(run_dir: Path = typer.Argument(..., help="A run directory under runs/.")):
    """Show the leaderboard and artifacts of a past run."""
    db = run_dir / "ledger.db"
    if not db.exists():
        console.print(f"[red]No ledger found at {db}[/red]")
        raise typer.Exit(1)
    ledger = ExperimentLedger(db)
    table = Table(title=f"Experiments in {run_dir.name}")
    for column in ("#", "Iter", "Hypothesis", "Status", "Metric", "Value", "Insight"):
        table.add_column(column)
    for rec in ledger.all():
        table.add_row(
            str(rec.id),
            str(rec.iteration),
            (rec.hypothesis[:60] + "...") if len(rec.hypothesis) > 60 else rec.hypothesis,
            rec.status,
            rec.metric_name or "-",
            f"{rec.metric_value:.4f}" if rec.metric_value is not None else "-",
            (rec.insight[:60] + "...") if rec.insight and len(rec.insight) > 60 else (rec.insight or "-"),
        )
    console.print(table)
    report = run_dir / "report.md"
    if report.exists():
        console.print(f"\nFull report: [bold]{report}[/bold]")


@app.command()
def bench(
    task_files: list[Path] = typer.Argument(..., help="Task YAMLs to benchmark."),
    repeats: int = typer.Option(2, "--repeats", "-n", help="Runs per task."),
    offline: bool = typer.Option(False, "--offline", help="Use the scripted demo session."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider."),
    runs_root: Path = typer.Option(Path("runs"), help="Directory for run artifacts."),
):
    """Benchmark the agent system itself: goal-hit rate, efficiency, failure taxonomy."""
    from .bench import run_benchmark

    if offline:
        from .offline import build_offline_llm

        llm_factory = build_offline_llm
    else:
        from .llm.factory import build_llm_suite

        def llm_factory():
            return build_llm_suite(provider, Settings.from_env())

    run_benchmark(task_files, repeats, llm_factory, runs_root, console)


@app.command()
def dashboard(
    run_dir: Path = typer.Argument(..., help="A run directory under runs/ (live or finished)."),
    port: int = typer.Option(8787, "--port", help="Port to serve on."),
):
    """Serve a live web dashboard streaming the run's reasoning and results."""
    from .dashboard import serve_dashboard

    serve_dashboard(run_dir, port=port)


if __name__ == "__main__":
    app()
