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
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Human-in-the-loop: pause to approve/guide plans and review verdicts.",
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

    gate = None
    if interactive or task.human_in_the_loop:
        from .hitl import ConsoleGate

        gate = ConsoleGate(console)

    result = ResearchDirector(task, llm, runs_root=runs_root, console=console, gate=gate).run()
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
def ask(
    request: str = typer.Argument(
        ..., help='Plain-English research request, e.g. '
        '"predict churn from data/customers.csv, aim for AUC 0.85".'
    ),
    data: Optional[list[Path]] = typer.Option(
        None, "--data", "-d", help="Data file(s) to attach (also auto-detected from the request)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Run without confirmation."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Only compose and save the task spec; don't run it."
    ),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider."),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Human-in-the-loop: pause to approve/guide plans and review verdicts.",
    ),
    runs_root: Path = typer.Option(Path("runs"), help="Directory for run artifacts."),
):
    """Describe what you want researched in plain English; Ramanujan composes
    the task spec, shows it, and runs it."""
    import yaml as _yaml
    from rich.panel import Panel
    from rich.syntax import Syntax

    from .composer import compose_task, detect_data_files, save_task
    from .llm.factory import build_llm_suite

    try:
        suite = build_llm_suite(provider, Settings.from_env())
    except RuntimeError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(2)

    files = detect_data_files(request, list(data) if data else None)
    for path in files:
        if not path.exists():
            console.print(f"[red]Data file not found:[/red] {path}")
            raise typer.Exit(2)
    if files:
        console.print(f"[dim]Inspecting data file(s): {', '.join(str(f) for f in files)}[/dim]")

    console.print("[dim]Composing research task...[/dim]")
    task = compose_task(suite.planner, request, files)
    spec_path = save_task(task)
    spec_yaml = _yaml.safe_dump(task.model_dump(), sort_keys=False, allow_unicode=True)
    console.print(
        Panel(
            Syntax(spec_yaml, "yaml", background_color="default"),
            title=f"Composed task (saved to {spec_path})",
            border_style="cyan",
        )
    )
    if dry_run:
        console.print(f"Dry run - edit and launch later with: [bold]ramanujan run {spec_path}[/bold]")
        raise typer.Exit(0)
    if not yes:
        typer.confirm("Run this research task now?", abort=True)

    gate = None
    if interactive:
        from .hitl import ConsoleGate

        gate = ConsoleGate(console)

    result = ResearchDirector(task, suite, runs_root=runs_root, console=console, gate=gate).run()
    raise typer.Exit(0 if result.best is not None else 1)


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
