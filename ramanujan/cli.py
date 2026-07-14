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
    rich_markup_mode="rich",
    help=(
        "Ramanujan - an autonomous ML research engineer.\n\n"
        "Give it a research goal (a task YAML, or plain English via [bold]ask[/bold]) and it "
        "runs the scientific method as a loop: explores the data, forms hypotheses, writes "
        "and debugs real training code in a sandbox, analyzes results, and iterates until "
        "the goal is met or the budget says stop - then writes a research report.\n\n"
        "[bold]Typical session:[/bold]\n"
        "  1. Put an LLM API key in .env (GEMINI_API_KEY / OPENROUTER_API_KEY / OPENCODE_API_KEY)\n"
        "  2. [cyan]ramanujan ask \"predict churn from data/customers.csv, aim for AUC 0.85\"[/cyan]\n"
        "  3. Watch live: [cyan]ramanujan dashboard runs/<run_dir>[/cyan]\n\n"
        "No key yet? Try the full offline demo: "
        "[cyan]ramanujan run tasks/demo_breast_cancer.yaml --offline[/cyan]"
    ),
    no_args_is_help=True,
)
console = Console()

_PANEL_LLM = "LLM backend"
_PANEL_EXEC = "Execution"
_PANEL_OVERSIGHT = "Human oversight"
_PANEL_OUTPUT = "Output"

_EXECUTORS = ("local", "docker", "runpod")


def _apply_executor_choice(task: TaskSpec, executor: Optional[str]) -> TaskSpec:
    """Let the user decide where generated code runs: their own machine (CPU or
    local GPU), a local Docker sandbox, or a RunPod GPU pod. Choosing runpod
    asks for billing acknowledgement unless the spec already granted it."""
    if executor:
        executor = executor.lower()
        if executor not in _EXECUTORS:
            console.print(f"[red]Unknown executor '{executor}'. Choose from: {', '.join(_EXECUTORS)}.[/red]")
            raise typer.Exit(2)
        if executor != task.executor:
            updates: dict = {"executor": executor}
            if executor == "runpod":
                updates["environment_notes"] = (
                    f"{task.environment_notes.rstrip()} NOTE: execution target overridden "
                    "to a remote RunPod GPU pod (NVIDIA GPU, PyTorch + CUDA, torchvision); "
                    "ignore any earlier CPU-only statements."
                )
            task = task.model_copy(update=updates)
            console.print(f"[dim]Executor overridden to: {executor}[/dim]")
    if task.executor == "runpod" and not task.runpod.get("confirm_billing"):
        console.print(
            "[yellow]RunPod execution creates GPU pods that bill per minute on your "
            "RunPod account (requires runpodctl configured with an API key).[/yellow]"
        )
        if not typer.confirm("Proceed with billed GPU execution?"):
            raise typer.Exit(1)
        task = task.model_copy(update={"runpod": {**task.runpod, "confirm_billing": True}})
    return task


@app.command(
    epilog=(
        "Examples:\n\n"
        "  ramanujan run tasks/demo_breast_cancer.yaml --offline      (no API key needed)\n\n"
        "  ramanujan run tasks/churn_csv.yaml -n 3 -x docker          (3 rounds, sandboxed in Docker)\n\n"
        "  ramanujan run tasks/cifar10_runpod.yaml -x runpod -i       (GPU pod, human-approved rounds)"
    )
)
def run(
    task_file: Path = typer.Argument(
        ..., help="Task specification YAML (see tasks/ for examples)."
    ),
    offline: bool = typer.Option(
        False, "--offline",
        help="Replay a scripted demo session instead of calling a real LLM - the full "
        "system runs (real training, ledger, report, dashboard events) with zero API "
        "keys. Designed for tasks/demo_breast_cancer.yaml.",
        rich_help_panel=_PANEL_LLM,
    ),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", "-n",
        help="Override the task's max_iterations budget (number of planning rounds).",
        rich_help_panel=_PANEL_EXEC,
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: gemini | openrouter | opencode (Zen) | custom. Default: "
        "auto-detected from whichever API key is set in the environment/.env.",
        rich_help_panel=_PANEL_LLM,
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Human-in-the-loop mode: pause before each round so you can approve the "
        "plans, type guidance to force a re-plan, or stop - and review every critic "
        "verdict (override it in either direction).",
        rich_help_panel=_PANEL_OVERSIGHT,
    ),
    web: bool = typer.Option(
        False, "--web",
        help="Like --interactive, but decisions are made from the web dashboard: the "
        "run pauses at each checkpoint and shows approve/guide/stop buttons in the "
        "browser (serve it with `ramanujan dashboard <run_dir>`).",
        rich_help_panel=_PANEL_OVERSIGHT,
    ),
    executor: Optional[str] = typer.Option(
        None, "--executor", "-x",
        help="Where generated training code runs, overriding the task spec: "
        "'local' = this machine (a local NVIDIA GPU is auto-detected and offered to "
        "the agents; RAMANUJAN_FORCE_CPU=1 disables that), "
        "'docker' = a disposable network-isolated container (build the image once: "
        "docker build -t ramanujan-sandbox docker/), "
        "'runpod' = a billed RunPod GPU pod (asks for confirmation; needs runpodctl "
        "configured with an API key).",
        rich_help_panel=_PANEL_EXEC,
    ),
    runs_root: Path = typer.Option(
        Path("runs"), "--runs-root",
        help="Directory where run artifacts (code, ledger, events, report) are stored.",
        rich_help_panel=_PANEL_OUTPUT,
    ),
):
    """Run an autonomous research session from a task YAML.

    Loads the task spec, then loops: EDA -> plan -> engineer/run -> analyze ->
    critic verdict, until the goal is met or the budget runs out. Every run
    leaves a self-contained directory under --runs-root with the generated
    code, metrics, event stream, and an auto-written research report.
    """
    task = TaskSpec.from_yaml(task_file)
    if iterations is not None:
        task = task.model_copy(
            update={"budget": task.budget.model_copy(update={"max_iterations": iterations})}
        )
    task = _apply_executor_choice(task, executor)

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
    if not web and (interactive or task.human_in_the_loop):
        from .hitl import ConsoleGate

        gate = ConsoleGate(console)

    director = ResearchDirector(task, llm, runs_root=runs_root, console=console, gate=gate)
    if web:
        director.gate = _arm_web_gate(director)
    result = director.run()
    raise typer.Exit(0 if result.best is not None else 1)


def _arm_web_gate(director: ResearchDirector):
    """Web-gated runs serve their own dashboard (background thread, correct run
    directory guaranteed) and exchange decisions through files in the run dir."""
    from .dashboard import start_dashboard_server
    from .hitl import FileGate

    url_hint = ""
    try:
        server, port = start_dashboard_server(director.run_dir)
        director._dashboard_server = server  # keep it alive for the run's lifetime
        url_hint = f"open http://127.0.0.1:{port}"
        console.print(
            f"[bold cyan]Web gate armed.[/bold cyan] Dashboard for this run is live at "
            f"[bold]http://127.0.0.1:{port}[/bold] - decisions will appear there."
        )
    except OSError as exc:
        console.print(
            f"[yellow]Could not start the dashboard automatically ({exc}). "
            f"Serve it manually: ramanujan dashboard \"{director.run_dir}\"[/yellow]"
        )
    return FileGate(director.run_dir, console=console, url_hint=url_hint)


@app.command(epilog="Example:\n\n  ramanujan show runs/20260714_010606_breast-cancer-diagnosis_a488db")
def show(run_dir: Path = typer.Argument(..., help="A run directory under runs/.")):
    """Show a past run's experiment leaderboard in the terminal.

    Lists every experiment with its hypothesis, status, metric value and
    insight, plus the path to the full markdown report.
    """
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


@app.command(
    epilog=(
        "Examples:\n\n"
        "  ramanujan ask \"predict churn from data/customers.csv, aim for AUC 0.85\"\n\n"
        "  ramanujan ask \"classify sklearn digits as accurately as possible\" --dry-run\n\n"
        "  ramanujan ask \"model churn from data/customers.csv\" -i -x docker"
    )
)
def ask(
    request: str = typer.Argument(
        ..., help='Plain-English research request, e.g. '
        '"predict churn from data/customers.csv, aim for AUC 0.85". File paths '
        "mentioned in the request are detected and inspected automatically."
    ),
    data: Optional[list[Path]] = typer.Option(
        None, "--data", "-d",
        help="Data file(s) to attach explicitly (in addition to any auto-detected "
        "from the request text). CSV/TSV files are inspected - columns, types, "
        "row counts - so the composed spec describes the real schema.",
        rich_help_panel=_PANEL_EXEC,
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip the 'Run this research task now?' confirmation.",
        rich_help_panel=_PANEL_OVERSIGHT,
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Compose, show and save the task spec, but do not run it. Edit the saved "
        "YAML and launch later with `ramanujan run`.",
        rich_help_panel=_PANEL_EXEC,
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="LLM provider: gemini | openrouter | opencode (Zen) | custom. Default: "
        "auto-detected from whichever API key is set.",
        rich_help_panel=_PANEL_LLM,
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Human-in-the-loop mode: approve/guide each round's plans and review "
        "critic verdicts as the research runs.",
        rich_help_panel=_PANEL_OVERSIGHT,
    ),
    web: bool = typer.Option(
        False, "--web",
        help="Like --interactive, but decisions are made from the web dashboard "
        "(serve it with `ramanujan dashboard <run_dir>`).",
        rich_help_panel=_PANEL_OVERSIGHT,
    ),
    executor: Optional[str] = typer.Option(
        None, "--executor", "-x",
        help="Where generated training code runs: 'local' (this machine; local NVIDIA "
        "GPU auto-detected), 'docker' (network-isolated container), or 'runpod' "
        "(billed GPU pod, asks for confirmation).",
        rich_help_panel=_PANEL_EXEC,
    ),
    runs_root: Path = typer.Option(
        Path("runs"), "--runs-root", help="Directory for run artifacts.",
        rich_help_panel=_PANEL_OUTPUT,
    ),
):
    """Describe what you want researched in plain English.

    Ramanujan inspects any data files you mention, composes a validated task
    spec (metric, goal, budget, loading instructions), shows it for approval,
    saves it under tasks/generated/ for reproducibility, and runs it.
    """
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
    task = _apply_executor_choice(task, executor)
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
    if interactive and not web:
        from .hitl import ConsoleGate

        gate = ConsoleGate(console)

    director = ResearchDirector(task, suite, runs_root=runs_root, console=console, gate=gate)
    if web:
        director.gate = _arm_web_gate(director)
    result = director.run()
    raise typer.Exit(0 if result.best is not None else 1)


@app.command(
    epilog=(
        "Examples:\n\n"
        "  ramanujan bench tasks/demo_breast_cancer.yaml -n 5 --offline\n\n"
        "  ramanujan bench tasks/digits_multiclass.yaml tasks/churn_csv.yaml -n 3"
    )
)
def bench(
    task_files: list[Path] = typer.Argument(
        ..., help="One or more task YAMLs to benchmark."
    ),
    repeats: int = typer.Option(
        2, "--repeats", "-n", help="Full research runs per task.",
        rich_help_panel=_PANEL_EXEC,
    ),
    offline: bool = typer.Option(
        False, "--offline",
        help="Use the scripted demo session instead of a real LLM (no API key needed).",
        rich_help_panel=_PANEL_LLM,
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="LLM provider (default: auto-detect).",
        rich_help_panel=_PANEL_LLM,
    ),
    runs_root: Path = typer.Option(
        Path("runs"), "--runs-root", help="Directory for run artifacts.",
        rich_help_panel=_PANEL_OUTPUT,
    ),
):
    """Benchmark the agent system itself.

    Runs each task N times and reports goal-hit rate, mean best metric,
    experiments per run and a failure taxonomy - so changes to prompts,
    models or orchestration are judged with numbers, not anecdotes. Writes
    a markdown report under <runs-root>/benchmarks/.
    """
    from .bench import run_benchmark

    if offline:
        from .offline import build_offline_llm

        llm_factory = build_offline_llm
    else:
        from .llm.factory import build_llm_suite

        def llm_factory():
            return build_llm_suite(provider, Settings.from_env())

    run_benchmark(task_files, repeats, llm_factory, runs_root, console)


@app.command(
    epilog=(
        "Examples:\n\n"
        "  ramanujan dashboard                    (serves the LATEST run automatically)\n\n"
        "  ramanujan dashboard runs/<run_dir>     (serve a specific run)\n\n"
        "Works while the run is still executing (start it in a second terminal) "
        "and as a replay of any finished run. Note: `run --web` starts a dashboard "
        "for its own run automatically - no separate command needed."
    )
)
def dashboard(
    run_dir: Path = typer.Argument(
        Path("runs"),
        help="A run directory - or a runs root, in which case the most recent "
        "run is served (default: latest under runs/).",
    ),
    port: int = typer.Option(8787, "--port", help="Local port to serve the dashboard on."),
):
    """Serve a live web dashboard for a run.

    Streams the run's reasoning into the browser: planner hypotheses, engineer
    tool calls, metrics, analyst insights, critic verdicts and human
    interventions, with a live best-metric tracker. With no argument, serves
    the most recent run.
    """
    from .dashboard import resolve_run_dir, serve_dashboard

    resolved = resolve_run_dir(run_dir)
    if resolved != run_dir:
        console.print(f"[dim]Serving latest run: {resolved}[/dim]")
    serve_dashboard(resolved, port=port)


if __name__ == "__main__":
    app()
