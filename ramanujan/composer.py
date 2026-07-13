"""Natural-language task composer: `ramanujan ask "..."`.

Turns a plain-English research request into a validated TaskSpec:
1. file paths mentioned in the request (or passed with --data) are inspected -
   real column names, types and row counts go into the composer's context,
2. one structured LLM call drafts the spec (metric, goal, budget, dataset
   loading instructions), validated against a Pydantic schema with self-repair,
3. the result is saved as a normal task YAML so every ad-hoc request stays
   reproducible and hand-editable.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .agents.base import ask_json
from .llm.base import LLMClient
from .task import BudgetSpec, MetricSpec, TaskSpec

COMPOSER_SYSTEM = """You are the research director of an autonomous machine-learning system.
A user states, in plain language, what they want researched. You translate that into a
precise, runnable research task specification.

Rules:
- The metric must be computable with scikit-learn (e.g. accuracy, roc_auc, f1, rmse, mae, r2).
- Pick a realistic goal: ambitious enough to require real work, not obviously impossible.
  If the user names a target, respect it.
- The dataset field must tell an engineer EXACTLY how to load the data:
  * if data files are provided, reference them by bare filename (they will be placed in
    the working directory) and describe their columns, including which column is the target;
  * otherwise name a sklearn.datasets loader or precise synthetic-data recipe.
- Budget: default 3-5 rounds; use 2 parallel branches only when comparing model families
  is clearly valuable. Respect any budget the user states.
- The environment is CPU-only Python with scikit-learn, numpy and pandas. Do not assume
  other libraries.
- Keep the task name short, lowercase and hyphenated."""

_FILE_PATTERN = re.compile(r"[\w.\\/:~-]+\.(?:csv|tsv|txt|json|parquet|xlsx)\b", re.IGNORECASE)

_MAX_SNIFF_ROWS = 500


class ComposedTask(BaseModel):
    name: str = Field(description="Short hyphenated task name, e.g. 'churn-prediction'.")
    description: str = Field(description="1-3 sentence statement of the research objective.")
    dataset: str = Field(description="Exact loading instructions for the engineer.")
    metric_name: str
    metric_goal: float
    metric_direction: Literal["maximize", "minimize"] = "maximize"
    max_iterations: int = Field(default=4, ge=1, le=8)
    parallel_branches: int = Field(default=1, ge=1, le=3)
    experiment_timeout_seconds: int = Field(default=300, ge=60, le=1800)


def detect_data_files(request: str, explicit: list[Path] | None = None) -> list[Path]:
    """File paths named in the request (that exist on disk) plus --data files."""
    found: list[Path] = []
    for match in _FILE_PATTERN.findall(request):
        path = Path(match.strip("'\""))
        if path.exists() and path.is_file():
            found.append(path)
    for path in explicit or []:
        if path not in found:
            found.append(path)
    return found


def inspect_data_file(path: Path) -> str:
    """Human/LLM-readable summary of a data file: shape, columns, inferred types."""
    if path.suffix.lower() not in (".csv", ".tsv"):
        size_kb = path.stat().st_size / 1024
        return f"{path.name}: {path.suffix} file, {size_kb:.0f} KB (contents not inspected)."
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return f"{path.name}: empty file."
        rows = []
        for i, row in enumerate(reader):
            if i >= _MAX_SNIFF_ROWS:
                break
            rows.append(row)
    lines = [f"{path.name}: {len(rows)}{'+' if len(rows) >= _MAX_SNIFF_ROWS else ''} data rows, "
             f"{len(header)} columns."]
    for idx, column in enumerate(header):
        values = [r[idx] for r in rows if idx < len(r) and r[idx] != ""]
        kind = _infer_kind(values)
        sample = ", ".join(list(dict.fromkeys(values))[:4])
        lines.append(f"  - {column} ({kind}; e.g. {sample})")
    return "\n".join(lines)


def _infer_kind(values: list[str]) -> str:
    if not values:
        return "empty"
    numeric = 0
    for value in values:
        try:
            float(value)
            numeric += 1
        except ValueError:
            pass
    if numeric == len(values):
        distinct = set(values)
        if distinct <= {"0", "1", "0.0", "1.0"}:
            return "binary numeric"
        return "numeric"
    return f"text ({min(len(set(values)), 51)}{'+' if len(set(values)) > 50 else ''} distinct)"


def compose_task(
    llm: LLMClient, request: str, data_files: list[Path] | None = None
) -> TaskSpec:
    data_files = data_files or []
    summaries = "\n\n".join(inspect_data_file(p) for p in data_files)
    data_block = (
        f"\n\nDATA FILES the user provided (will be staged into the working directory):\n{summaries}"
        if summaries
        else "\n\nNo data files were provided - the task must use a sklearn dataset "
        "loader or a synthetic-data recipe."
    )
    draft = ask_json(
        llm,
        system=COMPOSER_SYSTEM,
        prompt=f"USER REQUEST:\n{request}{data_block}\n\nCompose the research task.",
        model_cls=ComposedTask,
    )
    return TaskSpec(
        name=draft.name,
        description=draft.description,
        dataset=draft.dataset,
        metric=MetricSpec(
            name=draft.metric_name, goal=draft.metric_goal, direction=draft.metric_direction
        ),
        budget=BudgetSpec(
            max_iterations=draft.max_iterations,
            parallel_branches=draft.parallel_branches,
            experiment_timeout_seconds=draft.experiment_timeout_seconds,
        ),
        executor="local",
        data_files=[str(p) for p in data_files],
    )


def save_task(task: TaskSpec, directory: str | Path = "tasks/generated") -> Path:
    import time

    import yaml

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task.slug}_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
    path.write_text(
        yaml.safe_dump(task.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
