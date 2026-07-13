"""Research task specification, loaded from a YAML file.

A task tells the agent system WHAT to research (dataset, metric, goal) and under
WHICH constraints (iteration budget, timeouts, execution target). It never says
HOW - hypotheses and code are the agents' job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class MetricSpec(BaseModel):
    name: str = Field(description="Metric key the training script must report, e.g. 'roc_auc'.")
    goal: float = Field(description="Target value; reaching it lets the critic stop the run early.")
    direction: Literal["maximize", "minimize"] = "maximize"

    def is_better(self, a: float, b: float) -> bool:
        """True if value `a` beats value `b`."""
        return a > b if self.direction == "maximize" else a < b

    def goal_met(self, value: float) -> bool:
        return value >= self.goal if self.direction == "maximize" else value <= self.goal


class BudgetSpec(BaseModel):
    max_iterations: int = 5
    max_debug_attempts: int = 3
    experiment_timeout_seconds: int = 600
    parallel_branches: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Hypotheses the planner proposes per round; when >1 the critic "
        "allocates the experiment budget across them.",
    )
    max_experiments: int | None = Field(
        default=None,
        description="Total experiment cap across all rounds; defaults to "
        "max_iterations * parallel_branches.",
    )

    @property
    def experiment_cap(self) -> int:
        return self.max_experiments or self.max_iterations * self.parallel_branches


class TrackingSpec(BaseModel):
    wandb: bool = False
    wandb_project: str = "ramanujan"


class TaskSpec(BaseModel):
    name: str
    description: str
    dataset: str = Field(description="How to obtain/load the data, told verbatim to the engineer.")
    metric: MetricSpec
    budget: BudgetSpec = BudgetSpec()
    eda: bool = Field(
        default=True,
        description="Run the EDA agent before planning (skipped automatically for "
        "runpod tasks, whose data lives on the remote pod).",
    )
    executor: Literal["local", "docker", "runpod"] = "local"
    environment_notes: str = "Python with scikit-learn, numpy and pandas available. CPU only."
    runpod: dict = Field(default_factory=dict, description="RunPod executor options (gpu_type, image, ...).")
    docker: dict = Field(default_factory=dict, description="Docker executor options (image, memory, cpus).")
    data_files: list[str] = Field(
        default_factory=list,
        description="Local data files (e.g. CSVs) copied into each experiment's working "
        "directory, so generated scripts can read them by bare filename.",
    )
    tracking: TrackingSpec = TrackingSpec()

    def stage_data_files(self, dest: str | Path) -> list[str]:
        """Copy the task's data files into an experiment working directory.
        Returns the staged file names; raises if a file is missing."""
        import shutil

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        staged = []
        for entry in self.data_files:
            source = Path(entry)
            if not source.exists():
                raise FileNotFoundError(f"Task data file not found: {source}")
            shutil.copy2(source, dest / source.name)
            staged.append(source.name)
        return staged

    @field_validator("name")
    @classmethod
    def _slug_friendly(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task name must be non-empty")
        return v.strip()

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() else "-" for c in self.name.lower()).strip("-")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
