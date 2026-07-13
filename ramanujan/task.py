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


class TaskSpec(BaseModel):
    name: str
    description: str
    dataset: str = Field(description="How to obtain/load the data, told verbatim to the engineer.")
    metric: MetricSpec
    budget: BudgetSpec = BudgetSpec()
    executor: Literal["local", "runpod"] = "local"
    environment_notes: str = "Python with scikit-learn, numpy and pandas available. CPU only."
    runpod: dict = Field(default_factory=dict, description="RunPod executor options (gpu_type, image, ...).")

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
