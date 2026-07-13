"""Optional Weights & Biases experiment tracking.

Enabled per task via `tracking: {wandb: true, wandb_project: ...}`. Each
completed experiment becomes one W&B run (config = hypothesis/approach,
metrics logged, tags = ramanujan run id). Degrades to a no-op with a console
note when wandb is not installed or WANDB_API_KEY is not configured - a
missing tracker must never break research.
"""

from __future__ import annotations

import os

from .agents.roles import ExperimentPlan
from .task import TaskSpec


class ExperimentTracker:
    def __init__(self, task: TaskSpec, run_id: str, console=None):
        self.task = task
        self.run_id = run_id
        self.console = console
        self._wandb = None
        if not task.tracking.wandb:
            return
        try:
            import wandb  # type: ignore

            if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
                self._note(
                    "W&B tracking requested but WANDB_API_KEY is not set "
                    "(or set WANDB_MODE=offline); continuing without tracking."
                )
                return
            self._wandb = wandb
        except ImportError:
            self._note("W&B tracking requested but wandb is not installed "
                       "(pip install wandb); continuing without tracking.")

    @property
    def enabled(self) -> bool:
        return self._wandb is not None

    def log_experiment(
        self,
        *,
        experiment_id: int,
        iteration: int,
        plan: ExperimentPlan,
        status: str,
        metrics: dict | None = None,
        duration_seconds: float | None = None,
        error_summary: str | None = None,
    ) -> None:
        if self._wandb is None:
            return
        try:
            run = self._wandb.init(
                project=self.task.tracking.wandb_project,
                name=f"{self.run_id}-exp{experiment_id}",
                group=self.run_id,
                tags=["ramanujan", self.task.slug],
                config={
                    "task": self.task.name,
                    "iteration": iteration,
                    "hypothesis": plan.hypothesis,
                    "approach": plan.approach,
                    "status": status,
                },
                reinit=True,
            )
            payload: dict = {"duration_seconds": duration_seconds or 0.0}
            if metrics:
                payload.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            if error_summary:
                run.summary["error"] = error_summary[:500]
            run.log(payload)
            run.finish(exit_code=0 if status == "success" else 1)
        except Exception as exc:  # tracking must never break research
            self._note(f"W&B logging failed ({exc}); continuing without tracking.")
            self._wandb = None

    def _note(self, message: str) -> None:
        if self.console is not None:
            self.console.print(f"[yellow]{message}[/yellow]")
