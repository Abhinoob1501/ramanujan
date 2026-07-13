from .base import ExecutionResult, Executor
from .local import LocalExecutor

__all__ = ["ExecutionResult", "Executor", "LocalExecutor", "build_executor"]


def build_executor(task, workdir):
    """Pick the executor requested by the task spec."""
    if task.executor == "runpod":
        from .runpod import RunPodExecutor

        return RunPodExecutor(
            timeout_seconds=task.budget.experiment_timeout_seconds, **task.runpod
        )
    return LocalExecutor(timeout_seconds=task.budget.experiment_timeout_seconds)
