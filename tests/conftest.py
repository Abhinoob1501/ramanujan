import pytest

from ramanujan.task import BudgetSpec, MetricSpec, TaskSpec


@pytest.fixture
def task() -> TaskSpec:
    return TaskSpec(
        name="toy-task",
        description="A tiny task for tests.",
        dataset="No real data; scripts fabricate a score.",
        metric=MetricSpec(name="score", goal=0.9, direction="maximize"),
        budget=BudgetSpec(max_iterations=3, max_debug_attempts=2, experiment_timeout_seconds=30),
        executor="local",
        environment_notes="Plain Python, no extra libraries.",
    )
