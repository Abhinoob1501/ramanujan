from ramanujan.memory.ledger import ExperimentLedger
from ramanujan.task import MetricSpec


def make_ledger(tmp_path):
    return ExperimentLedger(tmp_path / "ledger.db")


def test_success_and_best(tmp_path):
    ledger = make_ledger(tmp_path)
    metric = MetricSpec(name="acc", goal=0.99)

    first = ledger.start_experiment(1, "baseline works", "logreg")
    ledger.record_success(
        first, metric_name="acc", metric_value=0.90, metrics={"metric_value": 0.90},
        duration_seconds=1.0, code_path="a.py",
    )
    second = ledger.start_experiment(2, "trees are better", "boosting")
    ledger.record_success(
        second, metric_name="acc", metric_value=0.95, metrics={"metric_value": 0.95},
        duration_seconds=2.0, code_path="b.py",
    )

    best = ledger.best(metric)
    assert best is not None and best.id == second and best.metric_value == 0.95


def test_minimize_direction(tmp_path):
    ledger = make_ledger(tmp_path)
    metric = MetricSpec(name="rmse", goal=0.1, direction="minimize")
    a = ledger.start_experiment(1, "h1", "a")
    ledger.record_success(a, metric_name="rmse", metric_value=0.5, metrics={},
                          duration_seconds=1, code_path="a.py")
    b = ledger.start_experiment(2, "h2", "b")
    ledger.record_success(b, metric_name="rmse", metric_value=0.2, metrics={},
                          duration_seconds=1, code_path="b.py")
    assert ledger.best(metric).id == b
    assert not metric.goal_met(0.2)
    assert metric.goal_met(0.05)


def test_failure_and_summary(tmp_path):
    ledger = make_ledger(tmp_path)
    metric = MetricSpec(name="acc", goal=0.99)
    exp = ledger.start_experiment(1, "this will break", "bad code")
    ledger.record_failure(exp, "ImportError: no such module")
    exp2 = ledger.start_experiment(2, "this works", "good code")
    ledger.record_success(exp2, metric_name="acc", metric_value=0.97, metrics={},
                          duration_seconds=1, code_path="x.py")
    ledger.record_insight(exp2, "simple models suffice")

    summary = ledger.summary_markdown(metric)
    assert "FAILED" in summary and "ImportError" in summary
    assert "simple models suffice" in summary
    assert "Current best" in summary and "0.9700" in summary
    assert ledger.best(metric).id == exp2


def test_empty_summary(tmp_path):
    ledger = make_ledger(tmp_path)
    assert "No experiments" in ledger.summary_markdown(MetricSpec(name="acc", goal=1.0))
    assert ledger.best(MetricSpec(name="acc", goal=1.0)) is None
