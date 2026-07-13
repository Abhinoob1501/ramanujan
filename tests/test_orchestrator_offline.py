"""End-to-end test: the full research loop with the scripted offline session.

This actually trains sklearn models in sandboxed subprocesses, exercises the
engineer's debug loop (iteration 2 starts with a deliberate ImportError),
writes the ledger and renders the report.
"""

import io

import pytest
from rich.console import Console

sklearn = pytest.importorskip("sklearn")

from ramanujan.memory.ledger import ExperimentLedger
from ramanujan.offline import build_offline_llm
from ramanujan.orchestrator import ResearchDirector
from ramanujan.task import TaskSpec

TASK_YAML = """
name: breast-cancer-diagnosis
description: Maximize ROC-AUC on the Wisconsin breast-cancer dataset.
dataset: sklearn.datasets.load_breast_cancer(return_X_y=True)
metric: {name: roc_auc, goal: 0.99, direction: maximize}
budget: {max_iterations: 4, max_debug_attempts: 3, experiment_timeout_seconds: 300}
executor: local
"""


@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    import yaml

    task = TaskSpec.model_validate(yaml.safe_load(TASK_YAML))
    console = Console(file=io.StringIO(), width=100)
    director = ResearchDirector(
        task, build_offline_llm(), runs_root=tmp_path_factory.mktemp("runs"), console=console
    )
    return director.run()


def test_run_completes_with_goal_met(run_result):
    assert run_result.stop_reason == "stop_goal_met"
    assert run_result.iterations_run == 2
    assert run_result.best is not None
    assert run_result.best.metric_value > 0.98  # real sklearn CV result


def test_ledger_records_both_experiments(run_result):
    ledger = ExperimentLedger(run_result.run_dir / "ledger.db")
    records = ledger.all()
    assert len(records) == 2
    assert all(r.status == "success" for r in records)
    assert all(r.insight for r in records)


def test_debug_loop_left_fixed_code_on_disk(run_result):
    code = (run_result.run_dir / "iter_02" / "train.py").read_text(encoding="utf-8")
    assert "HistGradientBoostingClassifier" in code  # fixed version, not the buggy one


def test_report_rendered(run_result):
    report = run_result.report_path.read_text(encoding="utf-8")
    assert "## Data exploration" in report  # EDA findings section
    assert "## Leaderboard" in report
    assert "## Conclusions" in report
    assert "roc_auc" in report
    assert "```python" in report  # best code appendix


def test_eda_ran_for_real(run_result):
    # the EDA script actually executed against sklearn data
    assert (run_result.run_dir / "eda" / "eda.py").exists()
    findings = (run_result.run_dir / "eda" / "findings.json").read_text(encoding="utf-8")
    assert "standardization" in findings.lower() or "standardized" in findings.lower()
