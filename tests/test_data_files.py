import pytest

from ramanujan.task import MetricSpec, TaskSpec


def make_task(files):
    return TaskSpec(
        name="csv-task",
        description="d",
        dataset="a csv",
        metric=MetricSpec(name="roc_auc", goal=0.8),
        data_files=files,
    )


def test_staging_copies_files(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    workdir = tmp_path / "iter_01"
    staged = make_task([str(source)]).stage_data_files(workdir)
    assert staged == ["data.csv"]
    assert (workdir / "data.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_missing_file_raises(tmp_path):
    task = make_task([str(tmp_path / "nope.csv")])
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        task.stage_data_files(tmp_path / "iter_01")


def test_bundled_churn_task_parses_and_stages(tmp_path):
    task = TaskSpec.from_yaml("tasks/churn_csv.yaml")
    assert task.budget.parallel_branches == 2
    assert task.budget.experiment_cap == 5
    staged = task.stage_data_files(tmp_path)
    assert staged == ["churn_sample.csv"]
    header = (tmp_path / "churn_sample.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "age,tenure_months,monthly_spend,support_tickets,is_premium,churned"
