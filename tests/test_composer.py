import json
from pathlib import Path

from ramanujan.composer import (
    ComposedTask,
    compose_task,
    detect_data_files,
    inspect_data_file,
    save_task,
)
from ramanujan.llm.base import LLMResponse
from ramanujan.llm.mock import MockLLM


def make_csv(tmp_path: Path) -> Path:
    path = tmp_path / "customers.csv"
    path.write_text(
        "age,plan,churned\n34,basic,0\n51,premium,1\n29,basic,0\n60,premium,1\n",
        encoding="utf-8",
    )
    return path


def test_detect_data_files_from_request_text(tmp_path, monkeypatch):
    csv = make_csv(tmp_path)
    monkeypatch.chdir(tmp_path)
    files = detect_data_files(f"predict churn from customers.csv please")
    assert files == [Path("customers.csv")]
    # nonexistent paths mentioned in prose are ignored
    assert detect_data_files("use imaginary_data.csv") == []


def test_detect_merges_explicit_files(tmp_path):
    csv = make_csv(tmp_path)
    files = detect_data_files("no file mentioned here", explicit=[csv])
    assert files == [csv]


def test_inspect_data_file_summarizes_schema(tmp_path):
    summary = inspect_data_file(make_csv(tmp_path))
    assert "customers.csv" in summary and "3 columns" in summary
    assert "age (numeric" in summary
    assert "plan (text" in summary
    assert "churned (binary numeric" in summary


def test_inspect_non_csv_reports_size_only(tmp_path):
    blob = tmp_path / "data.parquet"
    blob.write_bytes(b"\x00" * 2048)
    summary = inspect_data_file(blob)
    assert "not inspected" in summary


def composed_payload(**overrides) -> str:
    payload = {
        "name": "churn-prediction",
        "description": "Predict customer churn.",
        "dataset": "customers.csv with columns age, plan, churned (target).",
        "metric_name": "roc_auc",
        "metric_goal": 0.85,
        "metric_direction": "maximize",
        "max_iterations": 4,
        "parallel_branches": 2,
        "experiment_timeout_seconds": 300,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_compose_task_builds_valid_spec(tmp_path):
    csv = make_csv(tmp_path)
    llm = MockLLM(responses=[LLMResponse(text=composed_payload())])
    task = compose_task(llm, "predict churn from customers.csv, aim for auc 0.85", [csv])
    assert task.metric.name == "roc_auc" and task.metric.goal == 0.85
    assert task.budget.parallel_branches == 2
    assert task.executor == "local"
    assert task.data_files == [str(csv)]
    # the composer saw the real schema
    prompt = llm.calls[0].messages[0].content
    assert "churned (binary numeric" in prompt
    assert "predict churn from customers.csv" in prompt


def test_compose_without_data_mentions_sklearn_requirement():
    llm = MockLLM(responses=[LLMResponse(text=composed_payload(dataset="load_digits"))])
    compose_task(llm, "classify handwritten digits as well as possible")
    prompt = llm.calls[0].messages[0].content
    assert "No data files were provided" in prompt


def test_save_task_roundtrips_through_taskspec(tmp_path):
    from ramanujan.task import TaskSpec

    llm = MockLLM(responses=[LLMResponse(text=composed_payload())])
    task = compose_task(llm, "predict churn", [make_csv(tmp_path)])
    path = save_task(task, directory=tmp_path / "generated")
    reloaded = TaskSpec.from_yaml(path)
    assert reloaded.name == task.name
    assert reloaded.metric.goal == 0.85
    assert reloaded.data_files == task.data_files
