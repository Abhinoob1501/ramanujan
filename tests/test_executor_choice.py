import io

import pytest
import typer
from rich.console import Console

from ramanujan import hardware
from ramanujan.cli import _apply_executor_choice
from ramanujan.llm.mock import MockLLM
from ramanujan.orchestrator import ResearchDirector
from ramanujan.task import MetricSpec, TaskSpec


def make_task(**overrides) -> TaskSpec:
    base = dict(
        name="toy", description="d", dataset="none",
        metric=MetricSpec(name="score", goal=0.9), eda=False,
    )
    base.update(overrides)
    return TaskSpec.model_validate(base)


# ------------------------------------------------------------ hardware notes


@pytest.fixture(autouse=True)
def fresh_hardware_cache(monkeypatch):
    monkeypatch.delenv("RAMANUJAN_FORCE_CPU", raising=False)
    hardware.local_hardware_note.cache_clear()
    yield
    hardware.local_hardware_note.cache_clear()


def test_note_cpu_only_when_no_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_gpu_name", lambda: "")
    assert "no NVIDIA GPU" in hardware.local_hardware_note()


def test_note_gpu_with_cuda(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_gpu_name", lambda: "NVIDIA GeForce RTX 3050")
    monkeypatch.setattr(hardware, "_torch_cuda_available", lambda: True)
    note = hardware.local_hardware_note()
    assert "RTX 3050" in note and "MAY train on the GPU" in note


def test_note_gpu_without_torch(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_gpu_name", lambda: "NVIDIA GeForce RTX 3050")
    monkeypatch.setattr(hardware, "_torch_cuda_available", lambda: False)
    note = hardware.local_hardware_note()
    assert "not usable" in note and "CPU libraries" in note


def test_force_cpu_env_wins(monkeypatch):
    monkeypatch.setenv("RAMANUJAN_FORCE_CPU", "1")
    monkeypatch.setattr(hardware, "_detect_gpu_name", lambda: "NVIDIA GeForce RTX 4090")
    assert "CPU-only" in hardware.local_hardware_note()


def test_orchestrator_injects_hardware_note_for_local(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "_detect_gpu_name", lambda: "NVIDIA TestGPU")
    monkeypatch.setattr(hardware, "_torch_cuda_available", lambda: True)
    director = ResearchDirector(
        make_task(), MockLLM(), runs_root=tmp_path / "runs",
        console=Console(file=io.StringIO(), width=100),
    )
    assert "NVIDIA TestGPU" in director.task.environment_notes


def test_orchestrator_leaves_runpod_notes_alone(tmp_path):
    task = make_task(executor="runpod", runpod={"confirm_billing": True})
    director = ResearchDirector(
        task, MockLLM(), runs_root=tmp_path / "runs",
        console=Console(file=io.StringIO(), width=100),
    )
    assert "LOCAL HARDWARE" not in director.task.environment_notes


# --------------------------------------------------------- executor override


def test_no_flag_keeps_spec_executor():
    task = make_task(executor="docker")
    assert _apply_executor_choice(task, None).executor == "docker"


def test_flag_overrides_to_docker():
    assert _apply_executor_choice(make_task(), "docker").executor == "docker"


def test_unknown_executor_exits():
    with pytest.raises(typer.Exit):
        _apply_executor_choice(make_task(), "quantum")


def test_runpod_override_confirms_billing(monkeypatch):
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    task = _apply_executor_choice(make_task(), "runpod")
    assert task.executor == "runpod"
    assert task.runpod["confirm_billing"] is True
    assert "RunPod GPU pod" in task.environment_notes  # CPU-only notes superseded


def test_runpod_override_declined_aborts(monkeypatch):
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: False)
    with pytest.raises(typer.Exit):
        _apply_executor_choice(make_task(), "runpod")


def test_spec_level_billing_ack_skips_prompt(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("confirm should not be called")

    monkeypatch.setattr(typer, "confirm", explode)
    task = make_task(executor="runpod", runpod={"confirm_billing": True})
    assert _apply_executor_choice(task, None).executor == "runpod"
