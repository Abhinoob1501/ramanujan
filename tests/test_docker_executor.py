import shutil
import subprocess

import pytest

from ramanujan.executors.docker import DockerExecutor


def docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, timeout=20,
            ).returncode
            == 0
        )
    except Exception:
        return False


def test_missing_docker_is_a_clean_failure(tmp_path, monkeypatch):
    (tmp_path / "train.py").write_text("print('hi')", encoding="utf-8")
    executor = DockerExecutor(timeout_seconds=30)
    monkeypatch.setattr(
        DockerExecutor, "_docker_available", staticmethod(lambda: (False, "Docker is not installed"))
    )
    result = executor.run(tmp_path)
    assert not result.ok
    assert "Docker" in result.error


def test_missing_script_short_circuits(tmp_path):
    result = DockerExecutor(timeout_seconds=30).run(tmp_path)
    assert not result.ok
    assert "never written" in result.error


@pytest.mark.skipif(not docker_ready(), reason="docker daemon not available")
def test_real_container_run(tmp_path):
    (tmp_path / "train.py").write_text(
        "import json\n"
        "json.dump({'metric_name': 'score', 'metric_value': 0.5}, open('metrics.json', 'w'))\n",
        encoding="utf-8",
    )
    result = DockerExecutor(timeout_seconds=120, image="python:3.12-slim").run(tmp_path)
    assert result.ok, result.error
    assert result.metrics["metric_value"] == 0.5
