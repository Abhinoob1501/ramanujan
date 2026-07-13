import shutil
import subprocess

import pytest

from ramanujan.executors.docker import DockerExecutor


TEST_IMAGE = "python:3.12-slim"


def docker_ready() -> bool:
    """True only if the daemon is up AND the test image is obtainable.

    Registry hiccups / pull rate limits (seen on CI runners) must SKIP this
    test, not fail it - it verifies our executor logic, not Docker Hub."""
    if shutil.which("docker") is None:
        return False
    try:
        daemon = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, timeout=20,
        )
        if daemon.returncode != 0:
            return False
        pull = subprocess.run(
            ["docker", "pull", TEST_IMAGE], capture_output=True, timeout=300
        )
        return pull.returncode == 0
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


@pytest.mark.skipif(not docker_ready(), reason="docker daemon or test image unavailable")
def test_real_container_run(tmp_path):
    (tmp_path / "train.py").write_text(
        "import json\n"
        "json.dump({'metric_name': 'score', 'metric_value': 0.5}, open('metrics.json', 'w'))\n",
        encoding="utf-8",
    )
    result = DockerExecutor(timeout_seconds=120, image=TEST_IMAGE).run(tmp_path)
    assert result.ok, result.error
    assert result.metrics["metric_value"] == 0.5
