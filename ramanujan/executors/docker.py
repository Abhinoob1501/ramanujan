"""Docker-sandboxed executor: runs generated training scripts in a disposable,
network-isolated container.

Containment (much stronger than the plain LocalExecutor):
- --network=none  : generated code cannot reach the network at all
- fresh container per run, --rm, non-root not required (nothing persists)
- only the experiment directory is mounted (read-write, for metrics.json)
- memory / CPU caps and the same wall-clock timeout as other executors

The image must contain the task's Python stack. `docker/Dockerfile` in this
repo builds a suitable default (python + scikit-learn + pandas):

    docker build -t ramanujan-sandbox docker/
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .base import METRICS_FILENAME, ExecutionResult, tail


class DockerExecutor:
    def __init__(
        self,
        timeout_seconds: int = 600,
        image: str = "ramanujan-sandbox",
        memory: str = "2g",
        cpus: str = "2",
        **_ignored,
    ):
        self.timeout_seconds = timeout_seconds
        self.image = image
        self.memory = memory
        self.cpus = cpus

    def run(self, workdir: Path, script_name: str = "train.py") -> ExecutionResult:
        script = workdir / script_name
        if not script.exists():
            return ExecutionResult(
                ok=False, duration_seconds=0.0, error=f"{script_name} was never written"
            )
        available, why = self._docker_available()
        if not available:
            return ExecutionResult(ok=False, duration_seconds=0.0, error=why)

        metrics_file = workdir / METRICS_FILENAME
        if metrics_file.exists():
            metrics_file.unlink()

        command = [
            "docker", "run", "--rm",
            "--network=none",
            f"--memory={self.memory}",
            f"--cpus={self.cpus}",
            "-v", f"{workdir.resolve()}:/work",
            "-w", "/work",
            self.image,
            "python", script_name,
        ]
        start = time.time()
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=self.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                ok=False,
                duration_seconds=time.time() - start,
                error=f"Timed out after {self.timeout_seconds}s (container killed).",
            )
        duration = time.time() - start

        if proc.returncode != 0:
            return ExecutionResult(
                ok=False,
                duration_seconds=duration,
                stdout_tail=tail(proc.stdout),
                stderr_tail=tail(proc.stderr),
                error=f"Container exited with code {proc.returncode}.",
            )
        if not metrics_file.exists():
            return ExecutionResult(
                ok=False,
                duration_seconds=duration,
                stdout_tail=tail(proc.stdout),
                stderr_tail=tail(proc.stderr),
                error=f"Script exited 0 but wrote no {METRICS_FILENAME} (required by contract).",
            )
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return ExecutionResult(
                ok=False, duration_seconds=duration,
                stdout_tail=tail(proc.stdout),
                error=f"{METRICS_FILENAME} is not valid JSON: {exc}",
            )
        return ExecutionResult(
            ok=True,
            duration_seconds=duration,
            stdout_tail=tail(proc.stdout),
            stderr_tail=tail(proc.stderr),
            metrics=metrics,
        )

    @staticmethod
    def _docker_available() -> tuple[bool, str]:
        try:
            probe = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False, "Docker is not installed or not on PATH (executor: docker requires it)."
        if probe.returncode != 0:
            return False, f"Docker daemon not reachable: {tail(probe.stderr, 300)}"
        return True, ""
