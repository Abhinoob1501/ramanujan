"""Runs generated training scripts locally in a constrained subprocess.

Containment measures (documented honestly - this is a guardrail, not a jail;
see README "Safety" for the Docker roadmap item):
- runs in its own working directory under runs/
- secrets (env vars matching KEY/TOKEN/SECRET/...) are stripped from the child env
- hard wall-clock timeout, after which the process tree is killed
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .base import METRICS_FILENAME, ExecutionResult, tail

_SECRET_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE)


def _clean_env() -> dict:
    import os

    return {k: v for k, v in os.environ.items() if not _SECRET_PATTERN.search(k)}


class LocalExecutor:
    def __init__(self, timeout_seconds: int = 600):
        self.timeout_seconds = timeout_seconds

    def run(self, workdir: Path, script_name: str = "train.py") -> ExecutionResult:
        script = workdir / script_name
        if not script.exists():
            return ExecutionResult(
                ok=False, duration_seconds=0.0, error=f"{script_name} was never written"
            )
        metrics_file = workdir / METRICS_FILENAME
        if metrics_file.exists():
            metrics_file.unlink()  # stale metrics from a previous attempt must not leak through

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_name],
                cwd=workdir,
                env=_clean_env(),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                ok=False,
                duration_seconds=time.time() - start,
                stdout_tail=tail(exc.stdout or "" if isinstance(exc.stdout, str) else ""),
                error=f"Timed out after {self.timeout_seconds}s (wall-clock limit).",
            )
        duration = time.time() - start

        if proc.returncode != 0:
            return ExecutionResult(
                ok=False,
                duration_seconds=duration,
                stdout_tail=tail(proc.stdout),
                stderr_tail=tail(proc.stderr),
                error=f"Exited with code {proc.returncode}.",
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
                ok=False,
                duration_seconds=duration,
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
