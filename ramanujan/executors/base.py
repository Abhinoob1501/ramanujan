"""Executor contract.

An executor runs one training script inside a working directory and returns a
structured result. The script's side of the contract: write a `metrics.json`
file (or print a `RAMANUJAN_METRICS::{...}` line for remote executors) containing
at least the task's metric key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

METRICS_FILENAME = "metrics.json"
METRICS_SENTINEL = "RAMANUJAN_METRICS::"

TAIL_CHARS = 4000


@dataclass
class ExecutionResult:
    ok: bool
    duration_seconds: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    metrics: dict = field(default_factory=dict)
    error: str = ""

    def to_feedback(self) -> str:
        """Human/LLM-readable summary handed back to the engineer agent."""
        if self.ok:
            return (
                f"Script succeeded in {self.duration_seconds:.1f}s.\n"
                f"metrics.json: {self.metrics}\n"
                f"--- stdout (tail) ---\n{self.stdout_tail or '(empty)'}"
            )
        return (
            f"Script FAILED after {self.duration_seconds:.1f}s: {self.error}\n"
            f"--- stderr (tail) ---\n{self.stderr_tail or '(empty)'}\n"
            f"--- stdout (tail) ---\n{self.stdout_tail or '(empty)'}"
        )


class Executor(Protocol):
    def run(self, workdir: Path, script_name: str = "train.py") -> ExecutionResult: ...


def tail(text: str, limit: int = TAIL_CHARS) -> str:
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]
