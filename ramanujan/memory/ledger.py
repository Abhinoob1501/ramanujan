"""SQLite-backed experiment ledger - the system's long-term memory.

Every experiment (including failed ones) is recorded with its hypothesis,
outcome and the analyst's insight. Summaries of the ledger are injected into
the planner's context each iteration, so the system provably builds on what it
has already learned instead of re-running the same ideas.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..task import MetricSpec

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER NOT NULL,
    hypothesis TEXT NOT NULL,
    approach TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',   -- running | success | failed
    metric_name TEXT,
    metric_value REAL,
    metrics_json TEXT,
    insight TEXT,
    error_summary TEXT,
    duration_seconds REAL,
    code_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class ExperimentRecord:
    id: int
    iteration: int
    hypothesis: str
    approach: str
    status: str
    metric_name: str | None = None
    metric_value: float | None = None
    metrics: dict = field(default_factory=dict)
    insight: str | None = None
    error_summary: str | None = None
    duration_seconds: float | None = None
    code_path: str | None = None


class ExperimentLedger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------- write

    def start_experiment(self, iteration: int, hypothesis: str, approach: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO experiments (iteration, hypothesis, approach) VALUES (?, ?, ?)",
            (iteration, hypothesis, approach),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def record_success(
        self,
        experiment_id: int,
        *,
        metric_name: str,
        metric_value: float,
        metrics: dict,
        duration_seconds: float,
        code_path: str,
    ) -> None:
        self._conn.execute(
            """UPDATE experiments
               SET status='success', metric_name=?, metric_value=?, metrics_json=?,
                   duration_seconds=?, code_path=?
               WHERE id=?""",
            (metric_name, metric_value, json.dumps(metrics), duration_seconds, code_path, experiment_id),
        )
        self._conn.commit()

    def record_failure(self, experiment_id: int, error_summary: str) -> None:
        self._conn.execute(
            "UPDATE experiments SET status='failed', error_summary=? WHERE id=?",
            (error_summary[:2000], experiment_id),
        )
        self._conn.commit()

    def record_insight(self, experiment_id: int, insight: str) -> None:
        self._conn.execute(
            "UPDATE experiments SET insight=? WHERE id=?", (insight, experiment_id)
        )
        self._conn.commit()

    # -------------------------------------------------------------------- read

    def all(self) -> list[ExperimentRecord]:
        rows = self._conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
        return [self._to_record(r) for r in rows]

    def best(self, metric: MetricSpec) -> ExperimentRecord | None:
        best: ExperimentRecord | None = None
        for rec in self.all():
            if rec.status != "success" or rec.metric_value is None:
                continue
            if best is None or metric.is_better(rec.metric_value, best.metric_value):
                best = rec
        return best

    def summary_markdown(self, metric: MetricSpec) -> str:
        """Compact history for injection into agent prompts."""
        records = self.all()
        if not records:
            return "No experiments have been run yet."
        lines = []
        for rec in records:
            if rec.status == "success":
                outcome = f"{rec.metric_name}={rec.metric_value:.4f}"
            elif rec.status == "failed":
                outcome = f"FAILED ({(rec.error_summary or 'unknown error')[:160]})"
            else:
                outcome = "running"
            lines.append(f"- Experiment {rec.id} (iteration {rec.iteration}): {rec.hypothesis}")
            lines.append(f"  Approach: {rec.approach}")
            lines.append(f"  Outcome: {outcome}")
            if rec.insight:
                lines.append(f"  Insight: {rec.insight}")
        best = self.best(metric)
        if best is not None:
            lines.append(
                f"\nCurrent best: experiment {best.id} with {best.metric_name}={best.metric_value:.4f} "
                f"(goal: {metric.goal}, direction: {metric.direction})."
            )
        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()

    # ---------------------------------------------------------------- internal

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ExperimentRecord:
        return ExperimentRecord(
            id=row["id"],
            iteration=row["iteration"],
            hypothesis=row["hypothesis"],
            approach=row["approach"],
            status=row["status"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            insight=row["insight"],
            error_summary=row["error_summary"],
            duration_seconds=row["duration_seconds"],
            code_path=row["code_path"],
        )
