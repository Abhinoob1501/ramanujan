"""Append-only event stream for a research run.

The orchestrator emits one JSON line per event to `<run_dir>/events.jsonl`.
Because it's a plain file, the live dashboard (ramanujan/dashboard.py) can
tail it from a separate process while the run is still executing, and any
finished run can be replayed after the fact.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def emit(
        self,
        phase: str,
        kind: str,
        payload: dict | None = None,
        *,
        iteration: int | None = None,
        agent: str | None = None,
    ) -> None:
        self._seq += 1
        event = {
            "seq": self._seq,
            "ts": time.time(),
            "phase": phase,
            "kind": kind,
            "iteration": iteration,
            "agent": agent,
            "payload": payload or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_events_since(path: str | Path, since_seq: int = 0) -> list[dict]:
    """Read events with seq > since_seq. Tolerates a torn final line while a
    writer is mid-append."""
    path = Path(path)
    if not path.exists():
        return []
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn write in progress; the next poll will get it
            if event.get("seq", 0) > since_seq:
                events.append(event)
    return events
