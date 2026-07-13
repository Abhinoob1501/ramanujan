"""Cross-run knowledge base: semantic retrieval over the insights of past runs.

Every finished run distills its per-experiment insights into a global SQLite
store (`<runs_root>/knowledge.db`). Before planning, the orchestrator retrieves
the insights most relevant to the current task and injects them into the
planner's context - so lessons learned on one task transfer to the next.

Embedders are pluggable behind a tiny protocol:
- HashingEmbedder (default): local hashed bag-of-words with L2 norm. Zero
  network, zero dependencies, deterministic - retrieval quality is lexical
  but it works offline and in tests.
- GeminiEmbedder (opt-in via RAMANUJAN_EMBEDDER=gemini): true semantic vectors
  from the Gemini embeddings API.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_SCHEMA = """
CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    approach TEXT NOT NULL,
    insight TEXT NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    embedding TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Hashed bag-of-words embedding: each token is hashed into one of `dim`
    buckets; the resulting count vector is L2-normalized."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class GeminiEmbedder:
    """Semantic embeddings via the Gemini API (requires GEMINI_API_KEY)."""

    def __init__(self, model: str = "gemini-embedding-001"):
        from google import genai

        from ..config import Settings

        settings = Settings.from_env()
        if not settings.gemini_api_key:
            raise RuntimeError("GeminiEmbedder requires GEMINI_API_KEY.")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        result = self._client.models.embed_content(model=self.model, contents=text)
        return list(result.embeddings[0].values)


def build_embedder() -> Embedder:
    import os

    if os.environ.get("RAMANUJAN_EMBEDDER", "").lower() == "gemini":
        return GeminiEmbedder()
    return HashingEmbedder()


@dataclass
class KnowledgeItem:
    task_name: str
    run_id: str
    hypothesis: str
    approach: str
    insight: str
    metric_name: str | None
    metric_value: float | None
    similarity: float


class KnowledgeBase:
    def __init__(self, db_path: str | Path, embedder: Embedder | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or build_embedder()
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def add_insight(
        self,
        *,
        run_id: str,
        task_name: str,
        hypothesis: str,
        approach: str,
        insight: str,
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> None:
        text = f"{task_name}\n{hypothesis}\n{approach}\n{insight}"
        embedding = self.embedder.embed(text)
        self._conn.execute(
            """INSERT INTO insights
               (run_id, task_name, hypothesis, approach, insight, metric_name, metric_value, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, task_name, hypothesis, approach, insight, metric_name, metric_value,
             json.dumps(embedding)),
        )
        self._conn.commit()

    def retrieve(
        self, query: str, top_k: int = 5, exclude_run: str | None = None
    ) -> list[KnowledgeItem]:
        query_vec = self.embedder.embed(query)
        items: list[KnowledgeItem] = []
        for row in self._conn.execute("SELECT * FROM insights").fetchall():
            if exclude_run and row["run_id"] == exclude_run:
                continue
            similarity = _cosine(query_vec, json.loads(row["embedding"]))
            items.append(
                KnowledgeItem(
                    task_name=row["task_name"],
                    run_id=row["run_id"],
                    hypothesis=row["hypothesis"],
                    approach=row["approach"],
                    insight=row["insight"],
                    metric_name=row["metric_name"],
                    metric_value=row["metric_value"],
                    similarity=similarity,
                )
            )
        items.sort(key=lambda item: item.similarity, reverse=True)
        return [item for item in items[:top_k] if item.similarity > 0]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]

    def close(self) -> None:
        self._conn.close()


def format_for_prompt(items: list[KnowledgeItem]) -> str:
    """Render retrieved insights as a prompt block; empty string if none."""
    if not items:
        return ""
    lines = ["PRIOR KNOWLEDGE from past research runs (may or may not transfer - judge relevance):"]
    for item in items:
        outcome = (
            f" [{item.metric_name}={item.metric_value:.4f}]"
            if item.metric_name and item.metric_value is not None
            else ""
        )
        lines.append(f"- ({item.task_name}{outcome}) {item.insight}")
    return "\n".join(lines)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
