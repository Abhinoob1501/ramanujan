"""Tests for the 10-change improvement batch: per-role routing, usage telemetry,
engineer pre-flight + debug nudges, fold-spread reporting, knowledge-base code
reuse, and benchmark mode."""

import io
import json

import pytest
from rich.console import Console

from ramanujan.agents.engineer import EngineerAgent
from ramanujan.agents.roles import ExperimentPlan
from ramanujan.bench import run_benchmark
from ramanujan.config import Settings
from ramanujan.executors.local import LocalExecutor
from ramanujan.llm.base import LLMResponse, LLMUsage, Pacer, ToolCall
from ramanujan.llm.factory import LLMSuite, build_llm_suite
from ramanujan.llm.mock import MockLLM
from ramanujan.llm.openai_compat import OpenAICompatClient
from ramanujan.memory.knowledge import HashingEmbedder, KnowledgeBase
from ramanujan.memory.ledger import _fold_spread
from ramanujan.task import MetricSpec, TaskSpec

PLAN = ExperimentPlan(hypothesis="h", approach="a", rationale="r")
FAST = Settings(min_seconds_between_llm_calls=0, max_llm_calls_per_run=100)


# ---------------------------------------------------------------- role routing


@pytest.fixture()
def suite_env(monkeypatch):
    monkeypatch.setattr("ramanujan.llm.factory.load_dotenv", lambda *a, **k: None)
    for env in ("GEMINI_API_KEY", "RAMANUJAN_PROVIDER", "RAMANUJAN_MODEL",
                "RAMANUJAN_BASE_URL", "RAMANUJAN_API_KEY", "OPENCODE_API_KEY",
                "OPENCODE_ZEN_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    for role in ("PLANNER", "ENGINEER", "ANALYST", "CRITIC", "REPORTER"):
        monkeypatch.delenv(f"RAMANUJAN_MODEL_{role}", raising=False)
    return monkeypatch


def test_suite_shares_one_client_without_overrides(suite_env):
    suite = build_llm_suite(settings=FAST)
    assert suite.planner is suite.engineer is suite.critic is suite.reporter


def test_suite_routes_roles_to_override_models(suite_env):
    suite_env.setenv("RAMANUJAN_MODEL_ENGINEER", "qwen/qwen3-coder")
    suite_env.setenv("RAMANUJAN_MODEL_CRITIC", "qwen/qwen3-coder")
    suite = build_llm_suite(settings=FAST)
    assert suite.engineer.model == "qwen/qwen3-coder"
    assert suite.engineer is suite.critic  # same override -> shared client
    assert suite.planner is not suite.engineer
    # all clients share one pacer so the provider rate limit holds
    assert suite.planner._pacer is suite.engineer._pacer


def test_suite_usage_aggregates_deduplicated():
    a, b = MockLLM(), MockLLM()
    a.usage.add(prompt_tokens=10, completion_tokens=5, cost_usd=0.01)
    b.usage.add(prompt_tokens=100, completion_tokens=50)
    suite = LLMSuite(planner=a, engineer=b, analyst=a, critic=a, reporter=a)
    total = suite.usage()
    assert total.calls == 2
    assert total.prompt_tokens == 110
    assert total.cost_usd == pytest.approx(0.01)


# ------------------------------------------------------------- usage telemetry


def test_openai_compat_records_usage_and_cost():
    client = OpenAICompatClient(
        base_url="https://example.test/v1", api_key="k", model="m", settings=FAST
    )
    client._post = lambda payload: type(
        "R", (), {
            "status_code": 200,
            "json": lambda self=None: {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "cost": 0.0004},
            },
            "text": "",
        },
    )()
    from ramanujan.llm.base import ChatMessage

    client.generate(system="s", messages=[ChatMessage(role="user", content="x")])
    assert client.usage.calls == 1
    assert client.usage.prompt_tokens == 12
    assert client.usage.cost_usd == pytest.approx(0.0004)


def test_pacer_is_shared_state():
    pacer = Pacer(0)
    pacer.wait()
    assert pacer._last_call_ts > 0


# --------------------------------------------------- engineer: pre-flight etc.


def toy_task(**overrides) -> TaskSpec:
    base = dict(
        name="toy", description="d", dataset="none",
        metric=MetricSpec(name="score", goal=0.9),
    )
    base.update(overrides)
    return TaskSpec.model_validate(base)


def test_preflight_rejects_missing_imports(tmp_path):
    script = "import definitely_not_installed_xyz\nprint('never runs')\n"
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": script})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="gave up"),
            LLMResponse(text="gave up again"),  # prose-only corrective retry
        ]
    )
    engineer = EngineerAgent(llm, toy_task(), LocalExecutor(30), tmp_path / "iter")
    engineer.implement(PLAN)
    feedback = [m.content for call in llm.calls for m in call.messages if m.role == "tool"]
    assert any("PRE-FLIGHT REJECTED" in f and "definitely_not_installed_xyz" in f for f in feedback)
    assert engineer.run_attempts == 0  # rejection must not consume a debug attempt


def test_failure_feedback_reports_remaining_attempts(tmp_path):
    good = "import json\njson.dump({'metric_name': 'score', 'metric_value': 1.0}, open('metrics.json','w'))\n"
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": "raise ValueError('x')\n"})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": good})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="fixed"),
        ]
    )
    outcome = EngineerAgent(llm, toy_task(), LocalExecutor(30), tmp_path / "iter").implement(PLAN)
    assert outcome.success
    feedback = [m.content for call in llm.calls for m in call.messages if m.role == "tool"]
    assert any("debug attempt(s) remaining" in f for f in feedback)


def test_reference_code_lands_in_prompt(tmp_path):
    good = "import json\njson.dump({'metric_name': 'score', 'metric_value': 1.0}, open('metrics.json','w'))\n"
    llm = MockLLM(
        responses=[
            LLMResponse(tool_calls=[ToolCall("write_file", {"filename": "train.py", "content": good})]),
            LLMResponse(tool_calls=[ToolCall("run_script", {})]),
            LLMResponse(text="done"),
        ]
    )
    engineer = EngineerAgent(llm, toy_task(), LocalExecutor(30), tmp_path / "iter")
    engineer.implement(PLAN, reference_code="# winning past solution")
    assert "winning past solution" in llm.calls[0].messages[0].content


# ------------------------------------------------------------- fold statistics


def test_fold_spread_from_scores_and_fallbacks():
    assert _fold_spread({"fold_scores": [0.8, 1.0]}) == pytest.approx(0.1)
    assert _fold_spread({"fold_std": 0.05}) == pytest.approx(0.05)
    assert _fold_spread({"metric_value": 0.9}) is None
    assert _fold_spread({"fold_scores": ["bad", "data"]}) is None


# ------------------------------------------------------- knowledge: code reuse


def test_knowledge_base_stores_and_retrieves_code(tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder())
    kb.add_insight(
        run_id="r1", task_name="tabular churn", hypothesis="boosting wins",
        approach="hist gradient boosting", insight="trees dominate tabular churn",
        code="print('winning script')",
    )
    item = kb.retrieve("tabular churn boosting", top_k=1)[0]
    assert item.code == "print('winning script')"


def test_old_knowledge_db_is_migrated(tmp_path):
    import sqlite3

    db = tmp_path / "kb.db"
    conn = sqlite3.connect(db)  # v1 schema: no `code` column
    conn.executescript(
        """CREATE TABLE insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
            task_name TEXT NOT NULL, hypothesis TEXT NOT NULL, approach TEXT NOT NULL,
            insight TEXT NOT NULL, metric_name TEXT, metric_value REAL,
            embedding TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')));"""
    )
    conn.execute(
        "INSERT INTO insights (run_id, task_name, hypothesis, approach, insight, embedding) "
        "VALUES ('r0', 'legacy', 'h', 'a', 'legacy insight', ?)",
        (json.dumps(HashingEmbedder().embed("legacy insight")),),
    )
    conn.commit()
    conn.close()

    kb = KnowledgeBase(db, embedder=HashingEmbedder())  # must migrate, not crash
    items = kb.retrieve("legacy insight", top_k=1)
    assert items and items[0].code == ""


# ------------------------------------------------------------------- benchmark


def test_benchmark_offline_two_repeats(tmp_path):
    pytest.importorskip("sklearn")
    from ramanujan.offline import build_offline_llm

    task_yaml = tmp_path / "demo.yaml"
    task_yaml.write_text(
        "name: breast-cancer-diagnosis\n"
        "description: demo\n"
        "dataset: sklearn breast cancer\n"
        "metric: {name: roc_auc, goal: 0.99}\n"
        "budget: {max_iterations: 4}\n",
        encoding="utf-8",
    )
    report = run_benchmark(
        [task_yaml], repeats=2, llm_factory=build_offline_llm,
        runs_root=tmp_path / "runs", console=Console(file=io.StringIO(), width=100),
    )
    text = report.read_text(encoding="utf-8")
    assert "breast-cancer-diagnosis" in text
    assert "100%" in text  # both offline runs hit the goal
    assert "- run 2:" in text
