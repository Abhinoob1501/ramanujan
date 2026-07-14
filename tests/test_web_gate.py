"""File-backed web gate: FileGate exchange protocol + dashboard HTTP API."""

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from ramanujan.agents.roles import ExperimentPlan, Verdict
from ramanujan.dashboard import make_handler
from ramanujan.hitl import FileGate

PLAN = ExperimentPlan(hypothesis="h", approach="a", rationale="r")


def respond_later(control_dir, action, guidance="", delay=0.1, force_id=None):
    def responder():
        time.sleep(delay)
        request = json.loads((control_dir / "gate_request.json").read_text(encoding="utf-8"))
        (control_dir / "gate_response.json").write_text(
            json.dumps({"id": force_id if force_id is not None else request["id"],
                        "action": action, "guidance": guidance}),
            encoding="utf-8",
        )

    thread = threading.Thread(target=responder, daemon=True)
    thread.start()
    return thread


def make_gate(tmp_path, **kwargs):
    from io import StringIO

    from rich.console import Console

    return FileGate(tmp_path, console=Console(file=StringIO(), width=100),
                    poll_interval=0.02, **kwargs)


def test_plan_review_revise_roundtrip(tmp_path):
    gate = make_gate(tmp_path)
    respond_later(tmp_path, "revise", guidance="only tree models")
    review = gate.review_plans([PLAN], iteration=1)
    assert review.action == "revise"
    assert review.guidance == "only tree models"
    # both control files consumed
    assert not (tmp_path / "gate_request.json").exists()
    assert not (tmp_path / "gate_response.json").exists()


def test_verdict_review_override_roundtrip(tmp_path):
    gate = make_gate(tmp_path)
    respond_later(tmp_path, "continue_anyway")
    review = gate.review_verdict(Verdict(decision="stop_goal_met", reasoning="r"), 1)
    assert review.action == "continue_anyway"


def test_stale_response_id_is_ignored(tmp_path):
    gate = make_gate(tmp_path)
    respond_later(tmp_path, "stop", delay=0.05, force_id=999)     # wrong id: ignore
    respond_later(tmp_path, "approve", delay=0.25)                # right id: accept
    review = gate.review_plans([PLAN], iteration=1)
    assert review.action == "approve"


def test_timeout_auto_approves(tmp_path):
    gate = make_gate(tmp_path, timeout_seconds=0.15)
    start = time.time()
    review = gate.review_plans([PLAN], iteration=1)
    assert review.action == "approve"
    assert time.time() - start < 5
    assert not (tmp_path / "gate_request.json").exists()


def test_request_file_describes_the_decision(tmp_path):
    gate = make_gate(tmp_path, timeout_seconds=0.15)
    gate.review_plans([PLAN], iteration=3)
    # inspect what a UI would have seen (re-run with a reader thread)
    seen = {}

    def reader():
        time.sleep(0.05)
        seen.update(json.loads((tmp_path / "gate_request.json").read_text(encoding="utf-8")))

    threading.Thread(target=reader, daemon=True).start()
    gate.review_plans([PLAN], iteration=3)
    assert seen["type"] == "plan" and seen["iteration"] == 3
    assert seen["payload"]["plans"][0]["hypothesis"] == "h"


# -------------------------------------------------- run-dir resolution & auto-serve


def test_resolve_prefers_exact_run_dir(tmp_path):
    from ramanujan.dashboard import resolve_run_dir

    run = tmp_path / "20260101_000000_task_aaaaaa"
    run.mkdir()
    (run / "events.jsonl").write_text("", encoding="utf-8")
    assert resolve_run_dir(run) == run


def test_resolve_picks_latest_run_under_root(tmp_path):
    from ramanujan.dashboard import resolve_run_dir

    for name in ("20260101_000000_old_aaaaaa", "20260102_000000_new_bbbbbb"):
        d = tmp_path / name
        d.mkdir()
        (d / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "not_a_run").mkdir()  # no events.jsonl -> ignored
    assert resolve_run_dir(tmp_path).name == "20260102_000000_new_bbbbbb"


def test_start_dashboard_server_binds_and_serves(tmp_path):
    from ramanujan.dashboard import start_dashboard_server

    (tmp_path / "events.jsonl").write_text(
        '{"seq": 1, "phase": "start", "kind": "run_started", "payload": {}}\n',
        encoding="utf-8",
    )
    server, port = start_dashboard_server(tmp_path, port=0)  # ephemeral port
    try:
        data = get_json(f"http://127.0.0.1:{port}/api/events?since=0")
        assert data["events"][0]["kind"] == "run_started"
    finally:
        server.shutdown()


# ------------------------------------------------------------- dashboard API


@pytest.fixture()
def server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as res:
        return json.loads(res.read().decode("utf-8"))


def test_dashboard_gate_api_roundtrip(server):
    run_dir, base = server
    assert get_json(f"{base}/api/gate") == {}  # nothing pending

    request = {"id": 7, "type": "plan", "iteration": 2,
               "payload": {"plans": [{"hypothesis": "h", "approach": "a"}]}}
    (run_dir / "gate_request.json").write_text(json.dumps(request), encoding="utf-8")
    assert get_json(f"{base}/api/gate")["id"] == 7

    body = json.dumps({"id": 7, "action": "revise", "guidance": "try boosting"}).encode()
    post = urllib.request.Request(
        f"{base}/api/gate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(post, timeout=10) as res:
        assert json.loads(res.read())["ok"] is True

    saved = json.loads((run_dir / "gate_response.json").read_text(encoding="utf-8"))
    assert saved == {"id": 7, "action": "revise", "guidance": "try boosting"}
    assert get_json(f"{base}/api/gate") == {}  # answered -> no longer pending


def test_dashboard_rejects_bad_decisions(server):
    run_dir, base = server
    body = json.dumps({"id": 1, "action": "launch_the_missiles"}).encode()
    post = urllib.request.Request(
        f"{base}/api/gate", data=body, headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(post, timeout=10)
    assert exc.value.code == 400
    assert not (run_dir / "gate_response.json").exists()
