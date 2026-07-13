from ramanujan.events import EventLog, read_events_since


def test_emit_and_incremental_read(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.emit("plan", "plan_proposed", {"hypothesis": "h1"}, iteration=1, agent="planner")
    log.emit("build", "experiment_result", {"status": "success"}, iteration=1)
    log.emit("judge", "verdict", {"decision": "continue"}, iteration=1, agent="critic")

    events = read_events_since(path, 0)
    assert [e["seq"] for e in events] == [1, 2, 3]
    assert events[0]["payload"]["hypothesis"] == "h1"
    assert events[0]["agent"] == "planner"

    assert [e["seq"] for e in read_events_since(path, 2)] == [3]
    assert read_events_since(path, 3) == []


def test_missing_file_and_torn_line(tmp_path):
    path = tmp_path / "events.jsonl"
    assert read_events_since(path, 0) == []

    log = EventLog(path)
    log.emit("start", "run_started", {})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 2, "kind": "half-writ')  # simulate a torn concurrent append
    events = read_events_since(path, 0)
    assert len(events) == 1  # torn line skipped, valid one returned
