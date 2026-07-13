import json
import os

from ramanujan.executors.local import LocalExecutor, _clean_env


def write_script(tmp_path, body: str):
    (tmp_path / "train.py").write_text(body, encoding="utf-8")


def test_success_with_metrics(tmp_path):
    write_script(
        tmp_path,
        "import json\n"
        "print('training...')\n"
        "json.dump({'metric_name': 'score', 'metric_value': 0.93}, open('metrics.json', 'w'))\n",
    )
    result = LocalExecutor(timeout_seconds=30).run(tmp_path)
    assert result.ok
    assert result.metrics["metric_value"] == 0.93
    assert "training..." in result.stdout_tail


def test_crash_is_reported(tmp_path):
    write_script(tmp_path, "raise RuntimeError('boom')\n")
    result = LocalExecutor(timeout_seconds=30).run(tmp_path)
    assert not result.ok
    assert "boom" in result.stderr_tail
    assert "code 1" in result.error


def test_missing_metrics_is_a_failure(tmp_path):
    write_script(tmp_path, "print('finished but wrote nothing')\n")
    result = LocalExecutor(timeout_seconds=30).run(tmp_path)
    assert not result.ok
    assert "metrics.json" in result.error


def test_stale_metrics_do_not_leak(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"metric_value": 0.99}))
    write_script(tmp_path, "raise SystemExit(1)\n")
    result = LocalExecutor(timeout_seconds=30).run(tmp_path)
    assert not result.ok  # old metrics must not make a failed run look successful


def test_timeout_kills_script(tmp_path):
    write_script(tmp_path, "import time\ntime.sleep(30)\n")
    result = LocalExecutor(timeout_seconds=2).run(tmp_path)
    assert not result.ok
    assert "Timed out" in result.error


def test_secrets_are_stripped_from_child_env():
    os.environ["RAMANUJAN_TEST_API_KEY"] = "supersecret"
    os.environ["RAMANUJAN_TEST_HARMLESS"] = "fine"
    try:
        env = _clean_env()
        assert "RAMANUJAN_TEST_API_KEY" not in env
        assert env["RAMANUJAN_TEST_HARMLESS"] == "fine"
    finally:
        del os.environ["RAMANUJAN_TEST_API_KEY"]
        del os.environ["RAMANUJAN_TEST_HARMLESS"]
