"""Live research dashboard.

`ramanujan dashboard <run_dir>` serves a single-page view of a run's event
stream (events.jsonl). Works on finished runs and LIVE ones: the page polls
/api/events incrementally, and since the orchestrator appends events from its
own process, you can watch planner hypotheses, engineer tool calls, metrics
and critic verdicts arrive in real time.

Deliberately dependency-free: stdlib http.server + a self-contained HTML page.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .events import read_events_since

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ramanujan - live research</title>
<style>
  :root { --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#e6edf3; --dim:#8b949e;
          --cyan:#58a6ff; --green:#3fb950; --red:#f85149; --amber:#d29922; --purple:#bc8cff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--border);
           padding:14px 22px; display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; }
  .badge { border:1px solid var(--border); border-radius:12px; padding:1px 10px; color:var(--dim); }
  .badge.live { color:var(--green); border-color:var(--green); }
  .badge.done { color:var(--cyan); border-color:var(--cyan); }
  #best { color:var(--green); font-weight:bold; }
  main { max-width:940px; margin:0 auto; padding:18px 22px 60px; }
  .round { color:var(--dim); margin:26px 0 10px; border-bottom:1px dashed var(--border);
           padding-bottom:4px; font-weight:bold; }
  .card { background:var(--card); border:1px solid var(--border); border-left:3px solid var(--dim);
          border-radius:6px; padding:10px 14px; margin:8px 0; overflow-wrap:anywhere; }
  .card .who { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:var(--dim); }
  .plan { border-left-color:var(--cyan); }
  .ok { border-left-color:var(--green); }
  .fail { border-left-color:var(--red); }
  .judge { border-left-color:var(--amber); }
  .analysis { border-left-color:var(--purple); }
  .toollog { color:var(--dim); font-size:12px; margin:2px 0 2px 12px; white-space:pre-wrap; }
  .metric { font-size:18px; color:var(--green); font-weight:bold; }
  .dim { color:var(--dim); }
  #gate { display:none; position:sticky; top:56px; z-index:10; background:#1c1608;
          border:1px solid var(--amber); border-left:4px solid var(--amber);
          border-radius:6px; padding:12px 16px; margin:10px 0; }
  #gate .who { color:var(--amber); }
  #gate textarea { width:100%; box-sizing:border-box; margin:8px 0; padding:8px;
          background:var(--bg); color:var(--text); border:1px solid var(--border);
          border-radius:4px; font:inherit; min-height:54px; }
  #gate button { background:var(--card); color:var(--text); border:1px solid var(--border);
          border-radius:5px; padding:7px 16px; margin-right:8px; font:inherit; cursor:pointer; }
  #gate button:hover { border-color:var(--cyan); }
  #gate button.primary { border-color:var(--green); color:var(--green); }
  #gate button.danger { border-color:var(--red); color:var(--red); }
</style>
</head>
<body>
<header>
  <h1>Ramanujan</h1>
  <span id="task" class="dim">waiting for events...</span>
  <span id="status" class="badge">connecting</span>
  <span id="best"></span>
</header>
<main>
  <div id="gate"></div>
  <div id="feed"></div>
</main>
<script>
let last = 0, direction = "maximize", metricName = "", best = null, finished = false;
const feed = document.getElementById("feed");

function card(cls, who, html) {
  const el = document.createElement("div");
  el.className = "card " + cls;
  el.innerHTML = `<div class="who">${who}</div>${html}`;
  feed.appendChild(el);
}
function esc(s) {
  return String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}
function render(ev) {
  const p = ev.payload || {};
  switch (ev.kind) {
    case "run_started":
      direction = p.direction; metricName = p.metric_name;
      document.getElementById("task").textContent =
        `${p.task} - ${p.metric_name} goal ${p.goal} - ${p.max_iterations} rounds x ${p.parallel_branches} branch(es)`;
      break;
    case "prior_knowledge_retrieved":
      card("plan", "knowledge base",
        (p.insights || []).map(i => `<div class="dim">(${esc(i.task)}) ${esc(i.insight)}</div>`).join(""));
      break;
    case "round_started": {
      const el = document.createElement("div");
      el.className = "round"; el.textContent = `ROUND ${ev.iteration}`;
      feed.appendChild(el); break;
    }
    case "plan_proposed":
      card("plan", `planner - hypothesis ${p.branch}`,
        `<b>${esc(p.hypothesis)}</b><div class="dim">${esc(p.approach)}</div>`);
      break;
    case "budget_allocated":
      card("judge", "budget allocator",
        `funding candidates <b>[${(p.selected || []).join(", ")}]</b><div class="dim">${esc(p.reasoning)}</div>`);
      break;
    case "agent_tool_call": {
      const el = document.createElement("div");
      el.className = "toollog"; el.textContent = `${ev.agent} -> ${p.detail}`;
      feed.appendChild(el); break;
    }
    case "experiment_result":
      if (p.status === "success") {
        const v = p.metric_value;
        if (best === null || (direction === "maximize" ? v > best : v < best)) best = v;
        document.getElementById("best").textContent = `best ${metricName}: ${Number(best).toFixed(4)}`;
        card("ok", `experiment ${p.experiment_id} - success`,
          `<span class="metric">${metricName} = ${Number(v).toFixed(4)}</span>` +
          `<span class="dim"> in ${Number(p.duration_seconds).toFixed(1)}s</span>`);
      } else {
        card("fail", `experiment ${p.experiment_id} - failed`, `<span class="dim">${esc(p.error)}</span>`);
      }
      break;
    case "analysis":
      card("analysis", "analyst", `${esc(p.insight)}` +
        (p.suspicion ? `<div style="color:var(--amber)">suspicion: ${esc(p.suspicion)}</div>` : ""));
      break;
    case "verdict":
      card("judge", "critic", `<b>${esc(p.decision)}</b><div class="dim">${esc(p.reasoning)}</div>`);
      break;
    case "run_finished":
      finished = true;
      card(p.goal_met ? "ok" : "fail", "run finished",
        `stop reason: <b>${esc(p.stop_reason)}</b>` +
        (p.best_metric_value !== null ? ` - best ${metricName} = ${Number(p.best_metric_value).toFixed(4)}` : "") +
        `<div class="dim">${esc(p.report)}</div>`);
      break;
  }
}
let gateShownId = null;
async function pollGate() {
  const el = document.getElementById("gate");
  try {
    const g = await (await fetch("/api/gate")).json();
    if (!g || !g.id) { el.style.display = "none"; gateShownId = null; return; }
    if (g.id === gateShownId) return;
    gateShownId = g.id;
    const p = g.payload || {};
    if (g.type === "plan") {
      el.innerHTML =
        `<div class="who">YOUR DECISION NEEDED - round ${g.iteration} plans</div>` +
        (p.plans || []).map(pl => `<div><b>${esc(pl.hypothesis)}</b>` +
          `<div class="dim">${esc(pl.approach)}</div></div>`).join("") +
        `<textarea id="guidance" placeholder="Optional guidance for the planner ` +
        `(used with 'Guide & re-plan'), e.g. 'only tree models'"></textarea>` +
        `<div><button class="primary" onclick="respond('approve')">Run these</button>` +
        `<button onclick="respond('revise')">Guide &amp; re-plan</button>` +
        `<button class="danger" onclick="respond('stop')">Stop research</button></div>`;
    } else {
      const alt = p.decision === "continue"
        ? ["stop_now", "Stop now"] : ["continue_anyway", "Continue anyway"];
      el.innerHTML =
        `<div class="who">YOUR DECISION NEEDED - critic verdict: ${esc(p.decision)}</div>` +
        `<div class="dim">${esc(p.reasoning)}</div>` +
        `<div style="margin-top:8px"><button class="primary" onclick="respond('accept')">Accept</button>` +
        `<button class="danger" onclick="respond('${alt[0]}')">${alt[1]}</button></div>`;
    }
    el.style.display = "block";
  } catch (e) { /* run not gated or server briefly away */ }
}
async function respond(action) {
  const box = document.getElementById("guidance");
  await fetch("/api/gate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: gateShownId, action: action, guidance: box ? box.value : "" }),
  });
  document.getElementById("gate").style.display = "none";
}
async function poll() {
  try {
    const res = await fetch(`/api/events?since=${last}`);
    const data = await res.json();
    const atBottom = window.innerHeight + window.scrollY >= document.body.offsetHeight - 80;
    for (const ev of data.events) { render(ev); last = ev.seq; }
    document.getElementById("status").textContent = finished ? "finished" : "live";
    document.getElementById("status").className = "badge " + (finished ? "done" : "live");
    if (data.events.length && atBottom) window.scrollTo(0, document.body.scrollHeight);
  } catch (e) {
    document.getElementById("status").textContent = "disconnected";
    document.getElementById("status").className = "badge";
  }
  await pollGate();
  setTimeout(poll, finished ? 5000 : 1500);
}
poll();
</script>
</body>
</html>
"""


def make_handler(run_dir: Path):
    events_path = Path(run_dir) / "events.jsonl"
    request_path = Path(run_dir) / "gate_request.json"
    response_path = Path(run_dir) / "gate_response.json"

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (stdlib API)
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
            elif parsed.path == "/api/events":
                since = int(parse_qs(parsed.query).get("since", ["0"])[0])
                events = read_events_since(events_path, since)
                body = json.dumps(
                    {"events": events, "last": events[-1]["seq"] if events else since}
                ).encode("utf-8")
                self._send(200, "application/json", body)
            elif parsed.path == "/api/gate":
                self._send(200, "application/json", json.dumps(self._pending_gate()).encode("utf-8"))
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):  # noqa: N802 (stdlib API)
            parsed = urlparse(self.path)
            if parsed.path != "/api/gate":
                self._send(404, "text/plain", b"not found")
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                decision = json.loads(self.rfile.read(length).decode("utf-8"))
                assert isinstance(decision.get("id"), int)
                assert decision.get("action") in (
                    "approve", "revise", "stop", "accept", "continue_anyway", "stop_now"
                )
            except Exception:
                self._send(400, "application/json", b'{"ok": false, "error": "bad decision"}')
                return
            response_path.write_text(
                json.dumps({"id": decision["id"], "action": decision["action"],
                            "guidance": str(decision.get("guidance", ""))}),
                encoding="utf-8",
            )
            self._send(200, "application/json", b'{"ok": true}')

        def _pending_gate(self) -> dict:
            """The open gate request, if the run is waiting on a human."""
            if not request_path.exists():
                return {}
            try:
                # utf-8-sig: tolerate a BOM in hand-written files on Windows
                request = json.loads(request_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError):
                return {}
            if response_path.exists():
                try:
                    answered = json.loads(response_path.read_text(encoding="utf-8-sig"))
                    if answered.get("id") == request.get("id"):
                        return {}  # already answered; waiting for the run to consume it
                except (json.JSONDecodeError, OSError):
                    pass
            return request

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # keep the console clean
            pass

    return DashboardHandler


def resolve_run_dir(path: str | Path) -> Path:
    """Accept either a specific run directory or a runs root; in the latter
    case serve the most recent run that has an event stream. Run directories
    are timestamped, so users pointing at yesterday's dir (or the wrong one)
    was a common footgun - `ramanujan dashboard` with no argument now just
    works."""
    path = Path(path)
    if (path / "events.jsonl").exists():
        return path
    if path.is_dir():
        candidates = sorted(
            (d for d in path.iterdir() if d.is_dir() and (d / "events.jsonl").exists()),
            key=lambda d: d.name,
        )
        if candidates:
            return candidates[-1]
    return path  # serve as-is; the page will wait for events


def start_dashboard_server(
    run_dir: str | Path, port: int = 8787, max_port_tries: int = 10
) -> tuple[ThreadingHTTPServer, int]:
    """Start the dashboard on a daemon thread (used by `run --web` so no second
    terminal is needed). If the port is busy, the next few are tried. Returns
    (server, bound_port)."""
    last_error: OSError | None = None
    for candidate in range(port, port + max_port_tries):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), make_handler(Path(run_dir)))
        except OSError as exc:
            last_error = exc
            continue
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, server.server_address[1]
    raise OSError(
        f"No free port in {port}-{port + max_port_tries - 1} for the dashboard: {last_error}"
    )


def serve_dashboard(run_dir: str | Path, port: int = 8787) -> None:
    run_dir = resolve_run_dir(run_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Path(run_dir)))
    print(f"Dashboard for {run_dir} -> http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
