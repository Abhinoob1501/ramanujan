# Ramanujan — an autonomous ML research engineer

[![CI](https://github.com/Abhinoob1501/ramanujan/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhinoob1501/ramanujan/actions/workflows/ci.yml)

A multi-agent system that **runs the scientific method as a loop**. Given a research
task (dataset + target metric + compute budget), it autonomously:

1. **forms a falsifiable hypothesis** about what will improve the metric,
2. **writes real training code** to test it,
3. **executes that code in a sandbox** — locally on CPU, or on a freshly provisioned
   **RunPod GPU pod** for heavy tasks,
4. **debugs its own failures** by reading tracebacks and rewriting the code,
5. **analyzes the result**, records the insight in a persistent experiment ledger,
6. **decides whether to continue** — and when it stops, **writes a research report**
   with a leaderboard, an experiment narrative and conclusions.

Named after Srinivasa Ramanujan, who generated extraordinary hypotheses and let the
results speak.

```
     KNOWLEDGE BASE ---- semantic recall of insights from past runs
        |
     EDA AGENT -------- explores the data first: writes & runs a real analysis
        |               script; findings (balance, scales, leakage risks)
        |               ground every planning round
        |
+--> PLANNER ---- 1 hypothesis, or k candidate branches per round
|       |
|   ALLOCATOR --- (branching) the critic funds only the branches worth running
|       |
|    ENGINEER --- agentic write / run / read-traceback / fix loop (sandboxed)
|       |
|    ANALYST ---- insight extraction, leakage skepticism
|       |
+--- CRITIC ----- continue | stop_goal_met | stop_diminishing_returns | stop_flawed
        |
     REPORT ----- leaderboard + narrative + LLM-written conclusions
        |
     every step streams to events.jsonl -> live web dashboard
```

## Why this is interesting (not another chat wrapper)

- **Genuine agency where it pays, structure where it doesn't.** The Engineer is a
  real tool-using agent (it decides when to write, run, and how to fix its own
  crashes). The Planner/Analyst/Critic are *decision nodes*: single structured-output
  calls validated against Pydantic schemas with automatic self-repair. Control flow,
  budgets and persistence are plain code. This split is a deliberate design position:
  agency is spent only where open-endedness earns its failure modes.
- **Closed experimental loop with verifiable artifacts.** Every claim the system makes
  is backed by a `train.py` it wrote, a `metrics.json` a subprocess produced, and a
  SQLite ledger row — not by model say-so.
- **Memory that changes behavior.** The ledger summary (hypotheses, outcomes,
  insights, failures) is injected into every planning step, so the system provably
  builds on what it learned instead of re-running ideas.
- **Budget-aware science.** Iteration caps, a total experiment cap, per-experiment
  wall-clock timeouts, a debug-retry budget, an LLM-call budget, and a Critic whose
  whole job is refusing to waste compute — including being suspicious of results
  that look *too good*.
- **Parallel experiment branches.** Set `budget.parallel_branches: k` and the
  planner proposes k competing hypotheses per round; the critic then acts as a
  budget authority, funding only the candidates worth compute (it may fund fewer
  than k). Each branch gets its own sandbox (`iter_01_a/`, `iter_01_b/`, ...).
- **Cross-task memory.** Finished runs distill their insights into a global
  knowledge base (`runs/knowledge.db`); new runs retrieve the most relevant past
  insights and inject them into planning — lessons transfer across tasks. Pluggable
  embedders: a local hashing embedder by default (zero network, works offline),
  Gemini semantic embeddings with `RAMANUJAN_EMBEDDER=gemini`.
- **Live web dashboard.** Every step is appended to `events.jsonl`;
  `ramanujan dashboard <run_dir>` serves a zero-dependency live page (stdlib HTTP
  server) where hypotheses, engineer tool calls, metrics and verdicts stream in
  as the run executes.
- **Bring your own data + experiment tracking.** Task specs can list `data_files`
  (CSVs are staged into each experiment's sandbox), and `tracking.wandb: true`
  mirrors every experiment to Weights & Biases (hypothesis and approach as config,
  metrics logged, grouped per run) — degrading to a no-op if wandb isn't configured.
- **Statistical honesty.** The metrics contract asks for per-fold CV scores; the
  ledger reports every result with its fold spread ("0.9839 +/- 0.005"), and the
  critic is instructed to treat sub-1-standard-deviation "improvements" as noise
  rather than spending budget chasing them.
- **Per-role model routing.** `RAMANUJAN_MODEL_ENGINEER=...` (and friends) put
  each agent role on its own model — e.g. a cheap analyst and a strong
  tool-calling engineer — while one shared pacer keeps the provider-wide rate
  limit intact. Born from a live observation: routed budget models were fine at
  judging but flaky at tool use.
- **Deterministic failure prevention, learned from live runs.** The engineer
  pre-flights every script's imports and rejects unavailable packages *before*
  wasting an execution; failure feedback states how many debug attempts remain;
  a prose-only reply (no tool calls) triggers one corrective retry.
- **Cost telemetry.** Every run reports its LLM footprint — calls, tokens, and
  real dollar cost where the provider reports it (OpenRouter does) — in the
  console, the report, and the event stream.
- **Self-benchmarking.** `ramanujan bench <tasks...> -n 5` measures the agent
  system itself: goal-hit rate, mean experiments per run, failure taxonomy —
  so prompt/model/orchestration changes are evaluated with numbers.
- **EDA before hypotheses.** A dedicated EDA agent explores the dataset before
  planning starts — it writes and executes a real analysis script (class balance,
  missingness, feature scales, target correlations, redundancy, leakage suspects),
  and the distilled findings are injected into every planning round and the final
  report. Planner hypotheses are grounded in the actual data, not priors. EDA
  failure never sinks a run; disable per task with `eda: false`.
- **You choose where code runs.** `--executor/-x local|docker|runpod` on `run`
  and `ask` overrides any task spec: your own machine, a network-isolated Docker
  container, or a RunPod GPU pod (with an explicit billing confirmation before
  any pod is created). For local runs the hardware is auto-detected — if an
  NVIDIA GPU with working PyTorch CUDA is present, every agent is told it may
  train on `device='cuda'`; otherwise they're told the truth about CPU-only.
  `RAMANUJAN_FORCE_CPU=1` pins runs to CPU regardless.
- **Human-in-the-loop, opt-in.** Run with `-i` (or set `human_in_the_loop: true`)
  and the loop pauses at the two checkpoints where human judgment is cheapest:
  before compute is spent (approve the round's plans, type free-text guidance to
  force a re-plan, or stop) and after the critic rules (accept, or override in
  either direction). Guidance is remembered and injected into every later
  planning round. The default remains fully autonomous.
- **Plain-English front door.** `ramanujan ask "..."` composes a full task spec
  from a natural-language request: data files named in the request are inspected
  (columns, types, target candidates) so the spec describes the *real* schema,
  the draft is schema-validated with self-repair, shown for confirmation, and
  saved as a normal YAML — ad-hoc requests stay reproducible.

## Sample session (real output, offline demo)

```
-------------------------------- Iteration 2/4 --------------------------------
| Hypothesis: Gradient-boosted trees capture non-linear feature interactions  |
| that the linear baseline misses, improving cross-validated ROC-AUC.         |
  engineer -> write_file(filename='train.py', ...)
  engineer -> run_script()
  engineer <- Script FAILED after 1.5s: Exited with code 1.        <- ImportError
  engineer -> write_file(filename='train.py', ...)                 <- self-repair
  engineer -> run_script()
  engineer <- Script succeeded in 2.9s.
| roc_auc = 0.9907 (goal 0.995) in 2.9s                                       |
| Critic: stop_goal_met - the linear baseline meets the target with a sound   |
| CV protocol, and a contrasting model family failed to beat it.              |
```

The training runs, the crash, and the fix above are all real — only the LLM
responses are scripted in offline mode.

## Quickstart

```bash
pip install -e ".[dev]"

# 0) Just say what you want (with any API key configured - see below):
#    Ramanujan inspects your data, composes a validated task spec, shows it,
#    and runs the research. The spec is saved so the run stays reproducible.
ramanujan ask "predict customer churn from data/customers.csv, aim for AUC 0.85"
ramanujan ask "classify sklearn digits as accurately as possible" --dry-run
ramanujan ask "model churn from data/customers.csv" -i   # human-in-the-loop mode
ramanujan ask "train an image classifier on CIFAR-10" -x runpod   # you pick the hardware

# 1) Zero-API-key demo: full system, scripted LLM, real sklearn training
ramanujan run tasks/demo_breast_cancer.yaml --offline

# 2) Live autonomous research - bring any one API key:
#    Gemini (free: https://aistudio.google.com/apikey), OpenRouter
#    (https://openrouter.ai/keys), or OpenCode Zen (https://opencode.ai/zen)
cp .env.example .env       # set GEMINI_API_KEY / OPENROUTER_API_KEY / OPENCODE_API_KEY
ramanujan run tasks/digits_multiclass.yaml            # provider auto-detected
ramanujan run tasks/digits_multiclass.yaml -p openrouter   # or force one

# 3) Watch a run live (or replay a finished one) in the browser
ramanujan dashboard runs/<run_dir>        # -> http://127.0.0.1:8787

# 4) Bring your own CSV + parallel branches (see the spec for the knobs)
ramanujan run tasks/churn_csv.yaml

# 5) Inspect any past run in the terminal
ramanujan show runs/<run_dir>

# 6) Benchmark the agent system itself (works offline too)
ramanujan bench tasks/demo_breast_cancer.yaml -n 5 --offline
```

Each run produces a self-contained directory:

```
runs/20260713_194247_breast-cancer-diagnosis_a1b2c3/
  ledger.db          # every experiment: hypothesis, metrics, insight, failures
  events.jsonl       # the full reasoning stream (feeds the live dashboard)
  iter_01/train.py   # the code the agent wrote
  iter_01/metrics.json
  iter_02_a/ ...     # parallel branches get suffixed sandboxes
  report.md          # auto-written research report with leaderboard + conclusions
runs/knowledge.db    # cross-run knowledge base (insights recalled by future runs)
```

## GPU experiments on RunPod

For tasks that need real hardware, set `executor: runpod` in the task spec
(see [tasks/cifar10_runpod.yaml](tasks/cifar10_runpod.yaml)). Per experiment, the
executor creates a GPU pod with `runpodctl`, ships the generated script to it,
runs it, parses metrics back from a `RAMANUJAN_METRICS::{...}` sentinel line, and
**always tears the pod down** — success or failure — so nothing keeps billing.

Cost guard: it refuses to start unless the task sets `runpod.confirm_billing: true`.
Requires `runpodctl` configured with an API key and account credit.

## LLM backends

Provider-agnostic by construction: agents speak to an `LLMClient` protocol
([ramanujan/llm/base.py](ramanujan/llm/base.py)), never to a vendor SDK. The
provider is auto-detected from whichever API key is set (or forced with
`--provider` / `RAMANUJAN_PROVIDER`):

| Provider | Key env var | Default model |
|---|---|---|
| **Gemini** | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| **OpenRouter** | `OPENROUTER_API_KEY` | `deepseek/deepseek-chat-v3.1:free` |
| **OpenCode Zen** | `OPENCODE_API_KEY` | `grok-code` |
| **Custom** (Groq, Together, Ollama, vLLM, ...) | `RAMANUJAN_API_KEY` + `RAMANUJAN_BASE_URL` | via `RAMANUJAN_MODEL` |
| **MockLLM** | — (`--offline`) | scripted session |

OpenRouter, OpenCode Zen and Custom all go through one OpenAI-compatible
backend ([ramanujan/llm/openai_compat.py](ramanujan/llm/openai_compat.py)).
Override any preset with `RAMANUJAN_MODEL` / `RAMANUJAN_BASE_URL` — but pick a
model with native function-calling support; the Engineer agent depends on it.

All network backends share the same free-tier discipline: request pacing,
exponential backoff on 429/5xx, and a hard per-run LLM-call budget.

## Safety posture

Generated code is contained, not merely trusted. Two tiers:

**`executor: local`** (default, zero setup):
- runs in an isolated per-experiment directory with a hard wall-clock timeout,
- secrets (`*KEY*`, `*TOKEN*`, `*SECRET*`, ...) are stripped from the child
  environment before execution,
- the engineer's `write_file` tool rejects paths escaping the experiment directory,
- stale `metrics.json` files are deleted before each run so a crash can never
  inherit a previous success.

**`executor: docker`** (real isolation):
- each experiment runs in a disposable container with `--network=none` (generated
  code cannot reach the network at all), memory/CPU caps, and only the experiment
  directory mounted. Build the sandbox image once with
  `docker build -t ramanujan-sandbox docker/`.

## Project layout

```
ramanujan/
  orchestrator.py      # Research Director: deterministic loop around agentic steps
  hitl.py              # human-in-the-loop gates (plan approval, verdict override)
  hardware.py          # local GPU/CPU detection (informs every agent's prompts)
  task.py              # YAML task spec (dataset, metric, budgets, branches, data files)
  report.py            # final research report renderer (incl. cost telemetry)
  composer.py          # natural-language -> TaskSpec (the `ask` command)
  bench.py             # self-benchmark harness (goal-hit rate, failure taxonomy)
  events.py            # append-only event stream per run (events.jsonl)
  dashboard.py         # zero-dependency live web dashboard over the event stream
  tracking.py          # optional Weights & Biases mirroring
  offline.py           # scripted demo session (zero-key end-to-end run)
  agents/
    base.py            # generic tool-loop Agent + schema-validated ask_json()
    eda.py             # data-exploration agent (runs before planning)
    engineer.py        # tool-using coder with a self-debug loop
    roles.py           # Planner / Allocator / Analyst / Critic decision nodes
    prompts.py         # all system prompts, reviewable in one place
  executors/
    local.py           # sandboxed subprocess executor
    docker.py          # network-isolated container executor
    runpod.py          # GPU pod lifecycle: create -> exec -> parse -> destroy
  llm/
    base.py            # provider-agnostic LLM protocol + usage telemetry + pacing
    factory.py         # provider auto-detection + per-role model routing (LLMSuite)
    gemini.py          # Gemini backend (rate-limited, retrying, budgeted)
    openai_compat.py   # OpenRouter / OpenCode Zen / any OpenAI-style server
    mock.py            # scripted backend for tests + offline mode
  memory/
    ledger.py          # SQLite experiment ledger (per-run memory)
    knowledge.py       # cross-run knowledge base with pluggable embedders
tasks/                 # example task specs (local CPU, CSV+branches, RunPod GPU)
tests/                 # 52 tests incl. full end-to-end research runs
```

## Testing

```bash
python -m pytest tests -v
```

100 tests cover executor selection (overrides, billing confirmation, hardware
detection and prompt injection), the human-in-the-loop gates (guidance re-planning, stop, verdict
overrides in both directions), the EDA agent (exploration, distillation, contained failure),
the natural-language task composer (file detection, schema
inspection, spec round-tripping), the ledger (incl. fold-spread reporting), the knowledge base
(retrieval ranking, run exclusion, code storage, schema migration), the local and
Docker executors (timeouts, crashes, stale-metrics protection, secret stripping,
daemon-missing degradation), the agent tool loop (error feedback, step limits,
JSON self-repair), the engineer (debug loop and budget nudges, import pre-flight,
prose-only corrective retry, path-escape rejection, reference-code injection),
the OpenAI-compatible backend (message/tool-call conversion, retry, degradation,
usage/cost accounting), provider auto-detection and per-role routing, the event
stream (incremental reads, torn writes), data-file staging, the benchmark
harness, and full end-to-end research runs — including a branched run with
budget allocation and cross-run knowledge transfer. CI runs the suite on Linux
and Windows on every push.

## Roadmap

- Concurrent branch execution (branches currently run sequentially within a round,
  which respects free-tier LLM rate limits)
- GPU-fleet mode: reuse one warm RunPod pod across experiments instead of
  create/destroy per experiment
- Approve/guide controls in the web dashboard (the CLI gate, file-backed, could
  be driven remotely)
