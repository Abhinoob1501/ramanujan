# Ramanujan — an autonomous ML research engineer

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
+--> PLANNER ---- hypothesis + concrete experiment design
|       |
|    ENGINEER --- agentic write / run / read-traceback / fix loop (sandboxed)
|       |
|    ANALYST ---- insight extraction, leakage skepticism
|       |
+--- CRITIC ----- continue | stop_goal_met | stop_diminishing_returns | stop_flawed
        |
     REPORT ----- leaderboard + narrative + LLM-written conclusions
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
- **Budget-aware science.** Iteration caps, per-experiment wall-clock timeouts, a
  debug-retry budget, an LLM-call budget, and a Critic whose whole job is refusing
  to waste compute — including being suspicious of results that look *too good*.

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

# 1) Zero-API-key demo: full system, scripted LLM, real sklearn training
ramanujan run tasks/demo_breast_cancer.yaml --offline

# 2) Live autonomous research - bring any one API key:
#    Gemini (free: https://aistudio.google.com/apikey), OpenRouter
#    (https://openrouter.ai/keys), or OpenCode Zen (https://opencode.ai/zen)
cp .env.example .env       # set GEMINI_API_KEY / OPENROUTER_API_KEY / OPENCODE_API_KEY
ramanujan run tasks/digits_multiclass.yaml            # provider auto-detected
ramanujan run tasks/digits_multiclass.yaml -p openrouter   # or force one

# 3) Inspect any past run
ramanujan show runs/<run_dir>
```

Each run produces a self-contained directory:

```
runs/20260713_194247_breast-cancer-diagnosis/
  ledger.db          # every experiment: hypothesis, metrics, insight, failures
  iter_01/train.py   # the code the agent wrote
  iter_01/metrics.json
  iter_02/...
  report.md          # auto-written research report with leaderboard + conclusions
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

Generated code is contained, not merely trusted:

- runs in an isolated per-experiment directory with a hard wall-clock timeout,
- secrets (`*KEY*`, `*TOKEN*`, `*SECRET*`, ...) are stripped from the child
  environment before execution,
- the engineer's `write_file` tool rejects paths escaping the experiment directory,
- stale `metrics.json` files are deleted before each run so a crash can never
  inherit a previous success.

This is a guardrail, not a jail — Docker-based isolation is the top roadmap item.

## Project layout

```
ramanujan/
  orchestrator.py      # Research Director: deterministic loop around agentic steps
  task.py              # YAML task spec (dataset, metric, budgets, executor)
  report.py            # final research report renderer
  offline.py           # scripted demo session (zero-key end-to-end run)
  agents/
    base.py            # generic tool-loop Agent + schema-validated ask_json()
    engineer.py        # tool-using coder with a self-debug loop
    roles.py           # Planner / Analyst / Critic decision nodes
    prompts.py         # all system prompts, reviewable in one place
  executors/
    local.py           # sandboxed subprocess executor
    runpod.py          # GPU pod lifecycle: create -> exec -> parse -> destroy
  llm/
    base.py            # provider-agnostic LLM protocol
    factory.py         # provider auto-detection (Gemini / OpenRouter / Zen / custom)
    gemini.py          # Gemini backend (rate-limited, retrying, budgeted)
    openai_compat.py   # OpenRouter / OpenCode Zen / any OpenAI-style server
    mock.py            # scripted backend for tests + offline mode
  memory/
    ledger.py          # SQLite experiment ledger (the system's long-term memory)
tasks/                 # example research task specs (local CPU + RunPod GPU)
tests/                 # 26 tests incl. a full end-to-end offline research run
```

## Testing

```bash
python -m pytest tests -v
```

38 tests cover the ledger, the sandbox executor (timeouts, crashes, stale-metrics
protection, secret stripping), the agent tool loop (error feedback, step limits,
JSON self-repair), the engineer (debug loop, budget enforcement, path-escape
rejection), the OpenAI-compatible backend (message/tool-call conversion, retry
and degradation behavior), provider auto-detection, and a full end-to-end
research run that trains real models.

## Roadmap

- Docker-sandboxed local execution
- Parallel experiment branches (planner proposes k hypotheses, critic allocates budget)
- Semantic retrieval over ledgers of *past runs* (cross-task transfer of insights)
- Dataset upload tasks (CSV path in the spec) and W&B experiment tracking
- A live web dashboard streaming the agent's reasoning per iteration
