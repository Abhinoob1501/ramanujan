"""The Engineer: the genuinely agentic role.

It receives one experiment plan and owns the write -> run -> read error -> fix
loop through real tool use, until the script produces valid metrics or the
debug budget runs out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..executors.base import ExecutionResult, Executor
from ..llm.base import LLMClient, ToolSpec
from ..task import TaskSpec
from . import prompts
from .base import Agent, AgentStep, EventHook
from .roles import ExperimentPlan

_WRITE_FILE_SPEC = ToolSpec(
    name="write_file",
    description="Create or overwrite a file in the experiment working directory.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Relative file name, e.g. train.py"},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["filename", "content"],
    },
)

_RUN_SCRIPT_SPEC = ToolSpec(
    name="run_script",
    description="Execute the training script and return its outcome, output tails and metrics.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Script to run (default train.py).",
            }
        },
        "required": [],
    },
)


@dataclass
class EngineerOutcome:
    success: bool
    summary: str
    metrics: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    code_path: str = ""
    error_summary: str = ""
    steps: list[AgentStep] = field(default_factory=list)


class EngineerAgent:
    def __init__(
        self,
        llm: LLMClient,
        task: TaskSpec,
        executor: Executor,
        workdir: Path,
        on_event: EventHook | None = None,
    ):
        self.task = task
        self.executor = executor
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.last_success: ExecutionResult | None = None
        self.last_failure: ExecutionResult | None = None
        self.run_attempts = 0

        contract_template = (
            prompts.METRICS_CONTRACT_REMOTE
            if task.executor == "runpod"
            else prompts.METRICS_CONTRACT_LOCAL
        )
        system = prompts.ENGINEER_SYSTEM.format(
            metrics_contract=contract_template.format(metric_name=task.metric.name),
            timeout=task.budget.experiment_timeout_seconds,
            environment_notes=task.environment_notes,
        )
        # step budget: each debug attempt costs ~2 tool steps (write + run), plus
        # the final text turn; keep a little slack for re-reads and mistakes.
        max_steps = 3 * (task.budget.max_debug_attempts + 1) + 2
        self._agent = Agent(
            name="engineer",
            llm=llm,
            system_prompt=system,
            tools={
                "write_file": (_WRITE_FILE_SPEC, self._tool_write_file),
                "run_script": (_RUN_SCRIPT_SPEC, self._tool_run_script),
            },
            max_steps=max_steps,
            on_event=on_event,
        )

    # ------------------------------------------------------------------ public

    def implement(self, plan: ExperimentPlan) -> EngineerOutcome:
        prompt = (
            f"Implement this experiment.\n\n"
            f"Hypothesis: {plan.hypothesis}\n"
            f"Approach: {plan.approach}\n"
            f"Rationale: {plan.rationale}\n\n"
            f"Dataset: {self.task.dataset}\n"
            f"Report the metric '{self.task.metric.name}'."
        )
        result = self._agent.run(prompt)

        if self.last_success is not None:
            return EngineerOutcome(
                success=True,
                summary=result.final_text,
                metrics=self.last_success.metrics,
                duration_seconds=self.last_success.duration_seconds,
                code_path=str(self.workdir / "train.py"),
                steps=result.steps,
            )
        error = (
            self.last_failure.error if self.last_failure else "engineer never ran the script"
        )
        return EngineerOutcome(
            success=False,
            summary=result.final_text,
            error_summary=error,
            steps=result.steps,
        )

    # ------------------------------------------------------------------- tools

    def _tool_write_file(self, args: dict) -> str:
        filename = str(args.get("filename", "train.py"))
        content = str(args.get("content", ""))
        target = self._safe_path(filename)
        if not content.strip():
            return "ERROR: refused to write an empty file."
        target.write_text(content, encoding="utf-8")
        return f"Wrote {filename} ({len(content)} chars)."

    def _tool_run_script(self, args: dict) -> str:
        if self.run_attempts > self.task.budget.max_debug_attempts:
            return (
                "ERROR: debug budget exhausted "
                f"({self.task.budget.max_debug_attempts} retries). Stop and reply with a "
                "plain-text summary of what went wrong."
            )
        self.run_attempts += 1
        filename = str(args.get("filename") or "train.py")
        result = self.executor.run(self.workdir, script_name=filename)
        if result.ok and self._metrics_valid(result.metrics):
            self.last_success = result
        elif result.ok:
            result = ExecutionResult(
                ok=False,
                duration_seconds=result.duration_seconds,
                stdout_tail=result.stdout_tail,
                error=(
                    f"metrics reported, but key check failed: expected metric_name="
                    f"'{self.task.metric.name}' and a numeric metric_value; got {result.metrics}"
                ),
            )
            self.last_failure = result
        else:
            self.last_failure = result
        return result.to_feedback()

    # ---------------------------------------------------------------- internal

    def _metrics_valid(self, metrics: dict) -> bool:
        return (
            metrics.get("metric_name") == self.task.metric.name
            and isinstance(metrics.get("metric_value"), (int, float))
        )

    def _safe_path(self, filename: str) -> Path:
        candidate = (self.workdir / filename).resolve()
        if not str(candidate).startswith(str(self.workdir.resolve())):
            raise ValueError(f"path escapes the experiment directory: {filename}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
