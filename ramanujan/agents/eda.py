"""The EDA agent: explores the data before any experiment is planned.

Same genuinely-agentic pattern as the engineer (write script -> run -> read
output -> fix), but the contract differs: the script prints findings to stdout
instead of producing metrics. A final structured call distills the transcript
into typed findings that are injected into every planning round and the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ..executors.base import ExecutionResult
from ..executors.local import LocalExecutor
from ..llm.base import LLMClient, ToolSpec
from ..task import TaskSpec
from . import prompts
from .base import Agent, EventHook, ask_json

_WRITE_FILE_SPEC = ToolSpec(
    name="write_file",
    description="Create or overwrite a file in the analysis working directory.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Relative file name, e.g. eda.py"},
            "content": {"type": "string", "description": "Full file content."},
        },
        "required": ["filename", "content"],
    },
)

_RUN_SCRIPT_SPEC = ToolSpec(
    name="run_script",
    description="Execute the exploration script and return its printed output.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Script to run (default eda.py)."}
        },
        "required": [],
    },
)


class EdaFindings(BaseModel):
    summary: str = Field(description="One-paragraph overview of the dataset.")
    key_findings: list[str] = Field(default_factory=list)
    data_quality_issues: list[str] = Field(default_factory=list)
    leakage_risks: list[str] = Field(default_factory=list)
    modeling_recommendations: list[str] = Field(default_factory=list)

    def to_prompt_block(self) -> str:
        lines = ["EDA FINDINGS (from actual exploration of this dataset):", self.summary]
        for title, items in (
            ("Key findings", self.key_findings),
            ("Data quality issues", self.data_quality_issues),
            ("LEAKAGE RISKS", self.leakage_risks),
            ("Modeling recommendations", self.modeling_recommendations),
        ):
            if items:
                lines.append(f"{title}:")
                lines.extend(f"- {item}" for item in items)
        return "\n".join(lines)


@dataclass
class EdaOutcome:
    success: bool
    findings: EdaFindings | None = None
    error: str = ""


class EdaAgent:
    MAX_RUNS = 3  # initial run + 2 fixes

    def __init__(
        self,
        llm: LLMClient,
        task: TaskSpec,
        workdir: Path,
        on_event: EventHook | None = None,
        timeout_seconds: int | None = None,
    ):
        self.llm = llm
        self.task = task
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.executor = LocalExecutor(
            timeout_seconds=timeout_seconds or task.budget.experiment_timeout_seconds,
            require_metrics=False,
        )
        self.last_success: ExecutionResult | None = None
        self._runs = 0
        system = prompts.EDA_SYSTEM.format(
            environment_notes=task.environment_notes,
            timeout=self.executor.timeout_seconds,
        )
        self._agent = Agent(
            name="eda",
            llm=llm,
            system_prompt=system,
            tools={
                "write_file": (_WRITE_FILE_SPEC, self._tool_write_file),
                "run_script": (_RUN_SCRIPT_SPEC, self._tool_run_script),
            },
            max_steps=2 * self.MAX_RUNS + 3,
            on_event=on_event,
        )

    # ------------------------------------------------------------------ public

    def explore(self, staged_files: list[str] | None = None) -> EdaOutcome:
        files_block = (
            f"\nData files already present in your working directory: {', '.join(staged_files)}\n"
            if staged_files
            else ""
        )
        prompt = (
            f"Explore this dataset before the research begins.\n\n"
            f"Task: {self.task.description}\n"
            f"Target metric: {self.task.metric.name}\n"
            f"Dataset: {self.task.dataset}\n{files_block}"
        )
        result = self._agent.run(prompt)
        if self.last_success is None:
            return EdaOutcome(success=False, error="EDA script never ran successfully.")
        try:
            findings = ask_json(
                self.llm,
                system=prompts.EDA_DISTILL_SYSTEM,
                prompt=(
                    f"EXPLORATION SCRIPT OUTPUT:\n{self.last_success.stdout_tail}\n\n"
                    f"ANALYST'S OWN SUMMARY:\n{result.final_text}\n\n"
                    "Distill the structured findings."
                ),
                model_cls=EdaFindings,
            )
        except ValueError as exc:
            return EdaOutcome(success=False, error=f"could not distill findings: {exc}")
        (self.workdir / "findings.json").write_text(
            json.dumps(findings.model_dump(), indent=2), encoding="utf-8"
        )
        return EdaOutcome(success=True, findings=findings)

    # ------------------------------------------------------------------- tools

    def _tool_write_file(self, args: dict) -> str:
        filename = str(args.get("filename", "eda.py"))
        content = str(args.get("content", ""))
        target = (self.workdir / filename).resolve()
        if not str(target).startswith(str(self.workdir.resolve())):
            raise ValueError(f"path escapes the analysis directory: {filename}")
        if not content.strip():
            return "ERROR: refused to write an empty file."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {filename} ({len(content)} chars)."

    def _tool_run_script(self, args: dict) -> str:
        if self._runs >= self.MAX_RUNS:
            return (
                "ERROR: exploration run budget exhausted. Reply with a plain-text "
                "summary of what you learned so far."
            )
        self._runs += 1
        filename = str(args.get("filename") or "eda.py")
        result = self.executor.run(self.workdir, script_name=filename)
        if result.ok:
            self.last_success = result
        return result.to_feedback()
