"""RunPod GPU executor.

Ships a generated training script to a freshly created RunPod GPU pod, runs it
there, and parses metrics back from stdout via the RAMANUJAN_METRICS:: sentinel
(remote scripts print metrics instead of writing metrics.json locally).

Lifecycle per experiment:
  1. `runpodctl create pod`   - spin up a GPU pod from a PyTorch image
  2. poll `runpodctl get pod` - wait until RUNNING
  3. `runpodctl exec python`  - upload + execute the local script on the pod
  4. parse sentinel line      - recover metrics dict
  5. `runpodctl remove pod`   - always torn down (also on failure) so no GPU
                                keeps billing after the experiment

Requirements: `runpodctl` on PATH and configured with an API key
(`runpodctl config --apiKey ...`), and billing credit on the RunPod account.
Cost guard: refuses to run unless the task spec sets `runpod.confirm_billing: true`.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from pathlib import Path

from .base import METRICS_SENTINEL, ExecutionResult, tail


class RunPodExecutor:
    def __init__(
        self,
        timeout_seconds: int = 1800,
        gpu_type: str = "NVIDIA GeForce RTX 4090",
        image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        container_disk_gb: int = 20,
        confirm_billing: bool = False,
        **_ignored,
    ):
        if not confirm_billing:
            raise RuntimeError(
                "RunPod execution creates billed GPU pods. Set `runpod.confirm_billing: true` "
                "in the task spec to acknowledge, or switch the task to `executor: local`."
            )
        self.timeout_seconds = timeout_seconds
        self.gpu_type = gpu_type
        self.image = image
        self.container_disk_gb = container_disk_gb

    # ------------------------------------------------------------------ public

    def run(self, workdir: Path, script_name: str = "train.py") -> ExecutionResult:
        script = workdir / script_name
        if not script.exists():
            return ExecutionResult(ok=False, duration_seconds=0.0, error=f"{script_name} missing")

        start = time.time()
        pod_id = ""
        try:
            pod_id = self._create_pod()
            self._wait_until_running(pod_id)
            output = self._exec_script(pod_id, script)
            return self._parse_output(output, time.time() - start)
        except Exception as exc:
            return ExecutionResult(
                ok=False, duration_seconds=time.time() - start, error=str(exc)
            )
        finally:
            if pod_id:
                self._remove_pod(pod_id)

    # ---------------------------------------------------------------- internal

    def _cli(self, *args: str, timeout: int = 120) -> str:
        proc = subprocess.run(
            ["runpodctl", *args], capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"runpodctl {' '.join(args[:2])} failed: {tail(proc.stderr, 800) or tail(proc.stdout, 800)}"
            )
        return proc.stdout

    def _create_pod(self) -> str:
        name = f"ramanujan-{uuid.uuid4().hex[:8]}"
        out = self._cli(
            "create", "pod",
            "--name", name,
            "--gpuType", self.gpu_type,
            "--gpuCount", "1",
            "--imageName", self.image,
            "--containerDiskSize", str(self.container_disk_gb),
            "--args", "bash -c 'sleep infinity'",
        )
        match = re.search(r'pod "([a-z0-9]+)"', out) or re.search(r"\b([a-z0-9]{13,14})\b", out)
        if not match:
            raise RuntimeError(f"Could not parse pod id from runpodctl output: {out[:400]}")
        return match.group(1)

    def _wait_until_running(self, pod_id: str, poll_seconds: int = 10, max_wait: int = 600) -> None:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            out = self._cli("get", "pod", pod_id)
            if "RUNNING" in out.upper():
                return
            time.sleep(poll_seconds)
        raise RuntimeError(f"Pod {pod_id} did not reach RUNNING within {max_wait}s.")

    def _exec_script(self, pod_id: str, script: Path) -> str:
        proc = subprocess.run(
            ["runpodctl", "exec", "python", str(script), "--pod_id", pod_id],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Remote execution failed:\n{tail(proc.stderr)}\n{tail(proc.stdout)}"
            )
        return proc.stdout

    def _remove_pod(self, pod_id: str) -> None:
        try:
            self._cli("remove", "pod", pod_id)
        except Exception:
            # Never mask the real result, but make the leak loud in logs.
            print(f"WARNING: failed to remove RunPod pod {pod_id}; remove it manually to stop billing.")

    @staticmethod
    def _parse_output(output: str, duration: float) -> ExecutionResult:
        for line in reversed(output.splitlines()):
            if line.startswith(METRICS_SENTINEL):
                try:
                    metrics = json.loads(line[len(METRICS_SENTINEL):])
                except json.JSONDecodeError as exc:
                    return ExecutionResult(
                        ok=False, duration_seconds=duration,
                        stdout_tail=tail(output), error=f"Bad metrics sentinel JSON: {exc}",
                    )
                return ExecutionResult(
                    ok=True, duration_seconds=duration, stdout_tail=tail(output), metrics=metrics
                )
        return ExecutionResult(
            ok=False,
            duration_seconds=duration,
            stdout_tail=tail(output),
            error=f"Script finished but printed no '{METRICS_SENTINEL}' line (required for remote runs).",
        )
