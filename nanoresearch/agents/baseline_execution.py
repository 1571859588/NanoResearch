"""Baseline Execution Agent — runs baseline commands in physical isolation.

This agent intercepts the ExecutionAgent lifecycle to:
1. Swap the runner command to `baseline_train_command`
2. Clear stale iteration checkpoints so the loop runs fresh
3. Preserve baseline metrics to `baseline_metrics.json`
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.project_runner import ensure_project_runner
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Checkpoint file used by the local runner iteration loop
_LOCAL_CHECKPOINT = "plans/execution_iteration_checkpoint.json"


class BaselineExecutionAgent(ExecutionAgent):
    stage = PipelineStage.BASELINE_EXECUTION

    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Override the full execution lifecycle for baseline-specific execution."""
        coding_output = dict(inputs.get("coding_output", {}))

        baseline_cmd = coding_output.get("baseline_train_command", "")
        if not baseline_cmd:
            self.log("WARNING: No baseline_train_command found in coding_output, skipping baseline execution")
            return {"status": "skipped", "reason": "no baseline command"}

        code_dir = Path(coding_output.get("code_dir", ""))
        if not code_dir.exists():
            raise RuntimeError(f"Code directory not found: {code_dir}")

        self.log(f"Intercepting execution loop to run baseline validation: {baseline_cmd}")

        # --- Step 1: Clear stale iteration checkpoints ---
        # The iteration loop resumes from checkpoints. Since BASELINE_EXECUTION
        # is a distinct stage, we must NOT resume from stale EXECUTION checkpoints.
        checkpoint_path = self.workspace.path / _LOCAL_CHECKPOINT
        if checkpoint_path.exists():
            self.log("Clearing stale iteration checkpoint to ensure fresh baseline execution")
            checkpoint_path.unlink()

        # --- Step 2: Regenerate the runner wrapper for the baseline command ---
        runner_assets = ensure_project_runner(code_dir, baseline_cmd)

        # --- Step 3: Forcefully overwrite ALL command references ---
        # This prevents the parent class from re-overwriting with the proposed method.
        coding_output["train_command"] = runner_assets["runner_command"]
        coding_output["entry_train_command"] = baseline_cmd
        coding_output["baseline_train_command"] = baseline_cmd
        coding_output["runner_script"] = runner_assets["runner_script"]
        coding_output["runner_config"] = runner_assets["runner_config"]

        inputs = dict(inputs)
        inputs["coding_output"] = coding_output

        # --- Step 4: Execute via the parent class ---
        result = await super().run(**inputs)

        # --- Step 5: Clone baseline metrics to prevent overwrite by proposed method ---
        metrics_file = code_dir / "results" / "metrics.json"
        baseline_metrics_file = code_dir / "results" / "baseline_metrics.json"
        if metrics_file.exists():
            shutil.copy2(str(metrics_file), str(baseline_metrics_file))
            self.log(f"Cloned baseline metrics output to {baseline_metrics_file.name} to preserve analysis isolation")

        # --- Step 6: Clear checkpoint again so EXECUTION stage starts fresh ---
        if checkpoint_path.exists():
            self.log("Clearing iteration checkpoint after baseline completion for clean proposed method execution")
            checkpoint_path.unlink()

        return result
