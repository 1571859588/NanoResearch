"""Baseline Execution Agent — runs baseline commands in physical isolation.

This agent intercepts the ExecutionAgent lifecycle to:
1. Swap the runner command to `baseline_train_command`
2. Clear stale iteration checkpoints so the loop runs fresh
3. Scan per-baseline subdirectories for metrics and merge them into `baseline_metrics.json`
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
        checkpoint_path = self.workspace.path / _LOCAL_CHECKPOINT
        if checkpoint_path.exists():
            self.log("Clearing stale iteration checkpoint to ensure fresh baseline execution")
            checkpoint_path.unlink()

        # --- Step 2: Regenerate the runner wrapper for the baseline command ---
        runner_assets = ensure_project_runner(code_dir, baseline_cmd)

        # --- Step 3: Forcefully overwrite ALL command references ---
        coding_output["train_command"] = runner_assets["runner_command"]
        coding_output["entry_train_command"] = baseline_cmd
        coding_output["baseline_train_command"] = baseline_cmd
        coding_output["runner_script"] = runner_assets["runner_script"]
        coding_output["runner_config"] = runner_assets["runner_config"]

        inputs = dict(inputs)
        inputs["coding_output"] = coding_output

        # --- Step 4: Execute via the parent class ---
        result = await super().run(**inputs)

        # --- Step 5: Collect per-baseline metrics and merge ---
        baseline_metrics = self._collect_per_baseline_metrics(code_dir)

        # Also copy the root metrics.json as a fallback
        root_metrics_file = code_dir / "results" / "metrics.json"
        baseline_metrics_file = code_dir / "results" / "baseline_metrics.json"

        if baseline_metrics:
            # Write the merged per-baseline metrics
            baseline_metrics_file.parent.mkdir(parents=True, exist_ok=True)
            baseline_metrics_file.write_text(
                json.dumps(baseline_metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.log(
                f"Merged metrics from {len(baseline_metrics)} baseline subdirectories "
                f"into {baseline_metrics_file.name}"
            )
        elif root_metrics_file.exists():
            shutil.copy2(str(root_metrics_file), str(baseline_metrics_file))
            self.log(f"Cloned root metrics to {baseline_metrics_file.name} (no per-baseline dirs found)")

        # --- Step 6: Clear checkpoint for EXECUTION stage ---
        if checkpoint_path.exists():
            self.log("Clearing iteration checkpoint after baseline completion")
            checkpoint_path.unlink()

        return result

    @staticmethod
    def _collect_per_baseline_metrics(code_dir: Path) -> list[dict]:
        """Scan baselines/<slug>/results/metrics.json for each baseline."""
        baselines_dir = code_dir / "baselines"
        if not baselines_dir.exists():
            return []

        all_metrics = []
        for slug_dir in sorted(baselines_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            metrics_file = slug_dir / "results" / "metrics.json"
            if metrics_file.exists():
                try:
                    data = json.loads(metrics_file.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for entry in data:
                            entry["baseline_slug"] = slug_dir.name
                        all_metrics.extend(data)
                    elif isinstance(data, dict):
                        data["baseline_slug"] = slug_dir.name
                        all_metrics.append(data)
                    logger.info("Collected metrics from baseline %s", slug_dir.name)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read metrics from %s: %s", metrics_file, exc)

        return all_metrics
