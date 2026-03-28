"""Baseline Execution Agent — runs baseline commands in physical isolation.

This agent intercepts the ExecutionAgent lifecycle to:
1. Swap the runner command to `baseline_train_command`
2. Auto-detect baseline entry points when the field is missing
3. Clear stale iteration checkpoints so the loop runs fresh
4. Scan per-baseline subdirectories for metrics and merge them into `baseline_metrics.json`
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
        code_dir = Path(coding_output.get("code_dir", ""))

        if not code_dir.exists():
            raise RuntimeError(f"Code directory not found: {code_dir}")

        # --- Step 0: Resolve baseline command (with fallback auto-detection) ---
        baseline_cmd = coding_output.get("baseline_train_command", "").strip()

        if not baseline_cmd:
            baseline_cmd = self._auto_detect_baseline_command(code_dir)

        if not baseline_cmd:
            self.log(
                "WARNING: No baseline_train_command found and no baselines/ "
                "directory detected. Skipping baseline execution."
            )
            return {"status": "skipped", "reason": "no baseline command or baselines directory"}

        self.log(f"Baseline execution command resolved to: {baseline_cmd}")

        # --- Step 0.5: Check for reusable published results ---
        blueprint = inputs.get("experiment_blueprint", {})
        reusable_metrics = self._check_published_results(blueprint)
        if reusable_metrics and len(reusable_metrics) == len(blueprint.get("baselines", [])):
            self.log(f"Optimization: Reusing published results for all {len(reusable_metrics)} baselines from paper summaries.")
            # Create a mock result structure matching ExecutionAgent.run
            metrics_file = code_dir / "results" / "baseline_metrics.json"
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.write_text(json.dumps(reusable_metrics, indent=2), encoding="utf-8")
            
            return {
                "status": "completed",
                "reason": "optimized_via_paper_summaries",
                "metrics": reusable_metrics,
                "global_run_id": "REUSED_FROM_PAPERS"
            }

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

    def _auto_detect_baseline_command(self, code_dir: Path) -> str:
        """Auto-detect the baseline execution command by scanning the code directory.

        Checks for:
        1. baselines/run_all.sh (primary, batch executor)
        2. baselines/*/train.py (individual baseline scripts → build run_all.sh)
        3. Any .sh or .py file under baselines/ as a last resort
        """
        baselines_dir = code_dir / "baselines"
        if not baselines_dir.exists():
            return ""

        # Check 1: run_all.sh exists
        run_all_sh = baselines_dir / "run_all.sh"
        if run_all_sh.exists():
            self.log(f"Auto-detected baseline entry: {run_all_sh.name}")
            return "bash baselines/run_all.sh"

        # Check 2: Per-baseline subdirectories with train.py
        baseline_scripts = []
        for slug_dir in sorted(baselines_dir.iterdir()):
            if slug_dir.is_dir():
                train_script = slug_dir / "train.py"
                if train_script.exists():
                    baseline_scripts.append(f"python baselines/{slug_dir.name}/train.py")

        if baseline_scripts:
            # Auto-generate run_all.sh
            self.log(f"Auto-generating run_all.sh for {len(baseline_scripts)} baseline(s)")
            run_all_content = "#!/bin/bash\nset -e\n\nSCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\nPROJECT_ROOT=\"$(dirname \"$SCRIPT_DIR\")\"\ncd \"$PROJECT_ROOT\"\n\n"
            for script in baseline_scripts:
                run_all_content += f'echo "=== Running: {script} ==="\n{script}\n\n'
            run_all_sh.write_text(run_all_content, encoding="utf-8")
            run_all_sh.chmod(0o755)
            return "bash baselines/run_all.sh"

        # Check 3: Any .py file directly under baselines/
        py_files = list(baselines_dir.glob("*.py"))
        if py_files:
            # Build a sequential command
            commands = [f"python baselines/{f.name}" for f in py_files]
            self.log(f"Auto-detected {len(commands)} baseline script(s) at baselines/ root")
            run_all_content = "#!/bin/bash\nset -e\n\nSCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\nPROJECT_ROOT=\"$(dirname \"$SCRIPT_DIR\")\"\ncd \"$PROJECT_ROOT\"\n\n"
            for cmd in commands:
                run_all_content += f'echo "=== Running: {cmd} ==="\n{cmd}\n\n'
            run_all_sh.write_text(run_all_content, encoding="utf-8")
            run_all_sh.chmod(0o755)
            return "bash baselines/run_all.sh"

        return ""

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

    def _check_published_results(self, blueprint: dict) -> list[dict]:
        """Check if baselines have results in their paper SUMMARY.md for the target dataset."""
        import re
        baselines = blueprint.get("baselines", [])
        if not baselines:
            return []
            
        # Try to identify the primary dataset/benchmark from the blueprint
        dataset = blueprint.get("dataset", "") or blueprint.get("benchmark", "unknown")
        if isinstance(dataset, dict):
            dataset = dataset.get("name", "unknown")
            
        reusable_metrics = []
        
        for bl in baselines:
            paper_id = bl.get("reference_paper_id")
            slug = bl.get("slug")
            if not paper_id or not slug:
                continue
                
            # Look for the markdown summary
            clean_pid = re.sub(r"[^a-zA-Z0-9]+", "_", paper_id.lower()).strip("_")
            md_path = self.workspace.global_references_dir / "papers" / f"{clean_pid}.md"
            
            if not md_path.exists():
                logger.debug("No summary found for paper %s at %s", paper_id, md_path)
                continue
                
            md_content = md_path.read_text(encoding="utf-8", errors="replace")
            
            # Look for a results table entry matching the dataset
            # Matches: | Dataset | Metric | Value |
            # We search for lines containing the dataset name
            lines = md_content.splitlines()
            found_for_this_baseline = False
            for line in lines:
                if "|" in line and dataset.lower() in line.lower():
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 3:
                        # Heuristic: [Dataset, Metric, Value, ...]
                        metric_name = parts[1]
                        metric_val_str = parts[2]
                        
                        # Clean the value (handle percentages, bolding like **85.2**, etc.)
                        val_clean = re.sub(r"[^\d.]", "", metric_val_str)
                        if val_clean:
                            try:
                                val_float = float(val_clean)
                                reusable_metrics.append({
                                    "baseline_slug": slug,
                                    "metric": metric_name,
                                    "value": val_float,
                                    "source": f"paper:{paper_id}",
                                    "dataset": dataset
                                })
                                found_for_this_baseline = True
                                break # Take first valid metric for this dataset
                            except ValueError:
                                continue
            
            if not found_for_this_baseline:
                logger.debug("No valid metrics found in summary for baseline %s on dataset %s", slug, dataset)
                            
        return reusable_metrics
