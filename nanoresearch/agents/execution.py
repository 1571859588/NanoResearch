"""Execution agent — submits SLURM jobs, monitors progress, debugs failures, collects results."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shlex
import shutil
import time
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.debug import DebugAgent, MAX_DEBUG_ROUNDS
from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.feedback_analyzer import FeedbackAnalyzer
from nanoresearch.agents.project_runner import (
    RUNNER_SCRIPT_NAME,
    ensure_project_runner,
    is_python_launcher_token,
)
from nanoresearch.agents.preflight import PreflightChecker
from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
from nanoresearch.schemas.iteration import ExperimentHypothesis, IterationState, RoundResult
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Poll interval and max wait time for SLURM jobs
POLL_INTERVAL = 30  # seconds
MAX_WAIT_TIME = 7 * 24 * 3600  # 7 days — real training can run for days
LOCAL_EXECUTION_CHECKPOINT = "plans/execution_iteration_checkpoint.json"


class ExecutionAgent(BaseResearchAgent):
    """Submits SLURM training jobs, monitors them, debugs failures, and collects results."""

    stage = PipelineStage.EXECUTION

    @property
    def stage_config(self):
        """Reuse experiment-stage model routing for execution-time reasoning."""
        return self.config.for_stage("experiment")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        coding_output: dict = inputs.get("coding_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})
        setup_output: dict = inputs.get("setup_output", {})
        topic: str = inputs.get("topic", "")

        code_dir = Path(coding_output.get("code_dir", ""))
        slurm_script = coding_output.get("slurm_script", "")

        if not code_dir.exists():
            raise RuntimeError(f"Code directory not found: {code_dir}")

        self.log(f"Starting execution in: {code_dir}")

        # Create logs directory
        (code_dir / "logs").mkdir(exist_ok=True)
        (code_dir / "results").mkdir(exist_ok=True)

        cluster_available = bool(slurm_script) and shutil.which("sbatch") is not None
        if not self.config.prefers_cluster_execution() or not cluster_available:
            if self.config.prefers_cluster_execution() and not cluster_available:
                self.log("Cluster execution requested but sbatch is unavailable, falling back to local mode")
            elif not slurm_script:
                self.log("No SLURM script produced by CODING, falling back to local mode")
            else:
                self.log(f"Execution profile '{self.config.execution_profile.value}' prefers local execution")
            final_result = await self._run_local_mode(
                code_dir,
                coding_output,
                experiment_blueprint,
                setup_output,
                topic,
            )
            self.workspace.write_json("plans/execution_output.json", final_result)
            return final_result

        # Pre-flight: fix common SLURM issues before first submission
        debug_agent = DebugAgent(self.workspace, self.config)
        preflight_fixed = debug_agent._fix_common_slurm_issues(code_dir)
        if preflight_fixed:
            self.log("Pre-flight: fixed common SLURM script issues")

        # Pre-flight: local syntax/import check before wasting SLURM queue time
        local_ok, local_err = await self._local_preflight(code_dir)
        if not local_ok:
            self.log(f"Pre-flight import check failed, fixing before submission")
            # Run a mini debug loop locally (no SLURM submission)
            for pre_round in range(MAX_DEBUG_ROUNDS):
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log="",
                    stderr_log=local_err,
                    job_status="IMPORT_ERROR",
                    debug_round=pre_round + 1,
                    previous_fixes=[],
                )
                if not debug_result.get("needs_resubmit", False):
                    break
                local_ok, local_err = await self._local_preflight(code_dir)
                if local_ok:
                    self.log(f"Pre-flight fixed after {pre_round + 1} round(s)")
                    break

        # Debug loop: submit → monitor → if failed, debug & retry
        previous_fixes: list[dict] = []
        final_result = None

        for debug_round in range(MAX_DEBUG_ROUNDS + 1):
            # On first round, check for existing job from a previous run (resume)
            existing = await self._find_existing_job(code_dir) if debug_round == 0 else None
            if existing:
                job_id, existing_status = existing
                self.log(f"Found existing SLURM job {job_id} (status: {existing_status})")
                if existing_status == "COMPLETED":
                    final_status = "COMPLETED"
                else:  # RUNNING or PENDING
                    final_status = await self._monitor_job(job_id, code_dir)
                    self.log(f"Existing job {job_id} finished: {final_status}")
            else:
                # Submit new SLURM job
                job_id = await self._submit_job(slurm_script)
                self.log(f"Submitted SLURM job: {job_id}")
                # Monitor job until completion
                final_status = await self._monitor_job(job_id, code_dir)
                self.log(f"Job {job_id} finished with status: {final_status}")

            # Collect results
            results = await self._collect_results(code_dir, job_id, final_status)
            self.log(f"Collected results: {list(results.keys())}")

            final_result = {
                "job_id": job_id,
                "final_status": final_status,
                "code_dir": str(code_dir),
                "debug_rounds": debug_round,
                **results,
            }

            # If job succeeded or we've exhausted debug rounds, stop
            if final_status == "COMPLETED":
                # Verify training actually produced results (not just exit code 0)
                has_metrics = bool(
                    results.get("metrics")
                    or results.get("parsed_metrics")
                    or results.get("training_log")
                    or results.get("training_log_csv")
                    or results.get("checkpoints")
                )
                if has_metrics:
                    self.log(f"Job completed successfully after {debug_round} debug round(s)")
                    break
                else:
                    # Check stdout/stderr for crash indicators
                    combined_log = results.get("stdout_log", "") + results.get("stderr_log", "")
                    crash_indicators = [
                        "RuntimeError", "Error(s) in loading", "Traceback",
                        "CUDA out of memory", "OOM", "Killed",
                        "Exception", "FileNotFoundError", "ModuleNotFoundError",
                    ]
                    has_crash = any(ind in combined_log for ind in crash_indicators)
                    if has_crash:
                        self.log(
                            "Job exited with code 0 but logs contain errors and no metrics produced. "
                            "Treating as FAILED."
                        )
                        final_status = "FAILED"
                        final_result["final_status"] = "FAILED"
                        # Fall through to debug loop
                    else:
                        self.log(f"Job completed after {debug_round} debug round(s) (no metrics found)")
                        break

            if debug_round >= MAX_DEBUG_ROUNDS:
                self.log(f"Max debug rounds ({MAX_DEBUG_ROUNDS}) reached, giving up")
                break

            # Job failed — enter debug loop
            self.log(f"Job failed, entering debug round {debug_round + 1}/{MAX_DEBUG_ROUNDS}")

            try:
                debug_result = await debug_agent.run(
                    code_dir=str(code_dir),
                    stdout_log=results.get("stdout_log", ""),
                    stderr_log=results.get("stderr_log", ""),
                    job_status=final_status,
                    debug_round=debug_round + 1,
                    previous_fixes=previous_fixes,
                )

                if not debug_result.get("needs_resubmit", False):
                    self.log("Debug agent determined no fix is possible, stopping")
                    break

                previous_fixes.append({
                    "diagnosis": debug_result.get("diagnosis", ""),
                    "patches": debug_result.get("patches", []),
                    "fixed_files": debug_result.get("fixed_files", []),
                })

                self.log(f"Debug round {debug_round + 1}: fixed {debug_result.get('fixed_files', [])}, resubmitting...")

            except Exception as e:
                self.log(f"Debug agent failed: {e}")
                break

        await debug_agent.close()

        self.workspace.write_json("plans/execution_output.json", final_result)
        return final_result

    async def _run_local_mode(
        self,
        code_dir: Path,
        coding_output: dict[str, Any],
        experiment_blueprint: dict[str, Any],
        setup_output: dict[str, Any],
        topic: str,
    ) -> dict[str, Any]:
        runner_script = code_dir / RUNNER_SCRIPT_NAME
        entry_train_command = str(
            coding_output.get("entry_train_command")
            or coding_output.get("train_command")
            or ""
        ).strip()
        if not runner_script.exists() and RUNNER_SCRIPT_NAME not in entry_train_command:
            runner_assets = ensure_project_runner(code_dir, entry_train_command)
            coding_output = {**coding_output, **runner_assets, "train_command": runner_assets["runner_command"]}
            self.log("Injected deterministic execution runner for compatibility")

        runtime_manager = RuntimeEnvironmentManager(self.config, self.log)
        runtime_env = await runtime_manager.prepare(code_dir)
        runtime_python = str(runtime_env.get("python", "python"))
        helper = ExperimentAgent(self.workspace, self.config)
        analyzer = FeedbackAnalyzer(self.config, self._dispatcher)
        base_command = self._build_local_command(code_dir, coding_output, runtime_python)
        blueprint_summary = self._build_execution_blueprint_summary(
            topic,
            experiment_blueprint,
            setup_output,
            coding_output,
        )
        max_rounds = max(
            1,
            1 if self.config.execution_profile.value == "fast_draft" else self.config.experiment_max_rounds,
        )
        iteration_state = IterationState(max_rounds=max_rounds)
        iteration_state, start_round = helper._load_iteration_checkpoint(
            iteration_state,
            LOCAL_EXECUTION_CHECKPOINT,
        )
        round_artifacts: dict[int, dict[str, Any]] = {}
        last_analysis = iteration_state.rounds[-1].analysis if iteration_state.rounds else None
        latest_hypothesis = ExperimentHypothesis(
            round_number=1,
            hypothesis="Validate generated deep-pipeline experiment locally",
            planned_changes=[],
            expected_signal="Dry-run passes and quick-eval produces metrics",
            rationale="Use the generated code as baseline before iterative repair.",
        )

        try:
            for round_num in range(start_round, max_rounds + 1):
                self.log(f"=== Local iteration round {round_num}/{max_rounds} ===")
                files_modified: list[str] = []

                if round_num > 1:
                    history_summary = helper._build_history_summary(iteration_state.rounds)
                    preflight_error_ctx = ""
                    if last_analysis and last_analysis.recommended_action:
                        preflight_error_ctx = (
                            "The previous round recommended this action:\n"
                            f"{last_analysis.recommended_action}\n"
                        )
                    latest_hypothesis = await helper._generate_iteration_hypothesis(
                        last_analysis,
                        history_summary,
                        blueprint_summary,
                        preflight_error_ctx=preflight_error_ctx,
                        code_dir=code_dir,
                    )
                    if latest_hypothesis.hypothesis == "__NO_NEW_IDEAS__":
                        iteration_state.final_status = "no_new_ideas"
                        self.log("Iteration loop exhausted new ideas, stopping")
                        break

                    files_modified = await helper._apply_iteration_changes(
                        latest_hypothesis,
                        code_dir,
                        runtime_python,
                    )
                    if not files_modified and latest_hypothesis.planned_changes:
                        self.log("Search-replace matched nothing, retrying with full-file rewrite")
                        files_modified = await helper._apply_iteration_changes_fullwrite(
                            latest_hypothesis,
                            code_dir,
                        )

                preflight = PreflightChecker(code_dir).run_all()
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_preflight.json",
                    preflight.model_dump(),
                )

                if preflight.overall_status == "failed":
                    error_message = "\n".join(preflight.blocking_failures)
                    analysis = await analyzer.analyze(
                        current_round=round_num,
                        metrics={},
                        previous_rounds=iteration_state.rounds,
                        stderr_snippet=error_message[:1000],
                        max_rounds=max_rounds,
                    )
                    round_result = RoundResult(
                        round_number=round_num,
                        hypothesis=latest_hypothesis,
                        preflight=preflight,
                        execution_status="skipped",
                        quick_eval_status="skipped",
                        metrics={},
                        analysis=analysis,
                        files_modified=files_modified,
                    )
                    iteration_state.rounds.append(round_result)
                    helper._save_iteration_checkpoint(iteration_state, LOCAL_EXECUTION_CHECKPOINT)
                    last_analysis = analysis
                    if not analysis.should_continue:
                        iteration_state.final_status = analysis.termination_reason or "preflight_failed"
                        break
                    continue

                execution = await self._run_local_dry_run_loop(
                    code_dir,
                    base_command,
                    blueprint_summary,
                    helper,
                )
                execution_status = execution.get("status", "failed")
                quick_eval = {"status": "skipped", "metrics": {}}
                if execution_status in ("success", "fixed"):
                    quick_eval = await self._run_local_quick_eval_loop(
                        code_dir,
                        base_command,
                        blueprint_summary,
                        helper,
                    )

                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_execution.json",
                    execution,
                )
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}_quick_eval.json",
                    quick_eval,
                )
                round_artifacts[round_num] = {
                    "execution": execution,
                    "quick_eval": quick_eval,
                }

                stderr_snippet = quick_eval.get("stderr", "") or execution.get("stderr", "")
                analysis = await analyzer.analyze(
                    current_round=round_num,
                    metrics=quick_eval.get("metrics", {}),
                    previous_rounds=iteration_state.rounds,
                    stderr_snippet=str(stderr_snippet)[:1000],
                    max_rounds=max_rounds,
                )

                round_result = RoundResult(
                    round_number=round_num,
                    hypothesis=latest_hypothesis,
                    preflight=preflight,
                    execution_status=execution_status,
                    quick_eval_status=quick_eval.get("status", "skipped"),
                    metrics=quick_eval.get("metrics", {}),
                    analysis=analysis,
                    files_modified=files_modified,
                )
                iteration_state.rounds.append(round_result)
                self._update_best_round(iteration_state, analysis)
                self.workspace.write_json(
                    f"logs/execution_round_{round_num}.json",
                    round_result.model_dump(),
                )
                helper._save_iteration_checkpoint(iteration_state, LOCAL_EXECUTION_CHECKPOINT)
                last_analysis = analysis

                self.log(
                    f"Round {round_num}: execution={execution_status}, "
                    f"quick_eval={quick_eval.get('status', 'skipped')}, "
                    f"continue={analysis.should_continue}"
                )
                if not analysis.should_continue:
                    iteration_state.final_status = analysis.termination_reason or "completed"
                    break
            else:
                iteration_state.final_status = "max_rounds"

            best_round_data = helper._get_best_round(iteration_state)
            best_round_number = iteration_state.best_round or (
                iteration_state.rounds[-1].round_number if iteration_state.rounds else None
            )
            best_artifact = (
                round_artifacts.get(best_round_number or -1)
                or self._load_local_round_artifacts(best_round_number)
            )
            execution = best_artifact.get("execution", {})
            quick_eval = best_artifact.get("quick_eval", {})
            artifact_results = self._collect_result_artifacts(code_dir)
            metrics = best_round_data.get("metrics") or quick_eval.get("metrics") or artifact_results.get("metrics", {})
            stdout_log = str(quick_eval.get("stdout") or execution.get("stdout") or "")[-10000:]
            stderr_log = str(quick_eval.get("stderr") or execution.get("stderr") or "")[-5000:]
            final_status = (
                "COMPLETED"
                if metrics or best_round_data.get("quick_eval_status") in ("success", "partial")
                else "FAILED"
            )

            final_result = {
                "job_id": "local",
                "execution_backend": "local",
                "runtime_env": runtime_env,
                "command": base_command,
                "code_dir": str(code_dir),
                "debug_rounds": max(0, len(iteration_state.rounds) - 1),
                "final_status": final_status,
                "execution_status": best_round_data.get("execution_status", "failed"),
                "quick_eval_status": best_round_data.get("quick_eval_status", "failed"),
                "experiment_status": best_round_data.get("quick_eval_status", "failed"),
                "metrics": metrics,
                "parsed_metrics": self._parse_metrics_from_log(stdout_log) if stdout_log else {},
                "experiment_results": metrics,
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "iteration_state": iteration_state.model_dump(),
                "experiment_summary": self._summarize_local_iteration(
                    iteration_state,
                    experiment_blueprint,
                ),
                **artifact_results,
            }
            final_result["metrics"] = metrics
            final_result["experiment_results"] = metrics
            return final_result
        finally:
            await helper.close()

    async def _run_local_dry_run_loop(
        self,
        code_dir: Path,
        base_command: list[str],
        blueprint_summary: str,
        helper: ExperimentAgent,
    ) -> dict[str, Any]:
        """Run dry-run with iterative batch-fix cycles."""
        max_fix_cycles = 5
        last_result: dict[str, Any] = {}
        fix_history: list[dict[str, Any]] = []

        for cycle in range(1, max_fix_cycles + 1):
            result = await self._run_subprocess(
                self._command_with_mode(base_command, "--dry-run"),
                cwd=code_dir,
                timeout=120,
            )
            last_result = result
            if result["returncode"] == 0:
                status = "success" if cycle == 1 else "fixed"
                return {"status": status, "attempts": cycle, **result}

            if cycle >= max_fix_cycles:
                break

            stderr_text = result.get("stderr", "")
            modified = await helper._batch_fix_errors(
                code_dir,
                stderr_text,
                blueprint_summary,
                mode="dry-run",
                previous_fixes=fix_history,
            )
            fix_history.append({"error_msg": stderr_text[:300], "cycle": cycle})
            if not modified:
                break

        return {"status": "failed", "attempts": cycle, **last_result}

    async def _run_local_quick_eval_loop(
        self,
        code_dir: Path,
        base_command: list[str],
        blueprint_summary: str,
        helper: ExperimentAgent,
    ) -> dict[str, Any]:
        """Run quick-eval with timeout handling and batch-fix cycles."""
        timeout = self.config.quick_eval_timeout
        max_fix_cycles = 5
        last_result: dict[str, Any] = {}
        fix_history: list[dict[str, Any]] = []

        metrics_path = code_dir / "results" / "metrics.json"
        for cycle in range(1, max_fix_cycles + 1):
            mtime_before = metrics_path.stat().st_mtime if metrics_path.exists() else None
            result = await self._run_subprocess(
                self._command_with_mode(base_command, "--quick-eval"),
                cwd=code_dir,
                timeout=timeout,
            )
            last_result = result
            if result["returncode"] == 0:
                return helper._collect_quick_eval_results(code_dir, result, attempt=cycle)

            if result["returncode"] == -1 and metrics_path.exists():
                mtime_after = metrics_path.stat().st_mtime
                if mtime_before is None or mtime_after > mtime_before:
                    metrics = helper._parse_metrics_json(code_dir)
                    if metrics:
                        return {
                            "status": "success",
                            "metrics": metrics,
                            "attempts": cycle,
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                        }

            if cycle >= max_fix_cycles:
                break

            if result["returncode"] == -1 and "timed out" in result.get("stderr", "").lower():
                modified = await helper._fix_timeout(code_dir)
            else:
                stderr_text = result.get("stderr", "")
                modified = await helper._batch_fix_errors(
                    code_dir,
                    stderr_text,
                    blueprint_summary,
                    mode="quick-eval",
                    previous_fixes=fix_history,
                )
                fix_history.append({"error_msg": stderr_text[:300], "cycle": cycle})
            if not modified:
                break

        return {"status": "failed", "metrics": {}, "attempts": cycle, **last_result}

    @staticmethod
    def _command_with_mode(base_command: list[str], mode_flag: str) -> list[str]:
        """Append a pipeline mode flag if it is not already present."""
        if mode_flag in base_command:
            return list(base_command)
        return [*base_command, mode_flag]

    @staticmethod
    def _build_execution_blueprint_summary(
        topic: str,
        blueprint: dict[str, Any],
        setup_output: dict[str, Any],
        coding_output: dict[str, Any],
    ) -> str:
        """Compact execution context used for iterative repair."""
        payload = {
            "topic": topic,
            "title": blueprint.get("title", ""),
            "proposed_method": blueprint.get("proposed_method", {}),
            "datasets": blueprint.get("datasets", []),
            "metrics": blueprint.get("metrics", []),
            "baselines": blueprint.get("baselines", []),
            "ablation_groups": blueprint.get("ablation_groups", []),
            "downloaded_resources": setup_output.get("downloaded_resources", []),
            "data_dir": setup_output.get("data_dir", ""),
            "models_dir": setup_output.get("models_dir", ""),
            "train_command": coding_output.get("train_command", ""),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _update_best_round(
        iteration_state: IterationState,
        analysis: Any,
    ) -> None:
        """Track the current best round using the primary metric heuristic."""
        if not analysis or not getattr(analysis, "metric_summary", None):
            return
        primary_key = next(iter(analysis.metric_summary), None)
        primary_value = analysis.metric_summary.get(primary_key) if primary_key else None
        best_value = (
            iteration_state.best_metrics.get(primary_key)
            if iteration_state.best_metrics and primary_key
            else None
        )
        lower_is_better = bool(
            primary_key and any(
                kw in primary_key.lower()
                for kw in ("loss", "error", "perplexity", "mse", "mae", "cer", "wer")
            )
        )
        if best_value is None or primary_value is None:
            is_improvement = best_value is None and primary_value is not None
        elif lower_is_better:
            is_improvement = primary_value < best_value
        else:
            is_improvement = primary_value > best_value
        if is_improvement:
            iteration_state.best_round = iteration_state.rounds[-1].round_number
            iteration_state.best_metrics = analysis.metric_summary

    def _load_local_round_artifacts(self, round_number: int | None) -> dict[str, Any]:
        """Best-effort reload of local round artifacts from disk."""
        if round_number is None:
            return {}
        execution_path = self.workspace.path / "logs" / f"execution_round_{round_number}_execution.json"
        quick_eval_path = self.workspace.path / "logs" / f"execution_round_{round_number}_quick_eval.json"
        data: dict[str, Any] = {}
        if execution_path.exists():
            data["execution"] = json.loads(execution_path.read_text(encoding="utf-8"))
        if quick_eval_path.exists():
            data["quick_eval"] = json.loads(quick_eval_path.read_text(encoding="utf-8"))
        return data

    @staticmethod
    def _summarize_local_iteration(
        iteration_state: IterationState,
        blueprint: dict[str, Any],
    ) -> str:
        """Create a concise experiment summary for downstream writing/analysis."""
        method_name = blueprint.get("proposed_method", {}).get("name", "the proposed method")
        lines = [
            f"Executed local iterative experiment loop for {method_name}.",
            f"Completed rounds: {len(iteration_state.rounds)} / {iteration_state.max_rounds}.",
        ]
        if iteration_state.best_round is not None:
            lines.append(f"Best round: {iteration_state.best_round}.")
        if iteration_state.best_metrics:
            metrics_text = ", ".join(
                f"{key}={value}" for key, value in iteration_state.best_metrics.items()
            )
            lines.append(f"Best metrics: {metrics_text}.")
        if iteration_state.rounds and iteration_state.rounds[-1].analysis:
            analysis = iteration_state.rounds[-1].analysis
            lines.append(f"Latest attribution: {analysis.attribution or 'unknown'}.")
            if analysis.recommended_action:
                lines.append(f"Latest recommended action: {analysis.recommended_action}.")
        lines.append(f"Termination: {iteration_state.final_status}.")
        return "\n".join(lines)

    async def _find_existing_job(self, code_dir: Path) -> tuple[str, str] | None:
        """Check if a previous SLURM job exists (from a crashed run).

        Returns (job_id, status) if found, None otherwise.
        """
        tracker = code_dir / "logs" / "active_job_id.txt"
        if not tracker.exists():
            return None

        job_id = tracker.read_text().strip()
        if not job_id or not job_id.isdigit():
            return None

        status = await self._get_job_status(job_id)
        if status in ("RUNNING", "PENDING", "COMPLETED"):
            return (job_id, status)

        return None  # FAILED/CANCELLED/UNKNOWN — need fresh submit

    async def _submit_job(self, slurm_script: str) -> str:
        """Submit a SLURM batch job and return the job ID."""
        if not Path(slurm_script).exists():
            raise RuntimeError(f"SLURM script not found: {slurm_script}")

        result = await self._run_shell(f"sbatch {slurm_script}")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        # Parse job ID from "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            raise RuntimeError(
                f"Failed to submit SLURM job. stdout: {stdout}, stderr: {stderr}"
            )

        job_id = match.group(1)

        # Save job ID for resume tracking
        tracker_path = Path(slurm_script).parent / "logs" / "active_job_id.txt"
        tracker_path.parent.mkdir(parents=True, exist_ok=True)
        tracker_path.write_text(job_id)

        return job_id

    async def _monitor_job(self, job_id: str, code_dir: Path) -> str:
        """Poll SLURM until job completes. Returns final status."""
        start_time = time.time()
        last_log_lines = 0

        while time.time() - start_time < MAX_WAIT_TIME:
            status = await self._get_job_status(job_id)

            # Stream training log if available
            log_files = list(code_dir.glob("logs/slurm_*.out"))
            if log_files:
                try:
                    content = log_files[-1].read_text(errors="replace")
                    lines = content.strip().split("\n")
                    if len(lines) > last_log_lines:
                        new_lines = lines[last_log_lines:]
                        for line in new_lines[-5:]:  # show last 5 new lines
                            self.log(f"[TRAIN] {line.strip()}")
                        last_log_lines = len(lines)
                except Exception:
                    pass

            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                return status

            if status == "PENDING":
                elapsed = int(time.time() - start_time)
                self.log(f"Job {job_id} pending... ({elapsed}s elapsed)")
            elif status == "RUNNING":
                elapsed = int(time.time() - start_time)
                self.log(f"Job {job_id} running... ({elapsed}s elapsed)")

            await asyncio.sleep(POLL_INTERVAL)

        # Timeout — cancel the job
        self.log(f"Job {job_id} exceeded max wait time ({MAX_WAIT_TIME}s), cancelling")
        await self._run_shell(f"scancel {job_id}")
        return "TIMEOUT"

    async def _get_job_status(self, job_id: str) -> str:
        """Query SLURM for job status."""
        result = await self._run_shell(
            f"squeue -j {job_id} -h -o '%T' 2>/dev/null || "
            f"sacct -j {job_id} -n -o State -X 2>/dev/null"
        )
        stdout = result.get("stdout", "").strip()

        if not stdout:
            # Job not in queue and not in accounting — might have just finished
            result2 = await self._run_shell(
                f"sacct -j {job_id} -n -o State -X"
            )
            stdout = result2.get("stdout", "").strip()

        # Parse status
        status = stdout.split("\n")[0].strip().upper() if stdout else "UNKNOWN"
        # Clean up status (sacct sometimes adds '+')
        status = status.rstrip("+").strip()

        return status

    async def _collect_results(
        self, code_dir: Path, job_id: str, status: str
    ) -> dict:
        """Collect training results from output files."""
        results: dict[str, Any] = {
            **self._collect_result_artifacts(code_dir),
            "stdout_log": "",
            "stderr_log": "",
        }

        # Read SLURM stdout/stderr (use the latest log files for this job)
        for log_file in sorted(code_dir.glob("logs/slurm_*.out")):
            if job_id in log_file.name:
                results["stdout_log"] = log_file.read_text(errors="replace")[-10000:]
                break
        else:
            # Fallback: read any .out file
            for log_file in code_dir.glob("logs/slurm_*.out"):
                results["stdout_log"] = log_file.read_text(errors="replace")[-10000:]

        for log_file in sorted(code_dir.glob("logs/slurm_*.err")):
            if job_id in log_file.name:
                results["stderr_log"] = log_file.read_text(errors="replace")[-5000:]
                break
        else:
            for log_file in code_dir.glob("logs/slurm_*.err"):
                results["stderr_log"] = log_file.read_text(errors="replace")[-5000:]

        # Parse metrics from stdout if metrics.json missing
        if not results["metrics"] and results["stdout_log"]:
            results["parsed_metrics"] = self._parse_metrics_from_log(results["stdout_log"])

        return results

    def _collect_result_artifacts(self, code_dir: Path) -> dict[str, Any]:
        results: dict[str, Any] = {
            "metrics": {},
            "training_log": [],
        }

        metrics_path = code_dir / "results" / "metrics.json"
        if metrics_path.exists():
            try:
                results["metrics"] = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                results["metrics"] = {"raw": metrics_path.read_text()[:5000]}

        log_csv = code_dir / "results" / "training_log.csv"
        if log_csv.exists():
            results["training_log_csv"] = log_csv.read_text(errors="replace")[:10000]

        for results_file in (code_dir / "results").glob("*"):
            if results_file.is_file() and results_file.name not in ("metrics.json", "training_log.csv"):
                try:
                    content = results_file.read_text(errors="replace")[:5000]
                    results[f"result_file_{results_file.name}"] = content
                except Exception:
                    pass

        checkpoints = (
            list((code_dir / "checkpoints").glob("*.pt"))
            if (code_dir / "checkpoints").exists()
            else []
        )
        results["checkpoints"] = [str(p) for p in checkpoints]
        return results

    def _collect_local_results(
        self,
        code_dir: Path,
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {
            **self._collect_result_artifacts(code_dir),
            "stdout_log": str(run_result.get("stdout", ""))[-10000:],
            "stderr_log": str(run_result.get("stderr", ""))[-5000:],
        }
        if not results["metrics"] and results["stdout_log"]:
            results["parsed_metrics"] = self._parse_metrics_from_log(results["stdout_log"])
        return results

    def _parse_metrics_from_log(self, log_text: str) -> dict:
        """Try to extract metrics from training log output."""
        metrics: dict[str, Any] = {}
        lines = log_text.split("\n")

        # Common patterns in training logs
        patterns = [
            # "Epoch 10: loss=0.123, accuracy=0.95"
            r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)",
            # "Test accuracy: 0.95"
            r"[Tt]est\s+(?:accuracy|acc)[=:\s]+([0-9.e-]+)",
            # "Best metric: 0.95"
            r"[Bb]est\s+\w+[=:\s]+([0-9.e-]+)",
            # "AUC: 0.95" / "F1: 0.85"
            r"(AUC|F1|RMSE|MAE|accuracy|precision|recall)[=:\s]+([0-9.e-]+)",
        ]

        epochs = []
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) == 2:
                        metrics[groups[0]] = groups[1]

            # Track epoch losses
            epoch_match = re.search(
                r"[Ee]poch\s+(\d+).*?loss[=:\s]+([0-9.e-]+)", line
            )
            if epoch_match:
                epochs.append({
                    "epoch": int(epoch_match.group(1)),
                    "loss": float(epoch_match.group(2)),
                })

        if epochs:
            metrics["epoch_losses"] = epochs
            metrics["final_loss"] = epochs[-1]["loss"]

        return metrics

    async def _local_preflight(self, code_dir: Path, python: str = "python") -> tuple[bool, str]:
        """Run local checks before submitting to SLURM.

        Tests:
        1. Python syntax check (py_compile) on all .py files
        2. Import check — try importing the entry point module
        3. Verify all cross-file imports resolve

        Returns (ok, error_message).
        """
        errors = []

        # 1. Syntax check all .py files
        for py_file in sorted(code_dir.glob("*.py")):
            result = await self._run_subprocess(
                [python, "-c", f"import py_compile; py_compile.compile(r'{py_file}', doraise=True)"],
                timeout=10,
            )
            if result["returncode"] != 0:
                errors.append(f"Syntax error in {py_file.name}:\n{result['stderr']}")

        if errors:
            return False, "\n".join(errors)

        # 2. Try importing the main modules to catch import errors
        # (run in the code directory so local imports work)
        py_modules = [f.stem for f in code_dir.glob("*.py")]
        for module in py_modules:
            result = await self._run_subprocess(
                [python, "-c", f"import {module}"],
                cwd=code_dir,
                timeout=30,
            )
            if result["returncode"] != 0:
                err_text = result["stdout"] + result["stderr"]
                # Ignore errors from missing heavy dependencies (torch, etc.)
                # — those will be installed on the cluster node
                if any(pkg in err_text for pkg in [
                    "No module named 'torch'",
                    "No module named 'torchvision'",
                    "No module named 'torchaudio'",
                    "No module named 'timm'",
                    "No module named 'transformers'",
                    "No module named 'torch_geometric'",
                    "No module named 'torch_scatter'",
                    "No module named 'torch_sparse'",
                    "No module named 'esm'",
                    "No module named 'dgl'",
                    "No module named 'accelerate'",
                    "No module named 'datasets'",
                    "No module named 'einops'",
                    "No module named 'wandb'",
                    "No module named 'scipy'",
                    "No module named 'sklearn'",
                    "No module named 'cv2'",
                    "No module named 'PIL'",
                    "CUDA",
                ]):
                    continue
                errors.append(f"Import error in {module}.py:\n{err_text}")

        if errors:
            return False, "\n".join(errors)

        return True, ""

    def _build_local_command(
        self,
        code_dir: Path,
        coding_output: dict[str, Any],
        runtime_python: str,
    ) -> list[str]:
        runner_script = str(coding_output.get("runner_script", "")).strip()
        if runner_script and Path(runner_script).exists():
            runner_path = Path(runner_script)
            runner_token = runner_path.name if runner_path.parent == code_dir else str(runner_path)
            return [runtime_python, runner_token]
        if (code_dir / RUNNER_SCRIPT_NAME).exists():
            return [runtime_python, RUNNER_SCRIPT_NAME]

        command = str(
            coding_output.get("entry_train_command")
            or coding_output.get("train_command")
            or coding_output.get("code_plan", {}).get("train_command", "")
            or ""
        ).strip()
        if command:
            try:
                tokens = shlex.split(command, posix=platform.system() != "Windows")
            except ValueError:
                tokens = []
            if tokens:
                if is_python_launcher_token(tokens[0]):
                    return [runtime_python, *tokens[1:]]
                if tokens[0] in {"-m", "-c"} or tokens[0].endswith(".py"):
                    return [runtime_python, *tokens]
                return tokens

        for candidate in ("main.py", "train.py", "run.py"):
            if (code_dir / candidate).exists():
                return [runtime_python, candidate]
        return [runtime_python, "main.py"]

    async def _run_local_training(
        self,
        code_dir: Path,
        command: list[str],
    ) -> dict[str, Any]:
        timeout = max(60, int(self.config.local_execution_timeout))
        result = await self._run_subprocess(command, cwd=code_dir, timeout=timeout)
        result["command"] = command
        result["timed_out"] = (
            result.get("returncode") == -1
            and "timed out" in result.get("stderr", "").lower()
        )
        return result

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        env = self._build_proxy_env()
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    def _build_proxy_env(self) -> dict[str, str]:
        env = {**os.environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re

            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                match = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if match:
                    proxy_url = match.group(1)
        if proxy_url:
            env.update(
                {
                    "http_proxy": proxy_url,
                    "https_proxy": proxy_url,
                    "HTTP_PROXY": proxy_url,
                    "HTTPS_PROXY": proxy_url,
                }
            )
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return env

    async def _run_subprocess(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 60,
    ) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._build_proxy_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        pass
