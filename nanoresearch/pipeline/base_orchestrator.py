"""Base pipeline orchestrator — shared checkpoint/resume/retry logic.

Concrete subclasses (PipelineOrchestrator, DeepPipelineOrchestrator) only
need to supply agent construction, input wiring, and stage lists.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable

from nanoresearch.agents.ideation import IdeationAgent
from nanoresearch.agents.planning import PlanningAgent
from nanoresearch.agents.setup import SetupAgent
from nanoresearch.agents.coding import CodingAgent
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.baseline_execution import BaselineExecutionAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.blueprint_validator import validate_blueprint
from nanoresearch.pipeline.cost_tracker import CostTracker
from nanoresearch.pipeline.progress import ProgressEmitter
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.pipeline.status_tracker import StatusTracker
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage

logger = logging.getLogger(__name__)

# Retry backoff settings (centralised in constants.py)
from nanoresearch.agents.constants import (
    RETRY_BACKOFF_FACTOR,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

# Progress callback type: (stage_name, status, message)
ProgressCallback = Callable[[str, str, str], None]


class BaseOrchestrator(ABC):
    """Abstract base for pipeline orchestrators with checkpoint/resume."""

    # --- subclass must override these class attributes ---
    _STAGE_KEY_MAP: dict[PipelineStage, str] = {}
    _OUTPUT_FILE_MAP: dict[PipelineStage, str] = {}
    _PIPELINE_MODE: PipelineMode | None = None  # None = STANDARD

    def __init__(
        self,
        workspace: Workspace,
        config: ResearchConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        self.progress_callback = progress_callback
        self.cost_tracker = CostTracker()
        self.progress_emitter = ProgressEmitter(workspace.path / "progress.json")
        self.status_tracker = StatusTracker(workspace.path)

        self.state_machine = PipelineStateMachine(
            workspace.manifest.current_stage,
            mode=self._PIPELINE_MODE or PipelineMode.STANDARD,
        )

        self._agents: dict[PipelineStage, BaseResearchAgent] = self._build_agents()
        # Wire each agent's dispatcher to feed the cost tracker
        for agent in self._agents.values():
            agent._dispatcher._usage_callback = self.cost_tracker.record

    # ------------------------------------------------------------------
    # Abstract / hook methods — subclasses must / may override
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_agents(self) -> dict[PipelineStage, BaseResearchAgent]:
        """Create one agent per pipeline stage."""

    @abstractmethod
    def _get_processing_stages(self) -> list[PipelineStage]:
        """Return the ordered list of stages to execute."""

    @abstractmethod
    def _prepare_inputs(
        self,
        stage: PipelineStage,
        topic: str,
        accumulated: dict,
        last_error: str,
    ) -> dict[str, Any]:
        """Build the kwargs dict for ``agent.run()``."""

    def _get_initial_results(self, topic: str) -> dict[str, Any]:
        """Initial results dict.  Override to add pipeline_mode, etc."""
        return {"topic": topic}

    def _post_pipeline(self, results: dict[str, Any]) -> None:
        """Hook called after all stages complete and cost is saved.

        Default is no-op.  Deep pipeline overrides for export logic.
        """

    # ------------------------------------------------------------------
    # Shared logic
    # ------------------------------------------------------------------

    def _report_progress(self, stage: str, status: str, message: str) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(stage, status, message)
            except Exception as exc:
                logger.debug("Progress callback error (non-fatal): %s", exc)

    async def close(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, topic: str) -> dict[str, Any]:
        """Run the full pipeline from current stage to DONE."""
        mode_label = "DEEP" if self._PIPELINE_MODE == PipelineMode.DEEP else "standard"
        logger.info("Starting %s pipeline for topic: %s", mode_label, topic)
        logger.info("Current stage: %s", self.state_machine.current.value)

        # Ensure manifest records the pipeline mode (deep only)
        if (
            self._PIPELINE_MODE == PipelineMode.DEEP
            and self.workspace.manifest.pipeline_mode != PipelineMode.DEEP
        ):
            self.workspace.update_manifest(pipeline_mode=PipelineMode.DEEP)

        self._reset_stale_running_stages()

        results = self._get_initial_results(topic)

        try:
            stages = self._get_processing_stages()
            for stage_idx, stage in enumerate(stages):
                # Skip already-completed stages (for resume)
                stage_record = self.workspace.manifest.stages.get(stage.value)
                if stage_record and stage_record.status == "completed":
                    logger.info("Skipping completed stage: %s", stage.value)
                    self._report_progress(
                        stage.value, "skipped",
                        f"[{stage_idx+1}/{len(stages)}] {stage.value} already completed",
                    )
                    output = self._load_stage_output(stage, require=True)
                    results.update(output)
                    # BUG-21 fix: use force_set() with logging instead of
                    # directly mutating private _current attribute.
                    self.state_machine.force_set(stage)
                    continue

                # Skip stages configured to be skipped
                if stage.value in self.config.skip_stages:
                    logger.info("Skipping stage %s (configured in skip_stages)", stage.value)
                    if self.state_machine.current != stage:
                        if self.state_machine.can_transition(stage):
                            self.state_machine.transition(stage)
                        else:
                            logger.warning(
                                "Skipping stage %s from non-adjacent state %s; "
                                "forcing state machine to match",
                                stage.value,
                                self.state_machine.current.value,
                            )
                            self.state_machine.force_set(stage)
                    self.workspace.update_manifest(current_stage=stage)
                    self._report_progress(
                        stage.value, "skipped",
                        f"[{stage_idx+1}/{len(stages)}] {stage.value} skipped by config",
                    )
                    continue

                # Check transition
                if not self.state_machine.can_transition(stage):
                    if self.state_machine.current == stage:
                        pass  # resuming this stage
                    else:
                        prior = self._load_stage_output(stage)
                        if prior:
                            results.update(prior)
                            logger.info("Loaded prior output for skipped stage %s", stage.value)
                        else:
                            logger.warning(
                                "Skipping stage %s (no transition from %s) and no prior output found",
                                stage.value, self.state_machine.current.value,
                            )
                        continue

                if self.state_machine.current != stage:
                    self.state_machine.transition(stage)

                self._report_progress(
                    stage.value, "started",
                    f"[{stage_idx+1}/{len(stages)}] Running {stage.value}...",
                )
                self.progress_emitter.stage_start(
                    stage.value, len(stages), stage_idx,
                    f"[{stage_idx+1}/{len(stages)}] Running {stage.value}...",
                )
                self.cost_tracker.set_stage(stage.value)

                # Run with retry
                t0 = time.monotonic()
                stage_result = await self._run_stage_with_retry(stage, topic, results)
                duration = time.monotonic() - t0
                logger.info("Stage %s completed in %.1fs", stage.value, duration)
                results.update(stage_result)

                # Cross-stage reference validation
                self._validate_cross_stage_refs(stage, results)

                # Blueprint semantic validation after PLANNING
                if stage == PipelineStage.PLANNING:
                    bp = results.get("experiment_blueprint", {})
                    issues = validate_blueprint(bp)
                    if issues:
                        for issue in issues:
                            logger.warning("Blueprint issue: %s", issue)
                        self.progress_emitter.substep(
                            stage.value,
                            f"Blueprint validation: {len(issues)} issue(s) found",
                        )

                # NOTE: Gate 1 (post-CODING) is checked inside _run_stage_with_retry
                # so that failures trigger the retry mechanism.

                # === QUALITY GATE 2: Post-EXECUTION Result Sanity Check ===
                if stage == PipelineStage.EXECUTION:
                    exec_out = results.get("execution_output", {})
                    if exec_out:
                        from pathlib import Path as _Path
                        from nanoresearch.pipeline.quality_gates import (
                            validate_post_execution,
                            format_gate_failure_message,
                        )
                        exec_code_dir = _Path(exec_out.get("code_dir", ""))
                        gate_passed, gate_issues = validate_post_execution(
                            exec_out,
                            results.get("experiment_blueprint", {}),
                            code_dir=exec_code_dir if exec_code_dir.exists() else None,
                        )
                        if not gate_passed:
                            self.progress_emitter.substep(
                                stage.value,
                                f"Quality Gate 2 FAILED: {len(gate_issues)} issue(s)",
                            )
                            # Gate 2 failure is a warning, not a hard stop,
                            # because the execution did complete — the analysis
                            # stage should interpret broken results correctly.
                            for gi in gate_issues:
                                logger.warning("Gate 2 issue: %s", gi)
                            exec_out["quality_gate_issues"] = gate_issues
                            exec_out["quality_gate_passed"] = False
                        else:
                            logger.info("Quality Gate 2 (post-EXECUTION) PASSED")
                            exec_out["quality_gate_passed"] = True

                # === QUALITY GATE 3: Post-ANALYSIS Scientific Review ===
                if stage == PipelineStage.ANALYSIS:
                    analysis_out = results.get("analysis_output", {})
                    exec_out = results.get("execution_output", {})
                    if analysis_out:
                        from nanoresearch.pipeline.quality_gates import (
                            validate_post_analysis,
                            format_gate_failure_message,
                        )
                        gate_passed, gate_issues = validate_post_analysis(
                            analysis_out,
                            results.get("experiment_blueprint", {}),
                            exec_out or {},
                        )
                        if not gate_passed:
                            self.progress_emitter.substep(
                                stage.value,
                                f"Quality Gate 3 FAILED: {len(gate_issues)} issue(s)",
                            )
                            logger.warning(
                                "Quality Gate 3 (post-ANALYSIS) FAILED — "
                                "looping back to CODING to fix %d identified issues.",
                                len(gate_issues),
                            )

                            # Save gate issues as context for the next CODING run
                            self.workspace.write_json(
                                "plans/quality_gate3_issues.json",
                                {
                                    "gate": "post-ANALYSIS",
                                    "issues": gate_issues,
                                    "remediation": (
                                        "Re-generate code to fix identified issues: "
                                        "correct dataset paths, ensure sufficient "
                                        "training epochs, fix baseline execution, "
                                        "and produce scientifically plausible results."
                                    ),
                                },
                            )

                            # Reset state machine to CODING
                            self.state_machine.force_set(PipelineStage.CODING)
                            self.workspace.update_manifest(
                                current_stage=PipelineStage.CODING,
                            )

                            # Erase completion status from CODING onwards
                            manifest = self.workspace.manifest
                            reset_stages = [
                                PipelineStage.CODING,
                                PipelineStage.BASELINE_EXECUTION,
                                PipelineStage.EXECUTION,
                                PipelineStage.ANALYSIS,
                                PipelineStage.FIGURE_GEN,
                                PipelineStage.WRITING,
                                PipelineStage.REVIEW,
                            ]
                            for s in reset_stages:
                                if s.value in manifest.stages:
                                    del manifest.stages[s.value]
                            self.workspace._write_manifest(manifest)

                            logger.info(
                                "Pipeline state reset to CODING. "
                                "Restarting pipeline loop."
                            )
                            return await self.run(topic)

                        logger.info("Quality Gate 3 (post-ANALYSIS) PASSED")

                self._report_progress(
                    stage.value, "completed",
                    f"[{stage_idx+1}/{len(stages)}] {stage.value} completed",
                )
                self.progress_emitter.stage_complete(
                    stage.value, len(stages), stage_idx,
                    f"[{stage_idx+1}/{len(stages)}] {stage.value} completed in {duration:.1f}s",
                )

                # Cyclical Baseline-Driven Scientific Regression Check
                if stage == PipelineStage.EXPERIMENT or stage == PipelineStage.EXECUTION:
                    exp_out = results.get("experiment_output", {}) if stage == PipelineStage.EXPERIMENT else results.get("execution_output", {}) # Fallback if standard mode
                    if exp_out.get("experiment_status") == "failed_to_beat_baseline":
                        logger.warning("Pipeline hit a structural wall: The explored method failed to beat the baseline.")
                        logger.warning("Triggering global pipeline regression back to IDEATION phase to formulate a new architecture.")

                        import shutil
                        import uuid
                        archive_id = uuid.uuid4().hex[:6]
                        archive_dir = self.workspace.path / f"experiment_failed_{archive_id}"
                        code_dir = self.workspace.path / "code"
                        if code_dir.exists():
                            shutil.move(str(code_dir), str(archive_dir))
                        logger.info(f"Failed experiment code archived to {archive_dir.name}")

                        # Accumulate failed experiments to pass to IDEATION
                        try:
                            failed_list = self.workspace.read_json("plans/failed_experiments_ledger.json")
                        except FileNotFoundError:
                            failed_list = []
                            
                        failed_list.append({
                            "archive": archive_dir.name,
                            "blueprint": results.get("experiment_blueprint", {}),
                            "metrics": exp_out.get("experiment_results", {})
                        })
                        self.workspace.write_json("plans/failed_experiments_ledger.json", failed_list)

                        # Reset state machine to IDEATION
                        self.state_machine.force_set(PipelineStage.IDEATION)
                        self.workspace.update_manifest(current_stage=PipelineStage.IDEATION)

                        # Erase completion status for all stages from IDEATION onwards
                        manifest = self.workspace.manifest
                        reset_stages = [
                            PipelineStage.IDEATION, PipelineStage.PLANNING, 
                            PipelineStage.SETUP, PipelineStage.BASELINE_CODING,
                            PipelineStage.BASELINE_EXECUTION, PipelineStage.CODING,
                            PipelineStage.EXECUTION, PipelineStage.EXPERIMENT,
                            PipelineStage.ANALYSIS, PipelineStage.FIGURE_GEN,
                            PipelineStage.WRITING, PipelineStage.REVIEW
                        ]
                        for s in reset_stages:
                            if s.value in manifest.stages:
                                del manifest.stages[s.value]
                        self.workspace._write_manifest(manifest)
                        
                        # Recursively restart pipeline loop
                        return await self.run(topic)

            # Mark pipeline as DONE
            self.state_machine.transition(PipelineStage.DONE)
            self.workspace.update_manifest(current_stage=PipelineStage.DONE)

            # Save cost summary
            cost_summary = self.cost_tracker.summary()
            self.workspace.write_json("logs/cost_summary.json", cost_summary)
            results["cost_summary"] = cost_summary
            if cost_summary["total_tokens"] > 0:
                logger.info(
                    "Cost summary: %d total tokens, %d calls, %.1fs total latency",
                    cost_summary["total_tokens"],
                    cost_summary["total_calls"],
                    cost_summary["total_latency_ms"] / 1000,
                )

            self.progress_emitter.pipeline_complete(
                True, f"{mode_label.capitalize()} pipeline completed successfully",
            )
            logger.info("%s pipeline completed!", mode_label.capitalize())

            self._post_pipeline(results)

            return results
        except Exception:
            # Save cost summary even on failure so users can see token usage
            try:
                cost_summary = self.cost_tracker.summary()
                self.workspace.write_json("logs/cost_summary.json", cost_summary)
                if cost_summary["total_tokens"] > 0:
                    logger.info(
                        "Cost summary (on failure): %d total tokens, %d calls, %.1fs total latency",
                        cost_summary["total_tokens"],
                        cost_summary["total_calls"],
                        cost_summary["total_latency_ms"] / 1000,
                    )
            except Exception as cost_exc:
                logger.debug("Failed to save cost summary on failure: %s", cost_exc)
            self.progress_emitter.pipeline_complete(False, f"{mode_label.capitalize()} pipeline failed")
            raise

    async def _run_stage_with_retry(
        self, stage: PipelineStage, topic: str, accumulated: dict
    ) -> dict[str, Any]:
        """Run a stage with retry logic."""
        max_retries = self.config.max_retries
        last_error = ""

        for attempt in range(max_retries + 1):
            try:
                self.workspace.mark_stage_running(stage)
                logger.info(
                    "Running %s (attempt %d/%d)",
                    stage.value, attempt + 1, max_retries + 1,
                )

                agent = self._agents[stage]
                inputs = self._prepare_inputs(stage, topic, accumulated, last_error)

                if stage in {PipelineStage.BASELINE_EXECUTION, PipelineStage.EXECUTION}:
                    blueprint = accumulated.get("experiment_blueprint", {})
                    paper_summary_check = {}
                    if isinstance(blueprint, dict):
                        paper_summary_check = blueprint.get("paper_summary_check", {}) or {}
                    missing_count = int(paper_summary_check.get("missing_count", 0) or 0)
                    if not paper_summary_check:
                        paper_summary_check = self.workspace.validate_baseline_paper_summaries(blueprint)
                        missing_count = int(paper_summary_check.get("missing_count", 0) or 0)
                        if isinstance(blueprint, dict):
                            blueprint["paper_summary_check"] = paper_summary_check
                            accumulated["experiment_blueprint"] = blueprint
                            self.workspace.write_json("plans/experiment_blueprint.json", blueprint)
                    if missing_count > 0:
                        queue_path = paper_summary_check.get("queue_path", "plans/paper_enrichment_queue.json")
                        raise RuntimeError(
                            "Baseline reference summaries are incomplete "
                            f"({missing_count} pending). Resolve queue first: {queue_path}. "
                            "Use ccr code with paper-reading skills to enrich missing fields."
                        )

                result = await agent.run(**inputs)

                # Post-execution validation: detect stages that returned but
                # didn't actually produce usable output (e.g. "skipped")
                if stage == PipelineStage.BASELINE_EXECUTION:
                    if isinstance(result, dict) and result.get("status") == "skipped":
                        reason = result.get("reason", "unknown")
                        raise RuntimeError(
                            f"BASELINE_EXECUTION was skipped ({reason}). "
                            "This blocks downstream experiment comparison. "
                            "Ensure the CodingAgent generates baselines/ directory "
                            "with run_all.sh or per-baseline train.py scripts."
                        )

                # === QUALITY GATE 1: Post-CODING Scientific Code Review ===
                # Must be inside the retry loop so failures trigger retries
                if stage == PipelineStage.CODING:
                    code_dir_str = result.get("code_dir", "")
                    if code_dir_str:
                        from pathlib import Path as _Path
                        from nanoresearch.pipeline.quality_gates import (
                            validate_post_coding,
                            autofix_gate1_issues,
                        )
                        _code_path = _Path(code_dir_str)
                        _blueprint = accumulated.get("experiment_blueprint", {})
                        _setup = accumulated.get("setup_output", {})

                        gate_passed, gate_issues = validate_post_coding(
                            _code_path, _blueprint, _setup,
                        )
                        if not gate_passed:
                            self.progress_emitter.substep(
                                stage.value,
                                f"Quality Gate 1 FAILED: {len(gate_issues)} issue(s) — applying targeted fixes",
                            )
                            # Apply surgical in-place fixes
                            fixes = autofix_gate1_issues(_code_path, gate_issues)
                            for fix in fixes:
                                logger.info("Gate 1 auto-fix: %s", fix)

                            # Re-validate after fixes
                            gate_passed_2, remaining_issues = validate_post_coding(
                                _code_path, _blueprint, _setup,
                            )
                            if not gate_passed_2:
                                # Log remaining issues as warnings but don't block
                                # — autofix handled what it could, the rest will be
                                # caught by Gate 2/3 after execution anyway.
                                for ri in remaining_issues:
                                    logger.warning("Gate 1 residual (non-blocking): %s", ri)
                                self.progress_emitter.substep(
                                    stage.value,
                                    f"Gate 1: {len(fixes)} fixed, {len(remaining_issues)} residual warnings",
                                )
                            else:
                                logger.info(
                                    "Quality Gate 1 PASSED after auto-fix (%d fixes applied)",
                                    len(fixes),
                                )
                        else:
                            logger.info("Quality Gate 1 (post-CODING) PASSED")

                self.workspace.mark_stage_completed(
                    stage, self._OUTPUT_FILE_MAP.get(stage, ""),
                )
                try:
                    self.status_tracker.generate_status_report()
                except Exception as exc:
                    logger.debug("STATUS.md update failed (non-fatal): %s", exc)
                logger.info("Stage %s completed", stage.value)
                return self._wrap_stage_output(stage, result)

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                logger.error("Stage %s failed: %s", stage.value, last_error)

                self.workspace.write_text(
                    f"logs/{stage.value.lower()}_error_{attempt}.txt",
                    f"Error: {last_error}\n\nTraceback:\n{tb}",
                )

                if attempt < max_retries:
                    self.workspace.increment_retry(stage)
                    delay = min(
                        RETRY_BASE_DELAY * (RETRY_BACKOFF_FACTOR ** attempt),
                        RETRY_MAX_DELAY,
                    )
                    logger.info(
                        "Retrying %s in %.0fs (attempt %d/%d)...",
                        stage.value, delay, attempt + 2, max_retries + 1,
                    )
                    self._report_progress(
                        stage.value, "retrying",
                        f"Retrying in {delay:.0f}s...",
                    )
                    await asyncio.sleep(delay)
                else:
                    self.workspace.mark_stage_failed(stage, last_error)
                    self.state_machine.fail()
                    raise RuntimeError(
                        f"Stage {stage.value} failed after {max_retries + 1} attempts: {last_error}"
                    ) from e

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _wrap_stage_output(self, stage: PipelineStage, result: dict) -> dict[str, Any]:
        """Wrap agent output with a stage-specific key."""
        key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
        return {key: result}

    def _load_stage_output(
        self, stage: PipelineStage, *, require: bool = False
    ) -> dict[str, Any]:
        """Load previously saved output for a completed stage."""
        path = self._OUTPUT_FILE_MAP.get(stage)
        if path:
            try:
                data = self.workspace.read_json(path)
                key = self._STAGE_KEY_MAP.get(stage, stage.value.lower())
                return {key: data}
            except FileNotFoundError:
                if require:
                    raise RuntimeError(
                        f"Stage {stage.value} is marked completed but output "
                        f"file '{path}' is missing. The workspace may be "
                        f"corrupted. Delete the workspace and re-run, or "
                        f"manually reset the stage status in manifest.json."
                    )
                logger.warning(
                    "Stage %s marked completed but output file %s not found",
                    stage.value, path,
                )
        return {}

    def _reset_stale_running_stages(self) -> None:
        """Reset stages stuck in 'running' status back to 'pending'."""
        manifest = self.workspace.manifest
        changed = False
        for stage_key, record in manifest.stages.items():
            if record.status == "running":
                logger.warning(
                    "Stage %s was left in 'running' status (likely from a crash). "
                    "Resetting to 'pending' for re-execution.",
                    stage_key,
                )
                record.status = "pending"
                record.error_message = ""
                changed = True
        if changed:
            self.workspace._write_manifest(manifest)

    def _validate_cross_stage_refs(
        self, stage: PipelineStage, results: dict[str, Any]
    ) -> None:
        """Validate cross-stage references.  Logs warnings, never errors."""
        if stage == PipelineStage.PLANNING:
            blueprint = results.get("experiment_blueprint", {})
            ideation = results.get("ideation_output", {})
            hyp_ref = blueprint.get("hypothesis_ref", "")
            if hyp_ref and ideation:
                hyp_ids = {
                    h.get("hypothesis_id", "")
                    for h in ideation.get("hypotheses", [])
                }
                if hyp_ref not in hyp_ids:
                    logger.warning(
                        "Cross-ref mismatch: blueprint.hypothesis_ref=%r "
                        "not found in ideation hypotheses %s",
                        hyp_ref, hyp_ids,
                    )

        elif stage == PipelineStage.EXPERIMENT:
            exp_out = results.get("experiment_output", {})
            blueprint = results.get("experiment_blueprint", {})
            bp_metrics = {
                m.get("name", "") for m in blueprint.get("metrics", [])
            }
            if bp_metrics and exp_out:
                for entry in exp_out.get("experiment_results", {}).get("main_results", []):
                    for metric in entry.get("metrics", []):
                        mname = metric.get("metric_name", "")
                        if mname and mname not in bp_metrics:
                            logger.warning(
                                "Cross-ref mismatch: experiment metric %r "
                                "not defined in blueprint metrics %s",
                                mname, bp_metrics,
                            )
