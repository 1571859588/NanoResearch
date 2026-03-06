"""Deep pipeline orchestrator — extends the base pipeline with code search, execution, and analysis."""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.agents.ideation import IdeationAgent
from nanoresearch.agents.planning import PlanningAgent
from nanoresearch.agents.setup import SetupAgent
from nanoresearch.agents.coding import CodingAgent
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.analysis import AnalysisAgent
from nanoresearch.agents.writing import WritingAgent
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Deep pipeline stages in order
DEEP_STAGES = [
    PipelineStage.IDEATION,
    PipelineStage.PLANNING,
    "SETUP",
    "CODING",
    "EXECUTION",
    "ANALYSIS",
    PipelineStage.WRITING,
]

# Map string stage names to agent classes
DEEP_STAGE_NAMES = [
    "IDEATION", "PLANNING", "SETUP", "CODING",
    "EXECUTION", "ANALYSIS", "WRITING",
]


class DeepPipelineOrchestrator:
    """Runs the deep research pipeline: ideation → planning → setup → coding → execution → analysis → writing."""

    def __init__(self, workspace: Workspace, config: ResearchConfig) -> None:
        self.workspace = workspace
        self.config = config
        self._agents: dict[str, BaseResearchAgent] = {
            "IDEATION": IdeationAgent(workspace, config),
            "PLANNING": PlanningAgent(workspace, config),
            "SETUP": SetupAgent(workspace, config),
            "CODING": CodingAgent(workspace, config),
            "EXECUTION": ExecutionAgent(workspace, config),
            "ANALYSIS": AnalysisAgent(workspace, config),
            "WRITING": WritingAgent(workspace, config),
        }

    async def close(self) -> None:
        for agent in self._agents.values():
            await agent.close()

    async def run(self, topic: str) -> dict[str, Any]:
        """Run the full deep pipeline."""
        logger.info(f"Starting DEEP pipeline for topic: {topic}")

        results: dict[str, Any] = {"topic": topic, "pipeline_mode": "deep"}

        for stage_name in DEEP_STAGE_NAMES:
            # Check if stage already completed
            stage_record = self.workspace.manifest.stages.get(stage_name)
            if stage_record and stage_record.status == "completed":
                logger.info(f"Skipping completed stage: {stage_name}")
                results.update(self._load_stage_output(stage_name))
                continue

            # Run stage with retry
            logger.info(f"Running deep stage: {stage_name}")
            stage_result = await self._run_stage_with_retry(
                stage_name, topic, results
            )
            results.update(stage_result)

        # Mark pipeline done
        self.workspace.update_manifest(current_stage=PipelineStage.DONE)
        logger.info("Deep pipeline completed!")

        # Export clean output folder
        try:
            export_path = self.workspace.export()
            logger.info(f"Exported project to: {export_path}")
            results["export_path"] = str(export_path)

            # Also copy experiment code and results into export
            exp_dir = self.workspace.path / "experiment"
            if exp_dir.exists():
                import shutil
                code_dest = export_path / "code"
                code_dest.mkdir(exist_ok=True)
                for f in list(exp_dir.glob("*.py")) + list(exp_dir.glob("*.txt")) + list(exp_dir.glob("*.slurm")):
                    shutil.copy2(f, code_dest / f.name)
                results_src = exp_dir / "results"
                if results_src.exists():
                    results_dest = export_path / "results"
                    results_dest.mkdir(exist_ok=True)
                    for f in results_src.iterdir():
                        if f.is_file() and f.suffix in (".json", ".csv", ".log"):
                            shutil.copy2(f, results_dest / f.name)
        except Exception as exc:
            logger.warning(f"Export failed (non-fatal): {exc}")

        return results

    async def _run_stage_with_retry(
        self, stage_name: str, topic: str, accumulated: dict
    ) -> dict[str, Any]:
        """Run a stage with retry logic."""
        max_retries = self.config.max_retries
        last_error = ""

        for attempt in range(max_retries + 1):
            try:
                self._mark_stage_running(stage_name)
                logger.info(
                    f"Running {stage_name} (attempt {attempt + 1}/{max_retries + 1})"
                )

                agent = self._agents[stage_name]
                inputs = self._prepare_inputs(stage_name, topic, accumulated, last_error)
                result = await agent.run(**inputs)

                self._mark_stage_completed(stage_name)
                logger.info(f"Stage {stage_name} completed")
                return self._wrap_stage_output(stage_name, result)

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                logger.error(f"Stage {stage_name} failed: {last_error}")

                self.workspace.write_text(
                    f"logs/{stage_name.lower()}_error_{attempt}.txt",
                    f"Error: {last_error}\n\nTraceback:\n{tb}",
                )

                if attempt < max_retries:
                    logger.info(f"Retrying {stage_name}...")
                else:
                    self._mark_stage_failed(stage_name, last_error)
                    raise RuntimeError(
                        f"Stage {stage_name} failed after {max_retries + 1} attempts: {last_error}"
                    ) from e

        raise RuntimeError("Unreachable")

    def _prepare_inputs(
        self, stage_name: str, topic: str, accumulated: dict, last_error: str
    ) -> dict[str, Any]:
        """Prepare inputs for each deep stage."""
        inputs: dict[str, Any] = {}

        if stage_name == "IDEATION":
            inputs["topic"] = topic

        elif stage_name == "PLANNING":
            inputs["ideation_output"] = accumulated.get("ideation_output", {})

        elif stage_name == "SETUP":
            inputs["topic"] = topic
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage_name == "CODING":
            inputs["topic"] = topic
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            inputs["setup_output"] = accumulated.get("setup_output", {})

        elif stage_name == "EXECUTION":
            inputs["coding_output"] = accumulated.get("coding_output", {})

        elif stage_name == "ANALYSIS":
            inputs["execution_output"] = accumulated.get("execution_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})

        elif stage_name == "WRITING":
            inputs["ideation_output"] = accumulated.get("ideation_output", {})
            inputs["experiment_blueprint"] = accumulated.get("experiment_blueprint", {})
            inputs["figure_output"] = accumulated.get("analysis_output", {}).get("figures", {})
            inputs["template_format"] = self.config.template_format
            # Inject real experiment results for writing
            exec_output = accumulated.get("execution_output", {})
            analysis = accumulated.get("analysis_output", {}).get("analysis", {})
            inputs["experiment_results"] = analysis if analysis else exec_output.get("parsed_metrics", {})
            inputs["experiment_status"] = exec_output.get("final_status", "pending")

        if last_error:
            inputs["_retry_error"] = last_error

        return inputs

    def _wrap_stage_output(self, stage_name: str, result: dict) -> dict[str, Any]:
        """Wrap agent output with a stage-specific key."""
        key_map = {
            "IDEATION": "ideation_output",
            "PLANNING": "experiment_blueprint",
            "SETUP": "setup_output",
            "CODING": "coding_output",
            "EXECUTION": "execution_output",
            "ANALYSIS": "analysis_output",
            "WRITING": "writing_output",
        }
        key = key_map.get(stage_name, stage_name.lower())
        return {key: result}

    def _load_stage_output(self, stage_name: str) -> dict[str, Any]:
        """Load previously saved output for resume."""
        file_map = {
            "IDEATION": "papers/ideation_output.json",
            "PLANNING": "plans/experiment_blueprint.json",
            "SETUP": "plans/setup_output.json",
            "CODING": "plans/coding_output.json",
            "EXECUTION": "plans/execution_output.json",
            "ANALYSIS": "plans/analysis_output.json",
        }
        path = file_map.get(stage_name)
        if path:
            try:
                data = self.workspace.read_json(path)
                key = self._wrap_stage_output(stage_name, {})
                actual_key = list(key.keys())[0]
                return {actual_key: data}
            except FileNotFoundError:
                pass
        return {}

    # ---- Stage tracking helpers (use manifest's stages dict directly) ----

    def _mark_stage_running(self, stage_name: str) -> None:
        from datetime import datetime, timezone
        from nanoresearch.schemas.manifest import StageRecord
        m = self.workspace.manifest
        if stage_name not in m.stages:
            # Create stage record for new deep stages
            m.stages[stage_name] = StageRecord(
                stage=PipelineStage.EXPERIMENT,  # placeholder
                status="running",
            )
        else:
            m.stages[stage_name].status = "running"
        m.stages[stage_name].started_at = datetime.now(timezone.utc)
        self.workspace._write_manifest(m)

    def _mark_stage_completed(self, stage_name: str) -> None:
        from datetime import datetime, timezone
        m = self.workspace.manifest
        if stage_name in m.stages:
            m.stages[stage_name].status = "completed"
            m.stages[stage_name].completed_at = datetime.now(timezone.utc)
        self.workspace._write_manifest(m)

    def _mark_stage_failed(self, stage_name: str, error: str) -> None:
        from datetime import datetime, timezone
        m = self.workspace.manifest
        if stage_name in m.stages:
            m.stages[stage_name].status = "failed"
            m.stages[stage_name].error_message = error
        self.workspace._write_manifest(m)
