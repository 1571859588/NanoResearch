"""Tests for deep-pipeline manifest and orchestrator consistency."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.deep_orchestrator import DeepPipelineOrchestrator
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import (
    PipelineMode,
    PipelineStage,
    StageRecord,
    WorkspaceManifest,
)


class MemoryWorkspace:
    """Minimal in-memory workspace stub for orchestrator unit tests."""

    def __init__(self) -> None:
        relevant_stages = [
            PipelineStage.INIT,
            *PipelineStateMachine.processing_stages(PipelineMode.DEEP),
        ]
        self.path = Path(".")
        self._manifest = WorkspaceManifest(
            session_id="deep001",
            topic="deep topic",
            pipeline_mode=PipelineMode.DEEP,
            current_stage=PipelineStage.INIT,
            stages={
                stage.value: StageRecord(stage=stage)
                for stage in relevant_stages
            },
        )

    @property
    def manifest(self) -> WorkspaceManifest:
        return self._manifest

    def _write_manifest(self, manifest: WorkspaceManifest) -> None:
        self._manifest = manifest

    def update_manifest(self, **kwargs) -> WorkspaceManifest:
        for key, value in kwargs.items():
            setattr(self._manifest, key, value)
        return self._manifest

    def mark_stage_running(self, stage: PipelineStage) -> None:
        record = self._manifest.stages[stage.value]
        record.stage = stage
        record.status = "running"
        self._manifest.current_stage = stage

    def mark_stage_completed(self, stage: PipelineStage, output_path: str = "") -> None:
        record = self._manifest.stages[stage.value]
        record.stage = stage
        record.status = "completed"
        record.output_path = output_path

    def mark_stage_failed(self, stage: PipelineStage, error: str) -> None:
        record = self._manifest.stages[stage.value]
        record.stage = stage
        record.status = "failed"
        record.error_message = error
        self._manifest.current_stage = PipelineStage.FAILED

    def increment_retry(self, stage: PipelineStage) -> int:
        record = self._manifest.stages[stage.value]
        record.retries += 1
        record.status = "pending"
        return record.retries

    def write_text(self, *_args, **_kwargs):
        return None

    def read_json(self, *_args, **_kwargs):
        raise FileNotFoundError

    def export(self) -> Path:
        return self.path


def test_normalize_manifest_data_repairs_legacy_deep_stage_records():
    data = {
        "session_id": "deep002",
        "topic": "legacy topic",
        "current_stage": "SETUP",
        "stages": {
            "SETUP": {
                "stage": "EXPERIMENT",
                "status": "failed",
                "started_at": None,
                "completed_at": None,
                "retries": 0,
                "error_message": "",
                "output_path": "",
            },
        },
    }

    normalized, changed = Workspace._normalize_manifest_data(data)

    assert changed is True
    assert normalized["pipeline_mode"] == PipelineMode.DEEP.value
    assert normalized["stages"]["SETUP"]["stage"] == "SETUP"


@pytest.mark.asyncio
async def test_deep_orchestrator_marks_real_stage_names():
    workspace = MemoryWorkspace()
    config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
    orchestrator = DeepPipelineOrchestrator(workspace, config)

    try:
        with patch.object(
            orchestrator._agents[PipelineStage.SETUP],
            "run",
            AsyncMock(return_value={"downloaded_resources": []}),
        ):
            result = await orchestrator._run_stage_with_retry(
                PipelineStage.SETUP,
                "deep topic",
                {
                    "ideation_output": {},
                    "experiment_blueprint": {},
                },
            )

        stage_record = workspace.manifest.stages["SETUP"]
        assert stage_record.stage == PipelineStage.SETUP
        assert stage_record.status == "completed"
        assert result == {"setup_output": {"downloaded_resources": []}}
    finally:
        await orchestrator.close()


@pytest.mark.asyncio
async def test_prepare_inputs_routes_real_results_and_summaries():
    workspace = MemoryWorkspace()
    config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
    orchestrator = DeepPipelineOrchestrator(workspace, config)

    accumulated = {
        "ideation_output": {"topic": "deep topic"},
        "experiment_blueprint": {"title": "Deep Research"},
        "setup_output": {"data_dir": "data/demo"},
        "coding_output": {"train_command": "python train.py"},
        "execution_output": {
            "metrics": {
                "main_results": [
                    {
                        "method_name": "DeepMethod",
                        "dataset": "DemoSet",
                        "is_proposed": True,
                        "metrics": [{"metric_name": "accuracy", "value": 0.93}],
                    }
                ],
                "ablation_results": [],
                "training_log": [],
            },
            "experiment_summary": "Local loop finished successfully.",
            "final_status": "COMPLETED",
        },
        "analysis_output": {
            "analysis": {
                "summary": "DeepMethod outperforms the baseline.",
                "final_metrics": {"accuracy": 0.93},
            },
            "figures": {
                "fig_results": {
                    "png_path": "figures/fig_results.png",
                    "pdf_path": "figures/fig_results.pdf",
                }
            },
            "experiment_summary": "# Experiment Summary\n\nDeepMethod converged cleanly.",
        },
    }

    try:
        execution_inputs = orchestrator._prepare_inputs(
            PipelineStage.EXECUTION,
            "deep topic",
            accumulated,
            "",
        )
        assert execution_inputs["topic"] == "deep topic"
        assert execution_inputs["setup_output"] == accumulated["setup_output"]
        assert execution_inputs["experiment_blueprint"] == accumulated["experiment_blueprint"]

        figure_inputs = orchestrator._prepare_inputs(
            PipelineStage.FIGURE_GEN,
            "deep topic",
            accumulated,
            "",
        )
        assert figure_inputs["experiment_results"] == accumulated["execution_output"]["metrics"]
        assert figure_inputs["experiment_analysis"] == accumulated["analysis_output"]["analysis"]
        assert figure_inputs["experiment_summary"] == accumulated["analysis_output"]["experiment_summary"]
        assert figure_inputs["experiment_status"] == "COMPLETED"

        writing_inputs = orchestrator._prepare_inputs(
            PipelineStage.WRITING,
            "deep topic",
            accumulated,
            "",
        )
        assert writing_inputs["figure_output"] == {"figures": accumulated["analysis_output"]["figures"]}
        assert writing_inputs["experiment_results"] == accumulated["execution_output"]["metrics"]
        assert writing_inputs["experiment_analysis"] == accumulated["analysis_output"]["analysis"]
        assert writing_inputs["experiment_summary"] == accumulated["analysis_output"]["experiment_summary"]
        assert writing_inputs["experiment_status"] == "COMPLETED"
    finally:
        await orchestrator.close()
