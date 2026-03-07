"""Tests for analysis output grounding."""

from __future__ import annotations

from nanoresearch.agents.analysis import AnalysisAgent


def test_render_experiment_summary_markdown_falls_back_to_execution_metrics() -> None:
    summary = AnalysisAgent._render_experiment_summary_markdown(
        {"summary": "Quick evaluation completed."},
        {
            "final_status": "COMPLETED",
            "metrics": {"accuracy": 0.91, "loss": 0.12},
            "parsed_metrics": {"ignored_nested": {"x": 1}},
        },
        {
            "proposed_method": {"name": "DeepMethod"},
            "datasets": [{"name": "DemoSet"}],
        },
    )

    assert "Quick evaluation completed." in summary
    assert "## Final Metrics" in summary
    assert "`accuracy`: 0.91" in summary
    assert "`loss`: 0.12" in summary


def test_render_experiment_summary_includes_comparison_with_baselines() -> None:
    summary = AnalysisAgent._render_experiment_summary_markdown(
        {
            "summary": "Model converged.",
            "final_metrics": {"accuracy": 0.93},
            "comparison_with_baselines": {
                "our_method": {"accuracy": 0.93},
                "BaselineA": {"accuracy": 0.88},
                "BaselineB": {"accuracy": 0.90},
            },
        },
        {"final_status": "COMPLETED"},
        {"proposed_method": {"name": "DeepMethod"}, "datasets": [{"name": "DS"}]},
    )
    assert "## Comparison with Baselines" in summary
    assert "BaselineA" in summary
    assert "BaselineB" in summary
    assert "0.88" in summary
    assert "0.9" in summary


def test_render_experiment_summary_includes_ablation_results() -> None:
    summary = AnalysisAgent._render_experiment_summary_markdown(
        {
            "summary": "OK",
            "ablation_results": [
                {"variant_name": "Full model", "metrics": [{"metric_name": "acc", "value": 0.93}]},
                {"variant_name": "w/o Attn", "metrics": [{"metric_name": "acc", "value": 0.87}]},
            ],
        },
        {"final_status": "COMPLETED"},
        {"proposed_method": {"name": "M"}, "datasets": []},
    )
    assert "## Ablation Results" in summary
    assert "Full model" in summary
    assert "w/o Attn" in summary
