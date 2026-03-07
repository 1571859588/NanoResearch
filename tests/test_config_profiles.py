"""Tests for unified pipeline profile configuration."""

from __future__ import annotations

from nanoresearch.config import ExecutionProfile, ResearchConfig, WritingMode


def test_snapshot_serializes_profile_fields() -> None:
    config = ResearchConfig(
        base_url="http://localhost:8000/v1/",
        api_key="test-key",
        execution_profile=ExecutionProfile.CLUSTER_FULL,
        writing_mode=WritingMode.REACT,
    )

    snapshot = config.snapshot()

    assert snapshot["execution_profile"] == "cluster_full"
    assert snapshot["writing_mode"] == "react"
    assert "api_key" not in snapshot


def test_prefers_cluster_execution_respects_profile_and_cluster_flag() -> None:
    base = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
    assert base.prefers_cluster_execution() is False

    cluster_profile = ResearchConfig(
        base_url="http://localhost:8000/v1/",
        api_key="test-key",
        execution_profile=ExecutionProfile.CLUSTER_FULL,
    )
    assert cluster_profile.prefers_cluster_execution() is True

    cluster_flag = ResearchConfig(
        base_url="http://localhost:8000/v1/",
        api_key="test-key",
        cluster={"enabled": True},
    )
    assert cluster_flag.prefers_cluster_execution() is True


def test_should_use_writing_tools_depends_on_mode_and_profile() -> None:
    default = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
    assert default.should_use_writing_tools("Results") is True

    fast_draft = ResearchConfig(
        base_url="http://localhost:8000/v1/",
        api_key="test-key",
        execution_profile=ExecutionProfile.FAST_DRAFT,
    )
    assert fast_draft.should_use_writing_tools("Results") is False
    assert fast_draft.should_use_writing_tools("Introduction") is True

    direct = ResearchConfig(
        base_url="http://localhost:8000/v1/",
        api_key="test-key",
        writing_mode=WritingMode.DIRECT,
    )
    assert direct.should_use_writing_tools("Introduction") is False

    react = ResearchConfig(
        base_url="http://localhost:8000/v1/",
        api_key="test-key",
        writing_mode=WritingMode.REACT,
    )
    assert react.should_use_writing_tools("Conclusion") is True
