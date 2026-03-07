"""Tests for shared runtime environment preparation."""

from __future__ import annotations

import subprocess
from pathlib import Path
import shutil
import uuid

import pytest

from nanoresearch.agents.runtime_env import RuntimeEnvironmentManager
from nanoresearch.config import ResearchConfig


@pytest.mark.asyncio
async def test_prepare_system_python_still_installs_dependencies() -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")

        config = ResearchConfig(
            base_url="http://localhost:8000/v1/",
            api_key="test-key",
            auto_create_env=False,
        )
        manager = RuntimeEnvironmentManager(config)

        async def fake_install(python: str, target_dir: Path) -> dict:
            assert python
            assert target_dir == code_dir
            return {"status": "installed", "source": "requirements.txt"}

        manager.install_requirements = fake_install  # type: ignore[method-assign]

        result = await manager.prepare(code_dir)

        assert result["kind"] == "system"
        assert result["dependency_install"] == {
            "status": "installed",
            "source": "requirements.txt",
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_install_requirements_falls_back_to_environment_yml(monkeypatch) -> None:
    tmp_dir = Path(f".test_runtime_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        (code_dir / "environment.yml").write_text(
            "\n".join(
                [
                    "name: demo",
                    "channels:",
                    "  - conda-forge",
                    "dependencies:",
                    "  - python=3.10",
                    "  - pip",
                    "  - pip:",
                    "      - torch",
                    "      - pandas>=2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        commands: list[list[str]] = []

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command: list[str], **_kwargs) -> Completed:
            commands.append(command)
            return Completed()

        monkeypatch.setattr(subprocess, "run", fake_run)

        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        manager = RuntimeEnvironmentManager(config)
        result = await manager.install_requirements("python-custom", code_dir)

        assert result == {"status": "installed", "source": "environment.yml"}
        assert commands
        assert commands[0][:4] == ["python-custom", "-m", "pip", "install"]
        assert "torch" in commands[0]
        assert "pandas>=2" in commands[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
