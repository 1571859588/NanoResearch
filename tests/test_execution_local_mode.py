"""Tests for local execution helpers in the unified pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock
import uuid

import pytest

from nanoresearch.agents.experiment import ExperimentAgent
from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.agents.project_runner import RUNNER_SCRIPT_NAME, ensure_project_runner
from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.workspace import Workspace


def test_build_local_command_uses_runtime_python() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec001",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "python train.py --epochs 1"},
            "C:/env/python.exe",
        )

        assert command[0] == "C:/env/python.exe"
        assert command[1:] == ["train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_rewrites_explicit_python_path() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_python_path",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "./.venv/Scripts/python.exe train.py --epochs 1"},
            "C:/runtime/python.exe",
        )

        assert command == ["C:/runtime/python.exe", "train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_prefixes_runtime_python_for_bare_script() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_bare_script",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "train.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "train.py --epochs 1"},
            "python-custom",
        )

        assert command == ["python-custom", "train.py", "--epochs", "1"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_falls_back_to_main_py() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec002",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / "main.py").write_text("print('ok')", encoding="utf-8")

        command = agent._build_local_command(code_dir, {}, "python-custom")

        assert command == ["python-custom", "main.py"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_local_command_prefers_deterministic_runner() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec_runner",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)
        (code_dir / RUNNER_SCRIPT_NAME).write_text("print('runner')", encoding="utf-8")
        (code_dir / "train.py").write_text("print('train')", encoding="utf-8")

        command = agent._build_local_command(
            code_dir,
            {"train_command": "python train.py"},
            "python-custom",
        )

        assert command == ["python-custom", RUNNER_SCRIPT_NAME]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_project_runner_executes_non_python_launcher() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        code_dir = tmp_dir / "code"
        code_dir.mkdir()
        invoked = (code_dir / "launcher_invoked.txt").resolve()
        (code_dir / "train.py").write_text("print('train entry')", encoding="utf-8")

        if platform.system() == "Windows":
            launcher = tmp_dir / "launcher.py"
            launcher.write_text(
                "from pathlib import Path\n"
                "import os\n"
                "import sys\n"
                f"Path(r'{invoked}').write_text("
                "f\"{os.environ.get('NANORESEARCH_QUICK_EVAL', '')} {' '.join(sys.argv[1:])}\", "
                "encoding='utf-8')\n",
                encoding="utf-8",
            )
            launcher_command = f"cmd /c {sys.executable} {launcher.resolve()} train.py --quick-eval"
        else:
            launcher = tmp_dir / "launcher.sh"
            launcher.write_text(
                "#!/usr/bin/env sh\n"
                f"printf '%s %s\\n' \"$NANORESEARCH_QUICK_EVAL\" \"$*\" > '{invoked.as_posix()}'\n"
                "exit 0\n",
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            launcher_command = f"sh {launcher.resolve()} train.py --quick-eval"

        ensure_project_runner(code_dir, launcher_command)

        result = subprocess.run(
            [sys.executable, RUNNER_SCRIPT_NAME, "--quick-eval"],
            cwd=str(code_dir),
            capture_output=True,
            text=True,
            env=dict(os.environ),
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        assert invoked.exists()
        logged = invoked.read_text(encoding="utf-8")
        assert "1" in logged
        assert "train.py" in logged
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_collect_local_results_reads_metrics_file() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec003",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        results_dir = code_dir / "results"
        checkpoints_dir = code_dir / "checkpoints"
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        (results_dir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.91}),
            encoding="utf-8",
        )
        (checkpoints_dir / "model.pt").write_text("weights", encoding="utf-8")

        results = agent._collect_local_results(
            code_dir,
            {"stdout": "done", "stderr": "", "returncode": 0},
        )

        assert results["metrics"] == {"accuracy": 0.91}
        assert len(results["checkpoints"]) == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_dry_run_loop_retries_with_batch_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec004",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        code_dir.mkdir(exist_ok=True)

        agent._run_subprocess = AsyncMock(
            side_effect=[
                {"returncode": 1, "stdout": "", "stderr": "ImportError: missing module"},
                {"returncode": 0, "stdout": "dry run ok", "stderr": ""},
            ]
        )
        helper._batch_fix_errors = AsyncMock(return_value=["train.py"])

        result = await agent._run_local_dry_run_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
        )

        assert result["status"] == "fixed"
        assert result["attempts"] == 2
        helper._batch_fix_errors.assert_awaited_once()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_run_local_quick_eval_loop_recovers_after_timeout_fix() -> None:
    tmp_dir = Path(f".test_exec_{uuid.uuid4().hex[:8]}")
    tmp_dir.mkdir()
    try:
        workspace = Workspace.create(
            topic="test",
            root=tmp_dir,
            session_id="exec005",
        )
        config = ResearchConfig(base_url="http://localhost:8000/v1/", api_key="test-key")
        agent = ExecutionAgent(workspace, config)
        helper = ExperimentAgent(workspace, config)

        code_dir = workspace.path / "experiment"
        results_dir = code_dir / "results"
        code_dir.mkdir(exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        async def fake_run_subprocess(*_args, **_kwargs):
            if not hasattr(fake_run_subprocess, "calls"):
                fake_run_subprocess.calls = 0
            fake_run_subprocess.calls += 1
            if fake_run_subprocess.calls == 1:
                return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
            (results_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "main_results": [
                            {
                                "method_name": "DeepMethod",
                                "dataset": "DemoSet",
                                "is_proposed": True,
                                "metrics": [{"metric_name": "accuracy", "value": 0.91}],
                            }
                        ],
                        "ablation_results": [],
                        "training_log": [],
                    }
                ),
                encoding="utf-8",
            )
            return {"returncode": 0, "stdout": "quick eval ok", "stderr": ""}

        agent._run_subprocess = AsyncMock(side_effect=fake_run_subprocess)
        helper._fix_timeout = AsyncMock(return_value=["main.py"])

        result = await agent._run_local_quick_eval_loop(
            code_dir,
            ["python", "train.py"],
            '{"topic": "demo"}',
            helper,
        )

        assert result["status"] == "success"
        assert result["attempts"] == 2
        assert result["metrics"]["main_results"][0]["method_name"] == "DeepMethod"
        helper._fix_timeout.assert_awaited_once()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
