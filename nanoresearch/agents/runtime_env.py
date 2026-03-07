"""Shared runtime environment helpers for experiment execution."""

from __future__ import annotations

import asyncio
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Callable

from nanoresearch.config import ResearchConfig


class RuntimeEnvironmentManager:
    """Prepare Python runtimes for local experiment execution."""

    def __init__(
        self,
        config: ResearchConfig,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._log = log_fn or (lambda _message: None)

    async def prepare(self, code_dir: Path) -> dict[str, Any]:
        requirements_path = code_dir / "requirements.txt"
        environment_file = code_dir / "environment.yml"
        conda_env = self.config.experiment_conda_env.strip()
        if conda_env:
            conda_python = self.find_conda_python(conda_env)
            if conda_python:
                self._log(f"Using existing conda env '{conda_env}': {conda_python}")
                install_info = await self.install_requirements(conda_python, code_dir)
                return {
                    "kind": "conda",
                    "python": conda_python,
                    "env_name": conda_env,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file.exists() else "",
                    "dependency_install": install_info,
                }
            if self.config.auto_create_env and shutil.which("conda"):
                created = await self.create_conda_env(conda_env, code_dir)
                if created:
                    conda_python = self.find_conda_python(conda_env)
                    if conda_python:
                        install_info = await self.install_requirements(conda_python, code_dir)
                        return {
                            "kind": "conda",
                            "python": conda_python,
                            "env_name": conda_env,
                            "created": True,
                            "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                            "environment_file": str(environment_file) if environment_file.exists() else "",
                            "dependency_install": install_info,
                        }
            self._log(f"Conda env '{conda_env}' not found, falling back to venv")

        if not self.config.auto_create_env:
            self._log("Automatic env creation disabled, using system Python")
            install_info = await self.install_requirements(sys.executable, code_dir)
            return {
                "kind": "system",
                "python": sys.executable,
                "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                "environment_file": str(environment_file) if environment_file.exists() else "",
                "dependency_install": install_info,
            }

        venv_dir = code_dir / ".venv"
        is_windows = platform.system() == "Windows"
        python_path = venv_dir / ("Scripts/python.exe" if is_windows else "bin/python")
        created = False

        if not python_path.exists():
            self._log(f"Creating isolated venv at {venv_dir} ...")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: venv.create(str(venv_dir), with_pip=True),
                )
                created = True
                self._log(f"Venv created (python: {python_path})")
            except (OSError, subprocess.CalledProcessError) as exc:
                self._log(f"Venv creation failed: {exc}, falling back to system Python")
                install_info = await self.install_requirements(sys.executable, code_dir)
                return {
                    "kind": "system",
                    "python": sys.executable,
                    "requirements_path": str(requirements_path) if requirements_path.exists() else "",
                    "environment_file": str(environment_file) if environment_file.exists() else "",
                    "dependency_install": install_info,
                }
        else:
            self._log(f"Reusing existing venv at {venv_dir}")

        install_info = await self.install_requirements(str(python_path), code_dir)
        return {
            "kind": "venv",
            "python": str(python_path),
            "env_path": str(venv_dir),
            "created": created,
            "requirements_path": str(requirements_path) if requirements_path.exists() else "",
            "environment_file": str(environment_file) if environment_file.exists() else "",
            "dependency_install": install_info,
        }

    @staticmethod
    def find_conda_python(env_name: str) -> str | None:
        """Find the Python executable for a named conda env."""
        try:
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                path = result.stdout.strip().split("\n")[-1].strip()
                if path and Path(path).exists():
                    return path
        except Exception:
            pass

        is_windows = platform.system() == "Windows"
        for base in [
            Path.home() / "anaconda3",
            Path.home() / "miniconda3",
            Path("D:/anaconda"),
            Path("C:/anaconda3"),
        ]:
            python_path = (
                base / "envs" / env_name / ("python.exe" if is_windows else "bin/python")
            )
            if python_path.exists():
                return str(python_path)
        return None

    async def install_requirements(self, python: str, code_dir: Path) -> dict[str, Any]:
        """Install requirements from requirements.txt or environment.yml."""
        requirements_path = code_dir / "requirements.txt"
        environment_file = code_dir / "environment.yml"
        install_args: list[str]
        source = ""
        if requirements_path.exists():
            install_args = ["-r", str(requirements_path)]
            source = "requirements.txt"
        else:
            pip_dependencies = self._extract_pip_dependencies(environment_file)
            if not pip_dependencies:
                self._log("No requirements.txt or pip dependencies in environment.yml, skipping pip install")
                return {"status": "skipped", "source": ""}
            install_args = pip_dependencies
            source = "environment.yml"

        self._log(f"Installing dependencies from {source} ...")
        loop = asyncio.get_running_loop()
        try:
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [python, "-m", "pip", "install", *install_args, "--quiet"],
                    cwd=str(code_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,
                ),
            )
            if proc_result.returncode == 0:
                self._log(f"Dependency install OK via {source}")
                return {"status": "installed", "source": source}
            else:
                stderr = (proc_result.stderr or "").strip()
                self._log(f"pip install returned rc={proc_result.returncode}: {stderr[:500]}")
                return {
                    "status": "failed",
                    "source": source,
                    "returncode": proc_result.returncode,
                    "stderr": stderr[:500],
                }
        except Exception as exc:
            self._log(f"pip install error: {exc}")
            return {"status": "error", "source": source, "error": str(exc)}

    async def create_conda_env(self, env_name: str, code_dir: Path) -> bool:
        """Create a conda environment when requested and missing."""
        env_file = code_dir / "environment.yml"
        self._log(f"Creating conda env '{env_name}' ...")
        loop = asyncio.get_running_loop()
        try:
            if env_file.exists():
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["conda", "env", "create", "-n", env_name, "-f", str(env_file)],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=1800,
                    ),
                )
            else:
                proc_result = await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["conda", "create", "-y", "-n", env_name, "python=3.10"],
                        cwd=str(code_dir),
                        capture_output=True,
                        text=True,
                        timeout=1200,
                    ),
                )
            if proc_result.returncode == 0:
                self._log(f"Conda env '{env_name}' created")
                return True
            stderr = (proc_result.stderr or "").strip()
            self._log(f"Failed to create conda env '{env_name}': {stderr[:500]}")
        except Exception as exc:
            self._log(f"Conda env creation error: {exc}")
        return False

    @staticmethod
    def _extract_pip_dependencies(environment_file: Path) -> list[str]:
        """Extract pip-installable dependencies from environment.yml."""
        if not environment_file.exists():
            return []

        try:
            lines = environment_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        dependencies: list[str] = []
        in_pip_block = False
        pip_indent = 0
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(raw_line) - len(raw_line.lstrip())
            if stripped in {"- pip:", "pip:"}:
                in_pip_block = True
                pip_indent = indent
                continue

            if in_pip_block:
                if indent <= pip_indent:
                    in_pip_block = False
                elif stripped.startswith("- "):
                    dependency = stripped[2:].strip()
                    if dependency:
                        dependencies.append(dependency)
                    continue

            if not in_pip_block and stripped.startswith("- "):
                dependency = stripped[2:].strip()
                if dependency and not dependency.startswith(("python", "pip")) and "=" not in dependency:
                    dependencies.append(dependency)

        return dependencies
