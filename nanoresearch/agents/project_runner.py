"""Utilities for deterministic experiment runner assets."""

from __future__ import annotations

import json
import platform
import re
import shlex
from pathlib import Path
from typing import Any


RUNNER_SCRIPT_NAME = "nanoresearch_runner.py"
RUNNER_CONFIG_NAME = "nanoresearch_runner.json"


def is_python_launcher_token(token: str) -> bool:
    """Return True when a command token refers to a Python launcher."""
    normalized = str(token or "").strip().strip("\"'")
    if not normalized:
        return False
    name = Path(normalized).name.lower()
    return bool(re.fullmatch(r"(python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?", name))


def normalize_target_command(train_command: str, code_dir: Path) -> list[str]:
    """Normalize a model-generated train command into runner target tokens."""
    command = str(train_command or "").strip()
    tokens: list[str] = []
    if command:
        try:
            tokens = shlex.split(command, posix=platform.system() != "Windows")
        except ValueError:
            tokens = command.split()

    if tokens and is_python_launcher_token(tokens[0]):
        tokens = tokens[1:]

    normalized = [token for token in tokens if token not in {"--dry-run", "--quick-eval"}]
    if normalized:
        return normalized

    for candidate in ("main.py", "train.py", "run.py"):
        if (code_dir / candidate).exists():
            return [candidate]
    return ["main.py"]


def ensure_project_runner(code_dir: Path, train_command: str) -> dict[str, Any]:
    """Write deterministic runner assets for a generated experiment project."""
    target_command = normalize_target_command(train_command, code_dir)
    runner_script = code_dir / RUNNER_SCRIPT_NAME
    runner_config = code_dir / RUNNER_CONFIG_NAME
    runner_config.write_text(
        json.dumps({"target_command": target_command}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    runner_script.write_text(_build_runner_script(), encoding="utf-8")
    return {
        "runner_script": str(runner_script),
        "runner_config": str(runner_config),
        "runner_command": f"python {RUNNER_SCRIPT_NAME}",
        "target_command": target_command,
    }


def _build_runner_script() -> str:
    return r'''"""Deterministic wrapper around model-generated training entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "nanoresearch_runner.json"


def _load_target_command() -> list[str]:
    if not CONFIG_PATH.exists():
        return ["main.py"]
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["main.py"]
    target = data.get("target_command")
    if isinstance(target, list) and all(isinstance(token, str) and token.strip() for token in target):
        return target
    return ["main.py"]


def _entry_script_path(tokens: list[str]) -> Path | None:
    if not tokens:
        return None
    for token in tokens:
        if token in {"-m", "-c"}:
            return None
        if token.endswith(".py"):
            candidate = Path(token)
            return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return None


def _is_python_launcher(token: str) -> bool:
    normalized = str(token or "").strip().strip("'").strip('"')
    if not normalized:
        return False
    name = Path(normalized).name.lower()
    return bool(re.fullmatch(r"(python(?:\d+(?:\.\d+)*)?|py)(?:\.exe)?", name))


def _supports_flag(entry_script: Path | None, flag: str) -> bool:
    if not entry_script or not entry_script.exists():
        return False
    try:
        content = entry_script.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    normalized_flag = flag.lstrip("-").replace("-", "_")
    return flag in content or normalized_flag in content


def _token_present(tokens: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in tokens)


def _build_quick_eval_tokens(
    target_tokens: list[str],
    entry_script: Path | None,
    passthrough: list[str],
) -> list[str]:
    tokens = list(target_tokens)
    if _supports_flag(entry_script, "--quick-eval") and not _token_present(tokens, "--quick-eval"):
        tokens.append("--quick-eval")

    speedups = [
        ("--epochs", "1"),
        ("--num-epochs", "1"),
        ("--max-steps", "2"),
        ("--steps", "2"),
        ("--batch-size", "8"),
        ("--batch_size", "8"),
        ("--num-workers", "0"),
        ("--num_workers", "0"),
        ("--workers", "0"),
        ("--subset-size", "64"),
        ("--subset_size", "64"),
        ("--train-size", "64"),
        ("--quick-eval-train-size", "64"),
        ("--limit-train-batches", "2"),
        ("--limit-val-batches", "1"),
    ]
    for option, value in speedups:
        if (
            _supports_flag(entry_script, option)
            and not _token_present(tokens, option)
            and not _token_present(passthrough, option)
        ):
            tokens.extend([option, value])
    return [*tokens, *passthrough]


def _run_target(tokens: list[str], mode: str) -> int:
    env = {**os.environ}
    env["NANORESEARCH_EXECUTION_MODE"] = mode
    if mode == "quick-eval":
        env["NANORESEARCH_QUICK_EVAL"] = "1"
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = _materialize_command(tokens)
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    return int(completed.returncode or 0)


def _materialize_command(tokens: list[str]) -> list[str]:
    if not tokens:
        return [sys.executable, "main.py"]
    first = tokens[0]
    if _is_python_launcher(first):
        return [sys.executable, *tokens[1:]]
    if first in {"-m", "-c"} or first.endswith(".py"):
        return [sys.executable, *tokens]
    return list(tokens)


def _compile_project() -> None:
    for py_file in PROJECT_ROOT.rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in py_file.parts):
            continue
        py_compile.compile(str(py_file), doraise=True)


def _ensure_output_dirs() -> None:
    for dirname in ("results", "checkpoints", "logs"):
        (PROJECT_ROOT / dirname).mkdir(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoResearch deterministic runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick-eval", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.dry_run and args.quick_eval:
        print("Only one of --dry-run/--quick-eval may be used.", file=sys.stderr)
        return 2

    _ensure_output_dirs()
    target_tokens = _load_target_command()
    entry_script = _entry_script_path(target_tokens)

    if args.dry_run:
        if _supports_flag(entry_script, "--dry-run"):
            return _run_target([*target_tokens, "--dry-run", *passthrough], "dry-run")
        _compile_project()
        print("NanoResearch runner fallback dry-run: syntax check passed.")
        return 0

    if args.quick_eval:
        quick_tokens = _build_quick_eval_tokens(target_tokens, entry_script, passthrough)
        return _run_target(quick_tokens, "quick-eval")

    return _run_target([*target_tokens, *passthrough], "train")


if __name__ == "__main__":
    raise SystemExit(main())
'''
