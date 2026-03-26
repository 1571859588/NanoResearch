"""Code execution: cluster, local subprocess, batch fix, env setup."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re as _re
import subprocess
import sys
from pathlib import Path
from typing import Any

from nanoresearch.agents.cluster_executor import ClusterExecutor
from nanoresearch.agents.experiment._code_runner_helpers import _CodeRunnerHelpersMixin

from . import (
    _decode_bytes,
    DRY_RUN_TIMEOUT_SECONDS,
    SUBPROCESS_OUTPUT_LIMIT,
    STDERR_SNIPPET_LIMIT,
    LLM_CONTEXT_TRUNCATION,
)

logger = logging.getLogger(__name__)


class _CodeRunnerMixin(_CodeRunnerHelpersMixin):
    """Mixin — code execution, batch fix, environment setup."""

    async def _run_on_cluster(
        self,
        cluster: "ClusterExecutor",
        code_dir: Path,
        round_num: int,
        cluster_code_path: str,
    ) -> tuple[dict, dict]:
        """Run experiment on SLURM cluster (local or remote).

        Returns (execution_dict, quick_eval_dict) in the same format as
        the local execution path.
        """
        session_id = self.workspace.path.name

        try:
            runner_command = self._build_legacy_runner_command(
                code_dir,
                mode="quick-eval",
            )
            if runner_command is None:
                return (
                    {
                        "status": "skipped",
                        "cluster_code_path": cluster_code_path,
                        "stderr": "No runnable entry script found (expected one of main.py/train.py/run.py)",
                    },
                    {"status": "skipped", "metrics": {}},
                )

            # Step 1: Prepare code on cluster
            if not cluster_code_path:
                self.log("Preparing code on cluster...")
                cluster_code_path = await cluster.prepare_code(code_dir, session_id)

                # Step 2: Create conda env + install deps (first round only)
                env_result = await cluster.setup_env(cluster_code_path)
                if not env_result["ok"]:
                    return (
                        {
                            "status": "failed",
                            "cluster_code_path": cluster_code_path,
                            "stderr": f"Environment setup failed:\n{env_result['output'][-2000:]}",
                        },
                        {"status": "skipped", "metrics": {}},
                    )
            else:
                # Re-sync code after LLM modifications
                self.log("Re-syncing code to cluster...")
                await cluster.reupload_code(code_dir, cluster_code_path)

            # Step 3: Submit SLURM job
            script_cmd = runner_command
            job_id = await cluster.submit_job(cluster_code_path, script_cmd)

            # Step 4: Wait for completion
            job_status = await cluster.wait_for_job(job_id)
            state = job_status.get("state", "UNKNOWN")

            # Step 5: Collect results or error logs
            if state == "COMPLETED":
                downloaded = await cluster.download_results(
                    cluster_code_path, self.workspace.path
                )
                if downloaded:
                    metrics = self._parse_metrics_json(code_dir)
                    if metrics:
                        self.log("Cluster experiment succeeded — real results collected!")
                        return (
                            {
                                "status": "success",
                                "cluster_code_path": cluster_code_path,
                                "job_id": job_id,
                                "stdout": f"Job {job_id} completed",
                                "stderr": "",
                            },
                            {"status": "success", "metrics": metrics},
                        )

                # Job completed but metrics.json missing/invalid
                log_text = await cluster.get_job_log(cluster_code_path, job_id)
                return (
                    {
                        "status": "failed",
                        "cluster_code_path": cluster_code_path,
                        "job_id": job_id,
                        "stdout": f"Job {job_id} rc=0 but metrics.json missing/invalid",
                        "stderr": log_text[-STDERR_SNIPPET_LIMIT:] if log_text else "",
                    },
                    {"status": "partial", "metrics": {}, "stderr": log_text},
                )
            else:
                # Job failed
                log_text = await cluster.get_job_log(cluster_code_path, job_id)
                self.log(f"Cluster job {job_id} failed ({state})")
                return (
                    {
                        "status": "failed",
                        "cluster_code_path": cluster_code_path,
                        "job_id": job_id,
                        "stdout": "",
                        "stderr": log_text[-STDERR_SNIPPET_LIMIT:] if log_text else f"Job {state}",
                    },
                    {"status": "failed", "metrics": {}, "stderr": log_text},
                )

        except Exception as e:
            self.log(f"Cluster execution error: {e}")
            return (
                {
                    "status": "failed",
                    "cluster_code_path": cluster_code_path,
                    "stderr": str(e),
                },
                {"status": "failed", "metrics": {}},
            )

    async def _execute_code_with_venv(
        self, generated_files: list[str], blueprint_summary: str
    ) -> tuple[dict, str]:
        """Run _execute_code and also return the venv python path for reuse."""
        code_dir = self.workspace.path / "code"
        entry_script = self._find_legacy_entry_script(code_dir)
        if entry_script is None:
            return (
                {
                    "status": "skipped",
                    "reason": "No runnable entry script found (expected one of main.py/train.py/run.py)",
                    "stdout": "",
                    "stderr": "",
                },
                "",  # no python path — caller must call _setup_venv before using
            )

        venv_python = await self._setup_venv(code_dir)
        result = await self._execute_code(
            generated_files, blueprint_summary,
            _code_dir=code_dir, _main_py=entry_script, _venv_python=venv_python,
        )
        return result, venv_python

    async def _execute_code(
        self,
        generated_files: list[str],
        blueprint_summary: str,
        *,
        _code_dir: Path | None = None,
        _main_py: Path | None = None,
        _venv_python: str | None = None,
    ) -> dict:
        """Execute main.py --dry-run with up to 5 batch-fix cycles.

        Each cycle: run -> collect all errors -> fix ALL affected files in one
        LLM call -> run again.  This is much more efficient than fixing one
        bug at a time.
        """
        code_dir = _code_dir or (self.workspace.path / "code")
        main_py = _main_py or self._find_legacy_entry_script(code_dir)

        if main_py is None or not main_py.exists():
            return {
                "status": "skipped",
                "reason": "No runnable entry script found (expected one of main.py/train.py/run.py)",
                "stdout": "",
                "stderr": "",
            }

        venv_python = _venv_python or await self._setup_venv(code_dir)

        max_fix_cycles = 5
        last_result: dict = {}
        fix_history: list[dict] = []  # Track previous fixes to avoid repeating

        for cycle in range(1, max_fix_cycles + 1):
            result = await self._run_main_py(code_dir, venv_python)
            last_result = result
            if result["returncode"] == 0:
                status = "success" if cycle == 1 else "fixed"
                return {"status": status, "attempts": cycle, **result}

            self.log(f"Code execution failed (attempt {cycle}): {result['stderr'][:200]}")

            if cycle >= max_fix_cycles:
                break

            # Batch fix: identify ALL affected files and fix them in one call
            stderr_text = result["stderr"]
            try:
                modified = await self._batch_fix_errors(
                    code_dir, stderr_text, blueprint_summary,
                    mode="dry-run",
                    previous_fixes=fix_history,
                )
                fix_history.append({"error_msg": stderr_text[:300], "cycle": cycle})
                if not modified:
                    self.log("Dry-run: no files modified by batch fix, stopping")
                    break
            except Exception as e:
                self.log(f"Batch fix error in cycle {cycle}: {e}")
                break

        return {"status": "failed", "attempts": cycle, **last_result}

    async def _batch_fix_errors(
        self,
        code_dir: Path,
        stderr: str,
        blueprint_summary: str,
        mode: str = "dry-run",
        previous_fixes: list[dict] | None = None,
        extra_context: str = "",
    ) -> list[str]:
        """Parse traceback, fix each affected file with a targeted LLM call.

        Surgical approach: for each file in the traceback, send ONLY that file
        + the error to the LLM -> get a search-replace patch -> apply.
        Uses 4-layer patch matching and syntax validation with rollback.

        Returns list of modified file paths.
        """
        import re as _re

        import asyncio
        import shlex

        abs_code_dir = code_dir.resolve()
        
        # Parse traceback to find affected files
        code_dir_str = str(abs_code_dir).replace("\\", "/")
        tb_entries = _re.findall(r'File "([^"]+)",\s*line\s+(\d+)', stderr)
        affected: list[tuple[Path, int]] = []
        seen_files: set[str] = set()
        for fpath, lineno in reversed(tb_entries):
            resolved = Path(fpath).resolve()
            resolved_norm = str(resolved).replace("\\", "/")
            if code_dir_str not in resolved_norm:
                continue
            try:
                rel = str(resolved.relative_to(abs_code_dir)).replace("\\", "/")
            except ValueError:
                continue
            if rel not in seen_files and resolved.exists():
                affected.append((resolved, int(lineno)))
                seen_files.add(rel)

        # Build previous fix history
        fix_history = ""
        if previous_fixes:
            fix_history = (
                "\n\nPrevious fix attempts that did NOT resolve the problem:\n"
                + "\n".join(
                    (
                        f"  Round {i+1}: "
                        f"{fx.get('diagnosis', fx.get('error_msg', ''))[:200]}"
                    )
                    for i, fx in enumerate(previous_fixes)
                )
                + "\nDo NOT repeat the same fixes. Try a different approach.\n"
            )

        extra_context_text = (
            f"Additional execution context:\n{extra_context}\n\n"
            if extra_context.strip()
            else ""
        )
        
        files_to_check = ", ".join([str(p.name) for p, _ in affected]) if affected else "the project files"
        
        prompt = (
            f"Fix the python execution crash in: {files_to_check}. "
            f"Important: Do not read logs or massive files, just focus on the code. Do not explain, just make the edits and exit.\n\n"
            f"Error details:\n"
            f"{stderr[-2000:]}\n"
            f"{extra_context_text}"
            f"{fix_history}"
        )
        
        escaped_prompt = shlex.quote(prompt)
        
        # Defensively protect Claude Code's context window from massive logs/outputs
        try:
            ignore_file = abs_code_dir / ".gitignore"
            ignores = {"logs/", "results/", "checkpoints/", "__pycache__/", "datasets/", "models/", "outputs/", "figures/"}
            existing = set()
            if ignore_file.exists():
                existing = set(ignore_file.read_text(encoding="utf-8").splitlines())
            missing = ignores - existing
            if missing:
                with ignore_file.open("a", encoding="utf-8") as f:
                    f.write("\n" + "\n".join(missing) + "\n")
        except OSError:
            pass

        worker_cmd = f"cd {abs_code_dir} && ccr restart && ccr code -p --permission-mode acceptEdits --dangerously-skip-permissions {escaped_prompt}"
        
        self.log(f"Delegating repair to Claude Code via: su - nyt_worker")
        
        try:
            proc = await asyncio.create_subprocess_exec(
                "su", "-", "nyt_worker", "-c", worker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Since Claude Code might take a while, wait up to 10 minutes
            stdout, stderr_out = await asyncio.wait_for(proc.communicate(), timeout=600)
            
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr_out.decode('utf-8', errors='replace')
            
            if proc.returncode == 0:
                self.log(f"Claude Code auto-fix completed successfully.\nSTDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}")
                return ["auto-fixed-by-ccr"]
            else:
                self.log(f"Claude Code returned non-zero (return_code={proc.returncode}):\nSTDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}")
                # Sometimes it makes changes but exits with non-zero.
                if "edited" in stdout_text.lower() or "saved" in stdout_text.lower():
                    self.log("Claude Code returned non-zero but possibly made edits. Continuing.")
                    return ["auto-fixed-by-ccr-partial"]
                return []
                
        except asyncio.TimeoutError:
            self.log("Claude Code execution timed out after 10 minutes.")
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return []
        except Exception as e:
            self.log(f"Failed to execute Claude Code repair: {e}")
            return []
