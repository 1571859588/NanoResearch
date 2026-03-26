"""Iteration helpers: checkpoint, hypothesis, changes, history, imports, syntax."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents._code_utils import _strip_code_fences
from nanoresearch.agents.repair_journal import (
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.schemas.iteration import (
    ExperimentHypothesis,
    FeedbackAnalysis,
    IterationState,
    RoundResult,
)
from nanoresearch.agents.experiment._iteration_helpers import _IterationHelpersMixin

logger = logging.getLogger(__name__)


class _IterationMixin(_IterationHelpersMixin):
    """Mixin — iteration checkpoint, hypothesis, changes, history."""

    def _save_iteration_checkpoint(
        self,
        state: IterationState,
        checkpoint_path: str = "logs/iteration_checkpoint.json",
    ) -> None:
        """Save iteration state checkpoint for crash recovery."""
        self.workspace.write_json(
            checkpoint_path,
            state.model_dump(),
        )

    def _load_iteration_checkpoint(
        self,
        default_state: IterationState,
        checkpoint_path: str = "logs/iteration_checkpoint.json",
    ) -> tuple[IterationState, int]:
        """Load iteration checkpoint if available.

        Returns (state, start_round) where start_round is the round to
        resume from (1 if no checkpoint exists).
        """
        try:
            data = self.workspace.read_json(checkpoint_path)
            if isinstance(data, dict) and data.get("rounds"):
                state = IterationState.model_validate(data)
                completed_rounds = len(state.rounds)
                start_round = completed_rounds + 1
                if start_round <= state.max_rounds:
                    logger.info(
                        "Resuming experiment from round %d (checkpoint has %d completed rounds)",
                        start_round, completed_rounds,
                    )
                    return state, start_round
                else:
                    logger.info(
                        "Checkpoint shows all %d rounds completed, starting fresh",
                        completed_rounds,
                    )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to load iteration checkpoint: %s", exc)
        return default_state, 1

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    async def _generate_iteration_hypothesis(
        self,
        analysis: FeedbackAnalysis | None,
        history_summary: str,
        blueprint: str,
        preflight_error_ctx: str = "",
        code_dir: Path | None = None,
    ) -> ExperimentHypothesis:
        """LLM generates the next iteration hypothesis from feedback."""
        analysis_text = ""
        if analysis:
            analysis_text = (
                f"Attribution: {analysis.attribution}\n"
                f"Recommended action: {analysis.recommended_action}\n"
                f"Metrics: {json.dumps(analysis.metric_summary)}\n"
                f"Training dynamics: convergence={analysis.training_dynamics.convergence_speed}, "
                f"overfitting={analysis.training_dynamics.overfitting_detected}, "
                f"stability={analysis.training_dynamics.loss_stability}\n"
                f"Error categories: {analysis.error_categories}"
        )

        # Collect actual file list from code_dir for the LLM
        code_dir = code_dir or (self.workspace.path / "code")
        actual_files = []
        if code_dir.exists():
            for f in sorted(code_dir.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f) and ".pyc" not in str(f):
                    actual_files.append(str(f.relative_to(code_dir)).replace("\\", "/"))
        file_list = "\n".join(f"  - {f}" for f in actual_files) if actual_files else "  (no files yet)"

        # Build list of previously tried hypotheses to prevent repetition
        prev_hypotheses = []
        if history_summary:
            for line in history_summary.split("\n"):
                if line.strip():
                    prev_hypotheses.append(line.strip())
        prev_hyp_block = "\n".join(prev_hypotheses) if prev_hypotheses else "None"

        prompt = f"""Based on the previous experiment round's feedback, generate a hypothesis for the next improvement iteration.
{preflight_error_ctx}
== Previous Analysis ==
{analysis_text or "No analysis available."}

== History ==
{history_summary or "No previous rounds."}

== PREVIOUSLY TRIED HYPOTHESES (DO NOT REPEAT) ==
{prev_hyp_block}

== Experiment Blueprint ==
{blueprint[:2000]}

== Actual Project Files ==
{file_list}

IMPORTANT RULES:
1. Only reference files that exist in the list above. Do NOT invent new file paths.
2. Use the EXACT paths shown above in your planned_changes.
3. The `--quick-eval` mode HARDCODES a small model and 3-5 epochs regardless of config.
   Changing epochs/batch_size/num_runs in config/default.yaml has NO EFFECT on quick-eval.
   DO NOT suggest increasing epochs or changing hyperparameters in config — it is USELESS.
4. Instead, focus on changes that actually affect quick-eval behavior:
   - Fix bugs in model architecture (src/model.py)
   - Fix bugs in training loop (src/trainer.py)
   - Fix evaluation/metrics collection (src/evaluate.py, src/utils.py)
   - Fix data loading/preprocessing (src/dataset.py)
   - Fix the quick-eval code path in main.py directly
   - Improve model architecture (e.g., add batch norm, better init, residual connections)
5. DO NOT repeat any hypothesis from the list above. Each round must try something DIFFERENT.
   If you cannot think of a genuinely new improvement, set "no_new_ideas": true.

Output a JSON object with:
{{
  "hypothesis": "<what you will change and why>",
  "planned_changes": ["<EXACT_FILE_PATH: specific change>", ...],
  "expected_signal": "<what metric improvement you expect>",
  "rationale": "<reasoning>",
  "no_new_ideas": false
}}"""

        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML experiment iteration planner. Generate a focused hypothesis for the next improvement round. Output ONLY valid JSON.",
                prompt,
                json_mode=True,
            )
            data = self._parse_llm_json_payload(raw)

            # If LLM says no new ideas, signal early stop
            if data.get("no_new_ideas"):
                logger.info("LLM reports no new ideas — will signal early stop")
                return ExperimentHypothesis(
                    round_number=0,
                    hypothesis="__NO_NEW_IDEAS__",
                    planned_changes=[],
                    expected_signal="",
                    rationale="LLM exhausted improvement ideas",
                )

            return ExperimentHypothesis(
                round_number=0,  # caller sets this
                hypothesis=data.get("hypothesis", "Iterative improvement"),
                planned_changes=data.get("planned_changes", []),
                expected_signal=data.get("expected_signal", ""),
                rationale=data.get("rationale", ""),
            )
        except Exception as exc:
            logger.warning("Failed to generate hypothesis: %s", exc)
            return ExperimentHypothesis(
                round_number=0,
                hypothesis="Retry with general improvements based on error feedback",
                planned_changes=["Fix errors from previous round"],
                expected_signal="Successful execution",
                rationale="Fallback hypothesis after LLM generation failure",
            )

    async def _apply_iteration_changes(
        self,
        hypothesis: ExperimentHypothesis,
        code_dir: Path,
        venv_python: str,
    ) -> list[str]:
        """LLM modifies specific files using search-replace edits (OpenClaw style).

        Uses precise search-replace blocks instead of full file rewrites to:
        1. Reduce token usage (LLM only outputs the diff, not entire files)
        2. Avoid accidental deletion of unchanged code
        3. Make changes auditable
        """
        import asyncio
        import shlex

        self._remember_mutation_snapshot_entry(None)
        
        prompt = (
            f"Apply the following improvement to the project.\n\n"
            f"== Planned Changes ==\n"
            f"{json.dumps(hypothesis.planned_changes, indent=2)}\n\n"
            f"Important: Only edit the files necessary. Do not explain, just edit and exit."
        )
        
        escaped_prompt = shlex.quote(prompt)
        abs_code_dir = code_dir.resolve()
        
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
        
        self.log(f"Delegating iterative improvement application to Claude Code via: su - nyt_worker")
        
        modified_files: list[str] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "su", "-", "nyt_worker", "-c", worker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr_out = await asyncio.wait_for(proc.communicate(), timeout=600)
            
            if proc.returncode == 0:
                self.log(f"Claude Code iteration completed successfully.")
                modified_files = ["auto-fixed-iterative-ccr"]
            else:
                stderr_text = stderr_out.decode('utf-8', errors='replace')
                stdout_text = stdout.decode('utf-8', errors='replace')
                self.log(f"Claude Code returned non-zero (return_code={proc.returncode}):\nSTDOUT:\n{stdout_text}\nSTDERR:\n{stderr_text}")
                if "edited" in stdout_text.lower() or "saved" in stdout_text.lower():
                    self.log("Claude Code returned non-zero but possibly made edits. Continuing.")
                    modified_files = ["auto-fixed-iterative-ccr-partial"]
        except asyncio.TimeoutError:
            self.log("Claude Code execution timed out after 10 minutes.")
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        except Exception as exc:
            logger.warning("Failed to apply iteration changes via ccr: %s", exc)

        return modified_files
