"""Analysis agent — parses real experiment results and generates figures from actual data."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseResearchAgent):
    """Analyzes real experiment results and generates publication figures from actual data."""

    stage = PipelineStage.FIGURE_GEN  # placeholder for base class

    @property
    def stage_config(self):
        """Use writing model config for result analysis (needs strong reasoning)."""
        return self.config.for_stage("writing")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        execution_output: dict = inputs.get("execution_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})

        self.log("Starting analysis of real experiment results")

        # Step 1: Analyze results
        analysis = await self._analyze_results(execution_output, experiment_blueprint)
        self.log(f"Analysis complete: {list(analysis.keys())}")

        # Step 2: Generate figures from real data
        figures = await self._generate_figures(analysis, experiment_blueprint)
        self.log(f"Generated {len(figures)} figures")

        result = {
            "analysis": analysis,
            "figures": figures,
            "execution_output": execution_output,
        }

        self.workspace.write_json("plans/analysis_output.json", result)
        return result

    async def _analyze_results(
        self, execution_output: dict, blueprint: dict
    ) -> dict:
        """Use LLM to interpret and summarize experiment results."""
        metrics = execution_output.get("metrics", {})
        parsed_metrics = execution_output.get("parsed_metrics", {})
        stdout_log = execution_output.get("stdout_log", "")[-5000:]
        training_log_csv = execution_output.get("training_log_csv", "")
        final_status = execution_output.get("final_status", "UNKNOWN")

        system_prompt = (
            "You are an ML researcher analyzing experiment results. "
            "Given the training logs and metrics, provide a comprehensive analysis. "
            "Be honest about results — if the model didn't converge or results are poor, say so. "
            "Return JSON only."
        )

        user_prompt = f"""Job Status: {final_status}

Metrics JSON:
{json.dumps(metrics, indent=2)[:3000]}

Parsed Metrics from Log:
{json.dumps(parsed_metrics, indent=2)[:2000]}

Training Log CSV (last part):
{training_log_csv[-3000:] if training_log_csv else 'N/A'}

Stdout Log (last part):
{stdout_log[-3000:]}

Expected Metrics: {json.dumps(blueprint.get('metrics', []), indent=2)[:500]}
Baselines: {json.dumps(blueprint.get('baselines', []), indent=2)[:500]}

Analyze these results. Return JSON:
{{
  "summary": "1-2 paragraph summary of results...",
  "converged": true/false,
  "final_metrics": {{"metric_name": value, ...}},
  "comparison_with_baselines": "How our method compares...",
  "training_dynamics": "Description of training curve behavior...",
  "key_findings": ["finding1", "finding2", ...],
  "limitations": ["limitation1", ...],
  "figures_to_generate": [
    {{
      "figure_id": "fig_training_curve",
      "title": "Training Loss Curve",
      "type": "line",
      "data_source": "training_log"
    }},
    {{
      "figure_id": "fig_results_comparison",
      "title": "Results Comparison with Baselines",
      "type": "bar",
      "data_source": "metrics"
    }},
    {{
      "figure_id": "fig_ablation",
      "title": "Ablation Study",
      "type": "bar",
      "data_source": "metrics"
    }}
  ]
}}"""

        result = await self.generate_json(system_prompt, user_prompt)
        return result if isinstance(result, dict) else {}

    async def _generate_figures(
        self, analysis: dict, blueprint: dict
    ) -> dict:
        """Generate publication-quality figures from real experiment data."""
        figures_output = {}
        figures_dir = self.workspace.path / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Get actual data for plotting
        execution_output = analysis.get("execution_output", {}) or {}
        final_metrics = analysis.get("final_metrics", {})
        training_log = analysis.get("training_dynamics", "")

        figure_specs = analysis.get("figures_to_generate", [])
        if not figure_specs:
            # Default figures
            figure_specs = [
                {"figure_id": "fig_training_curve", "title": "Training Loss Curve", "type": "line"},
                {"figure_id": "fig_results", "title": "Results", "type": "bar"},
            ]

        for fig_spec in figure_specs[:3]:
            fig_id = fig_spec.get("figure_id", "fig_unknown")
            fig_title = fig_spec.get("title", "Figure")

            self.log(f"Generating figure: {fig_id}")

            # Generate plotting code via LLM
            plot_code = await self._generate_plot_code(
                fig_spec, analysis, blueprint
            )

            # Write and execute the plotting code
            script_path = figures_dir / f"{fig_id}_plot.py"
            script_path.write_text(plot_code)

            png_path = figures_dir / f"{fig_id}.png"
            pdf_path = figures_dir / f"{fig_id}.pdf"

            try:
                result = await self._run_shell(
                    f"cd {figures_dir} && python3 {script_path}",
                    timeout=60,
                )
                if png_path.exists():
                    figures_output[fig_id] = {
                        "png_path": str(png_path),
                        "pdf_path": str(pdf_path) if pdf_path.exists() else "",
                        "caption": fig_title,
                        "script_path": str(script_path),
                    }
                    self.log(f"Figure generated: {fig_id}")
                else:
                    self.log(f"Figure script ran but no output: {result.get('stderr', '')[:200]}")
                    figures_output[fig_id] = {"error": "No output file generated"}
            except Exception as e:
                self.log(f"Failed to generate figure {fig_id}: {e}")
                figures_output[fig_id] = {"error": str(e)}

        return figures_output

    async def _generate_plot_code(
        self, fig_spec: dict, analysis: dict, blueprint: dict
    ) -> str:
        """Generate matplotlib plotting code for a specific figure."""
        fig_id = fig_spec.get("figure_id", "fig")
        fig_title = fig_spec.get("title", "Figure")
        fig_type = fig_spec.get("type", "bar")

        final_metrics = analysis.get("final_metrics", {})
        baselines = blueprint.get("baselines", [])
        metrics = blueprint.get("metrics", [])

        system_prompt = (
            "You are a data visualization expert. Write a complete matplotlib Python script "
            "that creates a publication-quality figure. The script must:\n"
            "1. Use matplotlib and seaborn\n"
            "2. Save the figure as both PNG (300 DPI) and PDF\n"
            "3. Use the ACTUAL experiment results provided (not made-up data)\n"
            "4. Have proper axis labels, title, legend\n"
            "5. Use a clean academic style\n"
            "Return ONLY the Python code."
        )

        user_prompt = f"""Generate a {fig_type} plot for: {fig_title}
Figure ID: {fig_id}

ACTUAL experiment results:
{json.dumps(final_metrics, indent=2)[:2000]}

Analysis summary: {analysis.get('summary', '')[:1000]}
Training dynamics: {analysis.get('training_dynamics', '')[:500]}

Baselines for comparison: {json.dumps(baselines, indent=2)[:500]}
Metrics definitions: {json.dumps(metrics, indent=2)[:300]}

IMPORTANT:
- Use the REAL numbers from the experiment results above
- If some metrics are missing, use reasonable placeholder values but note them
- Save as '{fig_id}.png' (dpi=300) and '{fig_id}.pdf'
- Use plt.tight_layout()
- Make the figure 8x5 inches

Return ONLY the Python code, no markdown fences."""

        code = await self.generate(system_prompt, user_prompt)

        # Strip markdown fences
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        env = {**__import__('os').environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re
            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                m = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if m:
                    proxy_url = m.group(1)
        if proxy_url:
            env.update({
                "http_proxy": proxy_url, "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url,
            })
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        pass
