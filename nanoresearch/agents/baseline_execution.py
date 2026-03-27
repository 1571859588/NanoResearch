import logging
from typing import Any
from pathlib import Path

from nanoresearch.agents.execution import ExecutionAgent
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

class BaselineExecutionAgent(ExecutionAgent):
    stage = PipelineStage.BASELINE_EXECUTION

    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Intercept coding output to execute baseline_train_command instead of train_command."""
        coding_output = dict(inputs.get("coding_output", {}))
        
        baseline_cmd = coding_output.get("baseline_train_command")
        if baseline_cmd:
            self.log(f"Intercepting execution loop to run baseline validation: {baseline_cmd}")
            code_dir = Path(coding_output.get("code_dir", ""))
            
            if code_dir.exists():
                from nanoresearch.agents.project_runner import ensure_project_runner
                # Re-generate runner assets for baseline execution specifically
                runner_assets = ensure_project_runner(code_dir, baseline_cmd)
                coding_output["train_command"] = runner_assets["runner_command"]
                
        inputs["coding_output"] = coding_output
        result = await super().run(**inputs)
        
        # Save baseline specific metrics file to prevent overwrite by proposed method execution
        code_dir = Path(coding_output.get("code_dir", ""))
        metric_file = code_dir / "results" / "metrics.json"
        baseline_metric_file = code_dir / "results" / "baseline_metrics.json"
        if metric_file.exists():
            import shutil
            shutil.copy(str(metric_file), str(baseline_metric_file))
            self.log(f"Cloned baseline metrics output to {baseline_metric_file.name} to preserve analysis isolation")
            
        return result
