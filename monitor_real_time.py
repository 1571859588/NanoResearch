#!/usr/bin/env python3
"""Real-time monitoring script for the CLIP-SAPI-CBM experiment."""

import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentMonitor:
    def __init__(self, workspace_path: Path, log_path: Path):
        self.workspace_path = workspace_path
        self.log_path = log_path
        self.start_time = datetime.now()
        self.stage_times = {}
        self.local_resources_used = False
        self.datasets_copied = []
        self.models_downloaded = []
        self.code_generated = False
        self.experiment_executed = False
        self.results_analyzed = False
        self.paper_drafted = False

    def monitor_continuously(self):
        """Continuously monitor the experiment progress."""
        logger.info("Starting real-time monitoring of CLIP-SAPI-CBM experiment")
        logger.info(f"Workspace: {self.workspace_path}")
        logger.info(f"Log file: {self.log_path}")

        while True:
            try:
                self.check_current_stage()
                self.check_resource_usage()
                self.check_code_generation()
                self.check_experiment_execution()
                self.check_results_analysis()
                self.check_paper_writing()
                self.print_summary()

                # Check if experiment is complete
                if self.is_experiment_complete():
                    logger.info("\n" + "="*60)
                    logger.info("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
                    logger.info("="*60)
                    self.print_final_summary()
                    break

                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                logger.info("\nMonitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                time.sleep(60)

    def check_current_stage(self):
        """Check the current stage from manifest.json."""
        manifest_path = self.workspace_path / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                current_stage = manifest.get("current_stage", "unknown")

                # Record stage transition times
                if current_stage not in self.stage_times:
                    self.stage_times[current_stage] = datetime.now()
                    logger.info(f"\n📍 Entered stage: {current_stage}")

                    # Special handling for SETUP stage
                    if current_stage == "SETUP":
                        self.check_setup_phase()

            except Exception as e:
                logger.error(f"Error reading manifest: {e}")

    def check_setup_phase(self):
        """Special monitoring for SETUP phase to track local resource usage."""
        logger.info("🔍 Monitoring SETUP phase for local resource usage...")

        # Check setup output
        setup_output = self.workspace_path / "plans" / "setup_output.json"
        if setup_output.exists():
            try:
                with open(setup_output) as f:
                    setup_data = json.load(f)

                resources = setup_data.get("downloaded_resources", [])
                for resource in resources:
                    if resource.get("source") == "local_resource":
                        self.local_resources_used = True
                        if resource.get("type") == "dataset":
                            self.datasets_copied.append(resource.get("name"))
                        elif resource.get("type") == "model":
                            self.models_downloaded.append(resource.get("name"))

                if self.local_resources_used:
                    logger.info("✅ Local resources successfully used!")
                    logger.info(f"   Datasets copied: {self.datasets_copied}")
                    logger.info(f"   Models used: {self.models_downloaded}")
                else:
                    logger.warning("⚠️  No local resources detected in setup")

            except Exception as e:
                logger.error(f"Error reading setup output: {e}")

    def check_resource_usage(self):
        """Check for actual resource usage in workspace."""
        datasets_dir = self.workspace_path / "datasets"
        models_dir = self.workspace_path / "models"

        # Check datasets
        if datasets_dir.exists():
            dataset_files = list(datasets_dir.iterdir())
            if dataset_files:
                logger.info(f"📊 Datasets in workspace: {[f.name for f in dataset_files]}")

                # Check if CUB dataset is properly copied
                cub_indicator = datasets_dir / "CUB_200_2011"
                if cub_indicator.exists() and cub_indicator.is_dir():
                    num_classes = len([d for d in (cub_indicator / "data" / "images").iterdir() if d.is_dir()])
                    logger.info(f"✅ CUB dataset detected with {num_classes} bird classes")
                    self.local_resources_used = True

        # Check models
        if models_dir.exists():
            model_files = list(models_dir.iterdir())
            if model_files:
                logger.info(f"🤖 Models in workspace: {[f.name for f in model_files]}")

    def check_code_generation(self):
        """Check if code generation has started/completed."""
        code_dir = self.workspace_path / "code"
        if code_dir.exists():
            py_files = list(code_dir.glob("*.py"))
            if py_files:
                self.code_generated = True
                logger.info(f"📝 Code generated: {[f.name for f in py_files]}")

                # Check for specific files we expect
                expected_files = ["train.py", "model.py", "dataset.py", "evaluate.py"]
                for expected in expected_files:
                    if any(expected in f.name for f in py_files):
                        logger.info(f"   ✅ Found {expected}")

    def check_experiment_execution(self):
        """Check if experiment has been executed."""
        results_dir = self.workspace_path / "experiment" / "results"
        if results_dir.exists():
            result_files = list(results_dir.glob("*"))
            if result_files:
                self.experiment_executed = True
                logger.info(f"🔬 Experiment results: {[f.name for f in result_files]}")

                # Check for specific result indicators
                metrics_files = list(results_dir.glob("*metrics*.json")) + list(results_dir.glob("*results*.json"))
                if metrics_files:
                    logger.info(f"   📈 Metrics files: {[f.name for f in metrics_files]}")

    def check_results_analysis(self):
        """Check if results have been analyzed."""
        analysis_file = self.workspace_path / "plans" / "analysis_output.json"
        if analysis_file.exists():
            self.results_analyzed = True
            logger.info("📊 Results analysis completed")

            # Try to read some analysis details
            try:
                with open(analysis_file) as f:
                    analysis_data = json.load(f)
                logger.info(f"   Key findings: {list(analysis_data.keys())}")
            except:
                pass

    def check_paper_writing(self):
        """Check if paper drafting has started/completed."""
        drafts_dir = self.workspace_path / "drafts"
        paper_file = drafts_dir / "main.tex"

        if paper_file.exists():
            self.paper_drafted = True
            logger.info("📄 Paper drafting started/completed")

            # Check paper size
            try:
                size = paper_file.stat().st_size
                logger.info(f"   Paper size: {size} bytes")
            except:
                pass

    def print_summary(self):
        """Print current status summary."""
        elapsed = datetime.now() - self.start_time
        logger.info(f"\n📋 Status Update (elapsed: {elapsed.total_seconds()/60:.1f} min)")
        logger.info(f"   Local resources used: {'✅' if self.local_resources_used else '❌'}")
        logger.info(f"   Code generated: {'✅' if self.code_generated else '❌'}")
        logger.info(f"   Experiment executed: {'✅' if self.experiment_executed else '❌'}")
        logger.info(f"   Results analyzed: {'✅' if self.results_analyzed else '❌'}")
        logger.info(f"   Paper drafted: {'✅' if self.paper_drafted else '❌'}")

        # Show current stage if available
        manifest_path = self.workspace_path / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                current_stage = manifest.get("current_stage", "unknown")
                logger.info(f"   Current stage: {current_stage}")
            except:
                pass

    def is_experiment_complete(self):
        """Check if the entire experiment is complete."""
        manifest_path = self.workspace_path / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                return manifest.get("current_stage") == "DONE"
            except:
                pass
        return False

    def print_final_summary(self):
        """Print final summary of the experiment."""
        total_time = datetime.now() - self.start_time

        logger.info(f"\nTotal experiment time: {total_time.total_seconds()/3600:.1f} hours")
        logger.info(f"Local resources utilized: {'✅ Yes' if self.local_resources_used else '❌ No'}")
        logger.info(f"Datasets copied from local: {len(self.datasets_copied)}")
        logger.info(f"Code successfully generated: {'✅ Yes' if self.code_generated else '❌ No'}")
        logger.info(f"Experiment executed: {'✅ Yes' if self.experiment_executed else '❌ No'}")
        logger.info(f"Results analyzed: {'✅ Yes' if self.results_analyzed else '❌ No'}")
        logger.info(f"Paper drafted: {'✅ Yes' if self.paper_drafted else '❌ No'}")

        # Check final outputs
        final_outputs = []
        if (self.workspace_path / "output" / "main.pdf").exists():
            final_outputs.append("📄 Final paper (PDF)")
        if (self.workspace_path / "output" / "main.tex").exists():
            final_outputs.append("📝 LaTeX source")
        if (self.workspace_path / "output" / "references.bib").exists():
            final_outputs.append("📚 References")

        if final_outputs:
            logger.info(f"\nFinal outputs generated:")
            for output in final_outputs:
                logger.info(f"   {output}")


def main():
    """Main function to start monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor CLIP-SAPI-CBM experiment")
    parser.add_argument("--workspace", type=str, required=True, help="Workspace directory path")
    parser.add_argument("--log", type=str, required=True, help="Log file path")

    args = parser.parse_args()

    workspace_path = Path(args.workspace)
    log_path = Path(args.log)

    if not workspace_path.exists():
        logger.error(f"Workspace not found: {workspace_path}")
        return

    monitor = ExperimentMonitor(workspace_path, log_path)
    monitor.monitor_continuously()


if __name__ == "__main__":
    main()