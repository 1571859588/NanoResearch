#!/usr/bin/env python3
"""Monitor script to track local resource usage in NanoResearch pipelines."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch")
import sys
sys.path.insert(0, str(project_root))

from nanoresearch.agents.resource_manager import ResourceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor local resource usage in NanoResearch workspaces."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.workspaces_dir = project_root / "workspaces"
        self.resource_manager = ResourceManager(project_root)

    def scan_workspaces(self) -> dict:
        """Scan all workspaces for resource usage patterns."""
        results = {
            "scan_time": datetime.now().isoformat(),
            "total_workspaces": 0,
            "workspaces_using_local": 0,
            "local_resources_used": 0,
            "download_fallbacks": 0,
            "failed_downloads": 0,
            "workspaces": []
        }

        if not self.workspaces_dir.exists():
            logger.warning(f"Workspaces directory not found: {self.workspaces_dir}")
            return results

        for workspace_path in self.workspaces_dir.iterdir():
            if workspace_path.is_dir():
                workspace_info = self.analyze_workspace(workspace_path)
                results["workspaces"].append(workspace_info)
                results["total_workspaces"] += 1

                # Aggregate statistics
                if workspace_info["local_resources"] > 0:
                    results["workspaces_using_local"] += 1
                results["local_resources_used"] += workspace_info["local_resources"]
                results["download_fallbacks"] += workspace_info["download_fallbacks"]
                results["failed_downloads"] += workspace_info["failed_downloads"]

        return results

    def analyze_workspace(self, workspace_path: Path) -> dict:
        """Analyze a single workspace for resource usage."""
        logger.info(f"Analyzing workspace: {workspace_path.name}")

        info = {
            "name": workspace_path.name,
            "path": str(workspace_path),
            "manifest_exists": False,
            "setup_exists": False,
            "local_resources": 0,
            "download_fallbacks": 0,
            "failed_downloads": 0,
            "resources": [],
            "recommendations": []
        }

        # Check for manifest
        manifest_path = workspace_path / "manifest.json"
        if manifest_path.exists():
            info["manifest_exists"] = True
            with open(manifest_path) as f:
                manifest = json.load(f)
                info["topic"] = manifest.get("topic", "")[:100] + "..." if manifest.get("topic") else ""
                info["current_stage"] = manifest.get("current_stage", "unknown")

        # Check setup output
        setup_path = workspace_path / "plans" / "setup_output.json"
        if setup_path.exists():
            info["setup_exists"] = True
            with open(setup_path) as f:
                setup_data = json.load(f)
                resources = setup_data.get("downloaded_resources", [])

                for resource in resources:
                    resource_info = {
                        "name": resource.get("name", "unknown"),
                        "type": resource.get("type", "unknown"),
                        "status": resource.get("status", "unknown"),
                        "source": resource.get("source", "unknown"),
                        "path": resource.get("path", ""),
                        "local_source": resource.get("local_source", ""),
                        "exists": False
                    }

                    # Check if resource exists
                    if resource_info["path"]:
                        p = Path(resource_info["path"])
                        resource_info["exists"] = p.exists()
                        if p.exists() and p.is_file():
                            resource_info["size_bytes"] = p.stat().st_size

                    info["resources"].append(resource_info)

                    # Count statistics
                    if resource_info["source"] == "local_resource":
                        info["local_resources"] += 1
                    elif resource_info["source"] not in ["unknown", ""]:
                        info["download_fallbacks"] += 1

                    if resource_info["status"] == "failed":
                        info["failed_downloads"] += 1

        # Generate recommendations
        info["recommendations"] = self.generate_recommendations(info)

        return info

    def generate_recommendations(self, workspace_info: dict) -> list[str]:
        """Generate recommendations based on workspace analysis."""
        recommendations = []

        # Check for failed downloads
        if workspace_info["failed_downloads"] > 0:
            recommendations.append(
                f"⚠️ {workspace_info['failed_downloads']} resources failed to download. "
                "Consider adding them to local_resources/"
            )

        # Check for high download usage
        if workspace_info["download_fallbacks"] > 3:
            recommendations.append(
                f"💡 High download usage ({workspace_info['download_fallbacks']} resources). "
                "Consider adding commonly used resources locally"
            )

        # Check for missing manifest
        if not workspace_info["manifest_exists"]:
            recommendations.append(
                "❌ No manifest found. Workspace may be incomplete"
            )

        # Check for missing setup
        if not workspace_info["setup_exists"]:
            recommendations.append(
                "❌ No setup output found. Run may have failed during setup stage"
            )

        # Success case
        if workspace_info["local_resources"] > 0 and not recommendations:
            recommendations.append(
                f"✅ Successfully used {workspace_info['local_resources']} local resources"
            )

        return recommendations

    def check_resource_health(self) -> dict:
        """Check the health of local resources."""
        logger.info("Checking local resource health...")

        self.resource_manager.load_resources()
        report = self.resource_manager.generate_resource_report()

        health_report = {
            "check_time": datetime.now().isoformat(),
            "datasets": {
                "total": report["datasets"]["total"],
                "available_locally": 0,
                "missing": []
            },
            "models": {
                "total": report["models"]["total"],
                "available_locally": 0,
                "missing": []
            }
        }

        # Check datasets
        for ds in report["datasets"]["available"]:
            if ds["exists_locally"]:
                health_report["datasets"]["available_locally"] += 1
            else:
                health_report["datasets"]["missing"].append(ds["name"])

        # Check models
        for model in report["models"]["available"]:
            if model["exists_locally"]:
                health_report["models"]["available_locally"] += 1
            else:
                health_report["models"]["missing"].append(model["name"])

        return health_report

    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        workspace_results = self.scan_workspaces()
        health_results = self.check_resource_health()

        report = []
        report.append("=" * 60)
        report.append("NANORESEARCH LOCAL RESOURCE MONITORING REPORT")
        report.append("=" * 60)
        report.append(f"Report generated: {workspace_results['scan_time']}")
        report.append("")

        # Resource Health Summary
        report.append("LOCAL RESOURCE HEALTH")
        report.append("-" * 30)
        report.append(f"Datasets: {health_results['datasets']['available_locally']}/{health_results['datasets']['total']} available locally")
        if health_results["datasets"]["missing"]:
            report.append(f"Missing datasets: {', '.join(health_results['datasets']['missing'])}")
        report.append(f"Models: {health_results['models']['available_locally']}/{health_results['models']['total']} available locally")
        if health_results["models"]["missing"]:
            report.append(f"Missing models: {', '.join(health_results['models']['missing'])}")
        report.append("")

        # Workspace Usage Summary
        report.append("WORKSPACE RESOURCE USAGE")
        report.append("-" * 30)
        report.append(f"Total workspaces: {workspace_results['total_workspaces']}")
        report.append(f"Using local resources: {workspace_results['workspaces_using_local']}")
        report.append(f"Total local resources used: {workspace_results['local_resources_used']}")
        report.append(f"Download fallbacks: {workspace_results['download_fallbacks']}")
        report.append(f"Failed downloads: {workspace_results['failed_downloads']}")
        report.append("")

        # Detailed Workspace Analysis
        if workspace_results["workspaces"]:
            report.append("DETAILED WORKSPACE ANALYSIS")
            report.append("-" * 40)

            for ws in workspace_results["workspaces"]:
                report.append(f"\nWorkspace: {ws['name']}")
                report.append(f"Stage: {ws['current_stage']}")
                report.append(f"Local resources: {ws['local_resources']}")
                report.append(f"Downloads: {ws['download_fallbacks']}")

                if ws["recommendations"]:
                    report.append("Recommendations:")
                    for rec in ws["recommendations"]:
                        report.append(f"  {rec}")

        # Overall Recommendations
        report.append("\n" + "=" * 60)
        report.append("OVERALL RECOMMENDATIONS")
        report.append("=" * 60)

        if health_results["datasets"]["missing"] or health_results["models"]["missing"]:
            report.append("\n📦 RESOURCE AVAILABILITY")
            if health_results["datasets"]["missing"]:
                report.append(f"• Add these datasets locally: {', '.join(health_results['datasets']['missing'][:3])}")
            if health_results["models"]["missing"]:
                report.append(f"• Add these models locally: {', '.join(health_results['models']['missing'][:3])}")

        if workspace_results["failed_downloads"] > 0:
            report.append("\n🚨 FAILED DOWNLOADS")
            report.append(f"• {workspace_results['failed_downloads']} resources failed to download")
            report.append("• Consider adding these to local_resources/")

        if workspace_results["download_fallbacks"] > workspace_results["local_resources_used"]:
            report.append("\n💡 OPTIMIZATION OPPORTUNITY")
            report.append("• More downloads than local resources used")
            report.append("• Consider adding frequently used resources locally")

        if workspace_results["workspaces_using_local"] == workspace_results["total_workspaces"]:
            report.append("\n✅ EXCELLENT")
            report.append("• All workspaces are using local resources!")
            report.append("• This significantly improves speed and reliability")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Monitor local resource usage in NanoResearch")
    parser.add_argument("--continuous", "-c", action="store_true", help="Run continuously")
    parser.add_argument("--interval", "-i", type=int, default=300, help="Interval in seconds for continuous mode")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    monitor = ResourceMonitor(project_root)

    if args.continuous:
        print(f"Running continuous monitoring every {args.interval} seconds...")
        print("Press Ctrl+C to stop")
        try:
            while True:
                report = monitor.generate_summary_report()
                print(report)
                print(f"\n{'='*60}\nWaiting {args.interval} seconds...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
    else:
        # Single run
        if args.json:
            results = {
                "workspace_analysis": monitor.scan_workspaces(),
                "resource_health": monitor.check_resource_health()
            }
            output = json.dumps(results, indent=2)
        else:
            output = monitor.generate_summary_report()

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report saved to: {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()