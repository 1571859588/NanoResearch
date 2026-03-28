"""Status Tracker — generates STATUS.md from the global results registry."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class StatusTracker:
    """Aggregates results from latest_index.json and generates a human-readable STATUS.md."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        # Use root results/ if available, otherwise session results/
        # (StatusTracker is typically called from a workspace context)
        self.results_dir = workspace_path / "results"
        self.index_path = self.results_dir / "latest_index.json"
        self.status_md_path = self.results_dir / "STATUS.md"

    def generate_status_report(self):
        """Generate/update STATUS.md based on latest_index.json and references."""
        if not self.index_path.exists():
            logger.debug("latest_index.json not found, skipping STATUS.md generation")
            return
            
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load latest_index.json: {e}")
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        md = f"# Research Status Manifest\n\n"
        md += f"*Last Updated: {now}*\n\n"
        
        # 1. Global Metrics Table
        md += "## 📊 Benchmark Results\n\n"
        md += "| Benchmark | Metric | Best Value | Baseline/Run | Source |\n"
        md += "|-----------|--------|------------|--------------|--------|\n"
        
        benchmarks = index.get("benchmarks", {})
        if not benchmarks:
            md += "| (none) | | | | |\n"
        else:
            for b_name, b_info in sorted(benchmarks.items()):
                metrics = b_info.get("metrics", {})
                for m_name, m_info in sorted(metrics.items()):
                    val = m_info.get("best_value", "N/A")
                    run_id = m_info.get("best_run_id", "N/A")
                    source = m_info.get("source", "experiment")
                    md += f"| {b_name} | {m_name} | {val} | {run_id} | {source} |\n"
        
        md += "\n"
        
        # 2. Baseline Paper Status (from CURRENT workspace references)
        md += "## 📚 Baseline Paper Summary Status\n\n"
        md += "| Baseline | Paper ID | Summary Status | Result Reuse |\n"
        md += "|----------|----------|----------------|--------------|\n"
        
        ref_dir = self.workspace_path / "references" / "papers"
        ref_count = 0
        if ref_dir.exists():
            for md_file in sorted(ref_dir.glob("*.md")):
                try:
                    content = md_file.read_text(encoding="utf-8", errors="replace")
                    
                    # Extract metadata from frontmatter or bullet points
                    baseline_match = re.search(r"- baseline_method_name: (.*)", content)
                    baseline = baseline_match.group(1).strip() if baseline_match else md_file.stem
                    
                    paper_id_match = re.search(r"- paper_id: (.*)", content)
                    paper_id = paper_id_match.group(1).strip() if paper_id_match else "unknown"
                    
                    is_tbd = "TBD" in content or "unknown" in content
                    status = "✅ Complete" if not is_tbd else "⚠️ Missing Info"
                    
                    # Check for result table presence
                    reuse = "✅ Yes" if "|" in content and ("Metric" in content or "Benchmark" in content) else "❌ No"
                    
                    md += f"| {baseline} | {paper_id} | {status} | {reuse} |\n"
                    ref_count += 1
                except Exception as e:
                    logger.warning("Failed to parse reference %s: %s", md_file.name, e)

        if ref_count == 0:
            md += "| (none) | | | |\n"
        
        md += "\n"
        
        # 3. Recent Execution Runs (from global index history)
        md += "## 🚀 Recent Execution History\n\n"
        md += "| Run ID | Stage | Status | Created At | Session |\n"
        md += "|--------|-------|--------|------------|---------|\n"
        
        history_map = index.get("history", {})
        if not history_map:
            md += "| (none) | | | | |\n"
        else:
            history = list(history_map.values())
            # Sort by run_id descending if possible, else alphabetical
            history.sort(key=lambda x: str(x.get("run_id", "")), reverse=True)
            
            for run in history[:20]: # show last 20
                rid = run.get("run_id", "???")
                stage = run.get("stage", "???")
                status = run.get("final_status") or run.get("experiment_status", "pending")
                created = run.get("created_at", "???")
                sid = run.get("session_id", "???")[:8]
                md += f"| `{rid}` | {stage} | {status} | {created} | `{sid}` |\n"
            
        self.status_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_md_path.write_text(md, encoding="utf-8")
        logger.info(f"Generated STATUS.md at {self.status_md_path}")
