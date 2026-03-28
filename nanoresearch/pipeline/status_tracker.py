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
        # The session results are local
        self.local_results_dir = workspace_path / "results"
        # The registry is global (at repo root)
        self.repo_root = workspace_path.parent.parent
        self.registry_dir = self.repo_root / "registry"
        
        self.index_path = self.registry_dir / "latest_index.json"
        self.status_md_path = self.local_results_dir / "STATUS.md"

    def generate_status_report(self):
        """Generate/update STATUS.md based on latest_index.json and references."""
        # Load session_id from manifest if possible
        session_id = None
        manifest_path = self.workspace_path / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    session_id = manifest.get("session_id")
            except Exception:
                pass

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
        md += f"*Last Updated: {now}*\n"
        if session_id:
            md += f"*Session ID: {session_id}*\n"
        md += "\n"
        
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
                    if isinstance(m_info, dict):
                        val = m_info.get("best_value", "N/A")
                        run_id = m_info.get("best_run_id", "N/A")
                        source = m_info.get("source", "experiment")
                    else:
                        val = m_info
                        run_id = "N/A"
                        source = "manual"
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
        
        # 3. Recent Execution Runs (filtered by session)
        history_map = index.get("history", {})
        if history_map:
            history = list(history_map.values())
            history.sort(key=lambda x: str(x.get("run_id", "")), reverse=True)
            
            # Filter into current session vs others
            current_session_runs = [r for r in history if r.get("session_id") == session_id]
            other_runs = [r for r in history if r.get("session_id") != session_id]
            
            if current_session_runs:
                md += "## 🚀 Current Session History\n\n"
                md += "| Run ID | Stage | Status | Created At |\n"
                md += "|--------|-------|--------|------------|\n"
                for run in current_session_runs[:15]:
                    rid = run.get("run_id", "???")
                    stage = run.get("stage", "???")
                    status = run.get("final_status") or run.get("experiment_status", "pending")
                    created = run.get("created_at", "???")
                    md += f"| `{rid}` | {stage} | {status} | {created} |\n"
                md += "\n"
                
            if other_runs:
                md += "## 🌐 Global Registry Context (Other Sessions)\n\n"
                md += "| Run ID | Topic | Stage | Status | Session |\n"
                md += "|--------|-------|-------|--------|---------|\n"
                for run in other_runs[:10]:
                    rid = run.get("run_id", "???")
                    # Try to get a readable topic name from workspace path or method_slug
                    topic = run.get("topic") or Path(run.get("workspace", "")).name
                    stage = run.get("stage", "???")
                    status = run.get("final_status") or run.get("experiment_status", "pending")
                    sid = run.get("session_id", "???")[:8]
                    md += f"| `{rid}` | {topic} | {stage} | {status} | `{sid}` |\n"
                md += "\n"
        else:
            md += "## 🚀 Execution History\n\n"
            md += "| (none) | | | | |\n\n"
            
        self.status_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_md_path.write_text(md, encoding="utf-8")
        logger.info(f"Generated STATUS.md at {self.status_md_path}")
