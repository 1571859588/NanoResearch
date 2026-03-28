"""Reference Manager Agent — downloads and enriches baseline paper summaries."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import httpx

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

ENRICHMENT_SYSTEM_PROMPT = """You are a research assistant tasked with enriching a paper summary markdown file using the provided full text of the paper.

Your goal is to fill in the missing fields and resolve "TBD" or "unknown" placeholders.
The required fields are:
- baseline_method_name: The name used in the paper for the method (if it is a baseline).
- baseline_slug: A snake_case version of the method name.
- open_source: "yes" or "no".
- open_source_url: URL to the code repository if available.
- requires_training: "yes" or "no".
- training_params: Key hyperparameters mentioned in the text (e.g., learning rate, batch size, epochs).
- model_scale: Parameter count or model size (e.g., "7B", "ResNet-50 backbone").
- Benchmark & Results table: A markdown table containing at least one benchmark dataset, metric, and result value found in the paper.
- Citation (BibTeX): The full BibTeX entry for the paper.

If information is absolutely not found in the text, you may leave it as "unknown" or "N/A", but try your best to find it.
For results, prefer the main results table in the paper.

Return the FULL updated markdown content."""


class ReferenceManagerAgent(BaseResearchAgent):
    """Agent that downloads PDFs and enriches SUMMARY.md stubs."""

    stage = PipelineStage.SCIENTIFIC_REVIEW  # Operates as part of scientific review/setup

    async def run(self, **inputs: Any) -> dict[str, Any]:
        """Process the paper enrichment queue."""
        self.log("Starting reference paper enrichment")
        
        # Load the queue
        queue_path = self.workspace.path / "plans" / "paper_enrichment_queue.json"
        if not queue_path.exists():
            self.log("No enhancement queue found, generating one now")
            # We need the blueprint to generate the queue
            blueprint_path = self.workspace.path / "plans" / "experiment_blueprint.json"
            if not blueprint_path.exists():
                return {"status": "skipped", "reason": "no blueprint found"}
            
            with open(blueprint_path, "r", encoding="utf-8") as f:
                blueprint = json.load(f)
            
            enrichment_stats = self.workspace.validate_baseline_paper_summaries(blueprint)
            if enrichment_stats.get("missing_count", 0) == 0:
                return {"status": "completed", "reason": "no missing summaries"}
        
        with open(queue_path, "r", encoding="utf-8") as f:
            queue = json.load(f)
        
        items = queue.get("items", [])
        if not items:
            return {"status": "completed", "reason": "queue is empty"}
        
        results = []
        for item in items:
            paper_id = item.get("paper_id")
            baseline = item.get("baseline")
            md_path = Path(item.get("path"))
            
            self.log(f"Processing reference: {baseline} ({paper_id})")
            
            try:
                # 1. Ensure PDF is downloaded
                pdf_path = self._get_pdf_path(paper_id)
                if not pdf_path.exists():
                    await self._download_pdf(paper_id, pdf_path)
                
                # 2. Extract text (limited to avoid token overflow)
                text = self._extract_text(pdf_path)
                
                # 3. Read current markdown
                current_md = md_path.read_text(encoding="utf-8", errors="replace")
                
                # 4. Prompt LLM to enrich
                user_prompt = f"Paper ID: {paper_id}\nBaseline Name: {baseline}\n\nCurrent Summary Markdown:\n{current_md}\n\nFull Text Snippet (first 15000 chars):\n{text[:15000]}"
                
                enriched_md = await self.generate(ENRICHMENT_SYSTEM_PROMPT, user_prompt)
                
                # 5. Save enriched markdown
                md_path.write_text(enriched_md, encoding="utf-8")
                results.append({"paper_id": paper_id, "status": "enriched", "path": str(md_path)})
                
            except Exception as e:
                logger.error(f"Failed to enrich {paper_id}: {e}")
                results.append({"paper_id": paper_id, "status": "failed", "error": str(e)})
        
        # Final validation to see if we're done
        blueprint_path = self.workspace.path / "plans" / "experiment_blueprint.json"
        with open(blueprint_path, "r", encoding="utf-8") as f:
            blueprint = json.load(f)
        final_check = self.workspace.validate_baseline_paper_summaries(blueprint)
        
        return {
            "status": "completed" if final_check.get("missing_count", 0) == 0 else "partially_completed",
            "enriched_items": results,
            "final_check": final_check
        }

    def _get_pdf_path(self, paper_id: str) -> Path:
        """Get path for the PDF in the global references/papers directory."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", paper_id.lower()).strip("_")
        return self.workspace.global_references_dir / "papers" / f"{slug}.pdf"

    async def _download_pdf(self, paper_id: str, dest: Path):
        """Download PDF from Arxiv or other sources."""
        url = ""
        if paper_id.startswith("arxiv:"):
            arxiv_id = paper_id.replace("arxiv:", "")
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif re.match(r"^\d{4}\.\d{4,5}", paper_id): # Arxiv ID alone
            url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        
        if not url:
            # Try searching if it looks like a title? 
            # For now only support direct Arxiv IDs
            raise ValueError(f"Unsupported paper_id format for auto-download: {paper_id}")
        
        self.log(f"Downloading PDF from {url}")
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, follow_redirects=True, timeout=60.0)
            resp.raise_for_status()
            dest.write_bytes(resp.content)

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
