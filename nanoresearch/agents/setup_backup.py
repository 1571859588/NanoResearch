"""Setup agent — searches GitHub for relevant code, clones repos, downloads models/data.

Uses a global cache at ~/.nanoresearch/cache/ so models/data are shared across pipeline runs.
Downloads models from ModelScope first (faster in China), falls back to HuggingFace.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import re
import shlex
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

from .setup_search import _SetupSearchMixin
from .setup_github import _SetupGithubMixin

logger = logging.getLogger(__name__)

# Global cache directory — shared across all pipeline runs
GLOBAL_CACHE_DIR = Path.home() / ".nanoresearch" / "cache"
GLOBAL_MODELS_DIR = GLOBAL_CACHE_DIR / "models"
GLOBAL_DATA_DIR = GLOBAL_CACHE_DIR / "data"
SUCCESS_RESOURCE_STATUSES = {"downloaded", "full", "config_only"}

# Regex for GitHub repo URLs (not raw file links)
_GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9._-]+)/(?P<repo>[A-Za-z0-9._-]+?)(?:\.git)?/?$"
)
# Patterns for extracting real download URLs from README / scripts inside a dataset repo
_DOWNLOAD_URL_RE = re.compile(
    r"(https?://(?:"
    r"drive\.google\.com/[^\s\)\]\"'>]+|"          # Google Drive
    r"docs\.google\.com/[^\s\)\]\"'>]+|"            # Google Docs exports
    r"dl\.fbaipublicfiles\.com/[^\s\)\]\"'>]+|"     # Meta / FAIR
    r"zenodo\.org/record[^\s\)\]\"'>]+|"            # Zenodo
    r"zenodo\.org/api/records[^\s\)\]\"'>]+|"        # Zenodo API
    r"huggingface\.co/datasets/[^\s\)\]\"'>]+|"     # HuggingFace datasets
    r"storage\.googleapis\.com/[^\s\)\]\"'>]+|"     # GCS
    r"s3\.amazonaws\.com/[^\s\)\]\"'>]+|"           # S3
    r"(?:[a-z0-9-]+\.)?s3[.-][^\s\)\]\"'>]+|"      # S3 regional
    r"dropbox\.com/[^\s\)\]\"'>]+|"                 # Dropbox
    r"figshare\.com/[^\s\)\]\"'>]+|"                # Figshare
    r"data\.dgl\.ai/[^\s\)\]\"'>]+|"                # DGL datasets
    r"people\.csail\.mit\.edu/[^\s\)\]\"'>]+|"      # MIT
    r"[^\s\)\]\"'>]+\.(?:zip|tar\.gz|tgz|tar\.bz2|gz|csv|tsv|json|jsonl|h5|hdf5|pt|pkl|npy|npz|parquet|txt)"
    r")"
    r")",
    re.IGNORECASE,
)


class SetupAgent(_SetupSearchMixin, _SetupGithubMixin, BaseResearchAgent):
    """Searches for relevant code repos, clones them, and downloads required resources."""

    stage = PipelineStage.SETUP

    @property
    def stage_config(self):
        """Reuse experiment-stage model routing for setup planning."""
        return self.config.for_stage("experiment")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        topic: str = inputs["topic"]
        ideation_output: dict = inputs.get("ideation_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})

        self.log("Starting setup: code search + resource download")

        # Step 1: Search GitHub for relevant repos
        search_plan = await self._plan_search(topic, ideation_output, experiment_blueprint)
        search_plan = self._augment_search_plan_with_blueprint_resources(
            search_plan,
            experiment_blueprint,
        )
        self.log(f"Search plan: {json.dumps(search_plan, indent=2)[:500]}")

        # Step 2: Search and clone repos
        cloned_repos = await self._search_and_clone(search_plan)
        self.log(f"Cloned {len(cloned_repos)} repos")

        # Step 3: Analyze cloned code
        code_analysis = await self._analyze_cloned_code(cloned_repos, experiment_blueprint)

        # Step 4: Download required resources (models, datasets)
        # Datasets → workspace-local `datasets/` dir (each task gets its own copy)
        # Models  → global cache (large, reusable across runs)
        datasets_dir = self.workspace.path / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        GLOBAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if self.config.auto_download_resources:
            resources = await self._download_resources(
                search_plan, datasets_dir, GLOBAL_MODELS_DIR
            )
        else:
            self.log("Automatic resource download disabled, skipping dataset/model fetch")
            resources = []

        # Workspace directories for generated code to reference
        data_dir = datasets_dir  # datasets live here directly, no symlink needed
        models_dir = self.workspace.path / "models"
        models_dir.mkdir(exist_ok=True)

        # Verify downloads — check file sizes
        verified_resources = []
        for r in resources:
            path = r.get("path", "")
            if path and Path(path).exists():
                if Path(path).is_dir():
                    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
                else:
                    size = Path(path).stat().st_size
                r["size_bytes"] = size
                if size == 0:
                    r["status"] = "empty"
                    self.log(f"WARNING: {r['name']} downloaded but file is empty!")
            verified_resources.append(r)

        # Check if all blueprint datasets were downloaded
        blueprint_datasets = {
            (ds.get("name", "") if isinstance(ds, dict) else str(ds)).lower().strip()
            for ds in experiment_blueprint.get("datasets", [])
        }
        downloaded_names = {
            r.get("name", "").lower().strip()
            for r in verified_resources
            if r.get("status") in ("downloaded", "full", "config_only")
        }
        missing_datasets = blueprint_datasets - downloaded_names
        if missing_datasets:
            self.log(f"WARNING: Blueprint datasets not downloaded: {missing_datasets}")
            # Add explicit entries so CODING knows these are unavailable
            for name in missing_datasets:
                if not any(r.get("name", "").lower().strip() == name for r in verified_resources):
                    verified_resources.append({
                        "name": name,
                        "type": "dataset",
                        "status": "not_downloaded",
                        "error": "Not found by SETUP agent",
                    })

        # Stage only models from cache → workspace (datasets are already local)
        staged_resources, workspace_aliases = self._stage_workspace_resources(
            verified_resources,
            data_dir,
            models_dir,
        )

        # Analyze local dataset metadata for CODING stage
        resource_metadata = {}
        for r in verified_resources:
            if r.get("type") == "dataset" and r.get("status") in ("downloaded", "found_locally"):
                path = r.get("path", "")
                name = r.get("name", "")
                if path and Path(path).exists():
                    metadata = self._extract_dataset_metadata(Path(path), name)
                    if metadata:
                        resource_metadata[name] = metadata

        # Format resources for CODING stage prompt
        resource_paths = []
        for r in staged_resources:
            if r.get("type") == "model":
                resource_paths.append(f"  - [model] {r['name']}: {r['workspace_path']} ({r.get('size_bytes', 0) / 1024 / 1024:.1f} MB)" if r.get('size_bytes') else f"  - [model] {r['name']}: {r['workspace_path']}")
            elif r.get("type") == "dataset":
                resource_paths.append(f"  - [dataset] {r['name']}: {r['workspace_path']}")

        # Format unavailable datasets
        unavailable = []
        for r in staged_resources:
            if r.get("type") == "dataset" and r.get("status") not in ("downloaded", "found_locally"):
                error = r.get("error", "Not available")
                unavailable.append(f"  - [dataset] {r['name']}: NOT AVAILABLE ({error})")

        result = {
            "search_plan": search_plan,
            "cloned_repos": cloned_repos,
            "code_analysis": code_analysis,
            "downloaded_resources": staged_resources,
            "datasets_dir": str(datasets_dir),
            "data_dir": str(data_dir),
            "models_dir": str(models_dir),
            "cache_data_dir": str(GLOBAL_DATA_DIR),
            "cache_models_dir": str(GLOBAL_MODELS_DIR),
            "workspace_resource_aliases": workspace_aliases,
            "resource_download_enabled": self.config.auto_download_resources,
            "resource_metadata": resource_metadata,
            # Format for CODING prompt
            "available_resources": resource_paths,
            "unavailable_resources": unavailable,
        }

        self.workspace.write_json("plans/setup_output.json", result)
        return result

    @staticmethod
    def _safe_alias_name(value: str, fallback: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
        return normalized or fallback

    @staticmethod
    def _stage_path(source: Path, dest: Path) -> str:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            return "existing"

        if source.is_dir():
            try:
                os.symlink(source, dest, target_is_directory=True)
                return "symlink"
            except OSError:
                shutil.copytree(source, dest)
                return "copytree"

        try:
            os.link(source, dest)
            return "hardlink"
        except OSError:
            try:
                os.symlink(source, dest)
                return "symlink"
            except OSError:
                shutil.copy2(source, dest)
                return "copy"

    @classmethod
    def _stage_workspace_resources(
        cls,
        resources: list[dict],
        data_dir: Path,
        models_dir: Path,
    ) -> tuple[list[dict], list[dict]]:
        staged_resources: list[dict] = []
        workspace_aliases: list[dict] = []

        for resource in resources:
            staged = dict(resource)
            status = str(resource.get("status", "")).strip()
            source_path = str(resource.get("path", "")).strip()
            resource_type = str(resource.get("type", "dataset")).strip().lower()
            target_root = models_dir if resource_type == "model" else data_dir

            if status not in SUCCESS_RESOURCE_STATUSES or not source_path:
                # For non-successful resources, still set workspace_path to indicate the expected location
                staged["workspace_path"] = str(target_root)
                staged_resources.append(staged)
                continue

            source = Path(source_path)
            if not source.exists():
                # Source doesn't exist - still set workspace_path
                staged["workspace_path"] = str(target_root)
                staged_resources.append(staged)
                continue

            alias_details: dict[str, Any] = {
                "name": staged.get("name", ""),
                "type": resource_type,
                "cache_path": str(source),
            }

            if source.is_dir() and staged.get("files"):
                staged_file_paths: list[str] = []
                strategies: list[str] = []
                for file_name in staged.get("files", []):
                    candidate = source / str(file_name)
                    if not candidate.exists():
                        continue
                    dest = target_root / candidate.name
                    strategy = cls._stage_path(candidate, dest)
                    staged_file_paths.append(str(dest))
                    strategies.append(strategy)

                if staged_file_paths:
                    staged["cache_path"] = str(source)
                    staged["path"] = str(target_root)
                    staged["workspace_path"] = str(target_root)
                    staged["workspace_files"] = staged_file_paths
                    staged["staging_strategy"] = (
                        strategies[0] if len(set(strategies)) == 1 else "mixed"
                    )
                    alias_details.update(
                        {
                            "workspace_path": str(target_root),
                            "workspace_files": staged_file_paths,
                            "staging_strategy": staged["staging_strategy"],
                        }
                    )
                    workspace_aliases.append(alias_details)
                else:
                    # No files staged, but still set workspace_path
                    staged["workspace_path"] = str(target_root)
                staged_resources.append(staged)
                continue

            alias_base = source.name or cls._safe_alias_name(
                str(staged.get("name", "resource")),
                "resource",
            )
            dest = target_root / alias_base
            strategy = cls._stage_path(source, dest)

            staged["cache_path"] = str(source)
            staged["path"] = str(dest)
            staged["workspace_path"] = str(dest)
            staged["staging_strategy"] = strategy
            alias_details.update(
                {
                    "workspace_path": str(dest),
                    "staging_strategy": strategy,
                }
            )
            workspace_aliases.append(alias_details)
            staged_resources.append(staged)

        return staged_resources, workspace_aliases

    def _augment_search_plan_with_blueprint_resources(
        self,
        search_plan: dict,
        blueprint: dict,
    ) -> dict:
        """Backfill downloadable dataset entries directly from the blueprint.

        Also prepends local resources if they exist (priority 1).
        """
        merged = dict(search_plan or {})
        datasets = list(merged.get("datasets", []))
        seen = {
            str(entry.get("name", "")).strip().lower()
            for entry in datasets
            if isinstance(entry, dict)
        }

        # Priority 1: Add local resources first (if they exist)
        local_datasets_dir = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_datasets")
        self.log(f"[SETUP] Checking local resources in: {local_datasets_dir}")
        self.log(f"[SETUP] Local dir exists: {local_datasets_dir.exists()}")
        if local_datasets_dir.exists():
            local_contents = list(local_datasets_dir.iterdir())
            self.log(f"[SETUP] Local dir contents: {local_contents}")
            self.log(f"[SETUP] Found {len(local_contents)} local dataset directories")
        for dataset in blueprint.get("datasets", []):
            if not isinstance(dataset, dict):
                continue
            name = str(dataset.get("name", "")).strip()
            if not name or name.lower() in seen:
                continue

            # Check for local resource (case-insensitive)
            search_names = [
                name.lower().replace(" ", "_").replace("-", "_"),
                name.lower(),
            ]
            self.log(f"[SETUP] Checking local resource for dataset '{name}' with search_names: {search_names}")
            local_path = None

            # First try exact match
            for search_name in search_names:
                candidate = local_datasets_dir / search_name
                self.log(f"[SETUP] Checking candidate: {candidate}, exists: {candidate.exists()}, is_dir: {candidate.is_dir() if candidate.exists() else False}")
                if candidate.exists() and candidate.is_dir():
                    local_path = candidate
                    self.log(f"[SETUP] Found exact match: {local_path}")
                    break

            # Then try case-insensitive match
            if local_path is None and local_datasets_dir.exists():
                self.log(f"[SETUP] Trying case-insensitive match for {search_names[0]}")
                for item in local_datasets_dir.iterdir():
                    self.log(f"[SETUP] Comparing {item.name.lower()} with {search_names[0]}")
                    if item.is_dir() and item.name.lower() == search_names[0]:
                        local_path = item
                        self.log(f"[SETUP] Found case-insensitive match: {local_path}")
                        break

            # Fuzzy containment match: e.g. "cub_200" matches "CUB_200_2011"
            if local_path is None and local_datasets_dir.exists():
                local_path = self._fuzzy_match_local_dataset(
                    name, local_datasets_dir
                )
                if local_path:
                    self.log(f"[SETUP] Found fuzzy match: {local_path}")

            if local_path is not None and local_path.exists():
                # Found local resource - insert at beginning with highest priority
                datasets.insert(0, {
                    "name": name,
                    "url": "LOCAL_RESOURCE",
                    "filename": str(local_path),
                    "source": "local_resource",
                    "local_path": str(local_path),
                })
                seen.add(name.lower())
                self.log(f"Priority 1: Using local resource for {name}: {local_path}")
                continue  # Skip to next dataset (don't add again below)

        # Priority 2: Add blueprint datasets with URL correction
        for dataset in blueprint.get("datasets", []):
            if not isinstance(dataset, dict):
                continue
            name = str(dataset.get("name", "")).strip()
            if not name or name.lower() in seen:
                continue
            source_url = str(dataset.get("source_url", "")).strip()

            # Check if it's a webpage URL (not direct download)
            is_webpage = not any(
                source_url.lower().endswith(ext)
                for ext in [".gz", ".zip", ".tar", ".tgz", ".bz2", ".pt", ".pth"]
            )

            if is_webpage:
                # Use KNOWN_DATASET_URLS instead
                known_urls = {
                    "cub-200-2011": ("https://www.cs.cornell.edu/~kb/cub-200-2011.tar.gz",
                                     "CUB_200_2011.tgz"),
                    "cub200": ("https://www.cs.cornell.edu/~kb/cub-200-2011.tar.gz",
                               "CUB_200_2011.tgz"),
                    "fgvc-aircraft": ("https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
                                      "fgvc-aircraft-images.tar.bz2"),
                    "fgvc-aircraft-2013b": ("https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
                                            "fgvc-aircraft-images.tar.bz2"),
                }
                lower_name = name.lower()
                if lower_name in known_urls:
                    real_url, real_filename = known_urls[lower_name]
                    datasets.append({
                        "name": name,
                        "url": real_url,
                        "filename": real_filename,
                        "source": "corrected_url",
                    })
                    logger.info("Priority 2: Corrected webpage URL to direct download for %s", name)
                    seen.add(lower_name)
                    continue

            if not source_url.startswith(("http://", "https://")):
                continue

            filename = source_url.split("/")[-1].split("?")[0] or f"{name.lower().replace(' ', '_')}.dat"
            datasets.append({
                "name": name,
                "url": source_url,
                "filename": filename,
                "source": "blueprint",
            })
            seen.add(name.lower())

        merged["datasets"] = datasets
        return merged

    async def _plan_search(
        self, topic: str, ideation: dict, blueprint: dict
    ) -> dict:
        """Use LLM to plan what to search, clone, and download.

        With strong fallbacks for when LLM returns empty response.
        """
        system_prompt = (
            "You are a research engineer planning the setup phase for a deep learning experiment. "
            "Given a research topic and experiment blueprint, determine:\n"
            "1. What GitHub repos to search for (relevant codebases to build upon)\n"
            "2. What pretrained models to download (e.g., ESM, ProtBERT from HuggingFace)\n"
            "3. What datasets to download\n\n"
            "For datasets, you can provide:\n"
            "  - Direct download URLs (preferred): https://example.com/data.zip\n"
            "  - GitHub repo URLs: https://github.com/owner/dataset-repo (we will clone it "
            "and automatically extract real download links from README/scripts)\n"
            "  - wget/curl commands: wget https://... -O file.gz\n"
            "  - HuggingFace dataset URLs: https://huggingface.co/datasets/owner/name\n"
            "For models, use HuggingFace model IDs.\n"
            "IMPORTANT: Return ONLY valid JSON with ```json code fences. Do not add any text outside the code block."
        )

        method = blueprint.get("proposed_method", {})
        datasets = blueprint.get("datasets", [])
        hypothesis = ideation.get("selected_hypothesis", "")
        rationale = ideation.get("rationale", "")

        # Build explicit dataset checklist from blueprint
        dataset_checklist = ""
        for ds in datasets:
            if isinstance(ds, dict):
                name = ds.get("name", "")
                url = ds.get("source_url", "")
                dataset_checklist += f"  - {name} (known url: {url or 'FIND URL'})\n"
            else:
                dataset_checklist += f"  - {ds}\n"

        # Build local resource info for the LLM
        local_datasets_info = self._load_local_datasets_info()
        local_models_info = self._load_local_models_info()
        local_resource_lines = []
        if local_datasets_info:
            local_resource_lines.append("AVAILABLE LOCAL DATASETS (use LOCAL_RESOURCE as URL):")
            for key, info in local_datasets_info.items():
                local_resource_lines.append(
                    f"  - {info['name']} at {info['path']}"
                )
        if local_models_info:
            local_resource_lines.append("AVAILABLE LOCAL MODELS:")
            for key, info in local_models_info.items():
                local_resource_lines.append(
                    f"  - {info['name']} at {info['path']}"
                )
        local_resource_block = "\n".join(local_resource_lines) if local_resource_lines else ""

        user_prompt = f"""TOPIC: {topic}

HYPOTHESIS: {hypothesis}
RATIONALE: {rationale}

PROPOSED METHOD: {json.dumps(method, indent=2)[:1000]}

DATASETS REQUIRED (from blueprint):
{dataset_checklist}

{local_resource_block}

IMPORTANT INSTRUCTIONS:
1. You MUST include ALL datasets from the checklist above
2. If a dataset is available locally (listed above), set its URL to "LOCAL_RESOURCE"
   and use the filename as the local path from the listing
3. For other datasets, provide a URL that is either:
   - Direct download URL (e.g., https://example.com/data.zip)
   - GitHub repo URL (e.g., https://github.com/owner/dataset-repo)
   - HuggingFace dataset URL (e.g., https://huggingface.co/datasets/owner/name)
4. If you don't know a URL, provide the most likely GitHub repo name

Return ONLY a JSON object in this exact format with ```json code fences:
```json
{{
  "github_queries": ["query1", "query2", "query3"],
  "target_repos": [{{"owner": "...", "repo": "...", "reason": "..."}}],
  "pretrained_models": [{{"name": "...", "source": "huggingface", "model_id": "...", "download_weights": true, "reason": "..."}}],
  "datasets": [{{"name": "...", "url": "...", "filename": "...", "reason": "..."}}]
}}
```"""

        result = await self._plan_search_with_fallback(
            system_prompt, user_prompt, blueprint, datasets
        )
        return result

    @staticmethod
    def _fuzzy_match_local_dataset(
        name: str, local_dir: Path
    ) -> Path | None:
        """Fuzzy-match a dataset name against local directory entries.

        Supports:
        - Containment: "cub_200" matches "CUB_200_2011"
        - Core keyword: "cub" or "birds" matches "CUB_200_2011"
        - Normalized comparison with stripped delimiters
        """
        if not local_dir.exists():
            return None

        # Normalize the search name
        norm = name.lower().replace(" ", "_").replace("-", "_")
        # Also create a version without trailing version numbers
        norm_no_version = re.sub(r'[_-]?\d+$', '', norm)
        # Create a version with no delimiters at all
        norm_flat = re.sub(r'[_\-\s]+', '', name.lower())

        best_match: Path | None = None
        best_score = 0

        for item in local_dir.iterdir():
            if not item.is_dir():
                continue

            item_norm = item.name.lower().replace(" ", "_").replace("-", "_")
            item_flat = re.sub(r'[_\-\s]+', '', item.name.lower())

            score = 0

            # Exact match (already handled, but for completeness)
            if item_norm == norm:
                return item

            # Containment: search name is substring of local dir name
            if norm in item_norm or item_norm in norm:
                score = max(score, len(norm))

            # Containment with stripped version
            if norm_no_version and len(norm_no_version) >= 3:
                if norm_no_version in item_norm or item_norm.startswith(norm_no_version):
                    score = max(score, len(norm_no_version))

            # Flat comparison (no delimiters)
            if norm_flat in item_flat or item_flat in norm_flat:
                score = max(score, len(norm_flat))

            if score > best_score:
                best_score = score
                best_match = item

        # Only return if match score is meaningful (at least 3 chars matched)
        return best_match if best_score >= 3 else None

    def _load_local_datasets_info(self) -> dict[str, dict]:
        """Load available local datasets from DATASETS.md."""
        local_datasets_dir = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_datasets")
        datasets_info = {}

        if not local_datasets_dir.exists():
            return datasets_info

        # Check each subdirectory for datasets
        for item in local_datasets_dir.iterdir():
            if not item.is_dir():
                continue

            dataset_name = item.name
            datasets_info[dataset_name.lower().replace(" ", "_").replace("-", "_")] = {
                "name": dataset_name,
                "path": str(item),
                "type": "dataset",
            }

            # Try to read dataset-specific README if exists
            readme_path = item / "README.md"
            if readme_path.exists():
                try:
                    datasets_info[dataset_name.lower().replace(" ", "_").replace("-", "_")][
                        "description"
                    ] = readme_path.read_text()[:2000]
                except Exception as e:
                    self.log(f"Failed to read {readme_path}: {e}")

        return datasets_info

    def _load_local_models_info(self) -> dict[str, dict]:
        """Load available local models from MODELS.md."""
        local_models_dir = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/local_models")
        models_info = {}

        if not local_models_dir.exists():
            return models_info

        # Check each subdirectory for models
        for item in local_models_dir.iterdir():
            if not item.is_dir():
                continue

            model_name = item.name
            models_info[model_name.lower().replace(" ", "_").replace("-", "_")] = {
                "name": model_name,
                "path": str(item),
                "type": "model",
            }

            # Try to read model-specific README if exists
            readme_path = item / "README.md"
            if readme_path.exists():
                try:
                    models_info[model_name.lower().replace(" ", "_").replace("-", "_")][
                        "description"
                    ] = readme_path.read_text()[:2000]
                except Exception as e:
                    self.log(f"Failed to read {readme_path}: {e}")

        return models_info

    async def _plan_search_with_fallback(
        self,
        system_prompt: str,
        user_prompt: str,
        blueprint: dict,
        blueprint_datasets: list,
    ) -> dict:
        """Call LLM with multiple fallback strategies.

        Strategy 1: Try generate_json with markdown code block format
        Strategy 2: Try generate() without json_mode, then extract JSON
        Strategy 3: Return blueprint-based fallback
        """
        # Strategy 1: Try generate_json first
        try:
            result = await self.generate_json(system_prompt, user_prompt)
            if isinstance(result, dict) and result:
                self.log("SUCCESS: LLM returned valid search plan")
                return result
        except Exception as e:
            self.log(f"Strategy 1 (generate_json) failed: {e}")

        # Strategy 2: Try generate() without json_mode, then manually extract JSON
        try:
            self.log("Strategy 2: Trying generate() without json_mode...")
            raw = await self.generate(system_prompt, user_prompt, json_mode=False)
            if raw and raw.strip():
                # Try to extract JSON from markdown code blocks
                from nanoresearch.agents._base_helpers import _extract_json_candidates
                from nanoresearch.agents._base_helpers import _fix_json_escapes

                for text in _extract_json_candidates(raw):
                    try:
                        fixed = _fix_json_escapes(text)
                        result = json.loads(fixed, strict=False)
                        if isinstance(result, dict):
                            self.log("Strategy 2: Successfully extracted JSON from LLM response")
                            return result
                    except json.JSONDecodeError:
                        continue

                # If text extraction failed, try the raw response directly
                try:
                    result = json.loads(raw.strip(), strict=False)
                    if isinstance(result, dict):
                        self.log("Strategy 2: Successfully parsed raw LLM response as JSON")
                        return result
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            self.log(f"Strategy 2 (generate + extract) failed: {e}")

        # Strategy 3: Return blueprint-based fallback with known dataset URLs
        self.log("Strategy 3: Using blueprint-based fallback")
        return self._build_fallback_search_plan(blueprint_datasets)

    def _build_fallback_search_plan(self, blueprint_datasets: list) -> dict:
        """Build a search plan directly from blueprint with known dataset URLs.

        Priority order:
        1. Direct download URLs from KNOWN_DATASET_URLS (if blueprint URL is a webpage)
        2. Blueprint source_url (only if it's a direct download link)
        """
        # Known dataset download URLs (direct links, not webpages)
        KNOWN_DATASET_URLS = {
            "cub-200-2011": {
                "url": "https://www.cs.cornell.edu/~kb/cub-200-2011.tar.gz",
                "filename": "CUB_200_2011.tgz",
            },
            "cub200": {
                "url": "https://www.cs.cornell.edu/~kb/cub-200-2011.tar.gz",
                "filename": "CUB_200_2011.tgz",
            },
            "fgvc-aircraft": {
                "url": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
                "filename": "fgvc-aircraft-images.tar.bz2",
            },
            "fgvc-aircraft-2013b": {
                "url": "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz",
                "filename": "fgvc-aircraft-images.tar.bz2",
            },
            "cifar100": {
                "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                "filename": "cifar-100-python.tar.gz",
            },
            "cifar-10": {
                "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                "filename": "cifar-10-python.tar.gz",
            },
        }

        def _is_direct_download_url(url: str) -> bool:
            """Check if URL is a direct download link (not a webpage)."""
            return any(
                url.lower().endswith(ext)
                for ext in [".gz", ".zip", ".tar", ".tar.gz", ".tgz", ".bz2", ".xz", ".pt", ".pth", ".h5", ".hdf5", ".npy"]
            )

        datasets = []
        for ds in blueprint_datasets:
            if isinstance(ds, dict):
                name = ds.get("name", "").lower()
                source_url = ds.get("source_url", "")
                display_name = ds.get("name", "")
            else:
                name = str(ds).lower()
                source_url = ""
                display_name = str(ds)

            url = None
            filename = None

            if name in KNOWN_DATASET_URLS and source_url:
                # Check if blueprint URL is a webpage (not direct download)
                if not _is_direct_download_url(source_url):
                    # Use KNOWN_DATASET_URLS instead
                    known = KNOWN_DATASET_URLS[name]
                    url = known["url"]
                    filename = known["filename"]
                    self.log(f"Blueprint URL is a webpage, using KNOWN_DATASET_URL for {display_name}")
                else:
                    url = source_url
                    filename = source_url.split("/")[-1].split("?")[0]
            elif name in KNOWN_DATASET_URLS:
                # No source_url, use KNOWN_DATASET_URLS
                known = KNOWN_DATASET_URLS[name]
                url = known["url"]
                filename = known["filename"]
            elif source_url:
                url = source_url
                filename = source_url.split("/")[-1].split("?")[0]

            if name:
                datasets.append({
                    "name": display_name,
                    "url": url or "NOT_AVAILABLE",
                    "filename": filename or f"{name.replace(' ', '_')}.data",
                    "reason": "Required by experiment blueprint",
                })

        self.log(f"Built fallback search plan with {len(datasets)} datasets from blueprint")
        return {
            "github_queries": ["concept bottleneck model", "structured inference vision", "CLIP fine-grained recognition"],
            "target_repos": [],
            "pretrained_models": [],
            "datasets": datasets,
        }

    def _extract_dataset_metadata(self, dataset_path: Path, name: str) -> dict | None:
        """Extract metadata from a dataset directory."""
        if not dataset_path.exists() or not dataset_path.is_dir():
            return None

        metadata = {
            "name": name,
            "path": str(dataset_path),
            "files": [],
            "structure": [],
            "labels": [],
            "class_count": 0,
        }

        # Get directory structure
        subdirs = []
        files = []
        for item in dataset_path.iterdir():
            if item.is_dir():
                count = len(list(item.iterdir()))
                subdirs.append({"name": item.name, "file_count": count})
                if count > 0:
                    metadata["class_count"] = max(metadata["class_count"], count)
            elif item.is_file():
                files.append({"name": item.name, "size": item.stat().st_size})

        metadata["structure"] = subdirs[:20]
        metadata["files"] = files[:20]

        # Try to find labels file
        label_candidates = ["labels", "train_labels", "class_labels", "classnames", "classes"]
        labels_file = None
        for label_name in label_candidates:
            candidate = dataset_path / label_name
            if candidate.exists():
                labels_file = candidate
                break

        # Also check with extensions
        if not labels_file:
            for ext in [".txt", ".csv", ".json"]:
                candidate = dataset_path / f"labels{ext}"
                if candidate.exists():
                    labels_file = candidate
                    break

        if labels_file:
            try:
                content = labels_file.read_text(errors="replace")
                lines = content.strip().split("\n")[:200]
                metadata["labels_preview"] = lines[:50]
                # Try to extract unique labels
                if lines and (" " in lines[0] or "," in lines[0]):
                    unique_labels = []
                    for line in lines:
                        if " " in line:
                            label = line.split()[-1]
                        elif "," in line:
                            label = line.split(",")[-1]
                        else:
                            label = line
                        if label and label not in unique_labels:
                            unique_labels.append(label)
                    metadata["unique_labels"] = unique_labels[:100]
                    metadata["class_count"] = len(unique_labels)
            except Exception as e:
                metadata["labels_error"] = str(e)

        # Check for common dataset splits
        for split in ["train", "val", "test", "eval"]:
            split_path = dataset_path / split
            if split_path.exists() and split_path.is_dir():
                metadata[f"{split}_count"] = len(list(split_path.iterdir()))

        return metadata if metadata["structure"] or metadata["files"] else None
