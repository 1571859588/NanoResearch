"""Setup agent search and download mixin."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import re
import shlex
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class _SetupSearchMixin:
    """Mixin — search, clone, analyze, and download resources."""

    async def _search_and_clone(self, search_plan: dict) -> list[dict]:
        """Search GitHub and clone relevant repos."""
        cloned = []
        repos_dir = self.workspace.path / "repos"
        repos_dir.mkdir(exist_ok=True)

        # Clone specific target repos first
        for repo_info in search_plan.get("target_repos", [])[:3]:
            owner = repo_info.get("owner", "")
            repo = repo_info.get("repo", "")
            if not owner or not repo:
                continue
            # Sanitize owner/repo to prevent command injection
            if not re.match(r'^[a-zA-Z0-9._-]+$', owner) or not re.match(r'^[a-zA-Z0-9._-]+$', repo):
                self.log(f"Skipping unsafe repo name: {owner}/{repo}")
                continue
            clone_url = f"https://github.com/{owner}/{repo}.git"
            dest = repos_dir / repo
            if dest.exists():
                cloned.append({"name": repo, "path": str(dest), "source": clone_url})
                continue
            try:
                result = await self._run_shell(
                    f"git clone --depth 1 {shlex.quote(clone_url)} {shlex.quote(str(dest))}", timeout=120
                )
                if dest.exists():
                    cloned.append({"name": repo, "path": str(dest), "source": clone_url})
                    self.log(f"Cloned {owner}/{repo}")
            except Exception as e:
                self.log(f"Failed to clone {owner}/{repo}: {e}")

        # Search GitHub API for additional repos
        for query in search_plan.get("github_queries", [])[:3]:
            if len(cloned) >= 5:
                break
            try:
                repos = await self._github_search(query)
                for r in repos[:2]:
                    if len(cloned) >= 5:
                        break
                    name = r.get("name", "")
                    clone_url = r.get("clone_url", "")
                    if not clone_url or (repos_dir / name).exists():
                        continue
                    dest = repos_dir / name
                    await self._run_shell(
                        f"git clone --depth 1 {shlex.quote(clone_url)} {shlex.quote(str(dest))}", timeout=120
                    )
                    if dest.exists():
                        cloned.append({
                            "name": name,
                            "path": str(dest),
                            "source": clone_url,
                            "stars": r.get("stargazers_count", 0),
                            "description": r.get("description", ""),
                        })
                        self.log(f"Cloned {name} ({r.get('stargazers_count', 0)} stars)")
            except Exception as e:
                self.log(f"GitHub search failed for '{query}': {e}")

        return cloned

    async def _github_search(self, query: str) -> list[dict]:
        """Search GitHub repos via API."""
        import urllib.parse
        encoded = urllib.parse.quote(query)
        full_url = (
            f"https://api.github.com/search/repositories"
            f"?q={encoded}&sort=stars&per_page=5&order=desc"
        )
        cmd = f"curl -s {shlex.quote(full_url)}"
        result = await self._run_shell(cmd, timeout=30)
        stdout = result.get("stdout", "")
        try:
            data = json.loads(stdout)
            return data.get("items", [])
        except json.JSONDecodeError:
            return []

    async def _analyze_cloned_code(
        self, cloned_repos: list[dict], blueprint: dict
    ) -> dict:
        """Analyze cloned repos to understand their structure and key components."""
        if not cloned_repos:
            return {"summary": "No repos cloned", "key_files": [], "reusable_components": []}

        # Collect file listings and key files from each repo
        repo_summaries = []
        for repo in cloned_repos[:3]:
            repo_path = Path(repo["path"])
            tree_result = await self._run_shell(
                f"find {repo_path} -maxdepth 3 -type f -name '*.py' | head -50",
                timeout=10,
            )
            files = tree_result.get("stdout", "").strip().split("\n")[:50]

            readme_content = ""
            for readme_name in ["README.md", "readme.md", "README.rst"]:
                readme_path = repo_path / readme_name
                if readme_path.exists():
                    readme_content = readme_path.read_text(errors="replace")[:3000]
                    break

            key_snippets = []
            for f in files:
                fname = Path(f).name.lower()
                if any(kw in fname for kw in ["model", "train", "config", "main", "run"]):
                    try:
                        content = Path(f).read_text(errors="replace")[:2000]
                        key_snippets.append({"file": f, "content": content})
                    except Exception as exc:
                        logger.debug("Failed to read repo snippet %s: %s", f, exc)
                    if len(key_snippets) >= 5:
                        break

            repo_summaries.append({
                "name": repo["name"],
                "files": files[:30],
                "readme": readme_content[:1500],
                "key_snippets": key_snippets,
            })

        system_prompt = (
            "You are a ML research engineer analyzing cloned code repositories. "
            "Identify reusable components, architecture patterns, training pipelines, "
            "and suggest how to build upon this code for the proposed experiment. "
            "Return JSON only."
        )

        method = blueprint.get("proposed_method", {})
        user_prompt = f"""Proposed method: {json.dumps(method, indent=2)[:800]}

Cloned repositories:
{json.dumps(repo_summaries, indent=2)[:8000]}

Return JSON:
{{
  "summary": "Overall analysis of available code...",
  "best_base_repo": "name of most relevant repo to build upon",
  "key_files": [
    {{"repo": "...", "file": "...", "purpose": "...", "reuse_plan": "..."}}
  ],
  "reusable_components": [
    {{"name": "...", "source_file": "...", "description": "...", "modifications_needed": "..."}}
  ],
  "missing_components": ["list of things we need to implement from scratch"],
  "recommended_approach": "How to combine/extend these codebases..."
}}"""

        result = await self.generate_json(system_prompt, user_prompt)
        return result if isinstance(result, dict) else {}

    async def _extract_archive(self, archive_path: Path, extract_dir: Path) -> str:
        """Extract archive and return the path to extracted content.

        Supports: .tar.gz, .tgz, .tar.bz2, .tar.xz, .zip, .gz
        Returns the path to the extracted data (directory or file).
        """
        archive_name = archive_path.name.lower()

        # Create extract directory
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Handle .tar.gz and .tgz
        if archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
            extract_path = extract_dir / archive_path.stem  # Remove .tar.gz
            extract_path.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=extract_path)
                self.log(f"Extracted: {archive_path.name} -> {extract_path.name}/")
                return str(extract_path)
            except Exception as e:
                logger.warning(f"Failed to extract {archive_path}: {e}")

        # Handle .tar.bz2
        elif archive_name.endswith(".tar.bz2"):
            extract_path = extract_dir / archive_path.stem  # Remove .tar.bz2
            extract_path.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(path=extract_path)
                self.log(f"Extracted: {archive_path.name} -> {extract_path.name}/")
                return str(extract_path)
            except Exception as e:
                logger.warning(f"Failed to extract {archive_path}: {e}")

        # Handle .tar.xz
        elif archive_name.endswith(".tar.xz") or archive_name.endswith(".txz"):
            extract_path = extract_dir / archive_path.stem  # Remove .tar.xz
            extract_path.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(archive_path, "r:xz") as tar:
                    tar.extractall(path=extract_path)
                self.log(f"Extracted: {archive_path.name} -> {extract_path.name}/")
                return str(extract_path)
            except Exception as e:
                logger.warning(f"Failed to extract {archive_path}: {e}")

        # Handle .gz (single file decompression)
        elif archive_name.endswith(".gz") and not archive_name.endswith(".tar.gz"):
            decompressed = extract_dir / archive_path.stem  # Remove .gz
            try:
                with gzip.open(archive_path, "rb") as f_in:
                    with open(decompressed, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                self.log(f"Decompressed: {archive_path.name} -> {decompressed.name}")
                return str(decompressed)
            except Exception as e:
                logger.warning(f"Failed to decompress {archive_path}: {e}")

        # Handle .zip
        elif archive_name.endswith(".zip"):
            try:
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                self.log(f"Extracted: {archive_path.name} -> {extract_dir.name}/")
                return str(extract_dir)
            except zipfile.BadZipFile as e:
                logger.warning(f"Not a valid zip file: {archive_path}: {e}")
            except Exception as e:
                logger.warning(f"Failed to extract {archive_path}: {e}")

        return str(archive_path)

    def _load_manifest_local_paths(self, workspace_path: Path) -> list[str]:
        """Load per-project local_resource_paths from manifest.json.

        Args:
            workspace_path: Path to the workspace directory containing manifest.json

        Returns:
            List of local resource paths configured for this project
        """
        manifest_path = workspace_path / "manifest.json"
        if not manifest_path.exists():
            return []

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = data.get("local_resource_paths", [])
            if isinstance(paths, list):
                return paths
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _get_auto_scan_paths(self) -> list[Path]:
        """Get automatically scanned local resource paths.

        When auto_scan_local_resources is enabled, these directories
        are automatically checked for supplementary datasets/models.

        Customizable via config:
        - local_datasets_dir: e.g., "/mnt/public/.../local_data" (default: ~/.nanoresearch/local_datasets)
        - local_models_dir: e.g., "/mnt/public/.../local_data" (default: ~/.nanoresearch/local_models)

        Returns:
            List of paths to auto-scan
        """
        if not getattr(self.config, "auto_scan_local_resources", True):
            return []

        paths = []

        # Use configured paths or fall back to defaults
        datasets_dir = self.config.local_datasets_dir
        if datasets_dir:
            datasets_path = Path(datasets_dir) / "datasets"
        else:
            datasets_path = Path.home() / ".nanoresearch" / "local_datasets"

        models_dir = self.config.local_models_dir
        if models_dir:
            models_path = Path(models_dir) / "models"
        else:
            models_path = Path.home() / ".nanoresearch" / "local_models"

        # Add paths if they exist
        if datasets_path.exists():
            paths.append(datasets_path)
        if models_path.exists():
            paths.append(models_path)

        return paths

    def _find_resource_in_local_paths(
        self,
        resource_name: str,
        resource_type: str,
        data_dir: Path,
        models_dir: Path,
        max_candidates: int = 3,
    ) -> list[str]:
        """Search for a resource in configured local paths.

        Priority order:
        1. Per-project local_resource_paths (from manifest.json) - HIGHEST priority
        2. Auto-scan default directories (if enabled):
           - ~/.nanoresearch/local_datasets/
           - ~/.nanoresearch/local_models/
        3. Project workspace models/datasets subdirectories
        4. Global local_resource_paths (from config.json)
        5. Workspace data_dir/models_dir

        Returns all candidates found, up to max_candidates.

        Args:
            resource_name: Name/identifier of the resource (e.g., "cub200", "clip-vit-b")
            resource_type: "dataset" or "model"
            data_dir: Workspace data directory
            models_dir: Workspace models directory
            max_candidates: Maximum number of candidates to return (for LLM verification)

        Returns:
            List of paths to resources if found, empty list otherwise.
        """
        # Normalize resource name for matching
        search_key = resource_name.lower().replace(" ", "_").replace("-", "_")

        # Build list of paths to search (prioritized)
        search_paths = []
        workspace_path = data_dir.parent  # workspace root

        # Priority 1: Per-project local_resource_paths (from manifest.json)
        project_paths = self._load_manifest_local_paths(workspace_path)
        for path_str in project_paths:
            if path_str:
                search_paths.append(Path(path_str))

        # Priority 2: Auto-scan default directories
        auto_paths = self._get_auto_scan_paths()
        search_paths.extend(auto_paths)

        # Priority 3: Project-specific models/datasets subdirectories
        if resource_type == "dataset":
            project_data_subdir = workspace_path / "datasets"
            if project_data_subdir.exists():
                search_paths.append(project_data_subdir)
        else:  # model
            project_models_subdir = workspace_path / "models"
            if project_models_subdir.exists():
                search_paths.append(project_models_subdir)

        # Priority 4: Global local_resource_paths (from config.json)
        for path_str in getattr(self.config, "local_resource_paths", []):
            if path_str:
                search_paths.append(Path(path_str))

        # Priority 5: Workspace directories
        if resource_type == "dataset":
            search_paths.append(data_dir)
            search_paths.append(data_dir.parent / "datasets" if data_dir.parent.exists() else data_dir)
        else:  # model
            search_paths.append(models_dir)
            search_paths.append(models_dir.parent / "models" if models_dir.parent.exists() else models_dir)

        # Search each path and collect candidates
        candidates = []
        seen_paths = set()
        for search_path in search_paths:
            if not search_path.exists():
                continue

            found = self._search_path_for_resource(search_path, search_key, resource_name, resource_type)
            if found and str(found) not in seen_paths:
                self.log(f"Found {resource_type} '{resource_name}' in local path: {found}")
                candidates.append(str(found))
                seen_paths.add(str(found))
                if len(candidates) >= max_candidates:
                    break

        return candidates

    def _search_path_for_resource(
        self,
        root_path: Path,
        search_key: str,
        resource_name: str,
        resource_type: str,
    ) -> Path | None:
        """Recursively search a path for a resource.

        Looks for matches by:
        1. Exact filename match
        2. Normalized key match (lowercase, underscores)
        3. Partial name match
        """
        root_path = root_path.resolve()

        # Common resource names to look for
        keywords = [
            search_key,
            resource_name.lower(),
            resource_name.replace(" ", "_").lower(),
            resource_name.replace("-", "_").lower(),
        ]

        # Remove common prefixes/suffixes
        clean_name = search_key.strip("_-").strip(".")

        for item in root_path.rglob("*"):
            item_name = item.name.lower()
            item_stem = item.stem.lower() if item.suffix else item_name

            # Skip archives (we want extracted content)
            if any(item_name.endswith(ext) for ext in [".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".gz", ".zip"]):
                continue

            # Check exact match
            if item_name in keywords or item_stem in keywords:
                if item.is_file() and not item.suffix.startswith("."):
                    return item
                if item.is_dir():
                    # Check if directory contains expected files
                    if self._is_valid_resource_dir(item, resource_type):
                        return item

            # Check partial match
            for kw in keywords:
                if kw in item_name or kw in item_stem:
                    if item.is_file() and not item.suffix.startswith("."):
                        return item
                    if item.is_dir():
                        if self._is_valid_resource_dir(item, resource_type):
                            return item

        return None

    def _is_valid_resource_dir(self, dir_path: Path, resource_type: str) -> bool:
        """Check if a directory appears to contain a valid resource.

        A valid resource directory should contain actual data files,
        not just subdirectories or config files.
        """
        if not dir_path.is_dir():
            return False

        # Count actual data files (skip common non-data patterns)
        data_patterns = {
            ".csv", ".txt", ".json", ".jsonl", ".pkl", ".parquet",
            ".npy", ".npz", ".pt", ".pth", ".bin", ".h5", ".hdf5",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp",
            ".txt", ".md", ".rst",
        }

        file_count = 0
        for item in dir_path.iterdir():
            if item.is_file() and item.suffix.lower() in data_patterns:
                file_count += 1
            elif item.is_file() and not any(
                item.name.lower().startswith(x) for x in [".git", "__", ".pyc", ".pyo"]
            ):
                file_count += 1

        # Consider it valid if we found enough data files
        return file_count >= 3  # At least 3 data files

    async def _download_resources(
        self, search_plan: dict, data_dir: Path, models_dir: Path
    ) -> list[dict]:
        """Download pretrained models and datasets to global cache.

        Download priority for models:
        1. LLM verification of local resources (if enabled)
        2. Check if already cached (skip download)
        3. Try ModelScope first (faster in China)
        4. Fall back to HuggingFace
        """
        downloaded = []

        # Download pretrained models
        for model_info in search_plan.get("pretrained_models", []):
            name = model_info.get("name", "unknown")
            model_id = model_info.get("model_id", "")
            source = model_info.get("source", "")
            download_weights = model_info.get("download_weights", True)

            if not model_id:
                continue

            safe_name = name.replace("/", "_").replace(" ", "_")
            dest = models_dir / safe_name

            # Priority 1: Find and verify local resources with LLM (if enabled)
            # TEMP: Disabled by default due to LLM empty response issues
            should_verify_llm = getattr(self.config, "verify_local_resources_with_llm", False)

            # Find all candidates
            all_candidates = self._find_resource_in_local_paths(
                name, "model", data_dir, models_dir, max_candidates=5
            )

            if all_candidates:
                verified_path = None
                if should_verify_llm:
                    self.log(f"Running LLM verification for local model '{name}'")
                    # Use LLM to verify the resource matches the requirement
                    verified_path = await self._verify_local_resource_with_llm(
                        required_name=name,
                        resource_type="model",
                        candidate_paths=all_candidates,
                        requirement=model_info.get("reason", ""),
                    )
                else:
                    # Skip LLM verification, use first candidate
                    verified_path = all_candidates[0]
                    self.log(f"Skipping LLM verification (disabled), using first candidate: {verified_path}")

                if verified_path:
                    # Copy to workspace instead of direct reference
                    copied_path = await self._copy_resource_to_workspace(
                        source_path=Path(verified_path),
                        dest_dir=models_dir,
                        resource_name=safe_name,
                        resource_type="model",
                    )
                    self.log(f"Verified and copied local model '{name}' to: {copied_path}")
                    downloaded.append({
                        "name": name, "type": "model",
                        "path": str(copied_path), "source": "local_resource",
                        "status": "found_locally", "cached": True,
                        "local_source": verified_path,
                    })
                    continue

            # Priority 2: Check if already cached in workspace
            if dest.exists() and any(dest.iterdir()):
                existing_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                if existing_size > 1000:  # more than just a few config files
                    self.log(f"Model already cached: {model_id} ({existing_size / 1024 / 1024:.0f} MB)")
                    status = "full" if existing_size > 100_000_000 else "config_only"
                    downloaded.append({
                        "name": name, "type": "model",
                        "path": str(dest), "source": model_id,
                        "status": status, "cached": True,
                    })
                    continue

            dest.mkdir(parents=True, exist_ok=True)
            self.log(f"Downloading model: {model_id}")

            # BUG-20 fix: validate model_id format before passing to shell.
            # Only allow characters valid in HuggingFace/ModelScope IDs.
            _MODEL_ID_RE = re.compile(r"^[a-zA-Z0-9_\-./]+$")
            if not _MODEL_ID_RE.match(model_id):
                self.log(f"Invalid model_id format, skipping: {model_id!r}")
                downloaded.append({
                    "name": name, "type": "model",
                    "path": str(dest), "source": model_id,
                    "status": "failed", "error": "invalid model_id format",
                })
                continue

            # Try ModelScope first (convert HuggingFace ID to ModelScope format)
            modelscope_id = await self._hf_to_modelscope_id(model_id)
            success = False

            # BUG-20 fix: pass model_id/dest via environment variables
            # instead of embedding in f-string python code, preventing
            # shell/Python injection from untrusted LLM-generated IDs.
            if modelscope_id:
                if not _MODEL_ID_RE.match(modelscope_id):
                    self.log(f"Invalid modelscope_id format, skipping ModelScope: {modelscope_id!r}")
                else:
                    try:
                        self.log(f"Trying ModelScope (no proxy): {modelscope_id}")
                        ms_env = {
                            "_NR_MODEL_ID": modelscope_id,
                            "_NR_CACHE_DIR": str(dest.parent),
                        }
                        if download_weights:
                            result = await self._run_shell_no_proxy(
                                'python3 -c "'
                                'import os; '
                                'from modelscope import snapshot_download; '
                                'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                                'cache_dir=os.environ[\'_NR_CACHE_DIR\'], '
                                'revision=\'master\')"',
                                timeout=1800, env=ms_env,
                            )
                        else:
                            result = await self._run_shell_no_proxy(
                                'python3 -c "'
                                'import os; '
                                'from modelscope import snapshot_download; '
                                'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                                'cache_dir=os.environ[\'_NR_CACHE_DIR\'], '
                                'revision=\'master\', '
                                'ignore_file_pattern=[\'*.bin\', \'*.safetensors\', \'*.h5\', \'*.msgpack\'])"',
                                timeout=300, env=ms_env,
                            )
                        if result.get("returncode", 1) == 0:
                            success = True
                            self.log(f"Downloaded from ModelScope: {modelscope_id}")
                    except Exception as e:
                        self.log(f"ModelScope download failed: {e}")

            # Fall back to HuggingFace (official endpoint)
            if not success:
                try:
                    self.log(f"Trying HuggingFace: {model_id}")
                    hf_env = {
                        "_NR_MODEL_ID": model_id,
                        "_NR_LOCAL_DIR": str(dest),
                    }
                    if download_weights:
                        result = await self._run_shell(
                            'python3 -c "'
                            'import os; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'])"',
                            timeout=1800, env=hf_env,
                        )
                    else:
                        result = await self._run_shell(
                            'python3 -c "'
                            'import os; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'], '
                            'ignore_patterns=[\'*.bin\', \'*.safetensors\', \'*.h5\', \'*.msgpack\'])"',
                            timeout=300, env=hf_env,
                        )
                    if result.get("returncode", 1) == 0:
                        success = True
                        self.log(f"Downloaded from HuggingFace: {model_id}")
                except Exception as e:
                    self.log(f"HuggingFace download failed: {e}")

            # Fall back to hf-mirror.com (China mirror, no proxy needed)
            if not success:
                try:
                    self.log(f"Trying hf-mirror.com: {model_id}")
                    mirror_env = {
                        "_NR_MODEL_ID": model_id,
                        "_NR_LOCAL_DIR": str(dest),
                        "HF_ENDPOINT": "https://hf-mirror.com",
                    }
                    if download_weights:
                        result = await self._run_shell_no_proxy(
                            'python3 -c "'
                            'import os; '
                            'os.environ[\'HF_ENDPOINT\'] = \'https://hf-mirror.com\'; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'])"',
                            timeout=1800, env=mirror_env,
                        )
                    else:
                        result = await self._run_shell_no_proxy(
                            'python3 -c "'
                            'import os; '
                            'os.environ[\'HF_ENDPOINT\'] = \'https://hf-mirror.com\'; '
                            'from huggingface_hub import snapshot_download; '
                            'snapshot_download(os.environ[\'_NR_MODEL_ID\'], '
                            'local_dir=os.environ[\'_NR_LOCAL_DIR\'], '
                            'ignore_patterns=[\'*.bin\', \'*.safetensors\', \'*.h5\', \'*.msgpack\'])"',
                            timeout=300, env=mirror_env,
                        )
                    if result.get("returncode", 1) == 0:
                        success = True
                        self.log(f"Downloaded from hf-mirror.com: {model_id}")
                except Exception as e:
                    self.log(f"hf-mirror download failed: {e}")

            status = "full" if (download_weights and success) else ("config_only" if success else "failed")
            downloaded.append({
                "name": name, "type": "model",
                "path": str(dest), "source": model_id,
                "status": status,
            })

        # Download datasets
        for ds_info in search_plan.get("datasets", []):
            name = ds_info.get("name", "unknown")
            url = ds_info.get("url", "") or ds_info.get("download_cmd", "")
            filename = ds_info.get("filename", "")

            if not url:
                continue

            # Check if already cached (including compressed versions)
            if filename:
                cached_file = data_dir / filename
                decompressed_paths = self._get_decompressed_paths(filename)
                cached_decompressed = None

                for decompressed_name in decompressed_paths:
                    cached_decompressed = data_dir / decompressed_name
                    if cached_decompressed.exists() and cached_decompressed.stat().st_size > 0:
                        if cached_decompressed.is_dir():
                            # Check if directory has content
                            if any(cached_decompressed.iterdir()):
                                self.log(f"Dataset already cached: {name} -> {cached_decompressed.name}/")
                                downloaded.append({
                                    "name": name, "type": "dataset",
                                    "path": str(cached_decompressed),
                                    "status": "downloaded", "cached": True,
                                })
                                cached_decompressed = None
                                break
                        else:
                            self.log(f"Dataset already cached: {name} -> {cached_decompressed.name}")
                            downloaded.append({
                                "name": name, "type": "dataset",
                                "path": str(cached_decompressed),
                                "status": "downloaded", "cached": True,
                            })
                            cached_decompressed = None
                            break

                # File exists but not yet decompressed (only check if no archive found)
                if cached_decompressed is None and cached_file.exists() and cached_file.stat().st_size > 0:
                    name_lower = filename.lower()
                    if not any(name_lower.endswith(s) for s in [".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".txz", ".gz", ".zip"]):
                        self.log(f"Dataset already cached: {name} -> {cached_file.name}")
                        downloaded.append({
                            "name": name, "type": "dataset",
                            "path": str(cached_file),
                            "status": "downloaded", "cached": True,
                        })
                        continue

            # Priority 1: Find and verify local resources with LLM (if enabled)
            # TEMP: Disabled by default due to LLM empty response issues
            should_verify_llm = getattr(self.config, "verify_local_resources_with_llm", False)

            # Find all candidates
            all_candidates = self._find_resource_in_local_paths(
                name, "dataset", data_dir, models_dir, max_candidates=5
            )

            if all_candidates:
                verified_path = None
                if should_verify_llm:
                    self.log(f"Running LLM verification for local dataset '{name}'")
                    # Use LLM to verify the resource matches the requirement
                    verified_path = await self._verify_local_resource_with_llm(
                        required_name=name,
                        resource_type="dataset",
                        candidate_paths=all_candidates,
                        requirement=f"Dataset for {name}",
                    )
                else:
                    # Skip LLM verification, use first candidate
                    verified_path = all_candidates[0]
                    self.log(f"Skipping LLM verification (disabled), using first candidate: {verified_path}")

                if verified_path:
                    # Copy to workspace instead of direct reference
                    local_path_obj = Path(verified_path)
                    if local_path_obj.is_file():
                        extracted_path = await self._extract_archive(local_path_obj, data_dir)
                        copied_path = await self._copy_resource_to_workspace(
                            source_path=Path(extracted_path),
                            dest_dir=data_dir,
                            resource_name=name,
                            resource_type="dataset",
                        )
                    else:
                        copied_path = await self._copy_resource_to_workspace(
                            source_path=Path(verified_path),
                            dest_dir=data_dir,
                            resource_name=name,
                            resource_type="dataset",
                        )
                    self.log(f"Verified and copied local dataset '{name}' to: {copied_path}")
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "path": str(copied_path), "source": "local_resource",
                        "status": "found_locally", "cached": True,
                        "local_source": verified_path,
                    })
                    continue

            self.log(f"Downloading dataset: {name}")

            # ── GitHub repo URL → clone + extract real data ──
            gh_match = self._is_github_repo_url(url)
            if gh_match:
                gh_owner, gh_repo = gh_match.group("owner"), gh_match.group("repo")
                ds_data_dir = data_dir / gh_repo
                ds_data_dir.mkdir(parents=True, exist_ok=True)
                result_entry = await self._handle_github_dataset(
                    name, gh_owner, gh_repo, ds_data_dir,
                )
                downloaded.append(result_entry)
                continue

            if url.startswith(("wget ", "curl ")):
                try:
                    # BUG-18 fix: sanitize LLM-generated download command.
                    # Tokenize with shlex and reject anything that isn't a
                    # flag or an http(s)/ftp URL to block shell injection.
                    try:
                        dl_parts = shlex.split(url)
                    except ValueError:
                        raise RuntimeError(f"Unparseable download command: {url[:200]}")
                    dl_cmd = dl_parts[0]
                    if dl_cmd not in ("wget", "curl"):
                        raise RuntimeError(f"Blocked download command: {dl_cmd}")
                    safe_dl = [dl_cmd]
                    for dl_arg in dl_parts[1:]:
                        if dl_arg.startswith("-"):
                            safe_dl.append(dl_arg)
                        elif dl_arg.startswith(("http://", "https://", "ftp://")):
                            safe_dl.append(dl_arg)
                        else:
                            logger.warning("Dropped suspicious arg in download cmd: %s", dl_arg[:120])
                    sanitized_dl = " ".join(shlex.quote(p) for p in safe_dl)
                    result = await self._run_shell(
                        f"cd {shlex.quote(str(data_dir))} && {sanitized_dl}", timeout=600
                    )
                    dl_files = list(data_dir.glob("*"))

                    # Check if downloaded a single archive file and extract it
                    extracted_path = None
                    for dl_file in dl_files:
                        if dl_file.is_file():
                            name_lower = dl_file.name.lower()
                            if any(name_lower.endswith(s) for s in [".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".txz", ".gz", ".zip"]):
                                extracted_path = await self._extract_archive(dl_file, data_dir)
                                break

                    if extracted_path:
                        downloaded.append({
                            "name": name, "type": "dataset",
                            "path": extracted_path, "status": "downloaded",
                            "extracted_from": str(dl_file),
                        })
                    else:
                        downloaded.append({
                            "name": name, "type": "dataset",
                            "path": str(data_dir), "status": "downloaded",
                            "files": [f.name for f in dl_files],
                        })
                    self.log(f"Downloaded dataset: {name}")
                except Exception as e:
                    self.log(f"Failed to download dataset {name}: {e}")
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "status": "failed", "error": str(e),
                    })
            elif url.startswith("http"):
                if not filename:
                    filename = url.split("/")[-1].split("?")[0]
                dest_file = data_dir / filename
                try:
                    result = await self._run_shell(
                        f"wget -q -O {shlex.quote(str(dest_file))} {shlex.quote(url)}", timeout=600
                    )
                    if dest_file.exists() and dest_file.stat().st_size > 0:
                        # Try to extract archive
                        extracted_path = await self._extract_archive(dest_file, data_dir)

                        if extracted_path != str(dest_file):
                            # Successfully extracted
                            downloaded.append({
                                "name": name, "type": "dataset",
                                "path": extracted_path,
                                "compressed_path": str(dest_file),
                                "status": "downloaded",
                            })
                        else:
                            # Not an archive or extraction failed
                            downloaded.append({
                                "name": name, "type": "dataset",
                                "path": str(dest_file),
                                "status": "downloaded",
                            })
                        self.log(f"Downloaded dataset: {name} -> {dest_file.name}")
                    else:
                        downloaded.append({
                            "name": name, "type": "dataset",
                            "status": "failed", "error": "Downloaded file is empty or missing",
                        })
                except Exception as e:
                    self.log(f"Failed to download dataset {name}: {e}")
                    downloaded.append({
                        "name": name, "type": "dataset",
                        "status": "failed", "error": str(e),
                    })

        return downloaded

    @staticmethod
    def _get_decompressed_paths(filename: str) -> list[str]:
        """Get list of possible decompressed filenames for a given archive.

        Returns paths in order of preference for matching.
        """
        name = filename.lower()
        paths = []

        # .tar.gz / .tgz -> remove .tar.gz or .tgz
        if name.endswith(".tar.gz"):
            paths.append(filename[:-7])  # Remove .tar.gz
        elif name.endswith(".tgz"):
            paths.append(filename[:-4])  # Remove .tgz

        # .tar.bz2 -> remove .tar.bz2
        elif name.endswith(".tar.bz2"):
            paths.append(filename[:-8])  # Remove .tar.bz2

        # .tar.xz / .txz -> remove .tar.xz or .txz
        elif name.endswith(".tar.xz"):
            paths.append(filename[:-7])  # Remove .tar.xz
        elif name.endswith(".txz"):
            paths.append(filename[:-4])  # Remove .txz

        # .gz (not tar.gz) -> remove .gz
        elif name.endswith(".gz"):
            paths.append(filename[:-3])  # Remove .gz

        # .zip -> directory name or first file name
        elif name.endswith(".zip"):
            # Try extracting directory name
            base = filename[:-4]  # Remove .zip
            paths.append(base)

        return paths

    # ------------------------------------------------------------------
    # GitHub dataset repo handling
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LLM-based local resource verification
    # ------------------------------------------------------------------

    async def _verify_local_resource_with_llm(
        self,
        required_name: str,
        resource_type: str,
        candidate_paths: list[str],
        requirement: str,
        max_preview_chars: int = 3000,
    ) -> str | None:
        """Use LLM to verify if local resources match the requirement.

        This addresses the issue that local resource names may not match
        their actual content. The LLM reads metadata files (README, description, etc.)
        to determine if the resource is appropriate.

        Context window management:
        - Each file is read with a maximum of max_preview_chars characters
        - Only metadata files are prioritized (README.md, description.txt, etc.)
        - For large files, only the beginning and key sections are read

        Args:
            required_name: Name of the required resource (e.g., "CUB-200")
            resource_type: "dataset" or "model"
            candidate_paths: List of local paths to candidates
            requirement: Description of what is needed
            max_preview_chars: Maximum characters to read per file

        Returns:
            Path to the best verified resource, or None if no match found
        """
        if not candidate_paths:
            return None

        # Collect resource metadata from each candidate
        resource_descriptions = []

        for candidate_path in candidate_paths:
            candidate_path_obj = Path(candidate_path)
            if not candidate_path_obj.exists():
                continue

            metadata = self._extract_resource_metadata(candidate_path_obj, max_preview_chars)
            resource_descriptions.append(metadata)

        if not resource_descriptions:
            return None

        # Build LLM prompt with context window management
        system_prompt = (
            "You are a research resource curator. Your task is to verify if local "
            "datasets or models match the project requirements by reading their metadata."
            " Return ONLY valid JSON, no other text."
        )

        # Limit the resource descriptions to avoid exceeding context window
        max_desc_length = 12000
        desc_json = json.dumps(resource_descriptions, indent=2, ensure_ascii=False)
        if len(desc_json) > max_desc_length:
            desc_json = desc_json[:max_desc_length] + "\n... (truncated)"

        user_prompt = f"""Project requires: {required_name}
Requirement: {requirement}
Resource type: {resource_type}

I found the following local candidates. Please analyze each one's metadata
(README, descriptions, etc.) and determine if it matches the requirement.

Candidate resources:
{desc_json}

Return JSON only in this exact format:
{{
  "best_match": "full_path of best matching resource or null if none",
  "confidence": 0.0 to 1.0,
  "reason": "short explanation",
  "alternative_suggestions": []
}}
"""

        try:
            result = await self.generate_json(system_prompt, user_prompt)
            if isinstance(result, dict) and result.get("best_match"):
                best_match = result["best_match"]
                confidence = result.get("confidence", 0.0)
                # Only accept if confidence is above threshold
                if confidence >= 0.5:
                    self.log(f"LLM verified '{required_name}' with confidence {confidence:.2f}: {best_match}")
                    return best_match
                else:
                    self.log(f"LLM verification low confidence ({confidence:.2f}) for '{required_name}'")
            else:
                self.log(f"LLM verification returned unexpected format: {result}")
        except Exception as e:
            self.log(f"LLM verification failed for '{required_name}': {e}")

        # Fallback: return first candidate if LLM verification fails
        if candidate_paths:
            self.log(f"Fallback: using first candidate as verification failed: {candidate_paths[0]}")
            return candidate_paths[0]
        return None

    def _extract_resource_metadata(self, resource_path: Path, max_chars: int = 3000) -> dict:
        """Extract metadata from a resource directory or file.

        Prioritizes metadata files and limits read size to manage context window.

        Args:
            resource_path: Path to resource (file or directory)
            max_chars: Maximum characters to read per file

        Returns:
            Dictionary with resource metadata
        """
        metadata = {
            "path": str(resource_path),
            "name": resource_path.name,
            "type": "directory" if resource_path.is_dir() else "file",
            "files": [],
            "preview": {},
        }

        if resource_path.is_file():
            # For files, just read the content
            try:
                content = resource_path.read_text(errors="replace")[:max_chars]
                metadata["preview"] = {"content": content}
            except Exception:
                metadata["preview"] = {"error": "Failed to read file"}
        else:
            # For directories, collect metadata files and structure
            metadata_files = []
            structure = []
            data_files = []

            for item in resource_path.rglob("*"):
                if item.is_file():
                    name_lower = item.name.lower()
                    rel_path = str(item.relative_to(resource_path))

                    # Prioritize metadata files
                    if any(name_lower.startswith(mf) for mf in [
                        "readme", "description", "license", "version",
                        "config", "manifest", "readme"
                    ]):
                        metadata_files.append(rel_path)
                    else:
                        structure.append(rel_path)
                        if len(structure) > 20:  # Limit structure preview
                            break

            metadata["metadata_files"] = metadata_files[:10]
            metadata["structure_sample"] = structure[:20]

            # Read metadata files with size limits
            previews = {}
            for meta_file in metadata_files[:5]:  # Limit to 5 metadata files
                file_path = resource_path / meta_file
                try:
                    content = file_path.read_text(errors="replace")[:max_chars]
                    previews[meta_file] = content
                except Exception as e:
                    previews[meta_file] = f"Error: {e}"
            metadata["file_previews"] = previews

        return metadata

    async def _copy_resource_to_workspace(
        self,
        source_path: Path,
        dest_dir: Path,
        resource_name: str,
        resource_type: str,
    ) -> str:
        """Copy a local resource to the workspace directory.

        This ensures the project is self-contained and prevents accidental
        modification of external resources.

        Args:
            source_path: Source path (file or directory)
            dest_dir: Destination directory (workspace models/ or datasets/)
            resource_name: Name of the resource for folder naming
            resource_type: "model" or "dataset"

        Returns:
            Path to the copied resource
        """
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create destination path with sanitized name
        safe_name = resource_name.replace("/", "_").replace(" ", "_").replace("\\", "_")
        dest_path = dest_dir / safe_name

        # Remove existing destination if it exists
        if dest_path.exists():
            try:
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()
            except OSError as e:
                self.log(f"Warning: Could not remove existing {dest_path}: {e}")

        try:
            if source_path.is_dir():
                # Copy directory
                shutil.copytree(source_path, dest_path)
                self.log(f" Copied directory: {source_path.name} -> {dest_path.name}/")
            else:
                # Copy file and create directory structure
                dest_path.mkdir(parents=True, exist_ok=True)
                # Copy all files from source to dest
                for item in source_path.parent.iterdir():
                    if item.name == source_path.name:
                        shutil.copy2(item, dest_path / item.name)
                        self.log(f" Copied file: {item.name} -> {dest_path.name}/")

            return str(dest_path.resolve())
        except OSError as e:
            self.log(f"Failed to copy resource from {source_path} to {dest_path}: {e}")
            # Return source path as fallback (will cause issues later if source is removed)
            return str(source_path.resolve())

