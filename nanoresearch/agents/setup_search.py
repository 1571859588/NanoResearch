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

    async def _download_resources(
        self, search_plan: dict, data_dir: Path, models_dir: Path
    ) -> list[dict]:
        """Download pretrained models and datasets to global cache.

        Download priority for models:
        1. Check if already cached (skip download)
        2. Try ModelScope first (faster in China)
        3. Fall back to HuggingFace
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

            # Check if already cached
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

