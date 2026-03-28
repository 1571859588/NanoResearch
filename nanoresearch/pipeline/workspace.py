"""Workspace directory management and manifest CRUD."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoresearch.schemas.manifest import (
    ArtifactRecord,
    DEEP_ONLY_STAGES,
    PaperMode,
    PipelineMode,
    PipelineStage,
    StageRecord,
    WorkspaceManifest,
    processing_stages_for_mode,
)

from nanoresearch.pipeline._workspace_helpers import (  # noqa: F401
    _WorkspaceExportMixin,
    _slugify,
    _copy_if_exists,
    _prepare_exported_paper_tex,
    _insert_into_preamble,
    _count_lines,
)

logger = logging.getLogger(__name__)


_DEFAULT_ROOT = Path("/mnt/public/sichuan_a/nyt/ai/NanoResearch/workspaces")

WORKSPACE_DIRS = [
    "papers",
    "plans",
    "drafts",
    "figures",
    "logs",
    "code",
    "baselines",
    "benchmarks",
    "results",
]


class Workspace(_WorkspaceExportMixin):
    """Manages a single research session workspace on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._manifest_path = path / "manifest.json"
        self._manifest_cache: WorkspaceManifest | None = None

    # ---- creation --------------------------------------------------------

    @classmethod
    def create(
        cls,
        topic: str,
        config_snapshot: dict | None = None,
        root: Path = _DEFAULT_ROOT,
        session_id: str | None = None,
        pipeline_mode: PipelineMode = PipelineMode.STANDARD,
        paper_mode: PaperMode = PaperMode.ORIGINAL_RESEARCH,
    ) -> "Workspace":
        sid = session_id or uuid.uuid4().hex[:12]
        ws_path = root / sid
        ws_path.mkdir(parents=True, exist_ok=True)
        for d in WORKSPACE_DIRS:
            (ws_path / d).mkdir(exist_ok=True)

        relevant_stages = [
            PipelineStage.INIT,
            *processing_stages_for_mode(pipeline_mode),
        ]

        manifest = WorkspaceManifest(
            session_id=sid,
            topic=topic,
            pipeline_mode=pipeline_mode,
            paper_mode=paper_mode,
            current_stage=PipelineStage.INIT,
            stages={
                stage.value: StageRecord(stage=stage)
                for stage in relevant_stages
            },
            config_snapshot=config_snapshot or {},
        )
        ws = cls(ws_path)
        ws._write_manifest(manifest)
        ws.ensure_global_research_layout()
        return ws

    @classmethod
    def load(cls, path: Path) -> "Workspace":
        if not path.exists():
            raise FileNotFoundError(f"Workspace directory not found: {path}")
        ws = cls(path)
        ws.manifest  # validate readable
        ws.ensure_global_research_layout()
        return ws

    # ---- manifest --------------------------------------------------------

    @property
    def manifest(self) -> WorkspaceManifest:
        if self._manifest_cache is not None:
            return self._manifest_cache
        if not self._manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest file not found: {self._manifest_path}"
            )
        try:
            raw = self._manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read manifest file {self._manifest_path}: {exc}"
            ) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Manifest file contains invalid JSON: {exc}"
            ) from exc
        data, normalized = self._normalize_manifest_data(data)
        manifest = WorkspaceManifest.model_validate(data)
        self._manifest_cache = manifest
        if normalized:
            self._write_manifest(manifest)
        return self._manifest_cache

    @staticmethod
    def _normalize_manifest_data(data: dict) -> tuple[dict, bool]:
        """Repair legacy manifests in-memory before validation."""

        if not isinstance(data, dict):
            return data, False

        normalized = False
        stages = data.get("stages")
        if not isinstance(stages, dict):
            return data, False

        deep_stage_names = {stage.value for stage in DEEP_ONLY_STAGES}
        inferred_deep = False
        current_stage = str(data.get("current_stage", ""))

        for stage_key, record in stages.items():
            try:
                stage_enum = PipelineStage(stage_key)
            except ValueError:
                continue

            if isinstance(record, dict):
                if record.get("stage") != stage_key:
                    record["stage"] = stage_key
                    normalized = True

                status = str(record.get("status", "pending"))
                if (
                    stage_key in deep_stage_names
                    and (
                        status != "pending"
                        or bool(record.get("output_path"))
                        or bool(record.get("error_message"))
                    )
                ):
                    inferred_deep = True
            elif stage_key in deep_stage_names:
                inferred_deep = True

            if current_stage == stage_enum.value and stage_enum in DEEP_ONLY_STAGES:
                inferred_deep = True

        if "pipeline_mode" not in data:
            data["pipeline_mode"] = (
                PipelineMode.DEEP.value if inferred_deep else PipelineMode.STANDARD.value
            )
            normalized = True

        return data, normalized

    def _write_manifest(self, m: WorkspaceManifest) -> None:
        """Atomic write: write to temp file then rename to avoid corruption."""
        m.updated_at = datetime.now(timezone.utc)
        self._manifest_cache = m
        content = m.model_dump_json(indent=2)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._manifest_path.parent), suffix=".tmp"
            )
            try:
                os.write(fd, content.encode("utf-8"))
                os.close(fd)
                fd = -1  # mark as closed
                # Atomic rename (on POSIX; best-effort on Windows)
                os.replace(tmp_path, str(self._manifest_path))
            except BaseException:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError:
            # Fallback to direct write if temp file approach fails
            self._manifest_path.write_text(content, encoding="utf-8")

    def update_manifest(self, **kwargs) -> WorkspaceManifest:
        m = self.manifest
        for k, v in kwargs.items():
            setattr(m, k, v)
        self._write_manifest(m)
        return m

    # ---- stage tracking --------------------------------------------------

    def mark_stage_running(self, stage: PipelineStage) -> None:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.status = "running"
        rec.started_at = datetime.now(timezone.utc)
        m.current_stage = stage
        self._write_manifest(m)

    def mark_stage_completed(self, stage: PipelineStage, output_path: str = "") -> None:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.status = "completed"
        rec.completed_at = datetime.now(timezone.utc)
        rec.output_path = output_path
        self._write_manifest(m)

    def mark_stage_failed(self, stage: PipelineStage, error: str) -> None:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.status = "failed"
        rec.completed_at = datetime.now(timezone.utc)
        rec.error_message = error
        rec.retries += 1
        m.current_stage = PipelineStage.FAILED
        self._write_manifest(m)

    def increment_retry(self, stage: PipelineStage) -> int:
        m = self.manifest
        rec = m.stages.get(stage.value)
        if rec is None:
            rec = StageRecord(stage=stage)
            m.stages[stage.value] = rec
        rec.retries += 1
        rec.status = "pending"
        rec.error_message = ""
        self._write_manifest(m)
        return rec.retries

    # ---- artifacts -------------------------------------------------------

    def register_artifact(
        self, name: str, file_path: Path, stage: PipelineStage
    ) -> ArtifactRecord:
        checksum = ""
        if file_path.is_file():
            checksum = hashlib.md5(file_path.read_bytes()).hexdigest()
        record = ArtifactRecord(
            name=name,
            path=str(file_path.relative_to(self.path)),
            stage=stage,
            checksum=checksum,
        )
        m = self.manifest
        m.artifacts.append(record)
        self._write_manifest(m)
        return record

    # ---- convenience paths -----------------------------------------------

    @property
    def papers_dir(self) -> Path:
        return self.path / "papers"

    @property
    def plans_dir(self) -> Path:
        return self.path / "plans"

    @property
    def drafts_dir(self) -> Path:
        return self.path / "drafts"

    @property
    def figures_dir(self) -> Path:
        return self.path / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def code_dir(self) -> Path:
        return self.path / "code"

    @property
    def baselines_dir(self) -> Path:
        return self.path / "baselines"

    @property
    def benchmarks_dir(self) -> Path:
        return self.path / "benchmarks"

    @property
    def results_dir(self) -> Path:
        return self.path / "results"

    @property
    def repo_root(self) -> Path:
        """Repository root (parent of workspaces directory)."""
        return self.path.parent.parent

    @property
    def global_references_dir(self) -> Path:
        return self.repo_root / "references"

    @property
    def global_results_dir(self) -> Path:
        return self.repo_root / "results"

    def ensure_global_research_layout(self) -> None:
        """Create global references/results registry structure if missing."""
        references_dirs = [
            self.global_references_dir,
            self.global_references_dir / "papers",
            self.global_references_dir / "benchmarks",
        ]
        results_dirs = [
            self.global_results_dir,
            self.global_results_dir / "history",
            self.global_results_dir / "by_baseline",
            self.global_results_dir / "by_benchmark",
            self.global_results_dir / "counters",
        ]
        for d in [*references_dirs, *results_dirs]:
            d.mkdir(parents=True, exist_ok=True)

        latest_index = self.global_results_dir / "latest_index.json"
        if not latest_index.exists():
            latest_index.write_text(
                json.dumps(
                    {
                        "updated_at": "",
                        "latest_run": {},
                        "baselines": {},
                        "benchmarks": {},
                        "methods": {},
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    @staticmethod
    def _slug(value: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower())
        return text.strip("_") or "unknown"

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _load_latest_index(self) -> dict[str, Any]:
        self.ensure_global_research_layout()
        latest_index = self.global_results_dir / "latest_index.json"
        try:
            payload = json.loads(latest_index.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {
            "updated_at": "",
            "latest_run": {},
            "baselines": {},
            "benchmarks": {},
            "methods": {},
        }

    def _save_latest_index(self, payload: dict[str, Any]) -> Path:
        self.ensure_global_research_layout()
        latest_index = self.global_results_dir / "latest_index.json"
        latest_index.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return latest_index

    def allocate_global_run_id(self) -> str:
        self.ensure_global_research_layout()
        counter_file = self.global_results_dir / "counters" / "global_run_counter.txt"
        current = 0
        if counter_file.exists():
            raw = counter_file.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                current = int(raw)
        current += 1
        counter_file.parent.mkdir(parents=True, exist_ok=True)
        counter_file.write_text(str(current), encoding="utf-8")
        return f"{current:05d}"

    def update_latest_index(self, run_record: dict[str, Any]) -> Path:
        payload = self._load_latest_index()
        now = datetime.now(timezone.utc).isoformat()

        run_id = self._safe_text(run_record.get("run_id"))
        baseline_slug = self._slug(self._safe_text(run_record.get("baseline_slug")))
        benchmark_slug = self._slug(self._safe_text(run_record.get("benchmark_slug")))
        method_slug = self._slug(self._safe_text(run_record.get("method_slug")))

        payload["updated_at"] = now
        payload["latest_run"] = run_record
        if baseline_slug:
            payload.setdefault("baselines", {})[baseline_slug] = run_record
        if benchmark_slug:
            payload.setdefault("benchmarks", {})[benchmark_slug] = run_record
        if method_slug:
            payload.setdefault("methods", {})[method_slug] = run_record
        if run_id:
            payload.setdefault("history", {})[run_id] = run_record

        return self._save_latest_index(payload)

    def rebuild_latest_index_from_history(self) -> Path:
        """Recompute results/latest_index.json from results/history/*.json."""
        self.ensure_global_research_layout()
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "latest_run": {},
            "baselines": {},
            "benchmarks": {},
            "methods": {},
            "history": {},
        }

        history_dir = self.global_results_dir / "history"
        latest_record: dict[str, Any] = {}
        for rec_path in sorted(history_dir.glob("*.json"), key=lambda p: p.stem):
            try:
                record = json.loads(rec_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue

            run_id = self._safe_text(record.get("run_id") or rec_path.stem)
            baseline_slug = self._slug(self._safe_text(record.get("baseline_slug")))
            benchmark_slug = self._slug(self._safe_text(record.get("benchmark_slug")))
            method_slug = self._slug(self._safe_text(record.get("method_slug")))

            if run_id:
                payload["history"][run_id] = record
            if baseline_slug:
                payload["baselines"][baseline_slug] = record
            if benchmark_slug:
                payload["benchmarks"][benchmark_slug] = record
            if method_slug:
                payload["methods"][method_slug] = record
            latest_record = record

        if latest_record:
            payload["latest_run"] = latest_record
        return self._save_latest_index(payload)

    def register_research_run(
        self,
        *,
        stage: PipelineStage,
        execution_output: dict[str, Any],
        experiment_blueprint: dict[str, Any],
        run_kind: str,
    ) -> dict[str, Any]:
        self.ensure_global_research_layout()
        run_id = self.allocate_global_run_id()
        now = datetime.now(timezone.utc).isoformat()

        proposed = experiment_blueprint.get("proposed_method", {})
        method_name = self._safe_text(proposed.get("name") if isinstance(proposed, dict) else "")

        baselines = experiment_blueprint.get("baselines", [])
        baseline_slug = ""
        if isinstance(baselines, list) and baselines:
            first = baselines[0] if isinstance(baselines[0], dict) else {}
            baseline_slug = self._safe_text(first.get("slug") or first.get("name"))

        datasets = experiment_blueprint.get("datasets", [])
        benchmark_slug = ""
        if isinstance(datasets, list) and datasets:
            first_ds = datasets[0] if isinstance(datasets[0], dict) else {}
            benchmark_slug = self._safe_text(first_ds.get("name"))

        record = {
            "run_id": run_id,
            "run_kind": run_kind,
            "created_at": now,
            "session_id": self.manifest.session_id,
            "workspace": str(self.path),
            "stage": stage.value,
            "baseline_slug": self._slug(baseline_slug),
            "benchmark_slug": self._slug(benchmark_slug),
            "method_slug": self._slug(method_name or "proposed_method"),
            "baseline_name": baseline_slug,
            "benchmark_name": benchmark_slug,
            "method_name": method_name,
            "experiment_status": self._safe_text(execution_output.get("experiment_status")),
            "final_status": self._safe_text(execution_output.get("final_status")),
            "metrics": execution_output.get("experiment_results")
            or execution_output.get("metrics")
            or {},
            "execution_output_path": self._safe_text(execution_output.get("_output_path")),
            "required_papers": [
                {
                    "paper_id": self._safe_text(b.get("reference_paper_id")),
                    "baseline": self._safe_text(b.get("name")),
                }
                for b in (baselines if isinstance(baselines, list) else [])
                if isinstance(b, dict)
            ],
        }

        history_record_path = self.global_results_dir / "history" / f"{run_id}.json"
        history_record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        baseline_dir = self.global_results_dir / "by_baseline" / record["baseline_slug"]
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / f"{run_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        benchmark_dir = self.global_results_dir / "by_benchmark" / record["benchmark_slug"]
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        (benchmark_dir / f"{run_id}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        self.update_latest_index(record)
        self.update_manifest(
            latest_global_run_id=run_id,
            latest_global_run_record=str(history_record_path),
            latest_global_index_path="results/latest_index.json",
        )
        return record

    def build_paper_summary_markdown(self, paper: dict[str, Any], role: str = "literature") -> str:
        title = self._safe_text(paper.get("title")) or "Untitled Paper"
        paper_id = self._safe_text(paper.get("paper_id") or paper.get("arxiv_id"))
        url = self._safe_text(paper.get("url"))
        year = self._safe_text(paper.get("year"))
        venue = self._safe_text(paper.get("venue"))
        abstract = self._safe_text(paper.get("abstract"))
        bibtex = self._safe_text(paper.get("bibtex"))
        method_text = self._safe_text(paper.get("method_text"))
        experiment_text = self._safe_text(paper.get("experiment_text"))

        return (
            f"# {title}\n\n"
            f"- paper_id: {paper_id}\n"
            f"- role: {role}\n"
            f"- year: {year}\n"
            f"- venue: {venue}\n"
            f"- url: {url}\n\n"
            "## Method / Baseline Mapping\n"
            "- baseline_method_name: TBD\n"
            "- baseline_slug: TBD\n"
            "- open_source: unknown\n"
            "- open_source_url: TBD\n"
            "- requires_training: unknown\n"
            "- training_params: TBD\n"
            "- model_scale: TBD\n\n"
            "## Benchmark & Results\n"
            "| benchmark | metric | result | note |\n"
            "|---|---|---:|---|\n"
            "| TBD | TBD | TBD | TBD |\n\n"
            "## Abstract\n"
            f"{abstract or 'TBD'}\n\n"
            "## Method Notes\n"
            f"{method_text or 'TBD'}\n\n"
            "## Experiment Notes\n"
            f"{experiment_text or 'TBD'}\n\n"
            "## Citation (BibTeX)\n"
            "```bibtex\n"
            f"{bibtex or 'TBD'}\n"
            "```\n"
        )

    def seed_references_from_ideation(self, ideation_output: dict[str, Any]) -> list[str]:
        self.ensure_global_research_layout()
        papers = ideation_output.get("papers", []) if isinstance(ideation_output, dict) else []
        written: list[str] = []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            paper_id = self._safe_text(paper.get("paper_id") or paper.get("arxiv_id"))
            title = self._safe_text(paper.get("title"))
            file_stem = self._slug(paper_id or title)
            if not file_stem:
                continue
            md_path = self.global_references_dir / "papers" / f"{file_stem}.md"
            if md_path.exists():
                continue
            md_path.write_text(
                self.build_paper_summary_markdown(paper),
                encoding="utf-8",
            )
            written.append(str(md_path))
        return written

    def validate_baseline_paper_summaries(
        self,
        experiment_blueprint: dict[str, Any],
    ) -> dict[str, Any]:
        self.ensure_global_research_layout()
        baselines = experiment_blueprint.get("baselines", []) if isinstance(experiment_blueprint, dict) else []
        missing: list[dict[str, Any]] = []
        required_tokens = [
            "open_source:",
            "open_source_url:",
            "requires_training:",
            "training_params:",
            "model_scale:",
            "## Benchmark & Results",
            "## Citation (BibTeX)",
        ]
        for bl in baselines:
            if not isinstance(bl, dict):
                continue
            paper_id = self._safe_text(bl.get("reference_paper_id"))
            baseline_name = self._safe_text(bl.get("name"))
            key = self._slug(paper_id or baseline_name)
            md_path = self.global_references_dir / "papers" / f"{key}.md"
            if not md_path.exists():
                missing.append(
                    {
                        "baseline": baseline_name,
                        "paper_id": paper_id,
                        "reason": "missing_markdown",
                        "path": str(md_path),
                    }
                )
                continue
            text = md_path.read_text(encoding="utf-8", errors="replace")
            absent = [token for token in required_tokens if token not in text]
            placeholder_hits = [tok for tok in ("TBD", "unknown") if tok in text]
            if absent or placeholder_hits:
                missing.append(
                    {
                        "baseline": baseline_name,
                        "paper_id": paper_id,
                        "reason": "incomplete_markdown",
                        "path": str(md_path),
                        "missing_tokens": absent,
                        "placeholder_tokens": placeholder_hits,
                    }
                )

        queue = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": missing,
            "instruction": (
                "Use ccr code with paper-reading skills to enrich each markdown file until no required "
                "tokens are missing and placeholders are resolved."
            ),
        }
        queue_path = self.write_json("plans/paper_enrichment_queue.json", queue)
        return {
            "ok": len(missing) == 0,
            "missing_count": len(missing),
            "missing": missing,
            "queue_path": str(queue_path),
        }

    # ---- utility ---------------------------------------------------------

    def write_json(self, subpath: str, data: dict | list) -> Path:
        p = self.path / subpath
        content: str | None = None
        tmp: Path | None = None
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            # Atomic write: temp file + os.replace to avoid corruption on crash
            tmp = p.with_suffix(".tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(str(tmp), str(p))
        except OSError as exc:
            # Cleanup temp file if it exists
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            # Fallback to direct write if os.replace fails (e.g. cross-device)
            if content is not None:
                try:
                    p.write_text(content, encoding="utf-8")
                    return p  # fallback succeeded
                except OSError:
                    pass
            raise RuntimeError(f"Failed to write JSON to {p}: {exc}") from exc
        return p

    def read_json(self, subpath: str) -> dict | list:
        p = self.path / subpath
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {p}: {exc}") from exc

    def write_text(self, subpath: str, text: str) -> Path:
        p = self.path / subpath
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Failed to write to {p}: {exc}") from exc
        return p

    def read_text(self, subpath: str) -> str:
        p = self.path / subpath
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        return p.read_text(encoding="utf-8")

    # export() is inherited from _WorkspaceExportMixin

    # _slugify, _copy_if_exists, _prepare_exported_paper_tex,
    # _insert_into_preamble, _count_lines are imported from
    # nanoresearch.pipeline._workspace_helpers and re-exported above.
