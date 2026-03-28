# Results Registry and References Workflow

This document describes the new global research artifact layout and the migration/validation workflow.

## 1) Global directories

NanoResearch now maintains repository-level registries:

- `references/`
  - `references/papers/` — per-paper structured markdown summaries
  - `references/benchmarks/` — benchmark reference docs
- `results/`
  - `results/history/` — immutable run records (`00001.json`, `00002.json`, ...)
  - `results/by_baseline/<baseline_slug>/` — indexed copies by baseline
  - `results/by_benchmark/<benchmark_slug>/` — indexed copies by benchmark
  - `results/counters/global_run_counter.txt` — monotonic run counter
  - `results/latest_index.json` — latest pointers + history map

Workspace creation/load automatically bootstraps this layout.

## 2) Global run IDs and latest index

After BASELINE_EXECUTION and EXECUTION stages finish, NanoResearch now:

1. Allocates a global run id (`00001+`)
2. Writes a run record to `results/history/<run_id>.json`
3. Updates baseline/benchmark index copies
4. Updates `results/latest_index.json`
5. Stores pointers in workspace outputs and manifest fields

Affected output fields include:

- `global_run_id`
- `global_run_record`
- `global_latest_index`

## 3) Paper-summary completeness gate

During PLANNING, baseline paper references are validated against required markdown fields.

- Queue file: `plans/paper_enrichment_queue.json`
- Blueprint field: `paper_summary_check`

Before BASELINE_EXECUTION or EXECUTION starts, orchestrator enforces this gate:

- If required baseline paper summaries are missing/incomplete, the stage raises a blocking error.
- Workflow expects CCR enrichment (e.g., `ccr code ...`) to fill the queue items.

## 4) Migration for legacy workspaces

Use the CLI command:

```bash
nanoresearch migrate-results --root <workspace_root>
```

or for a single workspace:

```bash
nanoresearch migrate-results --workspace <workspace_path>
```

Migration behavior:

- Reads legacy `plans/baseline_execution_output.json` and `plans/execution_output.json`
- Registers missing global run records
- Backfills output pointers (`global_run_id`, etc.)
- Rebuilds `results/latest_index.json` from `results/history/*.json`

## 5) Notes

- Migration is idempotent for already-migrated outputs (detected via existing run record files).
- `results/latest_index.json` is always rebuilt at the end of migration to guarantee consistency.
- Baseline and execution outputs are now persisted to stage-specific paths automatically from the execution agent base class.