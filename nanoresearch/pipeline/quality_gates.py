"""Quality gates for the NanoResearch pipeline.

Three gates validate scientific correctness at critical stage transitions:
- Gate 1 (post-CODING): Dataset paths, hyperparameters, baseline completeness
- Gate 2 (post-EXECUTION): NaN detection, class collapse, sample count
- Gate 3 (post-ANALYSIS): Baseline comparison, result plausibility, convergence
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate 1 — Post-CODING: Scientific Code Review
# ---------------------------------------------------------------------------

def validate_post_coding(
    code_dir: Path,
    blueprint: dict[str, Any],
    setup_output: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate generated experiment code before execution.

    Returns (passed, issues) where issues is a list of human-readable problems.
    """
    issues: list[str] = []

    if not code_dir.exists():
        issues.append(f"Code directory does not exist: {code_dir}")
        return False, issues

    # --- 1. Dataset path validation ---
    _check_dataset_paths(code_dir, setup_output, issues)

    # --- 2. Synthetic data fallback detection ---
    _check_synthetic_data_fallback(code_dir, issues)

    # --- 3. Hyperparameter sanity ---
    _check_hyperparameters(code_dir, blueprint, issues)

    # --- 4. Baseline completeness ---
    _check_baseline_completeness(code_dir, blueprint, issues)

    # --- 5. Metric logging ---
    _check_metric_logging(code_dir, issues)

    # --- 6. Entry point existence ---
    _check_entry_points(code_dir, issues)

    passed = len(issues) == 0
    if not passed:
        logger.warning(
            "Gate 1 (post-CODING) FAILED with %d issue(s):\n  - %s",
            len(issues), "\n  - ".join(issues),
        )
    else:
        logger.info("Gate 1 (post-CODING) PASSED")
    return passed, issues


def autofix_gate1_issues(
    code_dir: Path,
    issues: list[str],
    min_epochs: int = 30,
) -> list[str]:
    """Apply targeted, surgical fixes to generated code files.

    Instead of re-generating all files from scratch, this function
    directly patches the specific lines that failed Gate 1 checks.
    It handles ALL issue types:
    - Hyperparameter fixes (epochs, batch_size, quick-eval samples)
    - Dataset path fixes (double-nested directories)
    - Synthetic data fallback removal
    - Missing baselines/run_all.sh auto-generation
    - Missing baseline directory stub creation

    Returns a list of human-readable descriptions of fixes applied.
    """
    fixes_applied: list[str] = []

    for issue in issues:
        # --- Fix: epochs too low ---
        if "epochs=" in issue and "too low" in issue:
            fixed = _fix_epochs_in_file(code_dir, issue, min_epochs)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: batch_size too large ---
        elif "batch_size=" in issue or "Batch size=" in issue:
            fixed = _fix_batch_size_in_file(code_dir, issue)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: quick-eval sample count too low ---
        elif "quick-eval" in issue.lower() or "Quick-eval" in issue:
            fixed = _fix_quick_eval_samples(code_dir, issue)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: double-nested dataset path ---
        elif "Double-nested dataset path" in issue:
            fixed = _fix_double_nested_path(code_dir, issue)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: synthetic data fallback ---
        elif "synthetic data fallback" in issue.lower() or "random numpy data" in issue.lower():
            fixed = _fix_synthetic_data_fallback(code_dir, issue)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: missing baselines/run_all.sh ---
        elif "run_all.sh does not exist" in issue:
            fixed = _fix_missing_run_all_sh(code_dir)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: missing baseline directory ---
        elif "has no directory" in issue and "baselines/" in issue:
            fixed = _fix_missing_baseline_dir(code_dir, issue)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: missing baseline train.py ---
        elif "has no train.py entry point" in issue:
            fixed = _fix_missing_baseline_train(code_dir, issue)
            if fixed:
                fixes_applied.append(fixed)

        # --- Fix: missing train.py metric logging ---
        elif "does not appear to write to results/metrics.json" in issue:
            fixed = _fix_missing_metric_logging(code_dir)
            if fixed:
                fixes_applied.append(fixed)

    # Additionally, always fix epochs in ALL relevant files (config.py, train.py,
    # and baseline train.py files) to ensure consistency
    _fix_all_epoch_values(code_dir, min_epochs, fixes_applied)

    return fixes_applied


def _fix_epochs_in_file(code_dir: Path, issue: str, min_epochs: int) -> str | None:
    """Fix a specific epoch count issue identified by the gate."""
    # Extract filename from issue like "Training epochs=1 in config.py is too low"
    filename_match = re.search(r"in\s+(\S+\.py)\s+is", issue)
    if not filename_match:
        return None

    filename = filename_match.group(1)
    filepath = code_dir / filename
    if not filepath.exists():
        return None

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    # Replace epoch assignments
    epoch_pattern = re.compile(
        r"((?:num_epochs|epochs|max_epoch|n_epochs|NUM_EPOCHS|MAX_EPOCHS|default_epochs)\s*[=:]\s*)(\d+)"
    )
    new_content, count = epoch_pattern.subn(
        lambda m: m.group(1) + str(max(int(m.group(2)), min_epochs)),
        content,
    )

    if count > 0 and new_content != content:
        filepath.write_text(new_content, encoding="utf-8")
        logger.info("Auto-fixed epochs in %s: set to >= %d", filename, min_epochs)
        return f"Fixed epochs in {filename} → {min_epochs}"

    return None


def _fix_all_epoch_values(
    code_dir: Path, min_epochs: int, fixes_applied: list[str],
) -> None:
    """Sweep all Python files and fix any epoch values below min_epochs."""
    epoch_pattern = re.compile(
        r"((?:num_epochs|epochs|max_epoch|n_epochs|NUM_EPOCHS|MAX_EPOCHS|default_epochs)"
        r"\s*[=:]\s*)(\d+)"
    )

    for py_file in code_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        new_content = content
        for match in epoch_pattern.finditer(content):
            old_val = int(match.group(2))
            if old_val < min_epochs:
                new_content = new_content.replace(
                    match.group(0),
                    match.group(1) + str(min_epochs),
                    1,
                )

        if new_content != content:
            py_file.write_text(new_content, encoding="utf-8")
            rel_path = py_file.relative_to(code_dir)
            fix_msg = f"Fixed epochs in {rel_path} → {min_epochs}"
            if fix_msg not in fixes_applied:
                fixes_applied.append(fix_msg)
                logger.info("Auto-fixed epochs in %s", rel_path)


def _fix_batch_size_in_file(code_dir: Path, issue: str) -> str | None:
    """Fix an oversized batch_size."""
    filename_match = re.search(r"in\s+(\S+\.py)\s+is", issue)
    if not filename_match:
        return None

    filename = filename_match.group(1)
    filepath = code_dir / filename
    if not filepath.exists():
        return None

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    batch_pattern = re.compile(r"((?:batch_size|bs)\s*[=:]\s*)(\d+)")
    new_content, count = batch_pattern.subn(
        lambda m: m.group(1) + str(min(int(m.group(2)), 64)),
        content,
    )

    if count > 0 and new_content != content:
        filepath.write_text(new_content, encoding="utf-8")
        logger.info("Auto-fixed batch_size in %s: capped at 64", filename)
        return f"Fixed batch_size in {filename} → capped at 64"

    return None


def _fix_quick_eval_samples(code_dir: Path, issue: str) -> str | None:
    """Fix insufficient quick-eval sample count."""
    # Extract the recommended count from the issue
    count_match = re.search(r"Need at least (\d+)", issue)
    if not count_match:
        return None
    target = int(count_match.group(1))

    for filename in ("config.py", "train.py"):
        filepath = code_dir / filename
        if not filepath.exists():
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        qe_pattern = re.compile(
            r"(quick.?eval.{0,30}(?:num_samples|n_samples|max_samples)\s*[=:]\s*)(\d+)",
            re.IGNORECASE,
        )
        new_content, count = qe_pattern.subn(
            lambda m: m.group(1) + str(max(int(m.group(2)), target)),
            content,
        )
        if count > 0 and new_content != content:
            filepath.write_text(new_content, encoding="utf-8")
            logger.info("Auto-fixed quick-eval samples in %s: set to >= %d", filename, target)
            return f"Fixed quick-eval samples in {filename} → {target}"

    return None


def _fix_double_nested_path(code_dir: Path, issue: str) -> str | None:
    """Fix os.path.join(..., 'X', 'X') → os.path.join(..., 'X')."""
    # Extract filename from issue
    filename_match = re.search(r"in\s+(\S+\.py):", issue)
    if not filename_match:
        return None

    filename = filename_match.group(1)
    filepath = code_dir / filename
    if not filepath.exists():
        return None

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    # Replace os.path.join(..., 'X', 'X') → os.path.join(..., 'X')
    double_nest_pattern = re.compile(
        r"(os\.path\.join\s*\([^)]*['\"])(\w+)(['\"])\s*,\s*['\"](\2)['\"](\s*\))",
    )
    new_content, count = double_nest_pattern.subn(
        r"\1\2\3\5",
        content,
    )

    if count > 0 and new_content != content:
        filepath.write_text(new_content, encoding="utf-8")
        logger.info("Auto-fixed double-nested dataset path in %s", filename)
        return f"Fixed double-nested dataset path in {filename}"

    return None


def _fix_synthetic_data_fallback(code_dir: Path, issue: str) -> str | None:
    """Comment out synthetic/random data fallback code and add a clear error."""
    # Extract filename from issue
    filename_match = re.search(r"in\s+(\S+\.py)\.", issue)
    if not filename_match:
        return None

    filename = filename_match.group(1)
    filepath = code_dir / filename
    if not filepath.exists():
        return None

    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    modified = False
    lines = content.split("\n")
    new_lines = []
    i = 0
    while i < len(lines):
        line_lower = lines[i].lower().strip()

        # Detect synthetic data fallback patterns and comment them out
        if any(pat in line_lower for pat in [
            "synthetic data", "fake data", "random images",
            "fallback to synthetic", "fallback to random",
            "generate synthetic", "generate fake",
        ]):
            # Comment out this line and following indented block
            indent = len(lines[i]) - len(lines[i].lstrip())
            new_lines.append(" " * indent + "# [AUTO-FIX] Removed synthetic data fallback:")
            new_lines.append(" " * indent + "# " + lines[i].strip())
            modified = True
            i += 1
            # Comment out the indented block below
            while i < len(lines) and lines[i].strip():
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent > indent:
                    new_lines.append(" " * curr_indent + "# " + lines[i].strip())
                    i += 1
                else:
                    break
            # Add a proper error raise instead of fallback
            new_lines.append(
                " " * (indent + 4) + 'raise RuntimeError("Dataset loading failed — '
                'check data paths. Do NOT use synthetic data.")'
            )
            continue

        # Also handle np.random.rand patterns that look like dataset substitutes
        if re.search(r"np\.random\.rand.{0,40}(image|sample|data)", line_lower):
            indent = len(lines[i]) - len(lines[i].lstrip())
            new_lines.append(" " * indent + "# [AUTO-FIX] Commented out random data generation:")
            new_lines.append(" " * indent + "# " + lines[i].strip())
            modified = True
            i += 1
            continue

        new_lines.append(lines[i])
        i += 1

    if modified:
        filepath.write_text("\n".join(new_lines), encoding="utf-8")
        logger.info("Auto-fixed synthetic data fallback in %s", filename)
        return f"Removed synthetic data fallback in {filename}"

    return None


def _fix_missing_run_all_sh(code_dir: Path) -> str | None:
    """Auto-generate baselines/run_all.sh from existing baseline directories."""
    baselines_dir = code_dir / "baselines"
    if not baselines_dir.exists():
        baselines_dir.mkdir(parents=True, exist_ok=True)

    # Find all baseline subdirectories with train.py
    baseline_slugs = []
    for subdir in sorted(baselines_dir.iterdir()):
        if subdir.is_dir() and (subdir / "train.py").exists():
            baseline_slugs.append(subdir.name)

    if not baseline_slugs:
        # No baselines to run — check for any .py files directly under baselines/
        py_files = list(baselines_dir.glob("*.py"))
        if py_files:
            # Generate run_all.sh that runs these files
            script_lines = [
                "#!/bin/bash",
                "# Auto-generated by quality gate autofix",
                'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                'PARENT_DIR="$(dirname "$SCRIPT_DIR")"',
                "",
            ]
            for py_file in sorted(py_files):
                if py_file.name != "__init__.py" and py_file.name != "utils.py":
                    script_lines.append(f'echo "=== Running {py_file.name} ==="')
                    script_lines.append(
                        f'cd "$PARENT_DIR" && python "baselines/{py_file.name}" "$@"'
                    )
                    script_lines.append("")
            script_lines.append('echo "=== All baselines complete ==="')
            run_all = baselines_dir / "run_all.sh"
            run_all.write_text("\n".join(script_lines), encoding="utf-8")
            logger.info("Auto-generated baselines/run_all.sh from .py files")
            return "Generated baselines/run_all.sh from standalone .py files"
        return None

    # Generate run_all.sh
    script_lines = [
        "#!/bin/bash",
        "# Auto-generated by quality gate autofix",
        "set -e",
        "",
        '# Get the directory where this script lives',
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'PARENT_DIR="$(dirname "$SCRIPT_DIR")"',
        "",
    ]
    for slug in baseline_slugs:
        script_lines.extend([
            f'echo "======================================"',
            f'echo "=== Running baseline: {slug} ==="',
            f'echo "======================================"',
            f'cd "$PARENT_DIR"',
            f'python "baselines/{slug}/train.py" "$@"',
            f'echo "=== {slug} complete ==="',
            "",
        ])
    script_lines.append('echo "=== All baselines complete ==="')

    run_all = baselines_dir / "run_all.sh"
    run_all.write_text("\n".join(script_lines), encoding="utf-8")
    logger.info("Auto-generated baselines/run_all.sh for %d baselines", len(baseline_slugs))
    return f"Generated baselines/run_all.sh for {len(baseline_slugs)} baseline(s): {', '.join(baseline_slugs)}"


def _fix_missing_baseline_dir(code_dir: Path, issue: str) -> str | None:
    """Create a missing baseline directory with a minimal stub train.py."""
    # Extract slug from issue like "Baseline 'X' (slug=Y) has no directory at baselines/Y/"
    slug_match = re.search(r"slug=(\w+)", issue)
    name_match = re.search(r"Baseline '([^']+)'", issue)
    if not slug_match:
        return None

    slug = slug_match.group(1)
    name = name_match.group(1) if name_match else slug

    baseline_dir = code_dir / "baselines" / slug
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal train.py stub that inherits from the main project
    train_stub = f'''#!/usr/bin/env python3
"""Training script for baseline: {name}

This is an auto-generated stub. The code should import shared utilities
from the parent project directory.
"""
import sys
import os
import json
import argparse

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    parser = argparse.ArgumentParser(description="{name} baseline training")
    parser.add_argument("--data-dir", type=str, default="../../datasets")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quick-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"[{{args.epochs}} epochs] Training baseline: {name}")

    # TODO: Implement {name} baseline training
    # This stub should be replaced with a proper implementation

    # Save metrics
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    metrics = {{
        "method": "{name}",
        "is_proposed": False,
        "status": "stub_not_implemented",
    }}
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline {name} complete (stub)")

if __name__ == "__main__":
    main()
'''
    (baseline_dir / "train.py").write_text(train_stub, encoding="utf-8")
    logger.info("Created stub baseline directory: baselines/%s/", slug)
    return f"Created stub baseline directory: baselines/{slug}/ with train.py"


def _fix_missing_baseline_train(code_dir: Path, issue: str) -> str | None:
    """Create a missing train.py in an existing baseline directory."""
    # Extract slug from issue like "baselines/slug/ but has no train.py"
    slug_match = re.search(r"baselines/(\w+)/", issue)
    name_match = re.search(r"Baseline '([^']+)'", issue)
    if not slug_match:
        return None

    slug = slug_match.group(1)
    name = name_match.group(1) if name_match else slug
    return _fix_missing_baseline_dir(code_dir, f"Baseline '{name}' (slug={slug}) has no directory")


def _fix_missing_metric_logging(code_dir: Path) -> str | None:
    """Inject metric-saving code into train.py if it's missing."""
    train_py = code_dir / "train.py"
    if not train_py.exists():
        return None

    try:
        content = train_py.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    # Don't fix if already present
    if "metrics.json" in content or "metrics_path" in content.lower():
        return None

    # Find the main function or the end of the file, and inject metric saving
    inject_code = '''
# [AUTO-FIX] Injected metric logging
import json as _json
def _save_metrics(metrics_dict, results_dir="results"):
    """Save metrics to results/metrics.json."""
    import os
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "metrics.json")
    with open(path, "w") as f:
        _json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to {path}")
'''

    # Inject at the top of the file, after imports
    # Find the last import line
    lines = content.split("\n")
    last_import_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i

    # Insert after last import
    lines.insert(last_import_idx + 1, inject_code)
    train_py.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Auto-injected metric logging into train.py")
    return "Injected _save_metrics() helper into train.py"


def _check_dataset_paths(
    code_dir: Path, setup_output: dict[str, Any], issues: list[str],
) -> None:
    """Verify that dataset paths in the code actually exist on disk."""
    data_dir = setup_output.get("data_dir", "")
    if not data_dir:
        return

    downloaded = setup_output.get("downloaded_resources", [])
    available_paths: set[str] = set()
    for res in downloaded:
        if isinstance(res, dict):
            p = res.get("path", "")
            if p:
                available_paths.add(str(Path(p).resolve()))

    # Scan Python files for hardcoded dataset paths that don't exist
    for py_file in code_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Find os.path.join(..., 'DATASET_NAME', 'DATASET_NAME') double-nesting bug
        double_nest_pattern = re.compile(
            r"os\.path\.join\s*\([^)]*['\"](\w+)['\"]\s*,\s*['\"](\1)['\"]\s*\)",
        )
        for match in double_nest_pattern.finditer(content):
            issues.append(
                f"Double-nested dataset path detected in {py_file.name}: "
                f"os.path.join(..., '{match.group(1)}', '{match.group(1)}'). "
                f"This will create a non-existent subdirectory."
            )


def _check_synthetic_data_fallback(code_dir: Path, issues: list[str]) -> None:
    """Detect code that silently falls back to synthetic/random data."""
    danger_patterns = [
        (r"synthetic[_ ]data", "synthetic data fallback"),
        (r"torch\.randn\(.{0,30}dataset", "random tensor as dataset"),
        (r"fake[_ ]data", "fake data generation"),
        (r"random[_ ]images?", "random image generation"),
        (r"np\.random\.rand.{0,40}(image|sample|data)", "random numpy data"),
    ]
    for py_file in code_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue

        for pattern, description in danger_patterns:
            if re.search(pattern, content):
                issues.append(
                    f"Potential {description} detected in {py_file.name}. "
                    f"The code should use REAL dataset files, never synthetic substitutes."
                )


def _check_hyperparameters(
    code_dir: Path, blueprint: dict[str, Any], issues: list[str],
) -> None:
    """Check that training hyperparameters are reasonable for the task."""
    # Determine expected scale from blueprint
    datasets = blueprint.get("datasets", [])
    num_classes = 0
    for ds in datasets:
        desc = str(ds.get("description", "") or ds.get("size_info", "")).lower()
        # Try to extract num_classes from description
        match = re.search(r"(\d+)\s*(class|categor|species|type)", desc)
        if match:
            num_classes = max(num_classes, int(match.group(1)))

    # Scan config.py and train.py for epoch/batch settings
    for filename in ("config.py", "train.py"):
        filepath = code_dir / filename
        if not filepath.exists():
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Check epochs
        epoch_match = re.search(
            r"(?:num_epochs|epochs|max_epoch|n_epochs)\s*[=:]\s*(\d+)", content,
        )
        if epoch_match:
            epochs = int(epoch_match.group(1))
            if epochs < 5:
                issues.append(
                    f"Training epochs={epochs} in {filename} is too low. "
                    f"Fine-grained classification typically needs 30-50+ epochs. "
                    f"Set epochs to at least 30."
                )

        # Check batch size
        batch_match = re.search(
            r"(?:batch_size|bs)\s*[=:]\s*(\d+)", content,
        )
        if batch_match:
            batch_size = int(batch_match.group(1))
            if batch_size > 512:
                issues.append(
                    f"Batch size={batch_size} in {filename} is unusually large. "
                    f"For fine-grained tasks with limited data, 16-64 is typical."
                )

        # Check quick_eval sample count
        qe_match = re.search(
            r"quick.?eval.{0,30}(?:num_samples|n_samples|max_samples)\s*[=:]\s*(\d+)",
            content, re.IGNORECASE,
        )
        if qe_match:
            qe_samples = int(qe_match.group(1))
            if num_classes > 0 and qe_samples < num_classes * 3:
                issues.append(
                    f"Quick-eval uses only {qe_samples} samples for {num_classes}-class task. "
                    f"Need at least {num_classes * 5} samples for statistically meaningful evaluation."
                )


def _check_baseline_completeness(
    code_dir: Path, blueprint: dict[str, Any], issues: list[str],
) -> None:
    """Verify each blueprint baseline has corresponding code."""
    baselines = blueprint.get("baselines", [])
    if not baselines:
        return

    baselines_dir = code_dir / "baselines"
    if not baselines_dir.exists():
        issues.append(
            f"Blueprint defines {len(baselines)} baseline(s) but no baselines/ directory exists."
        )
        return

    run_all_sh = baselines_dir / "run_all.sh"
    if not run_all_sh.exists():
        issues.append("baselines/run_all.sh does not exist — baseline execution will fail.")

    for bl in baselines:
        if not isinstance(bl, dict):
            continue
        slug = bl.get("slug", "")
        name = bl.get("name", str(bl))
        if slug:
            slug_dir = baselines_dir / slug
            if not slug_dir.exists():
                issues.append(
                    f"Baseline '{name}' (slug={slug}) has no directory "
                    f"at baselines/{slug}/. Missing implementation."
                )
            elif not (slug_dir / "train.py").exists():
                issues.append(
                    f"Baseline '{name}' directory exists at baselines/{slug}/ "
                    f"but has no train.py entry point."
                )


def _check_metric_logging(code_dir: Path, issues: list[str]) -> None:
    """Verify that training code logs metrics to results/metrics.json."""
    train_py = code_dir / "train.py"
    if not train_py.exists():
        issues.append("No train.py found — cannot verify metric logging.")
        return

    try:
        content = train_py.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    if "metrics.json" not in content and "metrics_path" not in content.lower():
        issues.append(
            "train.py does not appear to write to results/metrics.json. "
            "Metrics must be logged for downstream analysis."
        )


def _check_entry_points(code_dir: Path, issues: list[str]) -> None:
    """Verify that the main entry point(s) exist."""
    if not (code_dir / "train.py").exists() and not (code_dir / "main.py").exists():
        issues.append(
            "Neither train.py nor main.py found in code directory. "
            "At least one entry point is required."
        )


# ---------------------------------------------------------------------------
# Gate 2 — Post-EXECUTION: Result Sanity Check
# ---------------------------------------------------------------------------

def validate_post_execution(
    execution_output: dict[str, Any],
    blueprint: dict[str, Any],
    code_dir: Path | None = None,
) -> tuple[bool, list[str]]:
    """Validate experiment execution results for scientific soundness.

    Returns (passed, issues).
    """
    issues: list[str] = []

    metrics = execution_output.get("experiment_results", {})
    if not metrics:
        metrics = execution_output.get("metrics", {})

    # --- 1. NaN / Inf detection ---
    for key, value in metrics.items():
        if isinstance(value, float):
            if math.isnan(value):
                issues.append(f"Metric '{key}' is NaN — training likely crashed or used invalid data.")
            elif math.isinf(value):
                issues.append(f"Metric '{key}' is Inf — numerical overflow in training.")

    # --- 2. Single-class predictor detection ---
    unique_preds = metrics.get("unique_predictions")
    num_classes = _extract_num_classes(blueprint)
    if unique_preds is not None and num_classes > 10 and unique_preds < 3:
        issues.append(
            f"Model predicts only {unique_preds} unique class(es) out of {num_classes}. "
            f"This indicates model collapse — the model is predicting the majority class only."
        )

    # --- 3. Accuracy-F1 inconsistency ---
    accuracy = metrics.get("accuracy", metrics.get("top1_accuracy"))
    f1 = metrics.get("f1_macro", metrics.get("f1"))
    if accuracy is not None and f1 is not None:
        if isinstance(accuracy, (int, float)) and isinstance(f1, (int, float)):
            if accuracy > 0.5 and f1 < 0.05:
                issues.append(
                    f"Accuracy={accuracy:.4f} but F1-macro={f1:.4f} — "
                    f"this is a majority-class predictor, not real learning."
                )

    # --- 4. Evaluation sample count ---
    eval_samples = metrics.get("eval_samples", metrics.get("test_samples"))
    if eval_samples is not None and num_classes > 10:
        if isinstance(eval_samples, (int, float)) and eval_samples < num_classes * 3:
            issues.append(
                f"Only {int(eval_samples)} samples evaluated for {num_classes}-class task. "
                f"Need at least {num_classes * 5} for statistically reliable metrics."
            )

    # --- 5. Log-based detection: scan for synthetic data evidence ---
    if code_dir and code_dir.exists():
        log_dir = code_dir / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("*.txt"):
                try:
                    content = log_file.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    continue
                if "synthetic data" in content or "0 images loaded" in content:
                    issues.append(
                        f"Log file {log_file.name} indicates synthetic data was used. "
                        f"Training results are meaningless."
                    )
                    break
            for log_file in log_dir.glob("*.json"):
                try:
                    content = log_file.read_text(encoding="utf-8", errors="ignore").lower()
                except OSError:
                    continue
                if "synthetic" in content or "fallback" in content:
                    issues.append(
                        f"Log file {log_file.name} indicates fallback to synthetic data."
                    )
                    break

    # --- 6. Loss sanity ---
    loss = metrics.get("loss", metrics.get("final_loss", metrics.get("train_loss")))
    if loss is not None and isinstance(loss, (int, float)):
        if not math.isnan(loss) and not math.isinf(loss):
            if num_classes > 0:
                random_loss = math.log(num_classes)
                if loss > random_loss * 1.1:
                    issues.append(
                        f"Final loss={loss:.4f} is worse than random guessing "
                        f"(-log(1/{num_classes})={random_loss:.4f}). "
                        f"The model did not learn anything useful."
                    )

    passed = len(issues) == 0
    if not passed:
        logger.warning(
            "Gate 2 (post-EXECUTION) FAILED with %d issue(s):\n  - %s",
            len(issues), "\n  - ".join(issues),
        )
    else:
        logger.info("Gate 2 (post-EXECUTION) PASSED")
    return passed, issues


# ---------------------------------------------------------------------------
# Gate 3 — Post-ANALYSIS: Scientific Analysis Review
# ---------------------------------------------------------------------------

def validate_post_analysis(
    analysis_output: dict[str, Any],
    blueprint: dict[str, Any],
    execution_output: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate experiment analysis for completeness and plausibility.

    Returns (passed, issues).
    """
    issues: list[str] = []

    # --- 1. Baseline comparison completeness ---
    planned_baselines = blueprint.get("baselines", [])
    comparison = analysis_output.get("comparison_with_baselines", {})
    if planned_baselines and not comparison:
        issues.append(
            f"Blueprint defines {len(planned_baselines)} baseline(s) but analysis contains "
            f"no baseline comparison data. Baselines likely failed to produce results."
        )
    elif planned_baselines:
        for bl in planned_baselines:
            if isinstance(bl, dict):
                bl_name = bl.get("name", "")
                bl_slug = bl.get("slug", "")
                # Check if baseline appears in comparison
                found = any(
                    bl_name.lower() in k.lower() or (bl_slug and bl_slug in k.lower())
                    for k in comparison
                )
                if not found:
                    issues.append(
                        f"Baseline '{bl_name}' is missing from analysis comparison results."
                    )

    # --- 2. Proposed method result existence ---
    main_results = analysis_output.get("main_results", {})
    if not main_results:
        issues.append(
            "Analysis contains no main_results for the proposed method."
        )

    # --- 3. Result plausibility check ---
    metrics = execution_output.get("experiment_results", {})
    num_classes = _extract_num_classes(blueprint)
    accuracy = metrics.get("accuracy", metrics.get("top1_accuracy"))
    if accuracy is not None and isinstance(accuracy, (int, float)):
        if accuracy == 0.0:
            issues.append(
                "Accuracy is exactly 0.0%, indicating the model produced no correct predictions. "
                "This is almost certainly a bug (wrong evaluation, data loading failure, etc.)."
            )
        elif num_classes > 50 and accuracy > 0.99:
            issues.append(
                f"Accuracy is {accuracy*100:.1f}% on a {num_classes}-class task. "
                f"This is suspiciously high and may indicate data leakage or evaluation on training data."
            )

    # --- 4. Training convergence ---
    # Check if only 1 epoch was trained
    epochs_trained = metrics.get("epoch", metrics.get("epochs_completed"))
    if epochs_trained is not None and isinstance(epochs_trained, (int, float)):
        if epochs_trained <= 1:
            issues.append(
                f"Only {int(epochs_trained)} epoch(s) trained. "
                f"Most deep learning tasks require 20-100+ epochs for convergence. "
                f"Results from a single epoch are scientifically meaningless."
            )

    passed = len(issues) == 0
    if not passed:
        logger.warning(
            "Gate 3 (post-ANALYSIS) FAILED with %d issue(s):\n  - %s",
            len(issues), "\n  - ".join(issues),
        )
    else:
        logger.info("Gate 3 (post-ANALYSIS) PASSED")
    return passed, issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_num_classes(blueprint: dict[str, Any]) -> int:
    """Best-effort extraction of num_classes from blueprint datasets."""
    for ds in blueprint.get("datasets", []):
        if not isinstance(ds, dict):
            continue
        desc = str(ds.get("description", "") or ds.get("size_info", "")).lower()
        match = re.search(r"(\d+)\s*(class|categor|species|type|breed)", desc)
        if match:
            return int(match.group(1))
        name = str(ds.get("name", "")).lower()
        if "cub" in name or "cub_200" in name or "cub-200" in name:
            return 200
        if "cifar-100" in name or "cifar100" in name:
            return 100
        if "cifar-10" in name or "cifar10" in name:
            return 10
        if "imagenet" in name:
            return 1000
    return 0


def format_gate_failure_message(
    gate_name: str,
    issues: list[str],
    remediation_hint: str = "",
) -> str:
    """Format gate failure as a clear error message for the retry mechanism."""
    lines = [
        f"=== QUALITY GATE FAILURE: {gate_name} ===",
        f"The following {len(issues)} issue(s) MUST be fixed before proceeding:",
        "",
    ]
    for i, issue in enumerate(issues, 1):
        lines.append(f"  {i}. {issue}")
    if remediation_hint:
        lines.append("")
        lines.append(f"REMEDIATION: {remediation_hint}")
    lines.append("")
    lines.append(f"=== END {gate_name} ===")
    return "\n".join(lines)
