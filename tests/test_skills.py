"""Tests for K-Dense skill matching and context extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanoresearch.skills import (
    MAX_CHARS_PER_SKILL,
    MAX_TOTAL_CHARS,
    SkillContext,
    SkillEntry,
    SkillMatcher,
    _extract_high_value_sections,
    _extract_keywords,
    _extract_yaml_frontmatter,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def skill_tree(tmp_path: Path) -> Path:
    """Create a minimal skill tree for testing."""
    root = tmp_path / "scientific-skills"
    root.mkdir()

    # Skill 1: scanpy single-cell
    s1 = root / "scanpy-single-cell"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        '---\nname: scanpy-single-cell\n'
        'description: "Single-cell RNA-seq analysis with scanpy"\n---\n'
        "# Scanpy Single Cell\n\n"
        "## Quick Start\n"
        "Use scanpy for single-cell RNA-seq analysis.\n"
        "Filter mitochondrial genes, normalize, find variable genes.\n\n"
        "## Common Pitfalls\n"
        "- Forgetting to filter mitochondrial genes\n"
        "- Not using leiden clustering properly\n\n"
        "## Key Parameters\n"
        "- n_top_genes: 2000\n"
        "- resolution: 0.5 for leiden\n"
    )
    assets = s1 / "assets"
    assets.mkdir()
    (assets / "pipeline.py").write_text(
        "import scanpy as sc\n"
        "adata = sc.read_h5ad('data.h5ad')\n"
        "sc.pp.filter_cells(adata, min_genes=200)\n"
    )

    # Skill 2: pytorch-training
    s2 = root / "pytorch-training"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(
        '---\nname: pytorch-training\n'
        'description: "PyTorch model training best practices"\n---\n'
        "# PyTorch Training\n\n"
        "## Quick Start\n"
        "Set up training loop with torch DataLoader.\n\n"
        "## Best Practices\n"
        "- Use mixed precision training with torch.cuda.amp\n"
        "- Set pin_memory=True for DataLoader\n\n"
        "## Common Pitfalls\n"
        "- Forgetting model.eval() during validation\n"
    )

    # Skill 3: matplotlib (generic, low relevance)
    s3 = root / "matplotlib"
    s3.mkdir()
    (s3 / "SKILL.md").write_text(
        '---\nname: matplotlib\n'
        'description: "Data visualization with matplotlib"\n---\n'
        "# Matplotlib\n\n"
        "## Quick Start\n"
        "import matplotlib.pyplot as plt\n"
    )

    return root


@pytest.fixture
def writer_skill_tree(tmp_path: Path) -> Path:
    """Create a minimal writer skill tree for testing."""
    root = tmp_path / "writer-skills"
    root.mkdir()

    # scientific-writing skill
    sw = root / "scientific-writing"
    sw.mkdir()
    (sw / "SKILL.md").write_text(
        '---\nname: scientific-writing\n'
        'description: "Write scientific manuscripts with IMRAD structure"\n---\n'
        "# Scientific Writing\n\n"
        "## Quick Start\n"
        "Use IMRAD structure: Introduction, Methods, Results, Discussion.\n"
        "Write in formal academic English.\n\n"
        "## Best Practices\n"
        "- Use active voice where possible\n"
        "- Keep sentences concise\n"
    )

    # venue-templates skill
    vt = root / "venue-templates"
    vt.mkdir()
    (vt / "SKILL.md").write_text(
        '---\nname: venue-templates\n'
        'description: "LaTeX templates for NeurIPS, ICML, Nature"\n---\n'
        "# Venue Templates\n\n"
        "## Quick Start\n"
        "Select the correct template for your target venue.\n\n"
        "## Key Parameters\n"
        "- NeurIPS: 9 pages content, unlimited appendix\n"
        "- ICML: 8 pages content, unlimited appendix\n"
    )

    # citation-management skill
    cm = root / "citation-management"
    cm.mkdir()
    (cm / "SKILL.md").write_text(
        '---\nname: citation-management\n'
        'description: "BibTeX citation management and DOI validation"\n---\n'
        "# Citation Management\n\n"
        "## Quick Start\n"
        "Use authorYear format for citation keys.\n\n"
        "## Common Pitfalls\n"
        "- Missing DOI fields\n"
        "- Inconsistent author name formatting\n"
    )

    return root


@pytest.fixture
def sample_blueprint() -> dict:
    """A bioinformatics experiment blueprint."""
    return {
        "title": "Single-cell RNA-seq clustering with scanpy",
        "proposed_method": {
            "name": "ScDeepCluster",
            "description": "Deep clustering for single-cell RNA-seq",
            "key_components": ["autoencoder", "leiden clustering", "scanpy"],
        },
        "datasets": [
            {"name": "PBMC 3k", "description": "Peripheral blood mononuclear cells"},
        ],
        "metrics": [
            {"name": "ARI", "description": "Adjusted Rand Index"},
            {"name": "NMI", "description": "Normalized Mutual Information"},
        ],
        "baselines": [
            {"name": "Seurat", "description": "R-based single-cell toolkit"},
            {"name": "scVI", "description": "Variational inference for single-cell"},
        ],
        "ablation_groups": [],
    }


# ── Unit tests: helpers ───────────────────────────────────────────────────


def test_extract_yaml_frontmatter():
    text = '---\nname: my-skill\ndescription: "A test skill"\n---\n# Content'
    name, desc = _extract_yaml_frontmatter(text)
    assert name == "my-skill"
    assert desc == "A test skill"


def test_extract_yaml_frontmatter_missing():
    name, desc = _extract_yaml_frontmatter("# No frontmatter here")
    assert name == ""
    assert desc == ""


def test_extract_keywords():
    text = "Use scanpy for single-cell RNA-seq analysis with leiden clustering"
    kw = _extract_keywords(text)
    assert "scanpy" in kw
    assert "leiden" in kw
    assert "clustering" in kw
    # Stopwords removed
    assert "for" not in kw
    assert "with" not in kw


def test_extract_high_value_sections():
    md = (
        "# Title\n\nIntro paragraph\n\n"
        "## Quick Start\nDo this first.\n\n"
        "## Architecture\nNot extracted.\n\n"
        "## Common Pitfalls\nWatch out for X.\n"
    )
    result = _extract_high_value_sections(md)
    assert "Do this first" in result
    assert "Watch out for X" in result
    assert "Not extracted" not in result


def test_extract_high_value_sections_truncation():
    md = "## Quick Start\n" + "x" * 10000
    result = _extract_high_value_sections(md, max_chars=100)
    assert len(result) <= 120  # 100 + "... [truncated]"
    assert result.endswith("[truncated]")


# ── Unit tests: SkillMatcher ──────────────────────────────────────────────


def test_matcher_no_dir():
    """SkillMatcher with None dir produces empty results."""
    sm = SkillMatcher(None)
    assert sm.skill_count == 0
    assert sm.match({"title": "anything"}) == []


def test_matcher_missing_dir(tmp_path: Path):
    """SkillMatcher with non-existent dir produces empty results."""
    sm = SkillMatcher(tmp_path / "nonexistent")
    assert sm.skill_count == 0


def test_matcher_indexing(skill_tree: Path):
    """SkillMatcher indexes all SKILL.md files."""
    sm = SkillMatcher(skill_tree)
    assert sm.skill_count == 3


def test_matcher_match_scanpy(skill_tree: Path, sample_blueprint: dict):
    """Blueprint about scanpy matches the scanpy skill."""
    sm = SkillMatcher(skill_tree)
    matches = sm.match(sample_blueprint)
    names = [e.name for e, _ in matches]
    assert "scanpy-single-cell" in names


def test_matcher_match_score_ordering(skill_tree: Path, sample_blueprint: dict):
    """Matches are returned in descending score order."""
    sm = SkillMatcher(skill_tree)
    matches = sm.match(sample_blueprint)
    scores = [s for _, s in matches]
    assert scores == sorted(scores, reverse=True)


def test_extract_context(skill_tree: Path, sample_blueprint: dict):
    """extract_context produces non-empty phase1 and matched_skills."""
    sm = SkillMatcher(skill_tree)
    matches = sm.match(sample_blueprint)
    ctx = sm.extract_context(matches)
    assert ctx.matched_skills
    assert "DOMAIN EXPERT KNOWLEDGE" in ctx.phase1_context
    assert "scanpy" in ctx.phase1_context.lower()


def test_extract_context_has_assets(skill_tree: Path, sample_blueprint: dict):
    """extract_context includes asset template snippets in phase2_context."""
    sm = SkillMatcher(skill_tree)
    matches = sm.match(sample_blueprint)
    ctx = sm.extract_context(matches)
    # scanpy-single-cell has assets/pipeline.py
    if any(e.name == "scanpy-single-cell" for e, _ in matches):
        assert "pipeline.py" in ctx.phase2_context


def test_extract_context_empty():
    """extract_context with empty matches returns empty SkillContext."""
    sm = SkillMatcher(None)
    ctx = sm.extract_context([])
    assert ctx.phase1_context == ""
    assert ctx.phase2_context == ""
    assert ctx.matched_skills == []


# ── Unit tests: Writing skills ────────────────────────────────────────────


def test_match_writing_skills(writer_skill_tree: Path):
    """Writing skill matcher boosts priority skills."""
    sm = SkillMatcher(writer_skill_tree)
    matches = sm.match_writing_skills(
        topic="Graph neural networks for protein folding",
        template_format="neurips",
    )
    names = [e.name for e, _ in matches]
    # Priority skills should always be present
    assert "scientific-writing" in names
    assert "venue-templates" in names
    assert "citation-management" in names


def test_extract_writing_context(writer_skill_tree: Path):
    """extract_writing_context produces formatted guidance block."""
    sm = SkillMatcher(writer_skill_tree)
    matches = sm.match_writing_skills(topic="protein folding", template_format="neurips")
    ctx = sm.extract_writing_context(matches)
    assert "ACADEMIC WRITING GUIDELINES" in ctx
    assert "IMRAD" in ctx


def test_extract_writing_context_empty():
    """extract_writing_context with no matches returns empty string."""
    sm = SkillMatcher(None)
    assert sm.extract_writing_context([]) == ""


# ── Graceful degradation ─────────────────────────────────────────────────


def test_graceful_degradation_no_skills_dir():
    """System works normally when skills_dir is None."""
    sm = SkillMatcher(None)
    matches = sm.match({"title": "test", "proposed_method": {"name": "test"}})
    assert matches == []
    ctx = sm.extract_context(matches)
    assert ctx.phase1_context == ""
    assert ctx.phase2_context == ""

    wm = sm.match_writing_skills("test topic")
    assert wm == []
    assert sm.extract_writing_context(wm) == ""
