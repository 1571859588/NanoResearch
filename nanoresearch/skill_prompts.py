"""Condensed skill guidance extracted from K-Dense scientific skills.

Architecture: skill guidance is injected **per-call into user prompts** (not system
prompts) to minimize token overhead.  Only the guidance relevant to the current
section / task is included.

Functions return focused fragments (~100-200 tokens) instead of monolithic blocks.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════
# IDEATION — injected once per LLM call in _analyze_and_hypothesize
# Kept as a constant because ideation calls are few (~5-10 per run).
# ═══════════════════════════════════════════════════════════════════════════
IDEATION_SKILL = """\
LITERATURE STANDARDS:
- Prioritize Tier 1 venues (Nature, Science, NeurIPS, ICML, ICLR, CVPR, ACL, PNAS).
- Citation significance: 0-3yr 100+=influential, 3-7yr 500+=landmark, 7yr+ 1000+=foundational.
- Organize findings thematically, NOT study-by-study.
- Each hypothesis must be testable with concrete quantitative predictions.
- Identify which components drive the improvement (for ablation design)."""


# ═══════════════════════════════════════════════════════════════════════════
# WRITING — per-section injection via get_writing_guidance(section_heading)
# ═══════════════════════════════════════════════════════════════════════════
_WRITING_CORE = """\
WRITING PROCESS: First mentally outline key points and citations, then convert to
flowing prose. Every sentence connects logically. Use transitions (however, moreover,
in contrast). Integrate citations naturally within sentences — NOT as disconnected lists.
NEVER output bullet points — only LaTeX paragraphs."""

_WRITING_SECTIONS: dict[str, str] = {
    "introduction": (
        "Structure: Problem importance -> Literature gaps -> Research questions -> "
        "Novelty and contributions. End with a clear contribution list that maps 1:1 "
        "to experiments. Use present tense for established facts, past tense for "
        "specific prior work."
    ),
    "related work": (
        "Write THEMATIC synthesis, NOT study-by-study listing. Group by approach type, "
        "compare strengths/limitations within each theme, then position your method. "
        "BAD: 'X did A. Y did B. Z did C.' "
        "GOOD: 'Attention-based approaches (X; Y) improved Z but remain limited by W, "
        "which our method addresses via...'"
    ),
    "method": (
        "Reproducible detail: state every design choice with justification. "
        "Include equations for all non-trivial operations using \\begin{equation}. "
        "Define all notation on first use. Use algorithmic pseudocode for complex procedures. "
        "Every component must appear in experiments (ablation or main result)."
    ),
    "experiment": (
        "Must include: datasets (with stats), baselines (with citations), metrics, "
        "implementation details (lr, batch, epochs, hardware). Present results WITH analysis — "
        "don't just state numbers. Every contribution from Introduction needs evidence here. "
        "Include ablation study removing each proposed component individually."
    ),
    "conclusion": (
        "Concise summary of findings — no new information. Acknowledge limitations honestly "
        "(not just 'future work'). Future directions must be specific, not generic. "
        "Do NOT overstate claims beyond what experiments support."
    ),
}


def get_writing_guidance(section_heading: str) -> str:
    """Return focused writing guidance for a specific section (~100-150 tokens).

    Injected into the user prompt of _generate_section(), not the system prompt.
    """
    heading_lower = section_heading.lower()
    specific = ""
    for key, guidance in _WRITING_SECTIONS.items():
        if key in heading_lower:
            specific = f"\nSECTION FOCUS ({section_heading}): {guidance}"
            break
    return f"\n{_WRITING_CORE}{specific}\n"


# ═══════════════════════════════════════════════════════════════════════════
# REVIEW — per-section injection via get_review_guidance(section_heading)
# ═══════════════════════════════════════════════════════════════════════════
_REVIEW_CORE = """\
SCORING: 9-10=publication-ready, 7-8=solid with fixable issues, 5-6=significant problems, \
3-4=major rewrite, 1-2=fundamentally flawed.
FEEDBACK: Every issue must state (a) problem, (b) why it matters, (c) specific fix. No vague criticism."""

_REVIEW_SECTIONS: dict[str, str] = {
    "introduction": (
        "Check: Problem clearly defined? Key prior work cited? Gap explicitly stated? "
        "Contributions specific and testable? Each contribution maps to an experiment?"
    ),
    "related work": (
        "Check: Thematic organization (not study-by-study)? Seminal papers present? "
        "Clear positioning of proposed method vs. prior work? Balanced perspectives?"
    ),
    "method": (
        "Check: Equations correct? Notation consistent with other sections? "
        "All components justified? Reproducible from description alone? "
        "Design choices explained (why this architecture, not alternatives)?"
    ),
    "experiment": (
        "Check: Sufficient baselines? Ablation for each claimed contribution? "
        "Error bars / std present? Implementation details for reproducibility? "
        "Every Intro contribution has corresponding evidence here?"
    ),
    "conclusion": (
        "Check: Claims supported by actual results (no over-claiming)? "
        "Limitations honest and specific? Future directions actionable?"
    ),
}

_REVIEW_RED_FLAGS = (
    "RED FLAGS: Overstated conclusions, missing ablation, inconsistent notation, "
    "no error bars, causal claims from correlational data, selective reporting, "
    "SOTA claims without proper baselines."
)


def get_review_guidance(section_heading: str) -> str:
    """Return focused review guidance for a specific section (~100-150 tokens).

    Injected into the user prompt of _review_single_section().
    """
    heading_lower = section_heading.lower()
    specific = ""
    for key, guidance in _REVIEW_SECTIONS.items():
        if key in heading_lower:
            specific = f"\nFOCUS: {guidance}"
            break
    return f"\n{_REVIEW_CORE}{specific}\n{_REVIEW_RED_FLAGS}\n"


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE_GEN — appended to the per-figure user prompt, not system prompt.
# The existing CHART_CODE_SYSTEM already has detailed styling rules;
# this adds a concise quality checklist.
# ═══════════════════════════════════════════════════════════════════════════
FIGURE_CHECKLIST = """\
QUALITY CHECKLIST (verify before outputting):
- All axes labeled with units ("Accuracy (%)", "Latency (ms)")
- Error bars when std available (define SD/SEM in caption)
- No title inside figure (LaTeX caption = title)
- Redundant encoding: line styles + markers + colors
- Readable in grayscale (hatching for bars)
- Proposed method always COLORS[0], consistent across figures"""
