"""NanoResearch — Minimal AI-driven research engine."""

from pathlib import Path

__version__ = "0.1.0"


def get_style_files(format_name: str) -> list[Path]:
    templates_dir = Path(__file__).parent
    template_path = templates_dir / format_name
    if not template_path.is_dir():
        return []
    style_exts = {".sty", ".cls", ".bst"}
    return sorted(
        f for f in template_path.iterdir()
        if f.is_file() and f.suffix.lower() in style_exts
    )