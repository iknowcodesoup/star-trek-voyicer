"""Tool functions litert-lm calls during a CI fix attempt.

litert-lm loads this module for --preset and calls these functions directly
when the model decides to call them. Keep the tool surface narrow: the fix
loop only needs to read a file, write a file, and record what it changed.
"""

from pathlib import Path

skill_root = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[4]
assets_dir = skill_root / "assets"
fix_summary_path = assets_dir / "last_fix_summary.txt"
touched_files_path = assets_dir / "touched_files.txt"

# nx runs lint/test/typecheck with cwd set to the project folder, so the
# paths in ruff/pytest/eslint/tsc output are relative to the project, not
# the repo root. Discover every app folder instead of hardcoding names, so
# a new app under apps/ resolves without touching this file.
apps_dir = repo_root / "apps"
project_roots = [repo_root] + sorted(
    path for path in apps_dir.iterdir() if path.is_dir()
)


def resolve_path(path: str) -> Path:
    """Find the real file for a path copied from lint/test output."""
    given_path = Path(path)
    if given_path.is_absolute():
        return given_path
    for root in project_roots:
        candidate_path = root / given_path
        if candidate_path.exists():
            return candidate_path
    return repo_root / given_path


def read_file(path: str) -> str:
    """Return the text content of a file in the repository."""
    return resolve_path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    """Overwrite a file in the repository with new content."""
    target_path = resolve_path(path)
    target_path.write_text(content, encoding="utf-8")
    with touched_files_path.open("a", encoding="utf-8") as touched_file:
        touched_file.write(f"{target_path}\n")
    return f"wrote {target_path}"


def report_fix(summary: str) -> str:
    """Record a one-line summary of the fix just made, ending the turn."""
    fix_summary_path.write_text(summary, encoding="utf-8")
    return "recorded"
