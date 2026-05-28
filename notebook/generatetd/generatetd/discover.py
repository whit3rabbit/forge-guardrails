from __future__ import annotations

from pathlib import Path


def codex_log_paths(codex_root: Path) -> list[Path]:
    paths: list[Path] = []
    sessions = codex_root / "sessions"
    if sessions.exists():
        paths.extend(sessions.glob("**/*.jsonl"))
    archived = codex_root / "archived_sessions"
    if archived.exists():
        paths.extend(archived.glob("*.jsonl"))
    return sorted(p for p in paths if p.is_file())


def claude_log_paths(claude_root: Path) -> list[Path]:
    projects = claude_root / "projects"
    if not projects.exists():
        return []
    return sorted(p for p in projects.glob("**/*.jsonl") if p.is_file())


def claude_history_path(claude_root: Path) -> Path:
    return claude_root / "history.jsonl"
