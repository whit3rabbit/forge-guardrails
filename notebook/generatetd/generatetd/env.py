from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_env_file(path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Parse key-value pairs from an environment file and load them into os.environ."""
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value.strip())
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def env_default(primary: str, fallback: str, *aliases: str) -> str:
    """Retrieve an environment variable by name or aliases, falling back to a default value."""
    for key in (primary, *aliases):
        value = os.getenv(key)
        if value:
            return value
    return fallback


def _clean_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
