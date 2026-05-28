from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import FinalResponseObservation, ToolObservation


HOME_RE = re.compile(r"/Users/[^/\s\"']+")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
AUTH_RE = re.compile(r"(?i)\b(authorization\s*:\s*bearer)\s+[A-Za-z0-9._~+/=-]+")
SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|credential)\b\s*[:=]\s*['\"]?[^'\"\s,;]+"
)
KNOWN_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]+|hf_[A-Za-z0-9_]{16,}|xox[baprs]-[A-Za-z0-9-]{16,})\b"
)
LONG_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/=_-]{96,}\b")

PRIVATE_FINDINGS = [
    ("home_path", HOME_RE),
    ("email", EMAIL_RE),
    ("auth_header", AUTH_RE),
    ("secret_assignment", SECRET_ASSIGN_RE),
    ("known_token", KNOWN_TOKEN_RE),
]


def sanitize_text(value: Any, max_chars: int = 2000) -> str:
    text = "" if value is None else str(value)
    text = AUTH_RE.sub(r"\1 [REDACTED_TOKEN]", text)
    text = SECRET_ASSIGN_RE.sub("[REDACTED_SECRET]", text)
    text = KNOWN_TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = HOME_RE.sub("$HOME", text)
    text = LONG_BLOB_RE.sub("[REDACTED_BLOB]", text)
    text = _compact_large_diff(text)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def _compact_large_diff(text: str) -> str:
    lines = text.splitlines()
    if len(lines) < 80:
        return text
    diffish = sum(1 for line in lines if line.startswith(("+", "-", "@@", "diff --git")))
    if diffish < 40:
        return text
    kept = lines[:40]
    return "\n".join(kept + [f"...<truncated {len(lines) - len(kept)} diff lines>"])


def sanitize_json(value: Any, max_string_chars: int = 800, max_items: int = 40) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, max_string_chars)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                out["..."] = f"<truncated {len(value) - max_items} keys>"
                break
            out[str(key)] = sanitize_json(item, max_string_chars, max_items)
        return out
    if isinstance(value, list):
        items = [sanitize_json(item, max_string_chars, max_items) for item in value[:max_items]]
        if len(value) > max_items:
            items.append(f"<truncated {len(value) - max_items} items>")
        return items
    return value


def privacy_findings(value: Any) -> list[str]:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    findings: list[str] = []
    for name, pattern in PRIVATE_FINDINGS:
        if pattern.search(text):
            findings.append(name)
    return findings


def sanitize_path(path: Path) -> str:
    return sanitize_text(str(path), 400)


def sanitize_tool_observation(obs: ToolObservation) -> ToolObservation:
    args = sanitize_json(obs.arguments)
    effective = sanitize_json(obs.effective_arguments) if obs.effective_arguments is not None else None
    rewrite = sanitize_json(obs.hook_rewrite) if obs.hook_rewrite is not None else None
    return ToolObservation(
        source=obs.source,
        session_id=sanitize_text(obs.session_id, 160),
        cwd=sanitize_text(obs.cwd, 400),
        timestamp=obs.timestamp,
        user_request=sanitize_text(obs.user_request or "Tool call in agent session.", 1600),
        tool_name=sanitize_text(obs.tool_name, 160),
        arguments=args if isinstance(args, dict) else {"_raw": args},
        output=sanitize_text(obs.output, 2000),
        call_id=sanitize_text(obs.call_id, 160),
        source_path=obs.source_path,
        line_no=obs.line_no,
        exit_code=obs.exit_code,
        is_error=obs.is_error,
        effective_arguments=effective if isinstance(effective, dict) else None,
        hook_rewrite=rewrite if isinstance(rewrite, dict) else None,
    )


def sanitize_final_observation(obs: FinalResponseObservation) -> FinalResponseObservation:
    return FinalResponseObservation(
        source=obs.source,
        session_id=sanitize_text(obs.session_id, 160),
        cwd=sanitize_text(obs.cwd, 400),
        timestamp=obs.timestamp,
        user_request=sanitize_text(obs.user_request or "Terminal response in agent session.", 1600),
        final_text=sanitize_text(obs.final_text, 2000),
        tool_trace=[sanitize_text(item, 160) for item in obs.tool_trace[:40]],
        tool_results=[
            {
                "tool_name": sanitize_text(item.get("tool_name", ""), 160),
                "content": sanitize_text(item.get("content", ""), 1000),
            }
            for item in obs.tool_results[:20]
        ],
        source_path=obs.source_path,
        line_no=obs.line_no,
    )
