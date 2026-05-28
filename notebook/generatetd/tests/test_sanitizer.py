from __future__ import annotations

from generatetd.sanitizer import privacy_findings, sanitize_json, sanitize_text


def test_sanitizer_redacts_private_values():
    text = "Authorization: Bearer sk-secret1234567890 path /Users/alice/repo email a@example.com API_KEY=abc123"
    safe = sanitize_text(text)
    assert "sk-secret" not in safe
    assert "/Users/alice" not in safe
    assert "a@example.com" not in safe
    assert "abc123" not in safe
    assert not privacy_findings(safe)


def test_sanitize_json_bounds_strings_and_lists():
    value = {"path": "/Users/alice/repo", "items": list(range(50)), "blob": "x" * 1200}
    safe = sanitize_json(value, max_string_chars=20, max_items=3)
    assert safe["path"] == "$HOME/repo"
    assert safe["items"][-1].startswith("<truncated")
    assert safe["blob"] == "[REDACTED_BLOB]"
