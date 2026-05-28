from __future__ import annotations

import json

from generatetd.parsers import parse_claude_file, parse_codex_file


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_codex_parser_pairs_function_call_output(tmp_path):
    path = tmp_path / "session.jsonl"
    write_jsonl(
        path,
        [
            {"timestamp": "2026-05-01T00:00:00Z", "type": "session_meta", "payload": {"id": "s1", "cwd": "/repo"}},
            {
                "timestamp": "2026-05-01T00:00:01Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Run tests"}]},
            },
            {
                "timestamp": "2026-05-01T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": ["zsh", "-lc", "pytest"], "workdir": "/repo"}),
                    "call_id": "call1",
                },
            },
            {
                "timestamp": "2026-05-01T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call1",
                    "output": json.dumps({"output": "failed", "metadata": {"exit_code": 1}}),
                },
            },
        ],
    )
    result = parse_codex_file(path)
    assert len(result.tool_observations) == 1
    obs = result.tool_observations[0]
    assert obs.tool_name == "shell"
    assert obs.arguments["command"] == ["zsh", "-lc", "pytest"]
    assert obs.output == "failed"
    assert obs.exit_code == 1
    assert obs.is_error is True
    assert obs.user_request == "Run tests"


def test_claude_parser_applies_pretool_rewrite(tmp_path):
    path = tmp_path / "claude.jsonl"
    write_jsonl(
        path,
        [
            {
                "type": "user",
                "sessionId": "s1",
                "cwd": "/repo",
                "timestamp": "2026-05-01T00:00:00Z",
                "message": {"role": "user", "content": "Check status"},
            },
            {
                "type": "assistant",
                "sessionId": "s1",
                "cwd": "/repo",
                "timestamp": "2026-05-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool1",
                            "name": "Bash",
                            "input": {"command": "git status", "description": "status"},
                        }
                    ],
                },
            },
            {
                "type": "attachment",
                "sessionId": "s1",
                "cwd": "/repo",
                "timestamp": "2026-05-01T00:00:02Z",
                "attachment": {
                    "hookEvent": "PreToolUse",
                    "toolUseID": "tool1",
                    "stdout": json.dumps({"hookSpecificOutput": {"updatedInput": {"command": "rtk git status"}}}),
                },
            },
            {
                "type": "user",
                "sessionId": "s1",
                "cwd": "/repo",
                "timestamp": "2026-05-01T00:00:03Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tool1", "content": "clean"}],
                },
                "toolUseResult": {"stdout": "clean", "stderr": "", "interrupted": False},
            },
        ],
    )
    result = parse_claude_file(path)
    assert len(result.tool_observations) == 1
    obs = result.tool_observations[0]
    assert obs.tool_name == "Bash"
    assert obs.arguments["command"] == "git status"
    assert obs.effective_arguments == {"command": "rtk git status"}
    assert obs.output == "clean"
