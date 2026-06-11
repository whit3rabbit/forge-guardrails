from __future__ import annotations

import json

from generatetd.models import GenerateOptions, ReviewDecision, ReviewVerification, ToolObservation
from generatetd.pipeline import Quarantine, build_tool_row, dedupe_rows, generate, generate_synthetic_tool_rows
from generatetd.serialization import TOOLCALL_INPUT_SCHEMA_VERSION_V1, build_tool_spec


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_generate_no_api_outputs_private_positive_and_quarantine(tmp_path):
    codex_root = tmp_path / ".codex"
    session = codex_root / "sessions" / "2026" / "05" / "01" / "s.jsonl"
    write_jsonl(
        session,
        [
            {"timestamp": "2026-05-01T00:00:00Z", "type": "session_meta", "payload": {"id": "s1", "cwd": "/Users/alice/repo"}},
            {
                "timestamp": "2026-05-01T00:00:01Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
            },
            {
                "timestamp": "2026-05-01T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": ["zsh", "-lc", "ls"], "workdir": "/Users/alice/repo"}),
                    "call_id": "ok",
                },
            },
            {
                "timestamp": "2026-05-01T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "ok",
                    "output": json.dumps({"output": "README.md", "metadata": {"exit_code": 0}}),
                },
            },
            {
                "timestamp": "2026-05-01T00:00:04Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": ["zsh", "-lc", "false"], "workdir": "/Users/alice/repo"}),
                    "call_id": "bad",
                },
            },
            {
                "timestamp": "2026-05-01T00:00:05Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "bad",
                    "output": json.dumps({"output": "boom", "metadata": {"exit_code": 1}}),
                },
            },
        ],
    )
    out = tmp_path / "out"
    manifest = generate(
        GenerateOptions(
            out=out,
            include_codex=True,
            include_claude=False,
            no_api=True,
            provider="none",
            codex_root=codex_root,
            claude_root=tmp_path / ".claude",
        )
    )
    assert manifest["counts"]["tool_rows"] == 1
    assert manifest["counts"]["quarantine"] == 1
    row = json.loads((out / "tool_call_training.jsonl").read_text().splitlines()[0])
    assert row["label"] == "valid"
    assert row["review"]["private_agent_log"] is True
    assert "/Users/alice" not in json.dumps(row)
    quarantine = json.loads((out / "quarantine.jsonl").read_text().splitlines()[0])
    assert quarantine["reason"] == "needs_llm_review_for_failed_tool"


def test_generate_tool_calls_only_skips_final_response_candidates(tmp_path):
    codex_root = tmp_path / ".codex"
    session = codex_root / "sessions" / "2026" / "05" / "01" / "s.jsonl"
    write_jsonl(
        session,
        [
            {"timestamp": "2026-05-01T00:00:00Z", "type": "session_meta", "payload": {"id": "s1", "cwd": "/Users/alice/repo"}},
            {
                "timestamp": "2026-05-01T00:00:01Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]},
            },
            {
                "timestamp": "2026-05-01T00:00:02Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": json.dumps({"command": ["zsh", "-lc", "ls"], "workdir": "/Users/alice/repo"}),
                    "call_id": "ok",
                },
            },
            {
                "timestamp": "2026-05-01T00:00:03Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "ok",
                    "output": json.dumps({"output": "README.md", "metadata": {"exit_code": 0}}),
                },
            },
            {
                "timestamp": "2026-05-01T00:00:04Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "README.md exists."}]},
            },
        ],
    )
    out = tmp_path / "out"
    manifest = generate(
        GenerateOptions(
            out=out,
            include_codex=True,
            include_claude=False,
            no_api=True,
            provider="none",
            codex_root=codex_root,
            claude_root=tmp_path / ".claude",
            tool_calls_only=True,
        )
    )
    assert manifest["counts"]["tool_rows"] == 1
    assert manifest["counts"]["final_response_rows"] == 0
    assert manifest["counts"]["quarantine"] == 0


def test_dedupe_conflicts_on_same_input_different_label():
    input_obj = {
        "schema_version": TOOLCALL_INPUT_SCHEMA_VERSION_V1,
        "user_request": "Do it",
        "workflow_state": {
            "required_steps": [],
            "completed_steps": [],
            "pending_steps": [],
            "terminal_tools": [],
            "recent_errors": [],
        },
        "available_tools": [build_tool_spec("run", {"x": "1"})],
        "candidate_call": {"name": "run", "arguments": {"x": "1"}},
    }
    first = {"input": input_obj, "label": "valid", "review": {}}
    second = {"input": input_obj, "label": "wrong_arguments_semantic", "review": {}}
    rows, conflicts = dedupe_rows([first, second], "v1", "tool_call")
    assert len(rows) == 1
    assert len(conflicts) == 1


class LowConfidenceNegativeClient:
    def review_tool_call(self, payload):
        return ReviewDecision(
            disposition="training_row",
            label="wrong_tool_semantic",
            confidence=0.75,
            rationale="weak objection",
        )


class ValidReviewClient:
    provider_name = "fake"
    model = "fake-model"

    def review_tool_call(self, payload):
        return ReviewDecision(
            disposition="training_row",
            label="valid",
            confidence=0.95,
            rationale="visible exploratory call",
        )


class RejectingVerifier:
    provider_name = "fake-verifier"
    model = "fake-verifier-model"

    def verify_review(self, payload, decision):
        return ReviewVerification(
            approve_training_row=False,
            confidence=0.9,
            rationale="proposal is too speculative",
        )


class ApprovingVerifier:
    provider_name = "fake-verifier"
    model = "fake-verifier-model"

    def verify_review(self, payload, decision):
        return ReviewVerification(
            approve_training_row=True,
            confidence=0.93,
            rationale="proposal is supported",
        )


def test_low_confidence_negative_tool_review_is_quarantined(tmp_path):
    obs = ToolObservation(
        source="codex",
        session_id="s1",
        cwd="/Users/alice/repo",
        timestamp="2026-05-01T00:00:00Z",
        user_request="Clean up the repo",
        tool_name="shell",
        arguments={"command": ["bash", "-lc", "rg -n config"]},
        output="",
        call_id="c1",
        source_path=tmp_path / "session.jsonl",
        line_no=1,
    )
    try:
        build_tool_row(obs, GenerateOptions(out=tmp_path), LowConfidenceNegativeClient())
    except Quarantine as exc:
        assert exc.record["reason"] == "low_confidence_non_valid_review"
        assert exc.record["details"]["label"] == "wrong_tool_semantic"
    else:
        raise AssertionError("expected low confidence negative review to quarantine")


def test_verifier_rejection_quarantines_reviewed_row(tmp_path):
    obs = tool_observation(tmp_path)
    try:
        build_tool_row(obs, GenerateOptions(out=tmp_path), ValidReviewClient(), RejectingVerifier())
    except Quarantine as exc:
        assert exc.record["reason"] == "review_verifier_rejected"
        assert exc.record["details"]["verification"]["approve_training_row"] is False
    else:
        raise AssertionError("expected verifier rejection")


def test_verifier_approval_adds_review_metadata(tmp_path):
    row = build_tool_row(tool_observation(tmp_path), GenerateOptions(out=tmp_path), ValidReviewClient(), ApprovingVerifier())
    assert row["label"] == "valid"
    assert row["review"]["verified_by"] == "ApprovingVerifier"
    assert row["review"]["verifier_confidence"] == 0.93


def test_synthetic_tool_rows_are_bounded_and_labeled(tmp_path):
    row = build_tool_row(tool_observation(tmp_path), GenerateOptions(out=tmp_path), None)
    synthetic = generate_synthetic_tool_rows(
        [row],
        GenerateOptions(
            out=tmp_path,
            synthetic_missing_argument=1,
            synthetic_wrong_tool=1,
            synthetic_tool_not_needed=1,
        ),
    )
    assert [item["label"] for item in synthetic] == [
        "wrong_arguments_semantic",
        "tool_not_needed",
    ]
    assert all(item["review"]["synthetic"] is True for item in synthetic)
    assert synthetic[0]["input"]["candidate_call"]["arguments"] == {}
    # The tool_not_needed request is now chosen from a pool deterministically per
    # source row id, so we check it is one of the pool strings rather than a
    # specific hardcoded value.
    from generatetd.pipeline import _TOOL_NOT_NEEDED_REQUESTS
    assert synthetic[1]["input"]["user_request"] in _TOOL_NOT_NEEDED_REQUESTS
    assert "pool_index" in synthetic[1]["review"]["synthetic_mutation"]


def tool_observation(tmp_path):
    return ToolObservation(
        source="codex",
        session_id="s1",
        cwd="/Users/alice/repo",
        timestamp="2026-05-01T00:00:00Z",
        user_request="Clean up the repo",
        tool_name="shell",
        arguments={"command": ["bash", "-lc", "ls -la"]},
        output="README.md",
        call_id="c1",
        source_path=tmp_path / "session.jsonl",
        line_no=1,
    )
