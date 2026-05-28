from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You review sanitized agent tool-use transcripts for Forge verifier training. "
    "Output exactly one JSON object and no other text. Do not include hidden reasoning. "
    "Do not infer hidden facts. Prefer needs_human_review over guessing."
)


REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "disposition",
        "label",
        "confidence",
        "rationale",
        "corrected_candidate_call",
        "corrected_final_response",
        "required_facts",
        "metadata",
        "privacy_warnings",
    ],
    "additionalProperties": False,
    "properties": {
        "disposition": {"type": "string", "enum": ["training_row", "quarantine"]},
        "label": {
            "type": "string",
            "enum": [
                "valid",
                "wrong_tool_semantic",
                "wrong_arguments_semantic",
                "tool_not_needed",
                "needs_clarification",
                "deterministic_invalid",
                "valid_final_response",
                "missing_tool_fact",
                "contradicts_tool_result",
                "unsupported_claim",
                "failed_to_acknowledge_data_gap",
                "needs_human_review",
            ],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "corrected_candidate_call": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "additionalProperties": True,
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                },
            ]
        },
        "corrected_final_response": {"anyOf": [{"type": "null"}, {"type": "string"}]},
        "required_facts": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
        "privacy_warnings": {"type": "array", "items": {"type": "string"}},
    },
}

REVIEW_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "approve_training_row",
        "confidence",
        "rationale",
        "corrected_disposition",
        "corrected_label",
        "privacy_warnings",
    ],
    "additionalProperties": False,
    "properties": {
        "approve_training_row": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "corrected_disposition": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "enum": ["training_row", "quarantine"]},
            ]
        },
        "corrected_label": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "string",
                    "enum": [
                        "valid",
                        "wrong_tool_semantic",
                        "wrong_arguments_semantic",
                        "tool_not_needed",
                        "needs_clarification",
                        "deterministic_invalid",
                        "valid_final_response",
                        "missing_tool_fact",
                        "contradicts_tool_result",
                        "unsupported_claim",
                        "failed_to_acknowledge_data_gap",
                        "needs_human_review",
                    ],
                },
            ]
        },
        "privacy_warnings": {"type": "array", "items": {"type": "string"}},
    },
}


def review_schema_prompt() -> str:
    return (
        "Output fields, all required: disposition, label, confidence, rationale, "
        "corrected_candidate_call, corrected_final_response, required_facts, metadata, privacy_warnings.\n"
        "Allowed dispositions: training_row, quarantine.\n"
        "Allowed labels: valid, wrong_tool_semantic, wrong_arguments_semantic, tool_not_needed, "
        "needs_clarification, deterministic_invalid, valid_final_response, missing_tool_fact, "
        "contradicts_tool_result, unsupported_claim, failed_to_acknowledge_data_gap, needs_human_review.\n"
        "Use null for unavailable corrected fields, [] for no required_facts/privacy_warnings, "
        "and an empty object for metadata. Do not output or repeat a JSON Schema."
    )


def tool_call_review_prompt(payload: dict[str, Any]) -> str:
    return (
        "Review this sanitized tool-call transcript for a Forge tool-call verifier.\n"
        "Choose one tool-call label. If the call is a normal exploratory or verification "
        "tool use, label it valid even when the command output contains a project test failure. "
        "For shell commands that inspect project state during a broad coding request, default valid "
        "unless visible evidence proves the command cannot support the request. "
        "Only label wrong_arguments_semantic when the tool arguments themselves are wrong "
        "for the user request/workflow. If unsure, use label needs_human_review and disposition quarantine.\n\n"
        f"{review_schema_prompt()}\n\n"
        "Sanitized transcript JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def final_response_review_prompt(payload: dict[str, Any]) -> str:
    return (
        "Review this sanitized terminal response for a Forge final-response verifier.\n"
        "Choose a final-response label. Mark missing facts, contradictions, unsupported claims, "
        "or unacknowledged data gaps only when visible in the sanitized tool trace. "
        "If unsure, use label needs_human_review and disposition quarantine.\n\n"
        f"{review_schema_prompt()}\n\n"
        "Sanitized transcript JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def review_verification_prompt(payload: dict[str, Any], decision: dict[str, Any]) -> str:
    return (
        "Verify a proposed Forge verifier training row. You are a gatekeeper, not a second generator.\n"
        "Approve only when the proposed label and rationale are supported by the sanitized transcript. "
        "For valid tool-call labels, normal exploration, inspection, edits, and verification during a coding task are acceptable. "
        "For non-valid labels, require visible evidence that the call is truly wrong, unnecessary, or ambiguous. "
        "If the proposed row would train a weak or speculative label, set approve_training_row=false. "
        "Do not infer hidden facts.\n\n"
        "Output fields, all required: approve_training_row, confidence, rationale, "
        "corrected_disposition, corrected_label, privacy_warnings. Use null for corrected fields "
        "when no obvious correction exists, and [] for no privacy warnings.\n\n"
        "Sanitized transcript JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}\n\n"
        "Proposed review decision JSON:\n"
        f"{json.dumps(decision, ensure_ascii=False, sort_keys=True)}"
    )
