from __future__ import annotations

from typing import Any


WORKFLOW_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["required_steps", "completed_steps", "pending_steps", "terminal_tools", "recent_errors"],
    "additionalProperties": True,
    "properties": {
        "required_steps": {"type": "array", "items": {"type": "string"}},
        "completed_steps": {"type": "array", "items": {"type": "string"}},
        "pending_steps": {"type": "array", "items": {"type": "string"}},
        "terminal_tools": {"type": "array", "items": {"type": "string"}},
        "recent_errors": {"type": "array", "items": {"type": "string"}},
    },
}

TOOL_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "description", "parameters"],
    "additionalProperties": True,
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "parameters": {"type": "object"},
    },
}

TOOL_CALL_TRAINING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "input", "label", "review"],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"const": "toolcall-verifier-training/v1"},
        "input": {
            "type": "object",
            "required": [
                "schema_version",
                "user_request",
                "workflow_state",
                "available_tools",
                "candidate_call",
            ],
            "additionalProperties": True,
            "properties": {
                "schema_version": {
                    "enum": ["toolcall-verifier-input/v1", "toolcall-verifier-input/v2"]
                },
                "user_request": {"type": "string"},
                "workflow_state": WORKFLOW_STATE_SCHEMA,
                "available_tools": {"type": "array", "items": TOOL_SPEC_SCHEMA},
                "candidate_call": {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "additionalProperties": True,
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                },
                "metadata": {"type": "object"},
            },
        },
        "label": {
            "enum": [
                "valid",
                "wrong_tool_semantic",
                "wrong_arguments_semantic",
                "tool_not_needed",
                "needs_clarification",
                "deterministic_invalid",
            ]
        },
        "review": {"type": "object"},
        "corrected_positive": {"type": "object"},
    },
}

FINAL_RESPONSE_TRAINING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "input", "label", "review"],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"const": "final-response-verifier-training/v1"},
        "input": {
            "type": "object",
            "required": [
                "schema_version",
                "user_request",
                "workflow_state",
                "required_facts",
                "tool_trace",
                "tool_results",
                "candidate_final_response",
            ],
            "additionalProperties": True,
            "properties": {
                "schema_version": {"const": "final-response-verifier-input/v1"},
                "user_request": {"type": "string"},
                "workflow_state": WORKFLOW_STATE_SCHEMA,
                "required_facts": {"type": "array", "items": {"type": "string"}},
                "tool_trace": {"type": "array", "items": {"type": "string"}},
                "tool_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["tool_name", "content"],
                        "additionalProperties": True,
                        "properties": {
                            "tool_name": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                },
                "candidate_final_response": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
        "label": {
            "enum": [
                "valid_final_response",
                "missing_tool_fact",
                "contradicts_tool_result",
                "unsupported_claim",
                "failed_to_acknowledge_data_gap",
            ]
        },
        "review": {"type": "object"},
        "corrected_positive": {"type": "object"},
    },
}
