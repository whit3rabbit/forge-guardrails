from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator, ValidationError, validate

from .schemas import FINAL_RESPONSE_TRAINING_SCHEMA, TOOL_CALL_TRAINING_SCHEMA


def validate_tool_call_row(row: dict[str, Any]) -> None:
    """Validate a single tool call training data row against the schema and parameter definitions."""
    validate(instance=row, schema=TOOL_CALL_TRAINING_SCHEMA)
    call = row["input"]["candidate_call"]
    tool = _tool_by_name(row["input"]["available_tools"], call["name"])
    if tool is None:
        raise ValidationError(f"candidate tool {call['name']!r} is not in available_tools")
    parameters = tool.get("parameters")
    if isinstance(parameters, dict) and parameters:
        Draft202012Validator(parameters).validate(call.get("arguments", {}))


def validate_final_response_row(row: dict[str, Any]) -> None:
    """Validate a single final response training data row against the schema."""
    validate(instance=row, schema=FINAL_RESPONSE_TRAINING_SCHEMA)


def _tool_by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        if tool.get("name") == name:
            return tool
    return None
