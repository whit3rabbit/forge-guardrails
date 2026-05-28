from __future__ import annotations

import hashlib
import json
from typing import Any


TOOLCALL_INPUT_SCHEMA_VERSION_V1 = "toolcall-verifier-input/v1"
TOOLCALL_INPUT_SCHEMA_VERSION_V2 = "toolcall-verifier-input/v2"
FINAL_RESPONSE_INPUT_SCHEMA_VERSION = "final-response-verifier-input/v1"


def stable_id(*parts: Any, length: int = 16) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def compact_json(obj: Any, max_chars: int = 2500) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(obj)
    if len(text) > max_chars:
        return text[:max_chars] + "...<truncated>"
    return text


def normalize_tool_for_prompt(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function = tool["function"]
        return {
            "name": str(function.get("name", "unknown_tool")),
            "description": str(function.get("description", "")),
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        }
    return {
        "name": str(tool.get("name", tool.get("tool_name", "unknown_tool"))),
        "description": str(tool.get("description", tool.get("desc", ""))),
        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
    }


def serialize_tool(tool: dict[str, Any]) -> str:
    normalized = normalize_tool_for_prompt(tool)
    return (
        f"{normalized['name']}: {normalized['description']}\n"
        f"PARAMETERS: {compact_json(normalized['parameters'], 1200)}"
    )


def _json_or_null(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value, ensure_ascii=False)


def serialize_state_v1(input_obj: dict[str, Any]) -> str:
    ws = input_obj["workflow_state"]
    tool_text = "\n\n".join(serialize_tool(t) for t in input_obj["available_tools"])
    return f"""SCHEMA_VERSION:
{input_obj['schema_version']}

USER_REQUEST:
{input_obj['user_request']}

WORKFLOW_STATE:
required_steps={ws.get('required_steps', [])}
completed_steps={ws.get('completed_steps', [])}
pending_steps={ws.get('pending_steps', [])}
terminal_tools={ws.get('terminal_tools', [])}
recent_errors={ws.get('recent_errors', [])}

AVAILABLE_TOOLS:
{tool_text}

CANDIDATE_CALL:
{compact_json(input_obj['candidate_call'], 2400)}
""".strip()


def serialize_state_v2(input_obj: dict[str, Any]) -> str:
    metadata = input_obj.get("metadata") or {}
    base = serialize_state_v1(input_obj)
    return base + f"""

SCORING_METADATA:
scenario_family={_json_or_null(metadata.get('scenario_family'))}
requires_transform={_json_or_null(metadata.get('requires_transform'))}
requires_synthesis={_json_or_null(metadata.get('requires_synthesis'))}
requires_all_tool_facts={_json_or_null(metadata.get('requires_all_tool_facts'))}
must_acknowledge_missing_data={_json_or_null(metadata.get('must_acknowledge_missing_data'))}"""


def serialize_final_response_state_v1(input_obj: dict[str, Any]) -> str:
    ws = input_obj["workflow_state"]
    metadata = input_obj.get("metadata") or {}
    tool_results = "\n".join(
        f"{r.get('tool_name', '')}: {json.dumps(str(r.get('content', '')), ensure_ascii=False)}"
        for r in input_obj.get("tool_results", [])
    )
    return f"""SCHEMA_VERSION:
{input_obj['schema_version']}

USER_REQUEST:
{input_obj['user_request']}

WORKFLOW_STATE:
required_steps={ws.get('required_steps', [])}
completed_steps={ws.get('completed_steps', [])}
pending_steps={ws.get('pending_steps', [])}
terminal_tools={ws.get('terminal_tools', [])}
recent_errors={ws.get('recent_errors', [])}

REQUIRED_FACTS:
{input_obj.get('required_facts', [])}

TOOL_TRACE:
{input_obj.get('tool_trace', [])}

TOOL_RESULTS:
{tool_results}

CANDIDATE_FINAL_RESPONSE:
{input_obj.get('candidate_final_response', '')}

SCORING_METADATA:
scenario_family={_json_or_null(metadata.get('scenario_family'))}
requires_transform={_json_or_null(metadata.get('requires_transform'))}
requires_synthesis={_json_or_null(metadata.get('requires_synthesis'))}
requires_all_tool_facts={_json_or_null(metadata.get('requires_all_tool_facts'))}
must_acknowledge_missing_data={_json_or_null(metadata.get('must_acknowledge_missing_data'))}""".strip()


def scorer_input_hash(input_obj: dict[str, Any], serializer: str, kind: str) -> str:
    if kind == "final_response":
        serialized = serialize_final_response_state_v1(input_obj)
        serializer_name = "serialize_final_response_state_v1"
    elif serializer == "v2":
        serialized = serialize_state_v2(input_obj)
        serializer_name = "serialize_state_v2"
    else:
        serialized = serialize_state_v1(input_obj)
        serializer_name = "serialize_state_v1"
    return hashlib.sha256(f"{serializer_name}\n{serialized}".encode("utf-8")).hexdigest()


def json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def infer_parameters_schema(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            str(key): {"type": json_type(value)}
            for key, value in sorted(args.items(), key=lambda item: str(item[0]))
        },
    }


def build_tool_spec(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"Agent log tool observed as {name}.",
        "parameters": infer_parameters_schema(args),
    }


def scoring_metadata() -> dict[str, Any]:
    return {
        "scenario_family": "agent_log",
        "requires_transform": False,
        "requires_synthesis": False,
        "requires_all_tool_facts": False,
        "must_acknowledge_missing_data": False,
    }
