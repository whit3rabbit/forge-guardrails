from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ExtractionResult, FinalResponseObservation, ToolObservation


def _json_loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _dict_or_raw(value: Any) -> dict[str, Any]:
    loaded = _json_loads_maybe(value)
    if isinstance(loaded, dict):
        return loaded
    return {"_raw": loaded}


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append((line_no, obj))
    return rows


def _codex_message_text(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    pieces: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"input_text", "output_text", "text"}:
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    pieces.append(text)
    return "\n".join(pieces).strip()


def _is_environment_context(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("<environment_context>") or stripped.startswith("<user_info>")


def _parse_codex_output(output: Any) -> tuple[str, int | None, bool]:
    loaded = _json_loads_maybe(output)
    if isinstance(loaded, dict):
        text = str(loaded.get("output", loaded.get("content", "")))
        metadata = loaded.get("metadata") if isinstance(loaded.get("metadata"), dict) else {}
        exit_code = metadata.get("exit_code")
        exit_int = exit_code if isinstance(exit_code, int) else None
        return text, exit_int, bool(exit_int not in (None, 0))
    return str(output or ""), None, False


def parse_codex_file(path: Path) -> ExtractionResult:
    result = ExtractionResult(files_seen=1)
    session_id = path.stem
    cwd = ""
    current_user_request = ""
    pending: dict[str, ToolObservation] = {}
    recent_tool_results: list[dict[str, str]] = []
    recent_tool_trace: list[str] = []

    for line_no, obj in _read_jsonl(path):
        result.records_seen += 1
        typ = obj.get("type")
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        timestamp = str(obj.get("timestamp", ""))
        if typ == "session_meta":
            session_id = str(payload.get("id") or session_id)
            cwd = str(payload.get("cwd") or cwd)
            continue
        if typ != "response_item":
            continue
        item_type = payload.get("type")
        if item_type == "message":
            role = payload.get("role")
            text = _codex_message_text(payload)
            if role == "user" and text and not _is_environment_context(text):
                current_user_request = text
            elif role == "assistant" and text and recent_tool_results:
                result.final_observations.append(
                    FinalResponseObservation(
                        source="codex",
                        session_id=session_id,
                        cwd=cwd,
                        timestamp=timestamp,
                        user_request=current_user_request,
                        final_text=text,
                        tool_trace=list(recent_tool_trace),
                        tool_results=list(recent_tool_results),
                        source_path=path,
                        line_no=line_no,
                    )
                )
                recent_tool_results.clear()
                recent_tool_trace.clear()
            continue
        if item_type == "function_call":
            call_id = str(payload.get("call_id") or f"{path}:{line_no}")
            args = _dict_or_raw(payload.get("arguments", {}))
            pending[call_id] = ToolObservation(
                source="codex",
                session_id=session_id,
                cwd=cwd,
                timestamp=timestamp,
                user_request=current_user_request,
                tool_name=str(payload.get("name") or "unknown_tool"),
                arguments=args,
                output="",
                call_id=call_id,
                source_path=path,
                line_no=line_no,
            )
            continue
        if item_type == "function_call_output":
            call_id = str(payload.get("call_id") or "")
            output, exit_code, is_error = _parse_codex_output(payload.get("output"))
            obs = pending.pop(call_id, None)
            if obs is None:
                obs = ToolObservation(
                    source="codex",
                    session_id=session_id,
                    cwd=cwd,
                    timestamp=timestamp,
                    user_request=current_user_request,
                    tool_name="unknown_tool",
                    arguments={},
                    output="",
                    call_id=call_id or f"{path}:{line_no}",
                    source_path=path,
                    line_no=line_no,
                )
            obs.output = output
            obs.exit_code = exit_code
            obs.is_error = is_error
            result.tool_observations.append(obs)
            recent_tool_trace.append(obs.tool_name)
            recent_tool_results.append({"tool_name": obs.tool_name, "content": output})

    result.tool_observations.extend(pending.values())
    return result


def _load_claude_history(history_path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    if not history_path.exists():
        return prompts
    for _, obj in _read_jsonl(history_path):
        session_id = obj.get("sessionId")
        display = obj.get("display")
        if isinstance(session_id, str) and isinstance(display, str) and display.strip():
            prompts.setdefault(session_id, display.strip())
    return prompts


def _claude_text_content(message: dict[str, Any]) -> tuple[str, bool]:
    content = message.get("content")
    if isinstance(content, str):
        return content, False
    pieces: list[str] = []
    only_tool_results = True
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_result":
                continue
            only_tool_results = False
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                pieces.append(text)
    return "\n".join(pieces).strip(), only_tool_results


def _claude_tool_results(obj: dict[str, Any]) -> list[tuple[str, str, bool]]:
    message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    content = message.get("content")
    out: list[tuple[str, str, bool]] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            tool_id = str(item.get("tool_use_id") or "")
            text = item.get("content")
            if isinstance(text, list):
                text = json.dumps(text, ensure_ascii=False)
            out.append((tool_id, str(text or ""), bool(item.get("is_error"))))
    return out


def _hook_updated_input(attachment: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if attachment.get("hookEvent") != "PreToolUse":
        return None
    tool_id = attachment.get("toolUseID")
    if not isinstance(tool_id, str) or not tool_id:
        return None
    stdout = attachment.get("stdout")
    loaded = _json_loads_maybe(stdout)
    if not isinstance(loaded, dict):
        return None
    hook_output = loaded.get("hookSpecificOutput")
    if not isinstance(hook_output, dict):
        return None
    updated = hook_output.get("updatedInput")
    if isinstance(updated, dict):
        return tool_id, updated
    return None


def parse_claude_file(path: Path, history_prompts: dict[str, str] | None = None) -> ExtractionResult:
    history_prompts = history_prompts or {}
    result = ExtractionResult(files_seen=1)
    session_id = path.stem
    cwd = ""
    current_user_request = history_prompts.get(session_id, "")
    pending: dict[str, ToolObservation] = {}
    recent_tool_results: list[dict[str, str]] = []
    recent_tool_trace: list[str] = []

    for line_no, obj in _read_jsonl(path):
        result.records_seen += 1
        timestamp = str(obj.get("timestamp", ""))
        session_id = str(obj.get("sessionId") or session_id)
        cwd = str(obj.get("cwd") or cwd)
        typ = obj.get("type")
        if typ == "attachment" and isinstance(obj.get("attachment"), dict):
            updated = _hook_updated_input(obj["attachment"])
            if updated is not None:
                tool_id, rewritten = updated
                if tool_id in pending:
                    pending[tool_id].effective_arguments = rewritten
                    pending[tool_id].hook_rewrite = rewritten
            continue
        message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        if typ == "user":
            tool_results = _claude_tool_results(obj)
            if tool_results:
                tool_use_result = obj.get("toolUseResult") if isinstance(obj.get("toolUseResult"), dict) else {}
                for tool_id, text, item_error in tool_results:
                    obs = pending.pop(tool_id, None)
                    if obs is None:
                        continue
                    stdout = tool_use_result.get("stdout")
                    stderr = tool_use_result.get("stderr")
                    parts = []
                    if isinstance(stdout, str) and stdout:
                        parts.append(stdout)
                    if isinstance(stderr, str) and stderr:
                        parts.append(stderr)
                    obs.output = "\n".join(parts) if parts else text
                    obs.is_error = item_error or bool(tool_use_result.get("interrupted"))
                    result.tool_observations.append(obs)
                    recent_tool_trace.append(obs.tool_name)
                    recent_tool_results.append({"tool_name": obs.tool_name, "content": obs.output})
            else:
                text, _ = _claude_text_content(message)
                if text:
                    current_user_request = text
            continue
        if typ == "assistant":
            content = message.get("content")
            made_tool_call = False
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "tool_use":
                        continue
                    made_tool_call = True
                    tool_id = str(item.get("id") or f"{path}:{line_no}")
                    args = item.get("input") if isinstance(item.get("input"), dict) else {}
                    pending[tool_id] = ToolObservation(
                        source="claude",
                        session_id=session_id,
                        cwd=cwd,
                        timestamp=timestamp,
                        user_request=current_user_request,
                        tool_name=str(item.get("name") or "unknown_tool"),
                        arguments=args,
                        output="",
                        call_id=tool_id,
                        source_path=path,
                        line_no=line_no,
                    )
            text, only_tool_results = _claude_text_content(message)
            if text and not made_tool_call and not only_tool_results and recent_tool_results:
                result.final_observations.append(
                    FinalResponseObservation(
                        source="claude",
                        session_id=session_id,
                        cwd=cwd,
                        timestamp=timestamp,
                        user_request=current_user_request,
                        final_text=text,
                        tool_trace=list(recent_tool_trace),
                        tool_results=list(recent_tool_results),
                        source_path=path,
                        line_no=line_no,
                    )
                )
                recent_tool_results.clear()
                recent_tool_trace.clear()

    result.tool_observations.extend(pending.values())
    return result


def merge_results(results: list[ExtractionResult]) -> ExtractionResult:
    merged = ExtractionResult()
    for result in results:
        merged.files_seen += result.files_seen
        merged.records_seen += result.records_seen
        merged.tool_observations.extend(result.tool_observations)
        merged.final_observations.extend(result.final_observations)
    return merged
