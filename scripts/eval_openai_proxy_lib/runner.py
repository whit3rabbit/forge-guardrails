from __future__ import annotations

import inspect
import json
import sys
import time
from typing import Any

from forge.core.workflow import ToolSpec, Workflow
from tests.eval.eval_runner import _build_workflow_with_capture
from tests.eval.scenarios import EvalScenario

from .client import (
    OpenAIProxyClient,
    _http_error_message,
)
from .models import ProxyRunResult, ProxyToolCall


FORGE_EXTENSION_FIELD = "_forge"
FORGE_TOOL_STATUS_FIELD = "tool_status"
FORGE_TOOL_STATUS_OK = "ok"
FORGE_TOOL_STATUS_ERROR = "error"
REDACTED_TERMINAL_TEXT = "[REDACTED]"


def _proxy_tool_specs(workflow: Workflow) -> list[ToolSpec]:
    specs = workflow.get_tool_specs()
    if "respond" not in workflow.terminal_tools:
        return specs
    # `respond` is proxy-reserved. If a scenario supplies respond(answer=...),
    # the proxy will strip it as respond(message=...) and lose the answer.
    return [spec for spec in specs if spec.name != "respond"]


def _proxy_terminal_tools(workflow: Workflow) -> list[str]:
    terminal_tools = set(workflow.terminal_tools)
    if any(tool != "respond" for tool in terminal_tools):
        return sorted(tool for tool in terminal_tools if tool != "respond")
    return sorted(terminal_tools | {"respond"})


def _openai_tool_call(call: ProxyToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments_json,
        },
    }


def _tool_result_message(call: ProxyToolCall, content: str, status: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": content,
        FORGE_EXTENSION_FIELD: {FORGE_TOOL_STATUS_FIELD: status},
    }


async def _call_tool(fn: Any, args: dict[str, Any]) -> Any:
    result = fn(**args)
    if inspect.isawaitable(result):
        return await result
    return result


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _is_proxy_failure_text(text: str) -> bool:
    return text.startswith("Retries exhausted after ") or text.startswith(
        "Max iterations ("
    )


def _terminal_args_from_text(workflow: Workflow, text: str) -> dict[str, Any]:
    terminal = next(iter(workflow.terminal_tools))
    fields = workflow.tools[terminal].spec.parameters.model_fields
    args: dict[str, Any] = {}
    for name, field in fields.items():
        annotation = str(field.annotation)
        if field.annotation is str or "str" in annotation:
            args[name] = text
    if args:
        return args
    if fields:
        first = next(iter(fields))
        args[first] = text
    return args


def _terminal_text(args: dict[str, Any]) -> str:
    preferred = (
        "message",
        "answer",
        "content",
        "findings",
        "summary",
        "reason",
        "report",
        "diagnosis",
        "action",
        "rationale",
        "candidate",
    )
    for key in preferred:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    for value in args.values():
        if isinstance(value, str) and value:
            return value
    return ""


def _validate_result(
    scenario: EvalScenario,
    terminal_args: dict[str, Any] | None,
    validate_state_fn: Any,
) -> tuple[bool | None, str | None]:
    accuracy: bool | None = None
    validate_error: str | None = None
    if scenario.validate and terminal_args is not None:
        try:
            accuracy = scenario.validate(terminal_args)
        except Exception as exc:  # pragma: no cover - defensive parity path
            accuracy = None
            validate_error = type(exc).__name__
    if validate_state_fn is not None:
        try:
            state_ok = validate_state_fn()
            accuracy = state_ok if accuracy is None else accuracy and state_ok
        except Exception as exc:  # pragma: no cover - defensive parity path
            accuracy = False
            validate_error = f"validate_state: {type(exc).__name__}"
    return accuracy, validate_error


def _required_step_diagnostics(
    workflow: Workflow,
    tool_sequence: list[str],
) -> tuple[list[str], bool]:
    called = set(tool_sequence)
    missing = [step for step in workflow.required_steps if step not in called]
    return missing, not missing


async def run_proxy_scenario(
    client: OpenAIProxyClient,
    scenario: EvalScenario,
    *,
    stream: bool,
    budget_tokens: int | None,
    ablation: Any,
    verbose: bool = False,
) -> ProxyRunResult:
    """Run an evaluation scenario against the proxy server client."""
    workflow, capture, validate_state_fn = _build_workflow_with_capture(
        scenario, ablation=ablation,
    )
    max_tool_errors = (
        ablation.max_tool_errors if ablation is not None else scenario.max_tool_errors
    )
    consecutive_tool_errors = 0
    proxy_tools = _proxy_tool_specs(workflow)
    proxy_required_steps = list(workflow.required_steps)
    proxy_terminal_tools = _proxy_terminal_tools(workflow)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": workflow.build_system_prompt()},
        {"role": "user", "content": scenario.user_message},
    ]
    result = ProxyRunResult(
        scenario_name=scenario.name,
        completeness=False,
        iterations_used=0,
    )
    started = time.monotonic()

    import httpx

    for _ in range(scenario.max_iterations):
        try:
            turn = await client.chat(
                messages,
                tools=proxy_tools,
                sampling=None,
                stream=stream,
                required_steps=proxy_required_steps,
                terminal_tools=proxy_terminal_tools,
            )
        except httpx.HTTPError as exc:
            result.iterations_used += 1
            result.error_type = type(exc).__name__
            result.error_message = _http_error_message(exc)
            break
        result.iterations_used += 1
        result.input_tokens += turn.input_tokens
        result.output_tokens += turn.output_tokens

        if turn.kind == "text":
            result.final_text = turn.content
            if _is_proxy_failure_text(turn.content):
                result.error_type = "ToolCallError"
                result.error_message = turn.content
                break
            terminal_args = _terminal_args_from_text(workflow, turn.content)
            accuracy, validate_error = _validate_result(
                scenario, terminal_args, validate_state_fn,
            )
            result.completeness = True
            result.terminal_args = terminal_args
            result.accuracy = accuracy
            result.validate_error = validate_error
            result.proxy_terminal_source = "text"
            break

        if not turn.tool_calls:
            result.error_type = "ToolCallError"
            result.error_message = "Proxy returned an empty tool call batch"
            break

        if any(call.reasoning for call in turn.tool_calls):
            result.reasoning_msgs += 1

        messages.append({
            "role": "assistant",
            "content": turn.tool_calls[0].reasoning,
            "tool_calls": [_openai_tool_call(call) for call in turn.tool_calls],
        })
        batch_had_error = False
        terminal_args = None
        for call in turn.tool_calls:
            result.tool_sequence.append(call.name)
            result.tool_args.append(call.args)
            tool_status = FORGE_TOOL_STATUS_OK
            try:
                fn = workflow.get_callable(call.name)
                tool_result = await _call_tool(fn, call.args)
            except Exception as exc:
                batch_had_error = True
                tool_status = FORGE_TOOL_STATUS_ERROR
                content = f"[ToolError] {type(exc).__name__}: {exc}"
            else:
                content = _stringify_tool_result(tool_result)
                if call.name in workflow.terminal_tools:
                    terminal_args = call.args

            messages.append(_tool_result_message(call, content, tool_status))

        if batch_had_error:
            result.tool_errors += 1
            consecutive_tool_errors += 1
            if consecutive_tool_errors > max_tool_errors:
                result.error_type = "ToolExecutionError"
                result.error_message = "Too many consecutive tool execution errors"
                break
        else:
            consecutive_tool_errors = 0

        if terminal_args is not None:
            terminal_args = capture.get("args") or terminal_args
            accuracy, validate_error = _validate_result(
                scenario, terminal_args, validate_state_fn,
            )
            result.completeness = True
            result.terminal_args = terminal_args
            result.accuracy = accuracy
            result.validate_error = validate_error
            result.final_text = _terminal_text(terminal_args)
            result.proxy_terminal_source = "tool_call"
            break

        if verbose:
            names = ", ".join(call.name for call in turn.tool_calls)
            print(f"    [tool_call] {names}", file=sys.stderr, flush=True)
    else:
        result.error_type = "MaxIterationsError"
        result.error_message = (
            f"Max iterations ({scenario.max_iterations}) exceeded. "
            "The proxy did not return a terminal text response or terminal tool call."
        )

    missing_steps, steps_satisfied = _required_step_diagnostics(
        workflow, result.tool_sequence,
    )
    result.proxy_missing_required_steps = missing_steps
    result.proxy_required_steps_satisfied = steps_satisfied
    result.elapsed_seconds = time.monotonic() - started
    return result


def _proxy_failure_classification(result: ProxyRunResult) -> str | None:
    if not result.completeness:
        return result.error_type or "incomplete"
    if not result.proxy_required_steps_satisfied:
        return "proxy_contract_mismatch"
    if result.accuracy is not False:
        return None
    if _has_redacted_terminal_content(result):
        return "terminal_redacted"
    return "accuracy_false"


def _has_redacted_terminal_content(result: ProxyRunResult) -> bool:
    return _is_redacted_terminal_text(result.final_text) or any(
        _is_redacted_terminal_text(value)
        for value in (result.terminal_args or {}).values()
    )


def _is_redacted_terminal_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == REDACTED_TERMINAL_TEXT


def _result_row(
    result: ProxyRunResult,
    scenario: EvalScenario,
    run_idx: int,
    model: str,
    stream: bool,
    ablation: str,
    budget_tokens: int | None,
    backend_label: str,
    mode_label: str,
    eval_target_backend: str,
    proxy_backend_mode: str | None = None,
) -> dict[str, Any]:
    required_step_mismatch = not result.proxy_required_steps_satisfied
    success = bool(
        result.completeness
        and result.accuracy is not False
        and not required_step_mismatch
    )
    ideal_iterations = scenario.ideal_iterations or (
        len(scenario.workflow.required_steps) + 1
    )
    wasted_calls = (
        max(0, result.iterations_used - ideal_iterations)
        if result.completeness
        else None
    )
    proxy_failure_classification = _proxy_failure_classification(result)

    row = {
        "impl": "rust-proxy-oracle",
        "model": model,
        "backend": backend_label,
        "mode": mode_label,
        "eval_target_backend": eval_target_backend,
        "ablation": ablation,
        "tool_choice": "auto",
        "scenario": result.scenario_name,
        "user_message": scenario.user_message,
        "run": run_idx,
        "stream": stream,
        "completeness": result.completeness,
        "success": success,
        "accuracy": result.accuracy,
        "iterations": result.iterations_used,
        "elapsed_s": round(result.elapsed_seconds, 2),
        "error_type": result.error_type,
        "error_message": result.error_message,
        "budget_tokens": (
            budget_tokens if budget_tokens is not None else scenario.budget_tokens
        ),
        "retry_nudges": result.retry_nudges,
        "step_nudges": result.step_nudges,
        "tool_errors": result.tool_errors,
        "reasoning_msgs": result.reasoning_msgs,
        "tool_sequence": result.tool_sequence,
        "tool_args": result.tool_args,
        "final_text": result.final_text,
        "proxy_terminal_source": result.proxy_terminal_source,
        "proxy_missing_required_steps": result.proxy_missing_required_steps,
        "proxy_required_steps_satisfied": result.proxy_required_steps_satisfied,
        "missing_required_steps": result.proxy_missing_required_steps,
        "required_step_mismatch": required_step_mismatch,
        "proxy_failure_classification": proxy_failure_classification,
        "raw_response_on_failure": result.error_message if not result.completeness else None,
        "ideal_iterations": ideal_iterations,
        "wasted_calls": wasted_calls,
        "compaction_events": 0,
    }
    if proxy_backend_mode:
        row["proxy_backend_mode"] = proxy_backend_mode
    if result.input_tokens or result.output_tokens:
        row["input_tokens"] = result.input_tokens
        row["output_tokens"] = result.output_tokens
    return row
