#!/usr/bin/env python3
"""Run upstream Forge eval scenarios against an OpenAI-compatible proxy.

This wrapper lives outside the upstream `forge/` submodule on purpose. It uses
the Python eval scenarios as the oracle while targeting a Rust Forge proxy at
`--base-url`.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = ROOT / "forge"
sys.path.insert(0, str(FORGE_ROOT / "src"))
sys.path.insert(0, str(FORGE_ROOT))
_tests_package = sys.modules.get("tests")
if _tests_package is not None and hasattr(_tests_package, "__path__"):
    _forge_tests_path = str(FORGE_ROOT / "tests")
    _tests_paths = list(_tests_package.__path__)
    if _forge_tests_path not in _tests_paths:
        _tests_package.__path__ = [_forge_tests_path, *_tests_paths]


def _print_early_help() -> None:
    parser = argparse.ArgumentParser(
        description="Run upstream Python eval scenarios against an OpenAI proxy"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--scenario", nargs="*")
    parser.add_argument("--tags", nargs="*")
    parser.add_argument(
        "--ablation",
        choices=[
            "bare",
            "no_compact",
            "no_nudge",
            "no_recovery",
            "no_rescue",
            "no_steps",
            "reforged",
        ],
        default="reforged",
    )
    parser.add_argument("--output")
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--backend-label", default="openai-proxy")
    parser.add_argument("--mode-label", default="proxy")
    parser.add_argument("--proxy-backend-mode", choices=["native", "prompt"])
    parser.add_argument("--eval-target-backend", default="openai-proxy")
    parser.add_argument("--no-recommended-sampling", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.print_help()


if __name__ == "__main__" and any(arg in {"--help", "-h"} for arg in sys.argv[1:]):
    _print_early_help()
    raise SystemExit(0)


from forge.clients.base import format_tool
from forge.clients.sampling_defaults import get_sampling_defaults
from forge.core.workflow import ToolSpec, Workflow
from tests.eval.ablation import ABLATION_PRESETS
from tests.eval.eval_runner import _build_workflow_with_capture
from tests.eval.scenarios import ALL_SCENARIOS, EvalScenario


FORGE_EXTENSION_FIELD = "_forge"
FORGE_TOOL_STATUS_FIELD = "tool_status"
FORGE_TOOL_STATUS_OK = "ok"
FORGE_TOOL_STATUS_ERROR = "error"
REDACTED_TERMINAL_TEXT = "[REDACTED]"


@dataclass
class ProxyToolCall:
    id: str
    name: str
    args: dict[str, Any]
    arguments_json: str
    reasoning: str | None = None


@dataclass
class ProxyTurn:
    kind: str
    content: str = ""
    tool_calls: list[ProxyToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProxyRunResult:
    scenario_name: str
    completeness: bool
    iterations_used: int
    terminal_args: dict[str, Any] | None = None
    accuracy: bool | None = None
    validate_error: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retry_nudges: int = 0
    step_nudges: int = 0
    tool_errors: int = 0
    reasoning_msgs: int = 0
    tool_sequence: list[str] = field(default_factory=list)
    tool_args: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    proxy_terminal_source: str | None = None
    proxy_missing_required_steps: list[str] = field(default_factory=list)
    proxy_required_steps_satisfied: bool = True


class OpenAIProxyClient:
    """Minimal Python LLMClient for `/v1/chat/completions` proxies."""

    api_format = "openai"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        recommended_sampling: bool = True,
    ) -> None:
        self.base_url = _chat_completions_url(base_url)
        self.model = model
        self.timeout = timeout
        self.sampling_defaults = (
            get_sampling_defaults(model) if recommended_sampling else {}
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        stream: bool = False,
        required_steps: list[str] | None = None,
        terminal_tools: list[str] | None = None,
    ) -> ProxyTurn:
        import httpx

        body = self._body(
            messages,
            tools,
            sampling,
            stream=stream,
            required_steps=required_steps,
            terminal_tools=terminal_tools,
        )
        if stream:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.base_url, json=body) as response:
                    if response.status_code >= 400:
                        await response.aread()
                    response.raise_for_status()
                    return await _parse_openai_sse(response)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, json=body)
            response.raise_for_status()
        return _parse_openai_response(response.json())

    def _body(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None,
        stream: bool,
        required_steps: list[str] | None = None,
        terminal_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            body["tools"] = [format_tool(tool) for tool in tools]
        if required_steps:
            forge: dict[str, Any] = {"required_steps": required_steps}
            if terminal_tools:
                forge["terminal_tools"] = terminal_tools
            body["_forge"] = forge
        body.update(self.sampling_defaults)
        if sampling:
            body.update(sampling)
        return body


def _chat_completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


def _http_error_message(exc: Any) -> str:
    message = str(exc)
    response = getattr(exc, "response", None)
    if response is None:
        return message
    try:
        response_text = response.text
    except Exception:
        return message
    if not response_text:
        return message
    return f"{message}: {response_text}"


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int]:
    if not usage:
        return 0, 0
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return prompt, completion


def _parse_openai_response(data: dict[str, Any]) -> ProxyTurn:
    input_tokens, output_tokens = _usage_tokens(data.get("usage"))
    choices = data.get("choices") or []
    if not choices:
        return ProxyTurn("text", input_tokens=input_tokens, output_tokens=output_tokens)
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        parsed: list[ProxyToolCall] = []
        reasoning = message.get("content") or None
        for index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            arguments_json = _arguments_json(function.get("arguments"))
            parsed.append(ProxyToolCall(
                id=call.get("id") or f"call_{index}",
                name=function.get("name", ""),
                args=_parse_args(arguments_json),
                arguments_json=arguments_json,
                reasoning=reasoning if index == 0 else None,
            ))
        return ProxyTurn(
            "tool_call",
            tool_calls=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    return ProxyTurn(
        "text",
        content=message.get("content") or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _arguments_json(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return json.dumps(raw, separators=(",", ":"))
    return "{}"


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


async def _parse_openai_sse(response: Any) -> ProxyTurn:
    content = ""
    tool_parts: dict[int, dict[str, Any]] = {}
    input_tokens = 0
    output_tokens = 0
    final_reason: str | None = None

    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        data = json.loads(payload)

        usage = data.get("usage")
        if usage:
            input_tokens, output_tokens = _usage_tokens(usage)

        choices = data.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}

        if "content" in delta:
            text = delta.get("content") or ""
            content += text

        for part in delta.get("tool_calls") or []:
            index = int(part.get("index", 0))
            existing = tool_parts.setdefault(
                index,
                {
                    "id": f"call_{index}",
                    "name": "",
                    "arguments": "",
                    "reasoning": None,
                },
            )
            if part.get("id"):
                existing["id"] = part["id"]
            function = part.get("function") or {}
            if function.get("name"):
                existing["name"] = function["name"]
            if function.get("arguments"):
                existing["arguments"] += function["arguments"]

        if choice.get("finish_reason") in {"stop", "tool_calls"}:
            final_reason = choice["finish_reason"]

    if final_reason == "tool_calls":
        calls = [
            ProxyToolCall(
                id=part["id"],
                name=part["name"],
                args=_parse_args(part["arguments"]),
                arguments_json=part["arguments"] or "{}",
                reasoning=content if index == 0 and content else part.get("reasoning"),
            )
            for index, part in sorted(tool_parts.items())
        ]
        return ProxyTurn(
            "tool_call",
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    return ProxyTurn(
        "text",
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _select_scenarios(names: list[str] | None, tags: list[str] | None) -> list[EvalScenario]:
    scenarios = ALL_SCENARIOS
    if names:
        wanted = set(names)
        scenarios = [scenario for scenario in scenarios if scenario.name in wanted]
        missing = wanted - {scenario.name for scenario in scenarios}
        if missing:
            raise SystemExit(f"unknown scenarios: {', '.join(sorted(missing))}")
    if tags:
        scenarios = [
            scenario
            for scenario in scenarios
            if any(tag in scenario.tags for tag in tags)
        ]
        if not scenarios:
            raise SystemExit(f"no scenarios match tags: {', '.join(tags)}")
    return scenarios


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
        terminal_args: dict[str, Any] | None = None
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


async def main_async(args: argparse.Namespace) -> None:
    scenarios = _select_scenarios(args.scenario, args.tags)
    ablation = ABLATION_PRESETS[args.ablation]
    client = OpenAIProxyClient(
        args.base_url,
        args.model,
        timeout=args.timeout,
        recommended_sampling=not args.no_recommended_sampling,
    )

    output = Path(args.output) if args.output else None
    total_runs = len(scenarios) * args.runs
    completed_runs = 0
    print(
        f"Python oracle: {len(scenarios)} scenarios x {args.runs} runs "
        f"({total_runs} total)",
        file=sys.stderr,
        flush=True,
    )
    for scenario_index, scenario in enumerate(scenarios, 1):
        for run_idx in range(1, args.runs + 1):
            ordinal = completed_runs + 1
            print(
                f"[{ordinal}/{total_runs}] {scenario.name} "
                f"run {run_idx}/{args.runs}...",
                file=sys.stderr,
                flush=True,
            )
            result = await run_proxy_scenario(
                client,
                scenario,
                stream=args.stream,
                budget_tokens=args.budget_tokens,
                ablation=ablation,
                verbose=args.verbose,
            )
            row = _result_row(
                result,
                scenario,
                run_idx,
                args.model,
                args.stream,
                args.ablation,
                args.budget_tokens,
                args.backend_label,
                args.mode_label,
                args.eval_target_backend,
                args.proxy_backend_mode,
            )
            line = json.dumps(row, separators=(",", ":"))
            if output:
                with output.open("a") as handle:
                    handle.write(line + "\n")
            else:
                print(line)
            completed_runs += 1
            if not result.completeness:
                status = f"FAIL ({result.error_type})"
            elif result.accuracy is False:
                status = "OK (incorrect)"
            else:
                status = "OK"
            print(
                f"[{completed_runs}/{total_runs}] {scenario.name} "
                f"run {run_idx}/{args.runs}: {status}, "
                f"{result.iterations_used} iterations, "
                f"{result.elapsed_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run upstream Python eval scenarios against an OpenAI proxy"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--scenario", nargs="*")
    parser.add_argument("--tags", nargs="*")
    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATION_PRESETS.keys()),
        default="reforged",
    )
    parser.add_argument("--output")
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--backend-label", default="openai-proxy")
    parser.add_argument("--mode-label", default="proxy")
    parser.add_argument("--proxy-backend-mode", choices=["native", "prompt"])
    parser.add_argument("--eval-target-backend", default="openai-proxy")
    parser.add_argument("--no-recommended-sampling", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
