#!/usr/bin/env python3
"""Run upstream Forge eval scenarios against an OpenAI-compatible proxy.

This wrapper lives outside the upstream `forge/` submodule on purpose. It uses
the Python eval scenarios as the oracle while targeting a Rust Forge proxy at
`--base-url`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = ROOT / "forge"
sys.path.insert(0, str(FORGE_ROOT / "src"))
sys.path.insert(0, str(FORGE_ROOT))

import httpx

from forge.clients.base import ChunkType, StreamChunk, format_tool
from forge.core.workflow import TextResponse, ToolCall, ToolSpec
from tests.eval.ablation import ABLATION_PRESETS
from tests.eval.eval_runner import EvalConfig, run_scenario
from tests.eval.metrics import analyze_history
from tests.eval.scenarios import ALL_SCENARIOS, EvalScenario


class OpenAIProxyClient:
    """Minimal Python LLMClient for `/v1/chat/completions` proxies."""

    api_format = "openai"

    def __init__(self, base_url: str, model: str, timeout: float = 300.0) -> None:
        self.base_url = _chat_completions_url(base_url)
        self.model = model
        self.timeout = timeout
        self.last_usage: dict[str, int] | None = None

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> list[ToolCall] | TextResponse:
        body = self._body(messages, tools, sampling, stream=False)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, json=body)
            response.raise_for_status()
        data = response.json()
        self._record_usage(data.get("usage"))
        return _parse_openai_response(data)

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
    ):
        body = self._body(messages, tools, sampling, stream=True)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self.base_url, json=body) as response:
                response.raise_for_status()
                async for chunk in _parse_openai_sse(response):
                    if chunk.type == ChunkType.FINAL:
                        if isinstance(chunk.response, TextResponse):
                            self.last_usage = None
                        yield chunk
                    else:
                        yield chunk

    async def get_context_length(self) -> int | None:
        return None

    def _body(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        sampling: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            body["tools"] = [format_tool(tool) for tool in tools]
        if sampling:
            body.update(sampling)
        return body

    def _record_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            self.last_usage = None
            return
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        self.last_usage = {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": total,
        }


def _chat_completions_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


def _parse_openai_response(data: dict[str, Any]) -> list[ToolCall] | TextResponse:
    choices = data.get("choices") or []
    if not choices:
        return TextResponse(content="")
    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        parsed: list[ToolCall] = []
        reasoning = message.get("content") or None
        for index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            args = _parse_args(function.get("arguments"))
            parsed.append(
                ToolCall(
                    tool=function.get("name", ""),
                    args=args,
                    reasoning=reasoning if index == 0 else None,
                )
            )
        return parsed
    return TextResponse(content=message.get("content") or "")


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


async def _parse_openai_sse(response: httpx.Response):
    content = ""
    tool_parts: dict[int, dict[str, Any]] = {}

    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        data = json.loads(payload)
        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}

        if "content" in delta:
            text = delta.get("content") or ""
            content += text
            yield StreamChunk(type=ChunkType.TEXT_DELTA, content=text)

        for part in delta.get("tool_calls") or []:
            index = int(part.get("index", 0))
            existing = tool_parts.setdefault(
                index, {"name": "", "arguments": "", "reasoning": None}
            )
            function = part.get("function") or {}
            if function.get("name"):
                existing["name"] = function["name"]
            if function.get("arguments"):
                existing["arguments"] += function["arguments"]
            yield StreamChunk(type=ChunkType.TOOL_CALL_DELTA, content=json.dumps(part))

        if choice.get("finish_reason") in {"stop", "tool_calls"}:
            if choice["finish_reason"] == "tool_calls":
                calls = [
                    ToolCall(
                        tool=part["name"],
                        args=_parse_args(part["arguments"]),
                        reasoning=part.get("reasoning"),
                    )
                    for _, part in sorted(tool_parts.items())
                ]
                yield StreamChunk(type=ChunkType.FINAL, response=calls)
            else:
                yield StreamChunk(type=ChunkType.FINAL, response=TextResponse(content=content))


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


def _tool_trace(messages: list[Any] | None) -> tuple[list[str], list[dict[str, Any]]]:
    names: list[str] = []
    args: list[dict[str, Any]] = []
    if not messages:
        return names, args
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            names.append(call.name)
            args.append(dict(call.args or {}))
    return names, args


def _result_row(
    result: Any,
    scenario: EvalScenario,
    run_idx: int,
    model: str,
    stream: bool,
    ablation: str,
) -> dict[str, Any]:
    messages = result.messages
    stats = analyze_history(messages) if messages is not None else None
    tool_sequence, tool_args = _tool_trace(messages)
    success = bool(result.completeness and result.accuracy is not False)
    terminal_args = result.terminal_args or {}
    final_text = (
        terminal_args.get("message")
        or terminal_args.get("content")
        or terminal_args.get("findings")
        or ""
    )

    return {
        "impl": "rust-proxy-oracle",
        "model": model,
        "backend": "openai-proxy",
        "mode": "proxy",
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
        "retry_nudges": stats.retry_nudges if stats else None,
        "step_nudges": stats.step_nudges if stats else None,
        "tool_errors": stats.tool_errors if stats else None,
        "reasoning_msgs": stats.reasoning_messages if stats else None,
        "tool_sequence": tool_sequence,
        "tool_args": tool_args,
        "final_text": final_text,
        "raw_response_on_failure": result.error_message if not result.completeness else None,
        "ideal_iterations": scenario.ideal_iterations
        or (len(scenario.workflow.required_steps) + 1),
        "compaction_events": len(result.compaction_events),
    }


async def main_async(args: argparse.Namespace) -> None:
    scenarios = _select_scenarios(args.scenario, args.tags)
    ablation = ABLATION_PRESETS[args.ablation]
    client = OpenAIProxyClient(args.base_url, args.model, timeout=args.timeout)
    config = EvalConfig(
        runs_per_scenario=1,
        stream=args.stream,
        keep_message_history=not args.no_history,
        verbose=args.verbose,
    )

    output = Path(args.output) if args.output else None
    for scenario in scenarios:
        for run_idx in range(1, args.runs + 1):
            result = await run_scenario(client, scenario, config, ablation=ablation)
            row = _result_row(result, scenario, run_idx, args.model, args.stream, args.ablation)
            line = json.dumps(row, separators=(",", ":"))
            if output:
                with output.open("a") as handle:
                    handle.write(line + "\n")
            else:
                print(line)


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
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args(argv)


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
