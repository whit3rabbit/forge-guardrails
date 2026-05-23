#!/usr/bin/env python3
"""Generate Rust parity fixtures from the Python forge reference.

Run from the repository root:

    uv run --project forge python tests/parity/generate_fixtures.py

The output is committed under tests/parity/fixtures and consumed by Rust
integration tests. Keep this script small and deterministic: it should only
exercise reference behavior needed by parity tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "python_golden.json"


def _install_anthropic_stub() -> None:
    """Allow importing forge.clients.anthropic without the optional SDK."""
    if "anthropic" in sys.modules:
        return
    module = types.ModuleType("anthropic")

    class APIError(Exception):
        status_code = 0

    class AsyncAnthropic:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    module.APIError = APIError
    module.AsyncAnthropic = AsyncAnthropic
    sys.modules["anthropic"] = module


_install_anthropic_stub()

from forge.clients.anthropic import AnthropicClient
from forge.clients.base import ChunkType, StreamChunk
from forge.clients.llamafile import LlamafileClient, _extract_think_tags
from forge.clients.ollama import OllamaClient
from forge.context import ContextManager, NoCompact
from forge.core.inference import _build_tool_call_infos, run_inference
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType
from forge.core.workflow import TextResponse, ToolCall
from forge.errors import BackendError, StreamError, ToolCallError
from forge.guardrails import ErrorTracker, ResponseValidator
from forge.prompts import extract_tool_call
from forge.proxy.handler import _extract_sampling, handle_chat_completions
from forge.tools.respond import respond_spec


def _tool_calls_payload(calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "tool": call.tool,
            "args": call.args,
            "reasoning": call.reasoning,
        }
        for call in calls
    ]


def _messages_payload(messages: list[Message]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
            "type": message.metadata.type.value,
        }
        if message.tool_name is not None:
            item["tool_name"] = message.tool_name
        if message.tool_call_id is not None:
            item["tool_call_id"] = message.tool_call_id
        if message.tool_calls is not None:
            item["tool_calls"] = [
                {
                    "name": tc.name,
                    "args": tc.args,
                    "call_id": tc.call_id,
                }
                for tc in message.tool_calls
            ]
        result.append(item)
    return result


class AlwaysTextClient:
    api_format = "openai"

    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> TextResponse:
        del messages, tools, sampling
        idx = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        return TextResponse(content=self.contents[idx])

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ):
        del messages, tools, sampling
        raise AssertionError("send_stream should not be called")

    async def get_context_length(self) -> int | None:
        return 4096


class BackendErrorClient:
    api_format = "openai"

    def __init__(self) -> None:
        self.calls = 0

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> TextResponse:
        del messages, tools, sampling
        self.calls += 1
        raise BackendError(503, "backend down")

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ):
        del messages, tools, sampling
        raise AssertionError("send_stream should not be called")

    async def get_context_length(self) -> int | None:
        return 4096


class NoFinalStreamClient:
    api_format = "openai"

    def __init__(self) -> None:
        self.calls = 0

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> TextResponse:
        del messages, tools, sampling
        raise AssertionError("send should not be called")

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ):
        del messages, tools, sampling
        self.calls += 1
        yield StreamChunk(type=ChunkType.TEXT_DELTA, content="partial")

    async def get_context_length(self) -> int | None:
        return 4096


class RespondClient:
    api_format = "openai"

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> list[ToolCall]:
        del messages, tools, sampling
        return [ToolCall(tool="respond", args={"message": "done"})]

    async def send_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ):
        del messages, tools, sampling
        raise AssertionError("send_stream should not be called")

    async def get_context_length(self) -> int | None:
        return 4096


async def _retry_budget_fixture() -> dict[str, Any]:
    client = AlwaysTextClient(["bad 1", "bad 2", "bad 3", "bad 4"])
    messages = [
        Message(
            MessageRole.USER,
            "start",
            MessageMeta(MessageType.USER_INPUT),
        )
    ]
    context = ContextManager(NoCompact(), budget_tokens=4096)
    validator = ResponseValidator(["respond"], rescue_enabled=False)
    tracker = ErrorTracker(max_retries=2, max_tool_errors=2)
    try:
        await run_inference(
            messages=messages,
            client=client,
            context_manager=context,
            validator=validator,
            error_tracker=tracker,
            tool_specs=[respond_spec()],
            max_attempts=10,
        )
    except ToolCallError as exc:
        return {
            "input": {"max_retries": 2, "max_attempts": 10},
            "expected": {
                "error_type": "ToolCallError",
                "attempts": client.calls,
                "raw_response": exc.raw_response,
                "messages": _messages_payload(messages),
            },
        }
    raise AssertionError("expected ToolCallError")


async def _backend_error_fixture() -> dict[str, Any]:
    client = BackendErrorClient()
    messages = [
        Message(MessageRole.USER, "start", MessageMeta(MessageType.USER_INPUT))
    ]
    try:
        await run_inference(
            messages=messages,
            client=client,
            context_manager=ContextManager(NoCompact(), budget_tokens=4096),
            validator=ResponseValidator(["respond"], rescue_enabled=False),
            error_tracker=ErrorTracker(max_retries=2, max_tool_errors=2),
            tool_specs=[respond_spec()],
            max_attempts=10,
        )
    except BackendError as exc:
        return {
            "input": {"status_code": 503, "body": "backend down"},
            "expected": {
                "error_type": "BackendError",
                "attempts": client.calls,
                "message": str(exc),
            },
        }
    raise AssertionError("expected BackendError")


async def _stream_without_final_fixture() -> dict[str, Any]:
    client = NoFinalStreamClient()
    messages = [
        Message(MessageRole.USER, "start", MessageMeta(MessageType.USER_INPUT))
    ]
    try:
        await run_inference(
            messages=messages,
            client=client,
            context_manager=ContextManager(NoCompact(), budget_tokens=4096),
            validator=ResponseValidator(["respond"], rescue_enabled=False),
            error_tracker=ErrorTracker(max_retries=2, max_tool_errors=2),
            tool_specs=[respond_spec()],
            max_attempts=10,
            stream=True,
        )
    except StreamError as exc:
        return {
            "input": {"chunks": ["partial"]},
            "expected": {
                "error_type": "StreamError",
                "attempts": client.calls,
                "message": str(exc),
            },
        }
    raise AssertionError("expected StreamError")


async def _proxy_respond_fixture() -> dict[str, Any]:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
    }
    response = await handle_chat_completions(
        body,
        RespondClient(),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_retries=2,
        rescue_enabled=True,
    )
    choice = response["choices"][0]
    return {
        "input": body,
        "expected": {
            "content": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
        },
    }


def _anthropic_conversion_fixture() -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "First"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "abc",
                    "function": {
                        "name": "run",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "user", "content": "next"},
    ]
    system, converted = AnthropicClient._convert_messages(messages)
    return {
        "input": messages,
        "expected": {
            "system": system,
            "messages": converted,
        },
    }


def _anthropic_fallback_id_fixture() -> dict[str, Any]:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "user", "content": "next"},
    ]
    _, converted = AnthropicClient._convert_messages(messages)
    return {
        "input": messages,
        "expected": {"messages": converted},
    }


def _llamafile_reasoning_fixture() -> dict[str, Any]:
    content = '<think>reason</think>{"tool":"run","args":{}}'
    think_text, cleaned = _extract_think_tags(content)
    tool_calls = extract_tool_call(cleaned, ["run"])
    if tool_calls:
        client = LlamafileClient("t.gguf")
        tool_calls[0].reasoning = client._resolve_reasoning("", think_text)
    return {
        "input": {
            "content": content,
            "tools": ["run"],
        },
        "expected": {
            "cleaned": cleaned,
            "tool_calls": _tool_calls_payload(tool_calls),
        },
    }


def _ollama_thinking_fixture() -> dict[str, Any]:
    data = {
        "message": {
            "thinking": "reason",
            "content": "visible",
            "tool_calls": [
                {"function": {"name": "run", "arguments": {"x": 1}}},
            ],
        }
    }
    client = OllamaClient("reason-model", think=True)
    msg = data["message"]
    reasoning = client._resolve_reasoning(
        msg.get("thinking", ""),
        msg.get("content", ""),
    )
    tool_calls = [
        ToolCall(
            tool=tc["function"]["name"],
            args=tc["function"].get("arguments", {}),
            reasoning=reasoning if i == 0 else None,
        )
        for i, tc in enumerate(msg["tool_calls"])
    ]
    return {
        "input": data,
        "expected": {"tool_calls": _tool_calls_payload(tool_calls)},
    }


def _tool_call_id_fixture() -> dict[str, Any]:
    calls = [
        ToolCall(tool="first", args={}),
        ToolCall(tool="second", args={}),
    ]
    infos, counter = _build_tool_call_infos(calls, 0)
    return {
        "input": {"starting_counter": 0, "tools": ["first", "second"]},
        "expected": {
            "call_ids": [info.call_id for info in infos],
            "next_counter": counter,
        },
    }


def _proxy_sampling_fixture() -> dict[str, Any]:
    body = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "presence_penalty": 0.3,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": True},
        "frequency_penalty": 1.0,
        "response_format": {"type": "json_object"},
    }
    return {
        "input": body,
        "expected": _extract_sampling(body),
    }


async def build_fixtures() -> dict[str, Any]:
    return {
        "metadata": {
            "source": "forge Python reference submodule",
            "generator": "tests/parity/generate_fixtures.py",
            "command": "uv run --project forge python tests/parity/generate_fixtures.py",
        },
        "cases": {
            "anthropic_conversion_unpaired": _anthropic_conversion_fixture(),
            "anthropic_conversion_fallback_id": _anthropic_fallback_id_fixture(),
            "inference_retry_budget": await _retry_budget_fixture(),
            "backend_error_propagation": await _backend_error_fixture(),
            "streaming_without_final": await _stream_without_final_fixture(),
            "llamafile_reasoning_extraction": _llamafile_reasoning_fixture(),
            "ollama_thinking": _ollama_thinking_fixture(),
            "tool_call_id_generation": _tool_call_id_fixture(),
            "proxy_sampling_fields": _proxy_sampling_fixture(),
            "proxy_respond_stripping": await _proxy_respond_fixture(),
        },
    }


def main() -> None:
    fixtures = asyncio.run(build_fixtures())
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(fixtures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {FIXTURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
