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
from forge.clients.base import ChunkType, StreamChunk, format_tool
from forge.clients.llamafile import LlamafileClient, _extract_think_tags
from forge.clients.ollama import OllamaClient
from forge.context import ContextManager, NoCompact, TieredCompact
from forge.core.inference import _build_tool_call_infos, fold_and_serialize, run_inference
from forge.core.messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from forge.core.runner import WorkflowRunner
from forge.core.workflow import TextResponse, ToolCall, ToolDef, ToolSpec, Workflow
from forge.errors import (
    BackendError,
    MaxIterationsError,
    StreamError,
    ToolCallError,
    ToolExecutionError,
    ToolResolutionError,
)
from forge.guardrails import ErrorTracker, ResponseValidator, StepEnforcer
from forge.prompts import build_tool_prompt, extract_tool_call
from forge.prompts.nudges import unknown_tool_nudge
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
        if message.metadata.step_index is not None:
            item["step_index"] = message.metadata.step_index
        if message.metadata.original_type is not None:
            item["original_type"] = message.metadata.original_type.value
        if message.metadata.token_estimate is not None:
            item["token_estimate"] = message.metadata.token_estimate
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


class ScriptedClient:
    api_format = "openai"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls = 0

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        sampling: dict[str, Any] | None = None,
    ) -> Any:
        del messages, tools, sampling
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[idx]

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


SyntheticToolExecutionError = type("ToolExecutionError", (Exception,), {})


class FakeHttpResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self.status_code = 200
        self.text = ""
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


class FakeHttp:
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    async def post(self, *args: Any, **kwargs: Any) -> FakeHttpResponse:
        del args, kwargs
        return FakeHttpResponse(self.body)


def _empty_tool_spec(name: str, description: str | None = None) -> ToolSpec:
    return ToolSpec.from_json_schema(
        name,
        description or f"{name} tool",
        {"type": "object", "properties": {}},
    )


def _run_tool_spec() -> ToolSpec:
    return ToolSpec.from_json_schema(
        "run",
        "Run command",
        {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
            },
        },
    )


def _seed_messages() -> list[Message]:
    return [
        Message(MessageRole.SYSTEM, "sys", MessageMeta(MessageType.SYSTEM_PROMPT)),
        Message(MessageRole.USER, "start", MessageMeta(MessageType.USER_INPUT)),
    ]


def _ok_tool(**kwargs: Any) -> str:
    del kwargs
    return "ok"


def _lookup_tool(**kwargs: Any) -> str:
    del kwargs
    return "lookup ok"


def _analyze_tool(**kwargs: Any) -> str:
    del kwargs
    return "analyze ok"


def _respond_tool(message: str = "done", **kwargs: Any) -> str:
    del kwargs
    return message


def _soft_fail_tool(**kwargs: Any) -> str:
    del kwargs
    raise ToolResolutionError("try again")


def _hard_fail_tool(**kwargs: Any) -> str:
    del kwargs
    raise SyntheticToolExecutionError("boom")


def _workflow(
    tools: dict[str, ToolDef],
    required_steps: list[str] | None = None,
    terminal_tool: str | list[str] = "respond",
) -> Workflow:
    return Workflow(
        name="wf",
        description="Parity workflow",
        tools=tools,
        required_steps=required_steps or [],
        terminal_tool=terminal_tool,
        system_prompt_template="sys",
    )


async def _run_workflow_fixture(
    workflow: Workflow,
    responses: list[Any],
    *,
    max_iterations: int = 10,
    max_tool_errors: int = 2,
) -> tuple[Any, list[Message], int]:
    emitted: list[Message] = []
    runner = WorkflowRunner(
        ScriptedClient(responses),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_iterations=max_iterations,
        max_retries_per_step=2,
        max_tool_errors=max_tool_errors,
        on_message=emitted.append,
        rescue_enabled=False,
    )
    result = await runner.run(
        workflow,
        "ignored",
        initial_messages=_seed_messages(),
    )
    return result, emitted, runner.client.calls


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


async def _proxy_no_tools_tool_calls_fixture() -> dict[str, Any]:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": False,
    }
    response = await handle_chat_completions(
        body,
        ScriptedClient([[ToolCall(tool="search", args={"q": "x"})]]),
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


async def _proxy_retry_exhausted_raw_text_fixture() -> dict[str, Any]:
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
    max_retries = 1
    response = await handle_chat_completions(
        body,
        ScriptedClient([
            TextResponse(content="first bad"),
            TextResponse(content="raw final"),
        ]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_retries=max_retries,
        rescue_enabled=True,
    )
    choice = response["choices"][0]
    return {
        "input": {"body": body, "max_retries": max_retries},
        "expected": {
            "content": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
        },
    }


async def _proxy_rescue_success_fixture() -> dict[str, Any]:
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
        ScriptedClient([
            TextResponse(content='{"tool":"search","args":{"query":"rescued"}}'),
        ]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_retries=2,
        rescue_enabled=True,
    )
    choice = response["choices"][0]
    call = choice["message"]["tool_calls"][0]
    return {
        "input": body,
        "expected": {
            "finish_reason": choice["finish_reason"],
            "tool_name": call["function"]["name"],
            "tool_args": json.loads(call["function"]["arguments"]),
        },
    }


async def _proxy_rescue_failure_raw_text_fixture() -> dict[str, Any]:
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
        ScriptedClient([TextResponse(content="not a tool call")]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_retries=0,
        rescue_enabled=True,
    )
    choice = response["choices"][0]
    return {
        "input": {"body": body, "max_retries": 0},
        "expected": {
            "content": choice["message"]["content"],
            "finish_reason": choice["finish_reason"],
        },
    }


async def _proxy_unknown_tool_retry_fixture() -> dict[str, Any]:
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
        ScriptedClient([
            [ToolCall(tool="bogus", args={})],
            [ToolCall(tool="search", args={"query": "after nudge"})],
        ]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_retries=2,
        rescue_enabled=True,
    )
    choice = response["choices"][0]
    call = choice["message"]["tool_calls"][0]
    return {
        "input": body,
        "expected": {
            "finish_reason": choice["finish_reason"],
            "tool_name": call["function"]["name"],
            "tool_args": json.loads(call["function"]["arguments"]),
        },
    }


async def _proxy_mixed_respond_streaming_fixture() -> dict[str, Any]:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": True,
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
            },
            {
                "type": "function",
                "function": {
                    "name": "respond",
                    "description": "Respond",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                    },
                },
            },
        ],
    }
    events = await handle_chat_completions(
        body,
        ScriptedClient([
            [
                ToolCall(tool="respond", args={"message": "drop me"}),
                ToolCall(tool="search", args={"query": "keep me"}),
            ]
        ]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_retries=2,
        rescue_enabled=True,
    )
    tool_delta = next(
        event for event in events
        if event["choices"][0]["delta"].get("tool_calls")
    )
    final = events[-1]
    return {
        "input": body,
        "expected": {
            "tool_names": [
                call["function"]["name"]
                for call in tool_delta["choices"][0]["delta"]["tool_calls"]
            ],
            "finish_reason": final["choices"][0]["finish_reason"],
        },
    }


async def _proxy_text_sse_final_chunk_fixture() -> dict[str, Any]:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": True,
    }
    events = await handle_chat_completions(
        body,
        ScriptedClient([TextResponse(content="hello")]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_retries=2,
        rescue_enabled=True,
    )
    return {
        "input": body,
        "expected": {
            "first_content": events[0]["choices"][0]["delta"]["content"],
            "final_delta": events[-1]["choices"][0]["delta"],
            "finish_reason": events[-1]["choices"][0]["finish_reason"],
        },
    }


async def _workflow_step_nudge_exposure_fixture() -> dict[str, Any]:
    case = await _step_nudge_history_fixture()
    step_messages = [
        message
        for message in case["expected"]["messages"]
        if message["type"] == "step_nudge"
    ]
    return {
        "input": case["input"],
        "expected": {
            "step_nudge_count": len(step_messages),
            "first_tool_name": step_messages[0]["tool_name"],
            "first_content_prefix": step_messages[0]["content"].split(".", 1)[0],
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


def _rich_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
                "default": "rust",
            },
            "mode": {
                "type": "string",
                "enum": ["fast", "deep"],
                "description": "Mode",
            },
            "cfg": {
                "type": "object",
                "description": "Config",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Limit",
                        "default": 5,
                    }
                },
                "required": ["limit"],
            },
            "tags": {
                "type": "array",
                "description": "Tags",
                "items": {"type": "string", "description": "Tag"},
            },
        },
        "required": ["query", "cfg"],
    }


def _schema_output_fixture() -> dict[str, Any]:
    schema = _rich_tool_schema()
    spec = ToolSpec.from_json_schema("search_tool", "Search", schema)
    json_schema = spec.get_json_schema()
    return {
        "input": schema,
        "expected": {
            "schema": json_schema,
            "schema_json": json.dumps(json_schema),
        },
    }


def _format_tool_fixture() -> dict[str, Any]:
    schema = _rich_tool_schema()
    spec = ToolSpec.from_json_schema("search_tool", "Search", schema)
    formatted = format_tool(spec)
    return {
        "input": schema,
        "expected": {
            "tool": formatted,
            "tool_json": json.dumps(formatted),
        },
    }


def _tool_prompt_fixture() -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["fast", "deep"],
                "description": "Mode",
            }
        },
        "required": ["mode"],
    }
    spec = ToolSpec.from_json_schema("search", "Search docs", schema)
    return {
        "input": schema,
        "expected": build_tool_prompt([spec]),
    }


def _unknown_tool_order_fixture() -> dict[str, Any]:
    tools = ["zebra", "alpha", "middle"]
    return {
        "input": {"called_tool": "bogus", "available_tools": tools},
        "expected": unknown_tool_nudge("bogus", tools),
    }


async def _text_retry_history_fixture() -> dict[str, Any]:
    client = ScriptedClient([
        TextResponse(content="plain answer"),
        [ToolCall(tool="run", args={"x": 1})],
    ])
    messages = [
        Message(MessageRole.USER, "start", MessageMeta(MessageType.USER_INPUT))
    ]
    result = await run_inference(
        messages=messages,
        client=client,
        context_manager=ContextManager(NoCompact(), budget_tokens=4096),
        validator=ResponseValidator(["run"], rescue_enabled=False),
        error_tracker=ErrorTracker(max_retries=2, max_tool_errors=2),
        tool_specs=[_run_tool_spec()],
        max_attempts=3,
    )
    assert result is not None
    assert isinstance(result.response, list)
    return {
        "input": {"responses": ["text", "run"]},
        "expected": {
            "attempts": result.attempts,
            "next_counter": result.tool_call_counter,
            "messages": _messages_payload(messages),
            "response": _tool_calls_payload(result.response),
        },
    }


async def _unknown_tool_history_fixture() -> dict[str, Any]:
    client = ScriptedClient([
        [ToolCall(tool="bogus", args={})],
        [ToolCall(tool="run", args={"x": 1})],
    ])
    messages = [
        Message(MessageRole.USER, "start", MessageMeta(MessageType.USER_INPUT))
    ]
    result = await run_inference(
        messages=messages,
        client=client,
        context_manager=ContextManager(NoCompact(), budget_tokens=4096),
        validator=ResponseValidator(["run"], rescue_enabled=False),
        error_tracker=ErrorTracker(max_retries=2, max_tool_errors=2),
        tool_specs=[_run_tool_spec()],
        max_attempts=3,
    )
    assert result is not None
    assert isinstance(result.response, list)
    return {
        "input": {"responses": ["bogus", "run"]},
        "expected": {
            "attempts": result.attempts,
            "next_counter": result.tool_call_counter,
            "messages": _messages_payload(messages),
            "response": _tool_calls_payload(result.response),
        },
    }


async def _step_nudge_history_fixture() -> dict[str, Any]:
    workflow = _workflow(
        {
            "lookup": ToolDef(_empty_tool_spec("lookup"), _lookup_tool),
            "respond": ToolDef(respond_spec(), _respond_tool),
        },
        required_steps=["lookup"],
    )
    result, emitted, calls = await _run_workflow_fixture(
        workflow,
        [
            [ToolCall(tool="respond", args={"message": "too soon"})],
            [ToolCall(tool="lookup", args={})],
            [ToolCall(tool="respond", args={"message": "lookup complete"})],
        ],
    )
    return {
        "input": {"required_steps": ["lookup"]},
        "expected": {
            "result": result,
            "attempts": calls,
            "messages": _messages_payload(emitted),
        },
    }


async def _prerequisite_nudge_history_fixture() -> dict[str, Any]:
    workflow = _workflow(
        {
            "lookup": ToolDef(_empty_tool_spec("lookup"), _lookup_tool),
            "analyze": ToolDef(
                _empty_tool_spec("analyze"),
                _analyze_tool,
                prerequisites=["lookup"],
            ),
            "respond": ToolDef(respond_spec(), _respond_tool),
        },
        required_steps=["lookup", "analyze"],
    )
    result, emitted, calls = await _run_workflow_fixture(
        workflow,
        [
            [ToolCall(tool="analyze", args={})],
            [ToolCall(tool="lookup", args={})],
            [ToolCall(tool="analyze", args={})],
            [ToolCall(tool="respond", args={"message": "analysis complete"})],
        ],
    )
    return {
        "input": {"prerequisite": "lookup -> analyze"},
        "expected": {
            "result": result,
            "attempts": calls,
            "messages": _messages_payload(emitted),
        },
    }


async def _tool_resolution_soft_error_budget_fixture() -> dict[str, Any]:
    workflow = _workflow({
        "lookup": ToolDef(_empty_tool_spec("lookup"), _soft_fail_tool),
        "respond": ToolDef(respond_spec(), _respond_tool),
    })
    result, emitted, calls = await _run_workflow_fixture(
        workflow,
        [
            [ToolCall(tool="lookup", args={})],
            [ToolCall(tool="respond", args={"message": "soft resolution recovered"})],
        ],
        max_tool_errors=0,
    )
    return {
        "input": {"max_tool_errors": 0},
        "expected": {
            "result": result,
            "attempts": calls,
            "messages": _messages_payload(emitted),
        },
    }


async def _hard_tool_execution_error_budget_fixture() -> dict[str, Any]:
    workflow = _workflow({
        "explode": ToolDef(_empty_tool_spec("explode"), _hard_fail_tool),
        "respond": ToolDef(respond_spec(), _respond_tool),
    })
    emitted: list[Message] = []
    runner = WorkflowRunner(
        ScriptedClient([[ToolCall(tool="explode", args={})]]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_iterations=3,
        max_retries_per_step=2,
        max_tool_errors=0,
        on_message=emitted.append,
        rescue_enabled=False,
    )
    try:
        await runner.run(
            workflow,
            "ignored",
            initial_messages=_seed_messages(),
        )
    except ToolExecutionError as exc:
        return {
            "input": {"max_tool_errors": 0},
            "expected": {
                "error_type": "ToolExecutionError",
                "tool_name": exc.tool_name,
                "cause": str(exc.cause),
                "attempts": runner.client.calls,
                "messages": _messages_payload(emitted),
            },
        }
    raise AssertionError("expected ToolExecutionError")


def _fold_and_serialize_reasoning_fixture() -> dict[str, Any]:
    def tool_call(call_id: str, x: int) -> Message:
        return Message(
            MessageRole.ASSISTANT,
            "",
            MessageMeta(MessageType.TOOL_CALL),
            tool_calls=[ToolCallInfo("run", {"x": x}, call_id)],
        )

    cases = {
        "before_tool_call": [
            Message(
                MessageRole.ASSISTANT,
                "think first",
                MessageMeta(MessageType.REASONING),
            ),
            tool_call("call_000000000", 1),
        ],
        "orphan_before_user": [
            Message(
                MessageRole.ASSISTANT,
                "orphan",
                MessageMeta(MessageType.REASONING),
            ),
            Message(MessageRole.USER, "next", MessageMeta(MessageType.USER_INPUT)),
        ],
        "consecutive_before_tool_call": [
            Message(
                MessageRole.ASSISTANT,
                "first",
                MessageMeta(MessageType.REASONING),
            ),
            Message(
                MessageRole.ASSISTANT,
                "second",
                MessageMeta(MessageType.REASONING),
            ),
            tool_call("call_000000001", 2),
        ],
    }
    return {
        "input": {"api_format": "openai"},
        "expected": {
            name: fold_and_serialize(messages, "openai")
            for name, messages in cases.items()
        },
    }


async def _max_iterations_pending_error_fixture() -> dict[str, Any]:
    workflow = _workflow(
        {
            "lookup": ToolDef(_empty_tool_spec("lookup"), _lookup_tool),
            "analyze": ToolDef(_empty_tool_spec("analyze"), _analyze_tool),
            "respond": ToolDef(respond_spec(), _respond_tool),
        },
        required_steps=["lookup", "analyze"],
    )
    emitted: list[Message] = []
    runner = WorkflowRunner(
        ScriptedClient([[ToolCall(tool="lookup", args={})]]),
        ContextManager(NoCompact(), budget_tokens=4096),
        max_iterations=1,
        max_retries_per_step=2,
        max_tool_errors=2,
        on_message=emitted.append,
        rescue_enabled=False,
    )
    try:
        await runner.run(
            workflow,
            "ignored",
            initial_messages=_seed_messages(),
        )
    except MaxIterationsError as exc:
        return {
            "input": {
                "required_steps": ["lookup", "analyze"],
                "max_iterations": 1,
            },
            "expected": {
                "error_type": "MaxIterationsError",
                "completed": list(exc.completed_steps.keys()),
                "pending": exc.pending_steps,
                "messages": _messages_payload(emitted),
            },
        }
    raise AssertionError("expected MaxIterationsError")


def _compaction_phases_fixture() -> dict[str, Any]:
    messages: list[Message] = [
        Message(MessageRole.SYSTEM, "sys", MessageMeta(MessageType.SYSTEM_PROMPT)),
        Message(MessageRole.USER, "usr", MessageMeta(MessageType.USER_INPUT)),
    ]
    for step in range(3):
        messages.extend([
            Message(
                MessageRole.ASSISTANT,
                "thinking",
                MessageMeta(MessageType.REASONING, step_index=step),
            ),
            Message(
                MessageRole.ASSISTANT,
                "",
                MessageMeta(MessageType.TOOL_CALL, step_index=step),
                tool_calls=[ToolCallInfo("run", {"x": step}, f"call_{step}")],
            ),
            Message(
                MessageRole.TOOL,
                "x" * 250,
                MessageMeta(MessageType.TOOL_RESULT, step_index=step),
                tool_name="run",
                tool_call_id=f"call_{step}",
            ),
            Message(
                MessageRole.ASSISTANT,
                "text",
                MessageMeta(MessageType.TEXT_RESPONSE, step_index=step),
            ),
            Message(
                MessageRole.USER,
                "retry",
                MessageMeta(MessageType.RETRY_NUDGE, step_index=step),
            ),
        ])

    def run(thresholds: tuple[float, float, float]) -> dict[str, Any]:
        compacted, phase = TieredCompact(
            keep_recent=1,
            phase_thresholds=thresholds,
        ).compact(messages, 100)
        return {"phase": phase, "messages": _messages_payload(compacted)}

    return {
        "input": {"keep_recent": 1},
        "expected": {
            "phase1": run((0.0, 100.0, 100.0)),
            "phase2": run((0.0, 0.0, 100.0)),
            "phase3": run((0.0, 0.0, 0.0)),
        },
    }


async def _llamafile_malformed_args_fixture() -> dict[str, Any]:
    body = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "run",
                                "arguments": "{broken",
                            }
                        }
                    ],
                }
            }
        ]
    }
    client = LlamafileClient("t.gguf")
    client._http = FakeHttp(body)  # type: ignore[assignment]
    response = await client.send([{"role": "user", "content": "hi"}])
    assert isinstance(response, TextResponse)
    return {
        "input": body,
        "expected": {"content": response.content},
    }


def _pending_steps_fixture() -> dict[str, Any]:
    enforcer = StepEnforcer(
        required_steps=["lookup", "analyze"],
        terminal_tools=["respond"],
    )
    enforcer.record("lookup", {})
    return {
        "input": {"required_steps": ["lookup", "analyze"], "completed": ["lookup"]},
        "expected": {"pending": enforcer.pending()},
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
            "llamafile_malformed_args": await _llamafile_malformed_args_fixture(),
            "ollama_thinking": _ollama_thinking_fixture(),
            "tool_call_id_generation": _tool_call_id_fixture(),
            "toolspec_schema_output": _schema_output_fixture(),
            "format_tool_output": _format_tool_fixture(),
            "tool_prompt_text": _tool_prompt_fixture(),
            "unknown_tool_order": _unknown_tool_order_fixture(),
            "text_retry_history": await _text_retry_history_fixture(),
            "unknown_tool_history": await _unknown_tool_history_fixture(),
            "step_nudge_history": await _step_nudge_history_fixture(),
            "prerequisite_nudge_history": await _prerequisite_nudge_history_fixture(),
            "tool_resolution_soft_error_budget": await _tool_resolution_soft_error_budget_fixture(),
            "hard_tool_execution_error_budget": await _hard_tool_execution_error_budget_fixture(),
            "compaction_phases": _compaction_phases_fixture(),
            "fold_and_serialize_reasoning": _fold_and_serialize_reasoning_fixture(),
            "max_iterations_pending_error": await _max_iterations_pending_error_fixture(),
            "pending_steps_only": _pending_steps_fixture(),
            "proxy_sampling_fields": _proxy_sampling_fixture(),
            "proxy_respond_stripping": await _proxy_respond_fixture(),
            "proxy_no_tools_tool_calls": await _proxy_no_tools_tool_calls_fixture(),
            "proxy_retry_exhausted_raw_text": await _proxy_retry_exhausted_raw_text_fixture(),
            "proxy_rescue_success": await _proxy_rescue_success_fixture(),
            "proxy_rescue_failure_raw_text": await _proxy_rescue_failure_raw_text_fixture(),
            "proxy_unknown_tool_retry": await _proxy_unknown_tool_retry_fixture(),
            "proxy_mixed_respond_streaming": await _proxy_mixed_respond_streaming_fixture(),
            "proxy_text_sse_final_chunk": await _proxy_text_sse_final_chunk_fixture(),
            "workflow_step_nudge_exposure": await _workflow_step_nudge_exposure_fixture(),
        },
    }


def main() -> None:
    fixtures = asyncio.run(build_fixtures())
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(fixtures, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {FIXTURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
