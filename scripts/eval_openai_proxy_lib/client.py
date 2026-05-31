from __future__ import annotations

import json
from typing import Any

from forge.clients.base import format_tool
from forge.clients.sampling_defaults import get_sampling_defaults
from forge.core.workflow import ToolSpec

from .models import ProxyToolCall, ProxyTurn


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
        """Initialize the client with the proxy base URL and default model."""
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
        """Send a chat completions request to the proxy server."""
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
