from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
