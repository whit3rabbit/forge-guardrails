from __future__ import annotations

from tests.eval.scenarios import ALL_SCENARIOS
from tests.eval.ablation import ABLATION_PRESETS

from .models import ProxyToolCall, ProxyTurn, ProxyRunResult
from .client import (
    OpenAIProxyClient,
    _chat_completions_url,
    _http_error_message,
    _usage_tokens,
    _parse_openai_response,
    _arguments_json,
    _parse_args,
    _parse_openai_sse,
)
from .runner import (
    run_proxy_scenario,
    _result_row,
    _proxy_failure_classification,
    _has_redacted_terminal_content,
    _is_redacted_terminal_text,
    _proxy_tool_specs,
    _proxy_terminal_tools,
    _openai_tool_call,
    _tool_result_message,
    _call_tool,
    _stringify_tool_result,
    _is_proxy_failure_text,
    _terminal_args_from_text,
    _terminal_text,
    _validate_result,
    _required_step_diagnostics,
    FORGE_EXTENSION_FIELD,
    FORGE_TOOL_STATUS_FIELD,
    FORGE_TOOL_STATUS_OK,
    FORGE_TOOL_STATUS_ERROR,
    REDACTED_TERMINAL_TEXT,
)
from .main import (
    _print_early_help,
    _select_scenarios,
    main_async,
    parse_args,
    main,
)
