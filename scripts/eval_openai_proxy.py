#!/usr/bin/env python3
"""Run upstream Forge eval scenarios against an OpenAI-compatible proxy.

This wrapper lives outside the upstream `forge/` submodule on purpose. It uses
the Python eval scenarios as the oracle while targeting a Rust Forge proxy at
`--base-url`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORGE_ROOT = ROOT / "forge"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(FORGE_ROOT / "src"))
sys.path.insert(0, str(FORGE_ROOT))
_tests_package = sys.modules.get("tests")
if _tests_package is not None and hasattr(_tests_package, "__path__"):
    _forge_tests_path = str(FORGE_ROOT / "tests")
    _tests_paths = list(_tests_package.__path__)
    if _forge_tests_path not in _tests_paths:
        _tests_package.__path__ = [_forge_tests_path, *_tests_paths]

if __name__ == "__main__" and any(arg in {"--help", "-h"} for arg in sys.argv[1:]):
    from eval_openai_proxy_lib import _print_early_help
    _print_early_help()
    raise SystemExit(0)

from eval_openai_proxy_lib import (
    ALL_SCENARIOS,
    ABLATION_PRESETS,
    ProxyToolCall,
    ProxyTurn,
    ProxyRunResult,
    OpenAIProxyClient,
    _chat_completions_url,
    _http_error_message,
    _usage_tokens,
    _parse_openai_response,
    _arguments_json,
    _parse_args,
    _parse_openai_sse,
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
    main,
)

if __name__ == "__main__":
    main()
