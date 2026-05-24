# AGENTS.md

## Purpose

This repo is `forge-guardrails`, a Rust clean-room implementation inspired by `antoinezambelli/forge`.

It provides foundation types and runtime pieces for guarded LLM-agent workflows:
- workflow and step enforcement
- tool specs, tool execution, and terminal tools
- prompt rescue and tool-call parsing
- context tracking and compaction
- backend adapters for Anthropic, Llamafile, and Ollama
- anyllm runtime and sidecar client support for provider routing
- Anthropic/OpenAI request translation through `anyllm_translate`
- OpenAI-compatible proxy/server surfaces

The reference Python implementation is available in the [forge](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/forge) git submodule. Use its `src/` directory as the gold standard for behavioral reference, structure, and details. Ensure that all benchmark matrix scenarios are implemented so we can guarantee complete alignment with the Python implementation.

## Core layout

- `src/core/message.rs` - message roles, types, metadata, tool-call info
- `src/core/tool_spec.rs` - tool schema and callable definitions
- `src/core/workflow.rs` - workflow model, terminal tools, prerequisites
- `src/core/steps.rs` - step tracking and required-step state
- `src/guardrails/` - response validation, error tracking, step enforcement
- `src/core/runner.rs` - multi-turn workflow loop
- `src/core/inference.rs` - inference helpers and message conversion
- `src/tools/respond.rs` - built-in `respond` terminal-style tool
- `src/prompts/` - nudge prompts and JSON/tool-call rescue parsing
- `src/context/` - context manager, token budget, compaction callbacks, compaction strategies, hardware helpers
- `src/clients/base.rs` - `LLMClient`, streaming trait, OpenAI tool formatting
- `src/clients/` - Anthropic, Llamafile, Ollama, and anyllm runtime/sidecar clients
- `src/server.rs` - backend lifecycle and context-budget resolution
- `src/proxy/` - HTTP/OpenAI-compatible and Anthropic Messages proxy handling
- `tests/` - integration coverage for core behavior
- `tests/parity/` - Python-generated golden fixtures for Rust parity tests

## Commands

Run before committing:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Use `cargo fmt --all` to apply formatting.

Regenerate Python parity fixtures after intentional reference-behavior changes:

```bash
uv run --project forge python tests/parity/generate_fixtures.py
```

The generated `tests/parity/fixtures/python_golden.json` file is checked in.
Normal Rust test runs consume that JSON and should not invoke Python.

## Agent rules

Keep changes small and behavior-driven. Align design and behavior with the Python codebase by checking the reference implementation in the `forge` submodule.

Refer to the [Python forge src](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/forge/src) as the gold standard for logic, defaults, and API shapes. Implement and verify all benchmark matrices/evaluation scenarios to ensure complete parity.

Preserve these invariants:
- Tool-call IDs and tool-result IDs must stay paired.
- Step enforcement must not leave invalid tool-call history behind.
- Compaction must not produce protocol-invalid transcripts.
- Backend adapters must keep their wire formats separate.
- Per-call sampling overrides must not mutate client defaults.
- Retry and rescue logic should nudge the model without hiding hard failures.
- Server/proxy code must clearly separate passthrough behavior from guarded workflow behavior.
- Forge owns interception and nudging. Do not route guarded traffic through `anyllm_proxy` HTTP handlers before forge validates it.
- Use `anyllm_translate` for Anthropic/OpenAI compatibility instead of hand-rolling request or response translation.
- Use `AnyLlmRuntimeClient` for in-process anyllm routing when possible. It must use `anyllm_proxy::runtime::ChatCompletionService`, not the axum router.
- Use `AnyLlmProxyClient` for a separate sidecar process when admin UI, cache, metrics, batch, or standalone provider config is needed.
- Keep anyllm server-side tool execution out of Forge's guarded path. Forge must inspect tool calls before anything executes them.
- Keep `TokenUsage` token-only. Surface anyllm provider metadata, rate limits, warnings, cache state, and estimated cost through `LLMClient::last_call_info()`.
- Treat anyllm pricing-derived cost as observability, not billing authority.

When changing workflow execution:
1. Add or update tests first.
2. Cover blocked steps, malformed tool calls, terminal tools, and mixed tool batches.
3. Check both non-streaming and streaming paths when touching backend clients.

When changing context or compaction:
1. Preserve system/user setup messages.
2. Keep recent workflow state coherent.
3. Do not drop one side of a tool-call/tool-result pair unless the whole group is summarized as inert text.

When changing backend clients:
1. Avoid shared assumptions between Anthropic, Ollama, and OpenAI-compatible backends.
2. Mock HTTP responses in tests.
3. Assert request bodies, not only parsed outputs.

When changing Python parity behavior:
1. Update or add a fixture case in `tests/parity/generate_fixtures.py`.
2. Regenerate `tests/parity/fixtures/python_golden.json`.
3. Assert the Rust behavior against the generated fixture in `tests/parity_tests.rs`.

## Current status notes

The clean-room run summary reports:
- 8 of 8 units completed
- 487 tests passing
- 0 contamination incidents

Treat that as historical status. Re-run the local test suite after any meaningful change.
