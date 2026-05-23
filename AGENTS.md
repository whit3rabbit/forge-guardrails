# AGENTS.md

## Purpose

This repo is `forge-guardrails`, a Rust clean-room implementation inspired by `antoinezambelli/forge`.

It provides foundation types and runtime pieces for guarded LLM-agent workflows:
- workflow and step enforcement
- tool specs, tool execution, and terminal tools
- prompt rescue and tool-call parsing
- context tracking and compaction
- backend adapters for Anthropic, Llamafile, and Ollama
- OpenAI-compatible proxy/server surfaces

The reference Python implementation is available in the [forge](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/forge) git submodule. Use its `src/` directory as the gold standard for behavioral reference, structure, and details. Ensure that all benchmark matrix scenarios are implemented so we can guarantee complete alignment with the Python implementation.

## Core layout

- `src/message.rs` - message roles, types, metadata, tool-call info
- `src/tool_spec.rs` - tool schema and callable definitions
- `src/workflow.rs` - workflow model, terminal tools, prerequisites
- `src/steps.rs` - step tracking and required-step state
- `src/guardrails/` - response validation, error tracking, step enforcement
- `src/runner.rs` - multi-turn workflow loop
- `src/inference.rs` - inference helpers and message conversion
- `src/respond.rs` - built-in `respond` terminal-style tool
- `src/prompts/` - nudge prompts and JSON/tool-call rescue parsing
- `src/context.rs` - context manager, token budget, compaction callbacks
- `src/compact.rs` - no-op, sliding-window, and tiered compaction
- `src/client.rs` - `LLMClient`, streaming trait, OpenAI tool formatting
- `src/backends/` - Anthropic, Llamafile, and Ollama clients
- `src/server.rs` - backend lifecycle and context-budget resolution
- `src/http_server.rs`, `src/handler.rs`, `src/proxy.rs` - HTTP/OpenAI-compatible proxy
- `tests/` - integration coverage for core behavior

## Commands

Run before committing:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Use `cargo fmt --all` to apply formatting.

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

## Current status notes

The clean-room run summary reports:
- 8 of 8 units completed
- 487 tests passing
- 0 contamination incidents

Treat that as historical status. Re-run the local test suite after any meaningful change.
