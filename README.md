# forge-guardrails

A Rust clean-room implementation of [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge), built as a test of the [`clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill) workflow. See `docs/CLEANROOM.md` for the clean-room run summary.

`forge-guardrails` provides foundation types and runtime components for reliable LLM tool-calling workflows. It focuses on structured agent loops, response validation, retry nudges, prerequisite enforcement, context compaction, backend adapters, and an OpenAI-compatible proxy/server surface.

> Status: experimental clean-room port. The implementation was produced from clean behavioral artifacts and should be reviewed before production use.

## Clean-room provenance

This repository was produced as a clean-room migration of the original Python Forge project into Rust.

- Original project: [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge)
- Clean-room workflow/tooling: [`whit3rabbit/clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill)
- Local audit summary: `docs/CLEANROOM.md`
- Reported result: 8 / 8 units complete, 487 tests passing, 0 contamination incidents

This repository is not affiliated with, endorsed by, or maintained by the original Forge author unless stated elsewhere. Keep attribution to the original project and preserve license notices when redistributing.

## What this is

`forge-guardrails` is a Rust library for building agentic tool-calling loops around LLM backends.

It includes:

- workflow definitions with tools, required steps, prerequisites, and terminal tools
- guardrails for response validation, retry nudges, step enforcement, and error tracking
- typed message and streaming abstractions
- context-window management and compaction strategies
- backend clients for Anthropic, Llamafile / llama-server-compatible APIs, and Ollama
- sampling defaults for common model families
- a synthetic `respond` tool for forcing small models onto the tool channel
- OpenAI-compatible message conversion, proxy helpers, and HTTP server components
- a slot worker for serialized access to shared inference capacity

## What this is not

This is not a legal safe harbor, legal opinion, or proof that a clean-room process is sufficient for a given use case. It is an engineering artifact produced by a clean-room workflow. Review the generated artifacts, licenses, and implementation before relying on it.

This is also not a full multi-agent orchestration framework. It provides the lower-level pieces for one guarded LLM workflow loop and related proxy/server integration.

## Crate metadata

```toml
[package]
name = "forge-guardrails"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "Foundation types for an LLM-agent workflow framework"
```

## Major modules

```text
docs/
  CLEANROOM.md              Clean-room run summary and audit notes
src/
  backends/                 Anthropic, Llamafile, and Ollama adapters
  guardrails/               Validator, step enforcer, nudges, error tracking
  prompts/                  Tool-prompt builders and rescue parsers
  client.rs                 LLMClient trait, API format helpers, token usage
  compact.rs                NoCompact, SlidingWindowCompact, TieredCompact
  context.rs                ContextManager, compaction callbacks, thresholds
  error.rs                  ForgeError and backend/workflow error types
  handler.rs                OpenAI-compatible request handler
  hardware.rs               Hardware and context-budget helpers
  http_server.rs            Lightweight HTTP server wrapper
  inference.rs              Shared inference, validation, folding, serialization
  message.rs                Message, role, metadata, and tool-call types
  nudges.rs                 Corrective nudge text for guardrail failures
  proxy.rs                  OpenAI conversion and proxy response helpers
  respond.rs                Synthetic respond tool
  runner.rs                 WorkflowRunner orchestration loop
  sampling.rs               Model sampling defaults
  server.rs                 Backend lifecycle and budget setup
  slot_worker.rs            Serialized / queued inference slot worker
  steps.rs                  Step tracking and prerequisite checks
  streaming.rs              Stream chunks and LLM response union
  tool_spec.rs              Tool schema model
  workflow.rs               ToolDef and Workflow model
tests/
  *_tests.rs                Unit and integration-style tests
```

## Supported backends

The library exposes backend clients for:

- Anthropic Messages API
- Llamafile / llama-server-style OpenAI-compatible chat APIs
- Ollama chat API

Backend support is exposed through the shared `LLMClient` trait, plus common response types such as `LLMResponse`, `TextResponse`, `ToolCall`, `StreamChunk`, and `TokenUsage`.

## Public API surface

The crate re-exports the main building blocks from `src/lib.rs`, including:

```rust
use forge_guardrails::{
    AnthropicClient,
    LlamafileClient,
    OllamaClient,
    LLMClient,
    WorkflowRunner,
    Workflow,
    ToolDef,
    ToolSpec,
    ContextManager,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
    Guardrails,
    StepEnforcer,
    ErrorTracker,
    LLMResponse,
    StreamChunk,
    ToolCall,
};
```

## Development

Clone the repository and run the test suite:

```bash
git clone <repo-url>
cd forge-guardrails
cargo test
```

Run formatting and lint checks before submitting changes:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Usage modes

### 1. Workflow runner

Use `WorkflowRunner` when you want the library to manage the LLM loop: system prompt construction, message folding, validation, retries, tool execution, context compaction, and terminal-tool detection.

This is the direct library integration path for Rust applications that want Forge-like workflow behavior.

### 2. Guardrails components

Use the guardrail primitives directly when you already own the orchestration loop but want validation and policy enforcement.

Relevant pieces include:

- `ResponseValidator`
- `StepEnforcer`
- `StepTracker`
- `ErrorTracker`
- `retry_nudge`
- `step_nudge`
- `prerequisite_nudge`
- `unknown_tool_nudge`

### 3. OpenAI-compatible proxy/server layer

Use the proxy and HTTP server pieces when you need an OpenAI-compatible request/response boundary around a backend.

Relevant pieces include:

- `openai_to_messages`
- `tool_calls_to_openai`
- `text_response_to_openai`
- `text_to_sse_events`
- `tool_calls_to_sse_events`
- `handle_chat_completions`
- `HTTPServer`
- `ServerManager`
- `setup_backend`

## Testing scope

The clean-room summary reports:

- 487 passing tests
- 27 Rust source files
- 13 Rust test files
- 0 contamination incidents

Keep tests deterministic where possible. Backend integration tests should use mock servers unless they intentionally qualify a live backend.

## Known review areas before release

This clean-room implementation should be reviewed for protocol correctness and production hardening before publication or deployment. In particular, audit:

- tool-call ID pairing across assistant tool calls and tool results
- transcript validity after guardrail-blocked steps
- compaction behavior around tool-call / tool-result groups
- true progressive streaming behavior for each backend
- HTTP parsing and CORS/header handling if exposed beyond local development
- backend startup ordering and context-budget discovery
- serialization behavior for OpenAI, Ollama, and Anthropic formats

## Relationship to upstream Forge

The upstream Forge project is a Python reliability layer for self-hosted LLM tool-calling and multi-step agentic workflows. This repository is a Rust clean-room implementation inspired by that project’s behavior, not a direct source translation.

Use the upstream repository for the original Python implementation, documentation, paper citation, and release history:

- <https://github.com/antoinezambelli/forge>

## Relationship to clean-room-skill

This repository was created to exercise the Clean Room workflow, which separates source-reading roles from clean implementation roles and produces durable behavioral artifacts before implementation.

Use the clean-room skill repository for the workflow, installation instructions, boundary model, and CLI/runtime details:

- <https://github.com/whit3rabbit/clean-room-skill>

## License

MIT. See `LICENSE` if present in the repository.

The upstream Forge project is separately licensed by its author. Preserve upstream attribution and review license compatibility before redistribution.
