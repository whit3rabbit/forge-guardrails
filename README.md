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

## macOS / Apple Silicon backends

Apple Silicon is supported through the same backends. Ollama can be installed with Homebrew or the official macOS download. llama.cpp / llama-server can be installed with Homebrew or a Metal-enabled release build. llamafile works on macOS as a downloaded binary after `chmod +x`.

Managed llama.cpp and llamafile startup passes `-ngl 999`; on macOS that uses Metal rather than CUDA, so no NVIDIA driver setup is required. Apple Silicon uses unified memory shared with the OS. Automatic Ollama context budgets use the existing Rust VRAM tiers: less than 24 GB gets 4096 tokens, 24 GB to 47 GB gets 32768 tokens, and 48 GB or more gets 262144 tokens.

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

The HTTP server also accepts Anthropic Messages API requests at
`POST /v1/messages`. Those requests are translated with `anyllm_translate`,
run through the same guarded OpenAI-compatible handler used by
`POST /v1/chat/completions`, then translated back to Anthropic responses.

### 4. anyllm runtime and sidecar integration

Use `AnyLlmRuntimeClient` when you want in-process anyllm provider routing
without handing HTTP route ownership to anyllm. Build it from
`anyllm_proxy::runtime::ChatCompletionRuntime`, `Config`, or `MultiConfig`;
Forge still owns interception, validation, nudging, and tool-call execution.

```rust
use forge_guardrails::AnyLlmRuntimeClient;

let client = AnyLlmRuntimeClient::from_multi_config(
    "gpt-4o-mini",
    anyllm_proxy::config::MultiConfig::load().multi_config,
)
.with_context_length(128_000);
```

Use `AnyLlmProxyClient` when you prefer to run `anyllm_proxy` as a separate
sidecar process.

```rust
use forge_guardrails::AnyLlmProxyClient;

let client = AnyLlmProxyClient::new("gpt-4o-mini")
    .with_base_url("http://127.0.0.1:3000")
    .with_api_key("local-proxy-key")
    .with_context_length(128_000);
```

Run `anyllm_proxy` separately for its provider catalog, routing, config files,
admin UI, cache, metrics, and batch surfaces. Point `AnyLlmProxyClient` at the
sidecar. Clients can call forge-guardrails through either OpenAI
`/v1/chat/completions` or Anthropic `/v1/messages`; forge still performs the
guarded interception before any request reaches the sidecar.

Both anyllm clients expose provider observability through
`LLMClient::last_call_info()`. The runtime client reports selected backend,
mapped model, backend kind, provider id, degradation warnings, rate limits, and
estimated cost from anyllm pricing when usage is available. The sidecar client
captures response headers such as `x-anyllm-degradation`, `x-anyllm-cache`,
Anthropic rate-limit headers, and optional `x-anyllm-cost-usd`; otherwise it
uses response usage and anyllm pricing for a best-effort cost estimate. Token
counts remain available separately through `last_usage()`.

Do not embed anyllm's HTTP router for guarded traffic. That path owns request
handling and can bypass Forge guardrails. The runtime client uses anyllm's
OpenAI-native Chat Completions runtime, so provider-specific OpenAI-compatible
fields such as `seed`, `top_k`, `min_p`, `repeat_penalty`, and
`chat_template_kwargs` are preserved for compatible backends.

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

The upstream Forge project is separately licensed by its author as MIT as well. Preserve upstream attribution and review license compatibility before redistribution.
