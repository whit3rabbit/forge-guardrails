# forge-guardrails

A Rust implementation of [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge), produced via the [`clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill) workflow and subsequently brought to full behavioral parity with the Python reference. See [`docs/CLEANROOM.md`](docs/CLEANROOM.md) for the clean-room run summary and parity review.

`forge-guardrails` provides foundation types and runtime components for reliable LLM tool-calling workflows. It focuses on structured agent loops, response validation, retry nudges, prerequisite enforcement, context compaction, backend adapters, and an OpenAI-compatible proxy/server surface.

> Status: experimental. Behavioral parity with the Python reference has been verified through the parity test suite. Review for production hardening before deployment — see [Known review areas](#known-review-areas-before-release).

## Provenance

This repository was produced as a clean-room migration of the original Python Forge project into Rust, then brought to full behavioral parity with that reference implementation.

- Original project: [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge)
- Clean-room workflow: [`whit3rabbit/clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill)
- Full audit and parity review: [`docs/CLEANROOM.md`](docs/CLEANROOM.md)

This repository is not affiliated with, endorsed by, or maintained by the original Forge author unless stated elsewhere. Keep attribution to the original project and preserve license notices when redistributing.

## What this is

`forge-guardrails` is a Rust library for building agentic tool-calling loops around LLM backends.

It includes:

- workflow definitions with tools, required steps, prerequisites, and terminal tools
- guardrails for response validation, retry nudges, step enforcement, and error tracking
- typed message and streaming abstractions
- context-window management and compaction strategies
- backend clients for Anthropic, Llamafile / llama-server-compatible APIs, and Ollama
- anyllm-routed OpenAI-compatible upstreams for optional provider and local eval paths
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
  CLEANROOM.md              Clean-room run summary, parity review, and attribution
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
- anyllm-routed OpenAI-compatible chat APIs, including optional local eval endpoints

Backend support is exposed through the shared `LLMClient` trait, plus common response types such as `LLMResponse`, `TextResponse`, `ToolCall`, `StreamChunk`, and `TokenUsage`.

## macOS / Apple Silicon backends

Apple Silicon is supported through the same backends. Ollama can be installed with Homebrew or the official macOS download. llama.cpp / llama-server can be installed with Homebrew or a Metal-enabled release build. llamafile works on macOS as a downloaded binary after `chmod +x`.

Managed llama.cpp and llamafile startup passes `-ngl 999`; on macOS that uses Metal rather than CUDA, so no NVIDIA driver setup is required. Apple Silicon uses unified memory shared with the OS. Automatic Ollama context budgets use the existing Rust VRAM tiers: less than 24 GB gets 4096 tokens, 24 GB to 47 GB gets 32768 tokens, and 48 GB or more gets 262144 tokens.

MLX is supported as an optional macOS eval path through an OpenAI-compatible
server such as `mlx_lm.server`, routed by `AnyLlmRuntimeClient` or
`AnyLlmProxyClient`. It is not a managed `ServerManager` backend and is not a
Python-parity target. Prefer llama-server for parity runs; use MLX when the
goal is local Apple Silicon throughput or comparing MLX-format models. Treat
GGUF-on-MLX as experimental and server/model dependent: MLX's normal `mlx-lm`
path is MLX-compatible Hugging Face repos or local converted models, while
GGUF support has narrower quantization and architecture coverage than
llama.cpp.

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

### Docker proxy

The `forge-guardrails-proxy` binary runs Forge as an OpenAI-compatible and
Anthropic-compatible guardrail proxy. It preserves the inbound request `model`
for upstream anyllm routing, so the same container can serve plain env-based
backends or a `PROXY_CONFIG` model router.

Build and run locally:

```bash
docker build -t forge-guardrails:local .
docker run --rm -p 3000:3000 \
  -e OPENAI_API_KEY=sk-... \
  forge-guardrails:local
curl http://localhost:3000/health
```

Route to a local OpenAI-compatible backend such as Ollama:

```bash
docker run --rm -p 3000:3000 \
  -e OPENAI_BASE_URL=http://host.docker.internal:11434/v1 \
  -e OPENAI_API_KEY=dummy \
  forge-guardrails:local
```

Useful runtime env vars:

- `FORGE_HOST`, default `0.0.0.0`
- `FORGE_PORT`, fallback `PORT`, fallback `LISTEN_PORT`, default `3000`
- `FORGE_MODEL`, fallback `SMALL_MODEL`, default `gpt-4o-mini`
- `FORGE_CONTEXT_TOKENS`, default `128000`
- `FORGE_MAX_RETRIES`, default `3`
- `FORGE_RESCUE_ENABLED`, default `true`
- `FORGE_SERIALIZE_REQUESTS`, default `false`

Existing anyllm env and config are still honored, including `BACKEND`,
provider API keys, `OPENAI_BASE_URL`, `PROXY_CONFIG`, `BIG_MODEL`,
`SMALL_MODEL`, and LiteLLM aliases such as `LITELLM_CONFIG`.

This proxy does not enforce inbound authentication. Do not expose it publicly
without a reverse proxy, network policy, or another auth layer.

Publish to Docker Hub as `followthewhit3rabbit/forge-guardrails`:

```bash
docker login -u followthewhit3rabbit
docker buildx create --use --name forge-guardrails-builder || docker buildx use forge-guardrails-builder
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t followthewhit3rabbit/forge-guardrails:0.1.0 \
  -t followthewhit3rabbit/forge-guardrails:latest \
  --push .
```

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

### Live backend smoke recipes

These checks are manual. Keep CI on mock servers unless a test is explicitly
qualifying a live local backend.

For llama-server native function calling, start llama.cpp with Jinja tool
templates enabled:

```bash
llama-server -m path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf --jinja -ngl 999 --port 8080
```

Use `LlamafileClient::new(path).with_mode("native")` against
`http://localhost:8080/v1`. For prompt-injected fallback on the same server,
start without `--jinja` if desired and use `.with_mode("prompt")`.

For Ollama Python-parity behavior, use the dedicated native client path:

```bash
ollama pull ministral-3:8b-instruct-2512-q4_K_M
curl http://localhost:11434/api/chat -d '{
  "model": "ministral-3:8b-instruct-2512-q4_K_M",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "tools": [{"type": "function", "function": {"name": "calc", "description": "Math", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}}],
  "stream": false
}'
```

For generic OpenAI-compatible routing, configure anyllm with backend URLs such
as `http://localhost:8080/v1` for llama-server or
`http://localhost:11434/v1` for Ollama, then pass the resulting
`MultiConfig` to `AnyLlmRuntimeClient`. If routing by virtual model name is
needed, use `AnyLlmRuntimeClient::from_multi_config_with_model_router(...)`.

For optional MLX eval on macOS, run an OpenAI-compatible MLX server and route
it through anyllm as an `OpenAI` backend:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","messages":[{"role":"user","content":"Say ok"}],"max_tokens":16}'
```

Use `http://localhost:8080/v1` as the anyllm backend URL and keep Forge in
front of the request. Unless a specific MLX server/model combination is
qualified for native tool calls, evaluate it with Forge's prompt/rescue
guardrails rather than treating it as equivalent to llama-server `--jinja`.

For sidecar/admin/cache/metrics use, run `anyllm_proxy` separately and point
`AnyLlmProxyClient` at that sidecar. The guarded request path remains:
external client -> Forge `HTTPServer` or `handle_chat_completions` ->
`AnyLlmProxyClient` -> anyllm sidecar -> provider.

## Testing scope

Initial clean-room run metrics (see [`docs/CLEANROOM.md`](docs/CLEANROOM.md)):

- 487 passing tests
- 27 Rust source files
- 13 Rust test files
- 0 contamination incidents

After the clean-room phase, a full parity review against the Python reference established behavioral alignment across all major subsystems. The ongoing regression gate is `cargo test --test parity_tests`.

Keep tests deterministic where possible. Backend integration tests should use mock servers unless they intentionally qualify a live backend.

## Known review areas before release

The implementation should be reviewed for protocol correctness and production hardening before publication or deployment. Behavioral parity with the Python reference is covered by the parity test suite; the following areas need additional protocol and integration review:

- tool-call ID pairing across assistant tool calls and tool results
- transcript validity after guardrail-blocked steps
- compaction behavior around tool-call / tool-result groups
- true progressive streaming behavior for each backend
- HTTP parsing and CORS/header handling if exposed beyond local development
- backend startup ordering and context-budget discovery
- serialization behavior for OpenAI, Ollama, and Anthropic formats

## Relationship to upstream Forge

The upstream Forge project is a Python reliability layer for self-hosted LLM tool-calling and multi-step agentic workflows. This repository is a Rust implementation inspired by that project's behavior — not a direct source translation — and has been verified for full behavioral parity with the Python reference through the parity test suite.

The Python reference is included as the `forge/` git submodule for use in fixture generation and parity checks.

Use the upstream repository for the original Python implementation, documentation, paper citation, and release history:

- <https://github.com/antoinezambelli/forge>


The initial Rust implementation was produced using the Clean Room workflow as a deliberate test of that skill. The workflow separates source-reading roles from clean implementation roles and produces durable behavioral artifacts before any code is written. After the clean-room phase concluded, a full parity review was conducted against the Python reference to establish complete alignment.

See [`docs/CLEANROOM.md`](docs/CLEANROOM.md) for the full clean-room run summary and parity review narrative.

For the workflow itself, installation instructions, boundary model, and CLI/runtime details:

- <https://github.com/whit3rabbit/clean-room-skill>

## License

MIT. See `LICENSE` if present in the repository.

The upstream Forge project is separately licensed by its author as MIT as well. Preserve upstream attribution and review license compatibility before redistribution.
