# forge-guardrails

[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![Crates.io](https://img.shields.io/crates/v/forge-guardrails.svg)](https://crates.io/crates/forge-guardrails)
[![CI](https://github.com/whit3rabbit/forge-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/forge-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Rust implementation of [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge), produced via the [`clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill) workflow and subsequently verified for full behavioral parity with the Python reference. See [`docs/CLEANROOM.md`](docs/CLEANROOM.md) for the clean-room run summary and parity review.

A reliability layer for self-hosted LLM tool-calling. You give forge a set of tools; the model calls whichever it wants in whatever order. Workflow structure is opt-in — `required_steps`, `prerequisites`, and `terminal_tool` let you constrain the loop when you need to, but forge's guardrails (rescue parsing, retry nudges, response validation) apply with zero required steps too.

**What forge-guardrails isn't:**
- **Not an agent orchestrator.** Forge sits inside one agentic loop and makes its tool calls reliable. Multi-agent graphs, DAG planners, and cross-agent coordination are out of scope.
- **Not a coding harness.** Forge is domain-agnostic. If you're building a coding agent, [proxy mode](#proxy-server) lifts your existing harness with forge's guardrails — no rewrite.

**Three ways to use it:**

- **Proxy server** — Drop-in OpenAI-compatible and Anthropic-compatible proxy (`forge-guardrails-proxy` binary) that sits between any client and a local model server. Applies guardrails transparently. Also accepts Anthropic Messages API requests at `POST /v1/messages`, translated through `anyllm_translate`.

- **WorkflowRunner** — Define tools, pick a backend, run structured agent loops. Forge manages the full lifecycle: system prompts, tool execution, context compaction, and guardrails. **SlotWorker** adds priority-queued access to a shared inference slot with auto-preemption — for multi-agent architectures where specialist workflows share a GPU slot. Best when you're building on forge directly.

- **Guardrails middleware** — Use forge's reliability stack inside your own orchestration loop. You control the loop; forge validates responses, rescues malformed tool calls, and enforces required steps.

Supports Ollama, llama-server (llama.cpp), Llamafile, Anthropic, and anyllm-routed OpenAI-compatible upstreams as backends.

> Status: experimental. Behavioral parity with the Python reference has been verified through the parity test suite. Review for production hardening before deployment — see [Known review areas](#known-review-areas-before-release).

## Provenance

This repository was produced as a clean-room migration of the original Python Forge project into Rust, then brought to full behavioral parity with that reference implementation.

- Original project: [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge)
- Clean-room workflow: [`whit3rabbit/clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill)
- Full audit and parity review: [`docs/CLEANROOM.md`](docs/CLEANROOM.md)

This repository is not affiliated with, endorsed by, or maintained by the original Forge author unless stated elsewhere. Keep attribution to the original project and preserve license notices when redistributing.

## Requirements

- Rust 1.95+
- A running LLM backend (see below)

## Install

Add to your `Cargo.toml`:

```toml
[dependencies]
forge-guardrails = "0.1"
```

For development:

```bash
git clone https://github.com/whit3rabbit/forge-rs.git
cd forge-rs
cargo build
```

The `forge/` submodule contains the Python reference for fixture generation and parity checks. Initialize it with:

```bash
git submodule update --init --recursive
```

### Backend setup (pick one)

**llama-server** (recommended — top eval configs all run on llama-server):

Recommended model: [`mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf`](https://huggingface.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF/blob/main/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf) (bartowski / HuggingFace)

```bash
# Install from https://github.com/ggml-org/llama.cpp/releases
llama-server -m path/to/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf --jinja -ngl 999 --port 8080
```

**Ollama** (alternative — easier setup):
```bash
# Install from https://ollama.com/download
ollama pull ministral-3:8b-instruct-2512-q4_K_M
```

**Anthropic** (API, no local GPU needed):
```bash
export ANTHROPIC_API_KEY=sk-...
```

See [Backend Setup](docs/BACKEND_SETUP.md) for full instructions.

## Quick Start

Start llama-server (in a separate shell). Download the recommended model from [HuggingFace](https://huggingface.co/bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF/blob/main/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf) first:

```bash
llama-server -m path/to/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf --jinja -ngl 999 --port 8080
```

Then use the `WorkflowRunner` in your Rust code:

```rust
use forge_guardrails::{
    Workflow, ToolDef, ToolSpec, ParamModel,
    WorkflowRunner, LlamafileClient,
    ContextManager, TieredCompact,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = LlamafileClient::new("path/to/model.gguf")
        .with_mode("native");

    let ctx = ContextManager::new(
        Box::new(TieredCompact::new(2)),
        8192,
    );

    let workflow = Workflow {
        name: "weather".into(),
        description: "Look up weather for a city.".into(),
        tools: HashMap::new(), // populate with ToolDef entries
        required_steps: vec![],
        terminal_tool: Some("get_weather".into()),
        system_prompt_template: "You are a helpful assistant. Use the available tools.".into(),
        prerequisites: vec![],
    };

    let runner = WorkflowRunner::new(client, ctx);
    runner.run(&workflow, "What's the weather in Paris?", None).await?;
    Ok(())
}
```

For multi-step workflows, multi-turn conversations, and backend auto-management, see [Eval Guide](docs/EVAL_GUIDE.md) and [Backend Setup](docs/BACKEND_SETUP.md).

## Proxy Server

Drop-in OpenAI-compatible (and Anthropic-compatible) proxy that sits between any client and a local model server. Point your client at the proxy and forge applies its guardrails transparently.

This is the path for **using forge with an existing harness** (opencode, Continue, aider, Cline, anything that speaks the OpenAI chat-completions schema). No rewrite.

```bash
# External mode — you manage the backend, forge proxies it.
cargo run --bin forge-guardrails-proxy -- \
  --backend-url http://localhost:8080 \
  --port 8081

# Managed mode — forge starts the backend and proxy together.
cargo run --bin forge-guardrails-proxy -- \
  --backend llamaserver \
  --gguf path/to/model.gguf \
  --port 8081
```

Then configure OpenAI-compatible clients to use `http://localhost:8081/v1` as the API base URL. Anthropic-compatible clients should use `http://localhost:8081`; the proxy accepts Anthropic Messages API requests at `POST /v1/messages`.

**Backend compatibility:**

- **Managed mode** spins up the backend for you. Supported backends: `llamaserver`, `llamafile`, `ollama` (use `--gguf` for GGUF-based backends, or `--model` for Ollama).
- **External mode** is backend-agnostic — forge talks `POST /v1/chat/completions` to whatever you point `--backend-url` at, as long as it speaks the OpenAI schema. Tool calls must come back in OpenAI `tool_calls` format or in one of forge's rescue-parsed formats (Mistral `[TOOL_CALLS]`, Qwen `<tool_call>` XML, fenced JSON).
- **Env-routed mode** remains a Rust extension for Docker/provider routing. If neither `--backend-url` nor `--backend` is passed, the binary uses existing anyllm/provider env vars such as `PROXY_CONFIG`, `OPENAI_BASE_URL`, and `BACKEND`.

This proxy does not enforce inbound authentication. Do not expose it publicly without a reverse proxy, network policy, or another auth layer.

### What proxy mode fortifies

On every `POST /v1/chat/completions`, forge applies (in order):

1. **Response validation** — each tool call in the model's response is checked against the `tools` array in the request. Calls to unknown tool names or with malformed shapes are caught before the response returns to your client.
2. **Rescue parsing** — when the model emits tool calls in the wrong format (JSON in a code fence, Mistral's `[TOOL_CALLS]name{args}`, Qwen's `<tool_call>...\</tool_call>` XML), forge extracts the structured call and re-emits it in the canonical OpenAI `tool_calls` schema.
3. **Retry loop with error tracking** — if validation fails, forge retries inference up to `--max-retries` (default 3) with a corrective tool-result message on the canonical channel, rather than returning a malformed response.
4. **Synthetic `respond` tool injection** — when tools are present in the request, forge injects a synthetic `respond` tool the model calls instead of producing bare text. The `respond` call is stripped from the outbound response — the client sees a normal text response (`finish_reason: "stop"`) and never knows the tool exists. Essential for small local models (~8B) that can't reliably choose between text and tool calls.

### What proxy mode does *not* do

Proxy mode is single-shot per request; some forge features need multi-turn workflow state that the OpenAI chat-completions schema doesn't carry:

- **Prerequisite enforcement and step-ordering** — these need a workflow definition spanning turns. Available in `WorkflowRunner`.
- **Context compaction and session memory** — proxy mode forwards the inbound message list as-is; managing the rolling window is the client's job.
- **VRAM-aware budget detection** — opt in with `--budget-mode forge-full` or `--budget-mode forge-fast`; otherwise proxy uses the backend's reported budget. Env-routed mode can also use `FORGE_CONTEXT_TOKENS`.

### Useful flags

| Flag | Default | Purpose |
|---|---|---|
| `--backend-url URL` | — | External OpenAI-compatible backend |
| `--backend {llamaserver,llamafile,ollama}` | — | Managed backend type |
| `--model MODEL` | — | Model name, required for `ollama` |
| `--gguf PATH` | — | GGUF path, required for `llamaserver` / `llamafile` |
| `--backend-port N` | `8080` | Managed backend port |
| `--host HOST` | `127.0.0.1` | Proxy bind host in CLI mode |
| `--port N` | `8081` | Proxy bind port |
| `--max-retries N` | `3` | Retry budget per validation failure |
| `--no-rescue` | rescue on | Disable rescue parsing |
| `--budget-mode {backend,manual,forge-full,forge-fast}` | `backend` | Context budget source |
| `--budget-tokens N` | — | Manual token budget |
| `--serialize` / `--no-serialize` | auto | Force request serialization |
| `--extra-flags -- FLAG VALUE ...` | — | Pass additional flags to the managed backend |

### Useful environment variables (Docker / env-routed mode)

| Variable | Default | Purpose |
|---|---|---|
| `FORGE_HOST` | `0.0.0.0` | Bind address |
| `FORGE_PORT` / `PORT` / `LISTEN_PORT` | `8081` | Listen port |
| `FORGE_MODEL` / `SMALL_MODEL` | `gpt-4o-mini` | Default model |
| `FORGE_CONTEXT_TOKENS` | `128000` | Token budget |
| `FORGE_MAX_RETRIES` | `3` | Retry budget per validation failure |
| `FORGE_RESCUE_ENABLED` | `true` | Enable rescue parsing |
| `FORGE_SERIALIZE_REQUESTS` | `false` | Force request serialization |
| `BACKEND` | `openai` | anyllm provider id or first-party backend |
| `OPENAI_BASE_URL` | — | Route to a local OpenAI-compatible backend |
| `OPENAI_API_KEY` | — | API key forwarded to the upstream |

Existing anyllm env and config are still honored, including provider API keys, `PROXY_CONFIG`, `BIG_MODEL`, `SMALL_MODEL`, and LiteLLM aliases such as `LITELLM_CONFIG`.

### Docker

You can run the Forge proxy as a Docker container. Expose only the Forge proxy port (`8081`) to clients. The optional anyllm sidecar is a private upstream hop, not the public proxy URL, and is not enabled by default.

Build the image locally:

```bash
docker build -t forge-guardrails:local .
```

After publishing, replace `forge-guardrails:local` in these examples with `followthewhit3rabbit/forge-guardrails:latest`.

Run with OpenAI through the in-process anyllm runtime:

```bash
docker run --rm -p 8081:8081 \
  -e OPENAI_API_KEY=sk-... \
  -e FORGE_MODEL=gpt-4o-mini \
  forge-guardrails:local
```

Start Ollama on the host in another terminal:

```bash
ollama pull qwen2.5-coder:14b
ollama serve
```

Then run the proxy container:

```bash
docker run --rm -p 8081:8081 \
  -e BACKEND=ollama \
  -e OPENAI_BASE_URL=http://host.docker.internal:11434/v1 \
  -e OPENAI_API_KEY=dummy \
  -e FORGE_MODEL=qwen2.5-coder:14b \
  forge-guardrails:local
```

Start llama-server on the host in another terminal:

```bash
llama-server \
  -m /path/to/model.gguf \
  --jinja \
  --host 0.0.0.0 \
  --port 8080
```

Then run the proxy container:

```bash
docker run --rm -p 8081:8081 \
  -e OPENAI_BASE_URL=http://host.docker.internal:8080/v1 \
  -e OPENAI_API_KEY=dummy \
  -e FORGE_MODEL=local-llama \
  forge-guardrails:local
```

On Linux Docker engines that do not define `host.docker.internal`, add:

```bash
--add-host=host.docker.internal:host-gateway
```

Smoke the running proxy:

```bash
curl http://localhost:8081/health

curl http://localhost:8081/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-coder:14b","messages":[{"role":"user","content":"Say ok"}],"stream":false}'
```

OpenAI-compatible clients should use:

```text
base_url: http://localhost:8081/v1
api_key: dummy
model: qwen2.5-coder:14b
```

Claude Code can use the same Docker proxy through Forge's Anthropic-compatible endpoint:

```bash
unset ANTHROPIC_API_KEY
export ANTHROPIC_BASE_URL=http://127.0.0.1:8081
export ANTHROPIC_AUTH_TOKEN=dummy
export ANTHROPIC_MODEL=qwen2.5-coder:14b

claude --model qwen2.5-coder:14b
```

Do not add `/v1` to `ANTHROPIC_BASE_URL`; Claude Code sends Anthropic Messages requests and Forge serves those at `/v1/messages`. If you want Claude Code's model picker to query Forge's `/v1/models` endpoint, set `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`.

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

## Backends

| Backend | Best for | Native FC? |
|---------|----------|------------|
| **Ollama** | Easiest setup, model management built-in | Yes |
| **llama-server** | Best performance, full control | Yes (with `--jinja`) |
| **Llamafile** | Single binary, zero dependencies | No (prompt-injected) |
| **Anthropic** | Frontier baseline, hybrid workflows | Yes |
| **anyllm runtime** | In-process provider routing, OpenAI-compatible | Provider-dependent |
| **anyllm sidecar** | Separate process; admin UI, cache, metrics | Provider-dependent |

See [Backend Setup](docs/BACKEND_SETUP.md) for installation details.

## macOS / Apple Silicon

Apple Silicon is supported through all backends. Ollama can be installed with Homebrew or the official macOS download. llama.cpp / llama-server can be installed with Homebrew or a Metal-enabled release build. Llamafile works on macOS as a downloaded binary after `chmod +x`.

Managed llama.cpp and llamafile startup passes `-ngl 999`; on macOS that uses Metal rather than CUDA. Automatic Ollama context budgets use Rust VRAM tiers: <24 GB → 4096 tokens, 24–47 GB → 32768 tokens, ≥48 GB → 262144 tokens.

MLX is supported as an optional eval path on macOS through an OpenAI-compatible server such as `mlx_lm.server`, routed by `AnyLlmRuntimeClient` or `AnyLlmProxyClient`. It is not a managed `ServerManager` backend. Prefer llama-server for parity runs; treat GGUF-on-MLX as experimental.

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
```

## Running Tests

```bash
cargo test
```

```bash
# Parity suite only (requires the Python golden fixture)
cargo test --test parity_tests

# With coverage (requires cargo-llvm-cov)
cargo llvm-cov --all-targets
```

Regenerate the Python golden fixture after intentional reference-behavior changes:

```bash
uv run --project forge python tests/parity/generate_fixtures.py
```

## Eval Harness

The eval harness measures how reliably a model + backend combo navigates multi-step tool-calling workflows. See [Eval Guide](docs/EVAL_GUIDE.md) for full CLI reference.

```bash
# Python oracle against a running Rust proxy
python scripts/eval_openai_proxy.py \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 10 \
  --stream \
  --scenario basic_2step sequential_3step error_recovery \
  --output eval_results_rust_proxy.jsonl

# Native Rust smoke runner
cargo run --bin forge-eval -- \
  --backend openai-proxy \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 3 \
  --scenario basic_2step \
  --stream
```

The Rust smoke runner supports `basic_2step`, `sequential_3step`, and `error_recovery` scenarios and emits JSONL for quick CI/smoke checks.

## Project Structure

```
src/
  lib.rs                     Public API re-exports
  error.rs                   ForgeError hierarchy
  server.rs                  setup_backend(), ServerManager, BudgetMode
  core/
    message.rs               Message, MessageRole, MessageType, MessageMeta, ToolCallInfo
    tool_spec.rs             ToolSpec, ToolDef, ParamModel — tool schema and callable defs
    workflow.rs              Workflow model, terminal tools, prerequisites
    steps.rs                 StepTracker, step tracking and required-step state
    inference.rs             run_inference() — shared front half (compact, fold, validate, retry)
    runner.rs                WorkflowRunner — the agentic loop
    slot_worker.rs           SlotWorker — priority-queued slot access
  guardrails/
    guardrails.rs            Guardrails facade — applies the full stack in foreign loops
    nudge.rs                 Nudge dataclass
    response_validator.rs    ResponseValidator, ValidationResult
    step_enforcer.rs         StepEnforcer, StepCheck, StepPrerequisite
    error_tracker.rs         ErrorTracker
  clients/
    base.rs                  LLMClient trait, ChunkType, StreamChunk, LLMCallInfo, TokenUsage
    sampling.rs              Model sampling defaults, MODEL_SAMPLING_DEFAULTS
    anthropic/               AnthropicClient (frontier baseline, native FC)
    llamafile/               LlamafileClient (native FC or prompt-injected)
    ollama/                  OllamaClient (native FC)
    anyllm_proxy.rs          AnyLlmRuntimeClient, AnyLlmProxyClient
  context/
    manager.rs               ContextManager, CompactEvent
    strategies.rs            NoCompact, TieredCompact, SlidingWindowCompact
    hardware.rs              HardwareProfile, detect_hardware()
  prompts/
    mod.rs                   Tool prompt builders (prompt-injected path)
    nudges.rs                Retry and step-enforcement nudge templates
    parse_strategies.rs      Rescue parsing: Mistral, Qwen, fenced JSON
  tools/
    respond.rs               Synthetic respond tool (respond_tool(), respond_spec())
  proxy/
    handler.rs               Request handler — bridge between HTTP and run_inference
    proxy.rs                 OpenAI messages ↔ forge Messages conversion, SSE helpers
    server.rs                HTTPServer — axum HTTP/SSE server
  bin/
    forge-guardrails-proxy.rs  CLI proxy entry point
    forge-eval/                Native Rust eval smoke runner
tests/
  parity/                    Python-generated golden fixtures for Rust parity tests
  parity_tests.rs            Rust assertions against python_golden.json
  engine_tests.rs            WorkflowRunner / inference integration tests
  guardrails_tests.rs        Guardrails and step enforcement tests
  compact_tests.rs           Compaction strategy tests
  context_tests.rs           ContextManager tests
  *_tests.rs                 Unit and integration tests per subsystem
scripts/
  eval_openai_proxy.py       Python eval oracle wrapper for Rust proxy checks
docs/
  CLEANROOM.md               Clean-room run summary and parity review
  PARITY.md                  Parity contract and subsystem alignment status
  EVAL_GUIDE.md              Eval harness CLI reference
  BACKEND_SETUP.md           Backend installation and server setup
```

## Public API Surface

The crate re-exports the main building blocks from `src/lib.rs`:

```rust
use forge_guardrails::{
    // Backends
    AnthropicClient, LlamafileClient, OllamaClient,
    AnyLlmRuntimeClient, AnyLlmProxyClient,
    // Client trait and types
    LLMClient, LLMResponse, StreamChunk, TextResponse, ToolCall, TokenUsage,
    // Workflow
    WorkflowRunner, Workflow, ToolDef, ToolSpec, ParamModel,
    // Context
    ContextManager, NoCompact, SlidingWindowCompact, TieredCompact,
    // Guardrails
    Guardrails, StepEnforcer, ErrorTracker, ResponseValidator,
    // Step tracking
    StepTracker, SlotWorker,
    // Prompts and nudges
    retry_nudge, step_nudge, prerequisite_nudge, unknown_tool_nudge,
    rescue_tool_call, build_tool_prompt,
    // Proxy / server
    handle_chat_completions, handle_anthropic_messages,
    HTTPServer, ServerManager, setup_backend,
};
```

## Usage Modes

### 1. Workflow runner

Use `WorkflowRunner` when you want the library to manage the LLM loop: system prompt construction, message folding, validation, retries, tool execution, context compaction, and terminal-tool detection.

### 2. Guardrails middleware

Use the guardrail primitives directly when you already own the orchestration loop but want validation and policy enforcement.

Relevant pieces:

- `Guardrails` — composable facade for the full stack
- `ResponseValidator` / `ValidationResult`
- `StepEnforcer` / `StepCheck` / `StepPrerequisite`
- `ErrorTracker`
- `retry_nudge`, `step_nudge`, `prerequisite_nudge`, `unknown_tool_nudge`

### 3. OpenAI-compatible proxy / server layer

Use the proxy and HTTP server pieces when you need an OpenAI-compatible request/response boundary around a backend.

Relevant pieces:

- `openai_to_messages`, `tool_calls_to_openai`, `text_response_to_openai`
- `text_to_sse_events`, `tool_calls_to_sse_events`
- `handle_chat_completions`, `handle_anthropic_messages`
- `HTTPServer`, `ServerManager`, `setup_backend`

### 4. anyllm runtime and sidecar integration

Use `AnyLlmRuntimeClient` for in-process anyllm provider routing (no HTTP overhead; Forge still owns interception and nudging):

```rust
use forge_guardrails::AnyLlmRuntimeClient;

let client = AnyLlmRuntimeClient::from_multi_config(
    "gpt-4o-mini",
    anyllm_proxy::config::MultiConfig::load().multi_config,
)
.with_context_length(128_000);
```

Use `AnyLlmProxyClient` when you prefer to run `anyllm_proxy` as a separate sidecar process. The sidecar URL is an upstream hop from Forge to anyllm, not the public client-facing Forge proxy URL. Keep the sidecar private and expose only the Forge proxy unless you intentionally need direct anyllm access.

```rust
use forge_guardrails::AnyLlmProxyClient;

let client = AnyLlmProxyClient::new("gpt-4o-mini")
    .with_base_url("http://127.0.0.1:3000")
    .with_api_key("local-proxy-key")
    .with_context_length(128_000);
```

Both clients expose provider observability through `LLMClient::last_call_info()`.

## Testing Scope

- 487+ passing tests across 16 test files
- Deterministic parity suite against `tests/parity/fixtures/python_golden.json`
- 0 contamination incidents in the clean-room run

Keep tests deterministic where possible. Backend integration tests use mock servers (via `mockito`) unless they intentionally qualify a live backend.

## Known Review Areas Before Release

The implementation should be reviewed for protocol correctness and production hardening before publication or deployment. Behavioral parity with the Python reference is covered by the parity test suite; the following areas need additional protocol and integration review:

- tool-call ID pairing across assistant tool calls and tool results
- transcript validity after guardrail-blocked steps
- compaction behavior around tool-call / tool-result groups
- true progressive streaming behavior for each backend
- HTTP parsing and CORS/header handling if exposed beyond local development
- backend startup ordering and context-budget discovery
- serialization behavior for OpenAI, Ollama, and Anthropic formats

## Python Parity

The parity suite compares Rust behavior to synthetic golden outputs generated by the Python reference submodule. The source of truth for fixture generation is `tests/parity/generate_fixtures.py`; the checked-in output is `tests/parity/fixtures/python_golden.json`; Rust assertions live in `tests/parity_tests.rs`.

When updating parity behavior:
1. Add or update the Python fixture first.
2. Regenerate `tests/parity/fixtures/python_golden.json`.
3. Add or update the matching Rust assertion in `tests/parity_tests.rs`.
4. Run `cargo test --test parity_tests` before broader repo gates.

See [docs/PARITY.md](docs/PARITY.md) for the full parity contract.

## Relationship to Upstream Forge

The upstream Forge project is a Python reliability layer for self-hosted LLM tool-calling and multi-step agentic workflows. This repository is a Rust implementation inspired by that project's behavior — not a direct source translation — and has been verified for full behavioral parity with the Python reference through the parity test suite.

The Python reference is included as the `forge/` git submodule for use in fixture generation and parity checks.

Use the upstream repository for the original Python implementation, documentation, paper citation, and release history:

- <https://github.com/antoinezambelli/forge>

The forge guardrail framework and ablation study are published as:

> Zambelli, A. *Forge: A Reliability Layer for Self-Hosted LLM Tool-Calling.*
> [https://doi.org/10.1145/3786335.3813193](https://doi.org/10.1145/3786335.3813193)

## License

[MIT](LICENSE) — Rust implementation copyright (c) 2025-2026 whit3rabbit.

The upstream Forge project is separately licensed by its author as MIT as well. Preserve upstream attribution and review license compatibility before redistribution.
