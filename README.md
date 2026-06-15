# forge-guardrails

[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org/)
[![Crates.io](https://img.shields.io/crates/v/forge-guardrails.svg)](https://crates.io/crates/forge-guardrails)
[![CI](https://github.com/whit3rabbit/forge-guardrails/actions/workflows/ci.yml/badge.svg)](https://github.com/whit3rabbit/forge-guardrails/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Rust implementation of [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge). See python version for original.

This was mostly a test of my clean-room-skill repo to see if it could manage to reproduce in rust. It somewhat succeeded, but I made a lot of tweaks to more closely match the original project. 

# Summary

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

- Original project: [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge)

## Requirements

- Rust 1.95+
- A running LLM backend (see below)

## Install

Install the proxy binary:

```bash
# macOS, using the Homebrew cask
brew install --cask whit3rabbit/tap/forge-guardrails-proxy

# macOS or Linux, from crates.io
cargo install forge-guardrails --locked --bin forge-guardrails-proxy
```

Use it with an existing OpenAI-compatible local backend:

```bash
forge-guardrails-proxy \
  --backend-url http://localhost:8080 \
  --port 8081
```

Then point OpenAI-compatible clients at `http://localhost:8081/v1`.
Requests should include their own `model` field. The proxy does not pick a
default upstream model unless you explicitly set `--model`, `FORGE_MODEL`, or
`SMALL_MODEL`; managed `ollama` still requires `--model`, and managed
`llamaserver` / `llamafile` use `--gguf`.

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

The Makefile wraps common development and eval commands. `make build` builds
all targets with the default `classifier` feature; override with
`FEATURES=""` for a no-feature build.

```bash
make build
make test
make clippy
```

The `forge/` submodule contains the Python reference for fixture generation and parity checks. Initialize it with:

```bash
git submodule update --init --recursive
```

### Release

Release is tag-driven. After the version in `Cargo.toml` is ready and `main`
is pushed, create and push a matching tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow verifies the tag matches the crate version, runs format,
clippy, tests, `cargo package`, and `cargo publish`, builds platform archives
for `forge-guardrails-proxy`, publishes the GitHub release, then updates
`whit3rabbit/homebrew-tap` when `HOMEBREW_TAP_TOKEN` is configured. Users can
install the cask with:

```bash
brew install --cask whit3rabbit/tap/forge-guardrails-proxy
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

Run the proxy server as a reliability layer between your client agent and the LLM backend.

- **Run the proxy** on default port `8081` pointing to your local LLM backend:
  ```bash
  forge-guardrails-proxy --backend-url http://localhost:8080
  ```
- **Run on a different port** by specifying the `--port` flag:
  ```bash
  forge-guardrails-proxy --backend-url http://localhost:8080 --port 9090
  ```
- **Run with tool-output compression** using the `--tool-output-compression` flag to automatically compress prior tool results:
  ```bash
  forge-guardrails-proxy --backend-url http://localhost:8080 --tool-output-compression standard
  ```
- **Run with classifier/validator** using the `--classify` flag to score model tool calls against the verifier ONNX model:
  ```bash
  forge-guardrails-proxy --backend-url http://localhost:8080 --classify
  ```

### Client & Backend Integration

Configure your agent clients or model backends to route through the proxy.

#### Claude Code Env Variables
Set the Anthropic base URL environment variable to point to the proxy:
```bash
export ANTHROPIC_BASE_URL="http://localhost:8081"
export ANTHROPIC_API_KEY="dummy"  # Or your actual key if proxying to Anthropic
```

#### Backends (llama-server / LM Studio / Ollama)
Point the proxy's `--backend-url` to your running model server:
- **llama-server** (default port 8080): `--backend-url http://localhost:8080`
- **LM Studio** (default port 1234): `--backend-url http://localhost:1234`
- **Ollama** (default port 11434): `--backend-url http://localhost:11434`

### Library Usage (Rust)

For direct library integration, use the `WorkflowRunner` in your Rust code:

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

# Optional ONNX tool-call classifier shortcut.
cargo run --features classifier --bin forge-guardrails-proxy -- \
  --backend-url http://localhost:8080 \
  --classify \
  --port 8081

# Prefetch the quantized classifier artifact and print its location.
cargo run --features classifier --bin forge-guardrails-proxy -- --classify-download

# Convenience launcher for the recommended Ministral GGUF.
scripts/start_llamaserver_proxy.sh \
  /path/to/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf
```

The launcher uses managed `llamaserver` mode, verifies the GGUF path,
requires `llama-server` on `PATH`, checks that the proxy and backend ports are
free, and reuses an existing proxy binary from `PATH`, `CARGO_TARGET_DIR`, or
`target/`. If no binary is found, it falls back to `cargo build`.
Without an explicit path it searches for
`mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf`; set `FORGE_MODELS_DIR` or
`MODELS_DIR` to point at your model directory. Defaults are proxy port `8081`
and managed backend port `8080`; override them with `FORGE_PROXY_PORT` and
`FORGE_BACKEND_PORT`. Press Ctrl+C to stop the proxy and its managed
`llama-server` backend.

The `--classify` shortcut is opt-in and requires building with
`--features classifier`. It downloads the pinned quantized tool-call ONNX
classifier if needed, stores it outside `target/`, enables advisory mode, and
prints the artifact directory during startup. By default it uses
`FORGE_CLASSIFIER_CACHE_DIR`, then `XDG_CACHE_HOME`, then
`$HOME/.cache/forge-guardrails/classifiers`. Use `--classifier-dir` to provide
an explicit artifact directory.

Tool-output compression is also opt-in. It mutates only prior tool-result
content before forwarding a request upstream; tool calls, tool IDs, arguments,
and final responses are left unchanged. Start conservatively with `safe` or
`standard`; dictionary compression requires explicit `aggressive` mode.

```bash
cargo run --bin forge-guardrails-proxy -- \
  --backend-url http://localhost:8080 \
  --tool-output-compression standard \
  --port 8081

cargo run --bin forge-guardrails-proxy -- \
  --backend-url http://localhost:8080 \
  --tool-output-compression aggressive \
  --tool-output-compression-method auto \
  --port 8081
```

See [Tool Output Compression](docs/COMPRESSION.md) for modes, request-level
`_forge` overrides, and method details.

Then configure OpenAI-compatible clients to use `http://localhost:8081/v1` as the API base URL. Anthropic-compatible clients should use `http://localhost:8081`; the proxy accepts Anthropic Messages API requests at `POST /v1/messages`. Requests without `model` are rejected unless you explicitly configured a fallback with `--model`, `FORGE_MODEL`, or `SMALL_MODEL`.

**Backend compatibility:**

- **Managed mode** spins up the backend for you. Supported backends: `llamaserver`, `llamafile`, `ollama` (use `--gguf` for GGUF-based backends, or `--model` for Ollama).
- **External mode** is backend-agnostic — forge talks `POST /v1/chat/completions` to whatever you point `--backend-url` at, as long as it speaks the OpenAI schema. Tool calls must come back in OpenAI `tool_calls` format or in one of forge's rescue-parsed formats (Mistral `[TOOL_CALLS]`, Qwen `<tool_call>` XML, fenced JSON).
- **Anthropic-compatible inbound** uses `anyllm_translate` for Anthropic/OpenAI conversion by default. With `--backend-protocol anthropic`, external mode sends Anthropic Messages requests to an Anthropic-shape downstream. Path 1 preserves block-level `cache_control` only on clean calls; retries, compaction, and context warnings rebuild the request and drop block metadata. Path 2 drops Anthropic-only block metadata at the OpenAI boundary.
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
- **Context window management** — proxy mode does not choose or maintain the client's rolling message window. Opt-in tool-output compression can rewrite prior tool-result content, and dedup can use an explicit `session_id`, but the client still owns conversation memory.
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
| `--classify` | off | Enable the quantized tool-call ONNX classifier in advisory mode |
| `--classify-download` | off | Download the quantized tool-call ONNX classifier and exit |
| `--tool-output-compression {disabled,safe,standard,aggressive}` | `disabled` | Compress prior tool-result content before upstream forwarding |
| `--tool-output-compression-method {lzw,repair,auto}` | `lzw` | Aggressive dictionary method |
| `--no-rescue` | rescue on | Disable rescue parsing |
| `--budget-mode {backend,manual,forge-full,forge-fast}` | `backend` | Context budget source |
| `--budget-tokens N` | — | Manual token budget |
| `--serialize` / `--no-serialize` | auto | Force request serialization |
| `--extra-flags -- FLAG VALUE ...` | — | Pass additional flags to the managed backend |

### Useful environment variables (Docker / env-routed mode)

| Variable | Default | Purpose |
|---|---|---|
| `FORGE_HOST` | `0.0.0.0` | Bind address |
| `FORGE_PORT` / `PORT` / `LISTEN_PORT` | `8081` | Forge proxy listen port |
| `FORGE_MODEL` / `SMALL_MODEL` | `(none)` | Optional fallback model when a request omits `model` |
| `FORGE_CONTEXT_TOKENS` | `128000` | Token budget |
| `FORGE_MAX_RETRIES` | `3` | Retry budget per validation failure |
| `FORGE_RESCUE_ENABLED` | `true` | Enable rescue parsing |
| `FORGE_SERIALIZE_REQUESTS` | `false` | Force request serialization |
| `FORGE_SENTRY_ENABLED` | `false` | Opt in to Sentry crash and aggregate guardrail telemetry |
| `FORGE_CLASSIFIER_CACHE_DIR` | platform cache | User-facing classifier download cache root |
| `FORGE_CLASSIFIER_DIR` | — | Local ONNX tool-call classifier artifact directory |
| `FORGE_CLASSIFIER_MODE` | `shadow` | `disabled`, `shadow`, `advisory`, or `enforce` |
| `FORGE_CLASSIFIER_MODEL` | `quantized` | `quantized` or `full` classifier ONNX file |
| `FORGE_TOOL_OUTPUT_COMPRESSION` | `disabled` | `disabled`, `safe`, `standard`, or `aggressive` |
| `FORGE_TOOL_OUTPUT_COMPRESSION_METHOD` | `lzw` | `lzw`, `repair`, or `auto`; used only by aggressive mode |
| `FORGE_START_SIDECAR` | Docker: auto | Start the internal anyllm sidecar in Docker |
| `ANYLLM_LISTEN_PORT` | Docker: `3000` | Internal anyllm sidecar port; do not publish it |
| `FORGE_SIDECAR_API_KEY` / `PROXY_API_KEYS` | generated | Shared Forge-to-sidecar key in Docker |
| `BACKEND` | `openai` | anyllm provider id or first-party backend |
| `OPENAI_BASE_URL` | — | Route to a local OpenAI-compatible backend |
| `OPENAI_API_KEY` | — | API key forwarded to the upstream |

Existing anyllm env and config are still honored, including provider API keys, `PROXY_CONFIG`, `BIG_MODEL`, `SMALL_MODEL`, and LiteLLM aliases such as `LITELLM_CONFIG`.

`FORGE_SENTRY_ENABLED=true` enables Sentry for the proxy binary only. Sentry
events are limited to scrubbed crashes and aggregate guardrail signals such as
classifier labels, retry exhaustion reasons, counts, and tool names. Prompts,
messages, headers, request bodies, tool arguments, tool outputs, and final
responses are not sent. Use `FORGE_TRAINING_CAPTURE_LOG` or
`FORGE_CLASSIFIER_LOG` for private local JSONL training/eval examples.

### Docker

You can run the Forge proxy as a Docker container. The image starts Forge plus an internal anyllm sidecar by default, and exposes only the Forge proxy port (`8081`) to clients. The sidecar is an upstream hop from Forge to anyllm; do not publish the sidecar port.

Build the image locally:

```bash
docker build -t forge-guardrails:local .
```

The default `Dockerfile` builds the normal proxy image without ONNX classifier
support. Use `Dockerfile.classifier` when you want the quantized tool-call
classifier artifact downloaded into the image and loaded on proxy startup:

```bash
docker build -f Dockerfile.classifier -t forge-guardrails:classifier .
```

The classifier image sets:

```text
FORGE_CLASSIFIER_DIR=/opt/forge/classifiers/tool-call/onnx
FORGE_CLASSIFIER_MODE=advisory
FORGE_CLASSIFIER_MODEL=quantized
```

Set `FORGE_CLASSIFIER_MODE=disabled` at runtime to use the classifier image as a
plain proxy. This image bundles only the ONNX classifier artifact; it does not
bundle a GGUF or provider LLM.

After publishing, replace `forge-guardrails:local` in these examples with `followthewhit3rabbit/forge-guardrails:latest`.

Run with OpenAI through the internal anyllm sidecar:

```bash
docker run --rm -p 8081:8081 \
  -e OPENAI_API_KEY=sk-... \
  -e FORGE_MODEL=gpt-4o-mini \
  forge-guardrails:local
```

Run the classifier-ready image the same way:

```bash
docker run --rm -p 8081:8081 \
  -e OPENAI_API_KEY=sk-... \
  -e FORGE_MODEL=gpt-4o-mini \
  forge-guardrails:classifier
```

The entrypoint generates a private Forge-to-sidecar key unless you set `FORGE_SIDECAR_API_KEY` or `PROXY_API_KEYS`. It starts the sidecar with the upstream provider environment, then starts Forge with `OPENAI_API_KEY` set to the sidecar key and `--backend-url http://127.0.0.1:3000`.

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
scripts/publish_docker.sh
```

Override `VERSION`, `IMAGE`, `PLATFORMS`, or `BUILDER` when publishing a different tag or registry.

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
# 10-run release benchmark without classifier, with resource baseline enabled.
make eval-release

# 10-run release benchmark with classifier, with resource baseline enabled.
make eval-release-classify

# Fast smoke variants, also with resource baseline enabled.
make eval-smoke
make eval-smoke-classify

# Managed local smoke without the classifier
scripts/run_local_eval.sh --suite smoke --runs 1

# Managed local smoke with the user-cache classifier shortcut.
# Downloads or validates the quantized tool-call artifact before the proxy starts.
scripts/run_local_eval.sh --suite smoke --runs 1 \
  --classify \
  --classifier-mode shadow

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

Common Makefile overrides:

```bash
make eval-release OUTPUT_DIR=target/local-eval/release-baseline
make eval-release-classify CLASSIFIER_MODE=enforce
make eval-release RUNS=3 EVAL_ARGS="--skip-published-compare"
```

The Rust smoke runner supports `basic_2step`, `sequential_3step`, and `error_recovery` scenarios and emits JSONL for quick CI/smoke checks.

## Project Structure

```
src/
  lib.rs                     Public API re-exports
  error.rs                   ForgeError hierarchy
  server.rs                  setup_backend(), ServerManager, BudgetMode
  classifier_download.rs     Classifier artifact download logic (--features classifier)
  tool_output.rs             Tool-output compression pipeline (safe / standard / aggressive)
  tool_policy.rs             Per-request allowed/blocked tool sets and prerequisite policy
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
    scoring.rs               ScoringPipeline, ScoringExecutor — async classifier dispatch
    scoring_context.rs       ScoringContext — serialized input for ONNX scorer
    classifier_artifact.rs   Artifact loader, manifest validation, threshold policy
    onnx_scorer.rs           OnnxToolCallScorer, OnnxFinalResponseScorer (--features classifier)
    history.rs               Events timeline for validation results and violations
    policy.rs                Allowed/blocked tool policy based on sequence prerequisites
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
    nudges.rs                Retry, step-enforcement, and semantic classifier nudge templates
    parse_strategies.rs      Rescue parsing: Mistral, Qwen, fenced JSON
  tools/
    respond.rs               Synthetic respond tool (respond_tool(), respond_spec())
  proxy/
    handler.rs               Request handler — bridge between HTTP and run_inference
    proxy.rs                 OpenAI messages ↔ forge Messages conversion, SSE helpers
    server.rs                HTTPServer — axum HTTP/SSE server
  bin/
    forge-guardrails-proxy.rs  CLI proxy entry point
    download-classifier.rs     Standalone artifact downloader for eval / training paths
    forge-eval/                Native Rust eval smoke runner
model/
  README.md                  Artifact repository index and download commands
  MODEL.md                   Full model card: training config, metrics, labels, thresholds
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
  COMPRESSION.md             Tool-output compression modes and request overrides
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
    // Scoring pipeline (async classifier dispatch)
    ScoringPipeline, ScoringExecutor, ScoringContext,
    // Step tracking
    StepTracker, SlotWorker,
    // Prompts and nudges (including semantic classifier nudges)
    retry_nudge, step_nudge, prerequisite_nudge, unknown_tool_nudge,
    classifier_nudge, rescue_tool_call, build_tool_prompt,
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
- `retry_nudge`, `step_nudge`, `prerequisite_nudge`, `unknown_tool_nudge`, `classifier_nudge`
- `ScoringPipeline` / `ScoringExecutor` — async classifier dispatch for shadow, advisory, and enforce modes
- `ScoringContext` — serializes workflow state into the canonical `toolcall-verifier-input/v1` format for the ONNX scorer

The ONNX tool-call verifier and final-response verifier are built with `--features classifier`. Both start in `shadow` mode and are promoted only after eval replay proves safety. See [model/README.md](model/README.md) for artifact contracts, labels, thresholds, and promotion criteria.

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

Both clients expose provider observability through `LLMClient::last_call_info()`. Cost estimates, routing metadata, cache state, and rate-limit details come from anyllm runtime or sidecar metadata; Forge does not maintain separate pricing logic.

## Testing Scope

- 487+ passing tests across 16 test files
- Deterministic parity suite against `tests/parity/fixtures/python_golden.json`
- Classifier tests (`--test classifier_tests`) cover artifact loading, serializer parity, ONNX scorer output, and scoring pipeline routing
- 0 contamination incidents in the clean-room run

Keep tests deterministic where possible. Backend integration tests use mock servers (via `mockito`) unless they intentionally qualify a live backend. Classifier tests require `--features classifier` and the pinned ONNX artifact; they are gated separately from the core test suite.

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
