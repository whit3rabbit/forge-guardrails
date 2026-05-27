# Backend Setup Contract

This Rust port follows the upstream Python backend contract for parity runs.

## llama-server

Native function calling requires Jinja tool templates:

```bash
llama-server -m path/to/model.gguf --jinja -ngl 999 --port 8080
```

Rust managed startup preserves:

- `--jinja` for `llamaserver` native mode
- `-ngl 999`
- `-c <N>` for context overrides
- `--cache-type-k`
- `--cache-type-v`
- `--parallel`
- `--kv-unified`
- allowlisted reasoning `extra_flags`

Managed startup owns model path, host, port, GPU layers, context size, cache
flags, slots, and KV-unified settings. Do not pass those through
`--extra-flags`; they are rejected before any backend process is stopped or
spawned. Use first-class proxy CLI flags instead:

```bash
forge-guardrails-proxy \
  --backend llamaserver \
  --gguf path/to/model.gguf \
  --budget-mode manual \
  --budget-tokens 8192 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --slots 1 \
  --kv-unified
```

Reasoning-tagged models on recent llama.cpp builds may need:

```bash
--reasoning-budget 0
```

Managed startup accepts only `--reasoning-budget` and `--reasoning-format`
through `--extra-flags`; both require values and may also be provided as
first-class proxy flags. Model/path/host/port/context/server/security flags are
not accepted in `--extra-flags`.

The Rust smoke runner accepts `--reasoning-budget` so eval JSONL records the
intended setting, but it does not start or reconfigure a local server process.
When starting llama-server outside the runner, pass the same flag to that
server process.

## llamafile

Treat llamafile separately from llama-server. llamafile does not provide native
function calling reliably, so Rust defaults smoke runs to prompt mode:

```text
llamafile default: prompt
llamaserver default: native
```

Both paths use `LlamafileClient`, but the mode matters for parity.

Managed llamafile startup requires an explicit, trusted runtime binary path.
Forge does not discover or execute binaries from the GGUF/model directory.

```bash
forge-guardrails-proxy \
  --backend llamafile \
  --gguf path/to/model.gguf \
  --llamafile-runtime /opt/forge/bin/llamafile
```

The runtime path must be absolute, resolve to a regular file, and be
executable on Unix platforms.

## Ollama

Ollama uses `/api/chat`, not an OpenAI-compatible endpoint. Use
`OllamaClient` for parity, not a generic OpenAI client.

Rules:

- `setup_backend("ollama")` requires `--model`.
- `setup_backend("ollama")` rejects GGUF/file paths.
- `setup_backend("llamafile")` requires `gguf_path` and an explicit
  `llamafile_runtime`.
- Resolved budgets should be mirrored into Ollama as `num_ctx` when the client
  is used for evals.
- `forge-eval --backend ollama --num-ctx <N>` mirrors `<N>` into
  `OllamaClient::set_num_ctx(Some(N))` and uses the same value for the local
  context budget.

## Anthropic

Anthropic is a cloud API baseline. It has no local server smoke test; auth and
network failures surface on first inference. Keep it out of the default local
parity gate.

The proxy accepts Anthropic Messages requests at `POST /v1/messages`. The
default external and managed paths translate Anthropic inbound requests to an
OpenAI-compatible backend with `anyllm_translate`, then translate responses
back to Anthropic shape. Use `--backend-protocol anthropic` only in external
mode when the downstream already speaks Anthropic Messages.

`cache_control` is block-level Anthropic metadata. It is preserved on clean
Path 1 calls to an Anthropic-shape downstream. If Forge retries, compacts, or
injects a context warning, the retry request is rebuilt from Forge's internal
messages and block-level metadata is dropped. Path 2 always drops
Anthropic-only block metadata at the OpenAI protocol boundary.

## OpenAI-Compatible Proxy

Use the proxy path to verify client-visible behavior:

```bash
python scripts/eval_openai_proxy.py --base-url http://127.0.0.1:8081/v1 --model test-model --scenario basic_2step
cargo run --bin forge-eval -- --backend openai-proxy --base-url http://127.0.0.1:8081/v1 --model test-model --scenario basic_2step
```

This path is the right gate for `respond` stripping, retry exhaustion text,
empty text on unexpected no-tools tool calls, and streaming final chunk shape.
