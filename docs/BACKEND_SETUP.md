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
- raw `extra_flags`

Reasoning-tagged models on recent llama.cpp builds may need:

```bash
--reasoning-budget 0
```

The Rust smoke runner accepts `--reasoning-budget` so eval JSONL records the
intended setting. When starting llama-server outside the runner, pass the same
flag to the server process.

## llamafile

Treat llamafile separately from llama-server. llamafile does not provide native
function calling reliably, so Rust defaults smoke runs to prompt mode:

```text
llamafile default: prompt
llamaserver default: native
```

Both paths use `LlamafileClient`, but the mode matters for parity.

## Ollama

Ollama uses `/api/chat`, not an OpenAI-compatible endpoint. Use
`OllamaClient` for parity, not a generic OpenAI client.

Rules:

- `setup_backend("ollama")` requires `--model`.
- `setup_backend("ollama")` rejects GGUF/file paths.
- Resolved budgets should be mirrored into Ollama as `num_ctx` when the client
  is used for evals.

## Anthropic

Anthropic is a cloud API baseline. It has no local server smoke test; auth and
network failures surface on first inference. Keep it out of the default local
parity gate.

## OpenAI-Compatible Proxy

Use the proxy path to verify client-visible behavior:

```bash
python scripts/eval_openai_proxy.py --base-url http://127.0.0.1:8081/v1 --model test-model --scenario basic_2step
cargo run --bin forge-eval -- --backend openai-proxy --base-url http://127.0.0.1:8081/v1 --model test-model --scenario basic_2step
```

This path is the right gate for `respond` stripping, retry exhaustion text,
empty text on unexpected no-tools tool calls, and streaming final chunk shape.
