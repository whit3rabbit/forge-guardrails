# Rust Eval Guide

Rust eval support is intentionally small. Use it for smoke tests and proxy
regression checks. Use the upstream Python eval harness for large batch runs,
dashboards, reports, and model leaderboard comparisons.

## Python Oracle Against Rust Proxy

Start the Rust proxy separately, then run:

```bash
python scripts/eval_openai_proxy.py \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 10 \
  --stream \
  --scenario basic_2step sequential_3step error_recovery \
  --output eval_results_rust_proxy.jsonl
```

The wrapper imports scenarios from the upstream `forge/` submodule without
editing it. It emits the flat JSONL fields needed by the Python report path
plus Rust-parity fields such as `impl`, `success`, `tool_sequence`,
`tool_args`, `final_text`, and `raw_response_on_failure`. Streaming token
usage is recorded only when the proxy emits OpenAI `usage` chunks.

## Native Rust Smoke Runner

```bash
cargo run --bin forge-eval -- \
  --backend openai-proxy \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 3 \
  --num-ctx 8192 \
  --scenario basic_2step \
  --stream
```

Supported backends:

- `openai-proxy`
- `ollama`
- `llamaserver`
- `llamafile`
- `anthropic`

Supported initial scenarios:

- `basic_2step`
- `sequential_3step`
- `error_recovery`

The runner writes JSONL to stdout unless `--output` is provided. It does not
generate dashboards, manage local server processes, or clone the full Python
batch/reporting platform.

`--num-ctx` defaults to `8192` and controls the Rust `ContextManager` budget.
For `--backend ollama`, the same value is sent as Ollama `num_ctx` on each
request. `--reasoning-budget` is recorded in JSONL only; when using an external
llama-server, start that server with the same flag yourself.

## Parity Fields

Rows should include:

```text
impl, model, backend, mode, ablation, tool_choice, scenario, run, stream,
completeness, success, accuracy, iterations, ideal_iterations, wasted_calls,
elapsed_s, error_type, error_message, budget_tokens, compaction_events,
retry_nudges, step_nudges, tool_errors, reasoning_msgs, tool_sequence,
tool_args, final_text, raw_response_on_failure
```

Rows may also include `stream_retries`, `input_tokens`, and `output_tokens`
when the underlying eval result reports them.

Do not require exact parity for latency, token counts, generated IDs, provider
metadata, or JSON key ordering outside schema/prompt assertions.

## Verification

Run focused checks first:

```bash
cargo test --test parity_tests
cargo test proxy::handler
cargo test server::tests
cargo test --bin forge-eval
python scripts/eval_openai_proxy.py --help
```

Run live evals manually against real backends. CI should stay deterministic
unless a job is explicitly marked as live-backend integration.
