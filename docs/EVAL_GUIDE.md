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
editing it. It emits JSONL fields compatible with the Python reports plus
Rust-parity fields such as `impl`, `success`, `tool_sequence`, `tool_args`,
`final_text`, and `raw_response_on_failure`.

## Native Rust Smoke Runner

```bash
cargo run --bin forge-eval -- \
  --backend openai-proxy \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 3 \
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
generate dashboards.

## Parity Fields

Rows should include:

```text
impl, model, backend, mode, ablation, tool_choice, scenario, run, stream,
completeness, success, accuracy, iterations, elapsed_s, error_type,
error_message, retry_nudges, step_nudges, tool_errors, reasoning_msgs,
tool_sequence, tool_args, final_text, raw_response_on_failure
```

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
