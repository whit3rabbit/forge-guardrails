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
editing it. It runs a proxy-aware OpenAI tool loop instead of wrapping the
proxy in the upstream `WorkflowRunner`: returned `assistant.tool_calls` are
executed locally, matching `tool` results are appended with the original tool
call IDs, and final proxy-visible text is validated against the scenario
terminal schema where possible. The wrapper does not send scenario-owned
`respond(...)` tools to the proxy because `respond` is proxy-reserved.
It does not enforce scenario `required_steps` or prerequisites itself; if the
proxy returns premature text, the row completes with whatever accuracy the
scenario validator assigns. Rows include `proxy_missing_required_steps`,
`proxy_required_steps_satisfied`, and `proxy_failure_classification` so direct
runner step-enforcement deltas are not mislabeled as harness or protocol
failures.

Rows should label the managed backend and proxy mode separately. For a local
managed llama-server run, use `backend=llamaserver`, `mode=proxy`,
`eval_target_backend=openai-proxy`, and inspect `proxy_terminal_source` to see
whether the terminal answer came from proxy-visible text or a tool call. The
wrapper applies recommended sampling defaults by model unless
`--no-recommended-sampling` is passed.

It emits the flat JSONL fields needed by the Python report path plus
Rust-parity fields such as `impl`, `success`, `tool_sequence`, `tool_args`,
`final_text`, `proxy_terminal_source`, `proxy_missing_required_steps`,
`proxy_required_steps_satisfied`, `proxy_failure_classification`, and
`raw_response_on_failure`.
Streaming token usage is recorded only when the proxy emits OpenAI `usage`
chunks.

`scripts/summarize_proxy_eval.py <jsonl>` prints an outer summary that separates
completeness failures from completed-but-inaccurate rows. This avoids relying on
the upstream report's completeness-only "Weak" line when diagnosing proxy eval
results.

Known classifications for the local Ministral proxy comparison:

- `inconsistent_api_recovery_stateful`: proxy contract mismatch from skipped
  required setup; direct `WorkflowRunner` step enforcement nudges this path.
- `argument_transformation*`: model/scenario accuracy weakness in local runs.
- `grounded_synthesis*`: model/scenario weakness; published direct LS/N is also
  0 for these columns.
- `data_gap_recovery_extended*`: strict scorer/string-literal misses.

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
impl, model, backend, mode, eval_target_backend, ablation, tool_choice,
scenario, run, stream, completeness, success, accuracy, iterations,
ideal_iterations, wasted_calls, elapsed_s, error_type, error_message,
budget_tokens, compaction_events, retry_nudges, step_nudges, tool_errors,
reasoning_msgs, tool_sequence, tool_args, final_text, proxy_terminal_source,
proxy_missing_required_steps, proxy_required_steps_satisfied,
proxy_failure_classification, raw_response_on_failure
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

`scripts/run_local_eval.sh --suite release` writes proxy rows and runs the
published comparison helper. The helper skips direct published LS/N comparison
for proxy rows by default; pass `--force-published-compare` only when you
explicitly want that non-equivalent comparison.

Run live evals manually against real backends. CI should stay deterministic
unless a job is explicitly marked as live-backend integration.
