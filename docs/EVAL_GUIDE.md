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
call IDs and `_forge.tool_status` (`ok` or `error`), and final proxy-visible
text is validated against the scenario terminal schema where possible. The
wrapper does not send scenario-owned
`respond(...)` tools to the proxy because `respond` is proxy-reserved.
For workflow parity checks it sends a private `_forge` extension object with
`required_steps` and `terminal_tools` (`respond` plus any scenario terminal
tools). The proxy strips `_forge` before forwarding to the backend, uses it only
for step nudging/finalization, and still returns non-terminal client-owned tool
calls for the wrapper to execute. Rows include
`proxy_missing_required_steps`, `proxy_required_steps_satisfied`, and
`proxy_failure_classification` so required-step contract misses are not
mislabeled as successful accuracy results or generic protocol failures.

Rows should label the managed backend and proxy mode separately. For a local
managed llama-server run, use `backend=llamaserver`, `mode=proxy`,
`eval_target_backend=openai-proxy`, and inspect `proxy_terminal_source` to see
whether the terminal answer came from proxy-visible text or a tool call. The
wrapper applies recommended sampling defaults by model unless
`--no-recommended-sampling` is passed.

It emits the flat JSONL fields needed by the Python report path plus
Rust-parity fields such as `impl`, `success`, `tool_sequence`, `tool_args`,
`final_text`, `proxy_terminal_source`, `proxy_missing_required_steps`,
`proxy_required_steps_satisfied`, `missing_required_steps`,
`required_step_mismatch`, `proxy_failure_classification`, and
`raw_response_on_failure`. The `success` field is contract-gated: a completed
row with acceptable answer accuracy is still unsuccessful when
`required_step_mismatch` is true.
Streaming token usage is recorded only when the proxy emits OpenAI `usage`
chunks.

`scripts/summarize_proxy_eval.py <jsonl>` prints an outer summary that separates
completeness failures from completed-but-inaccurate rows. This avoids relying on
the upstream report's completeness-only "Weak" line when diagnosing proxy eval
results.

Known classifications for the local Ministral proxy comparison:

- `terminal_redacted`: completed tool flow, but the model submitted exact
  terminal content `[REDACTED]`; this remains an accuracy failure.
- `inconsistent_api_recovery_stateful`: if this appears as a failed contract
  mismatch, the proxy workflow extension path regressed.
- `argument_transformation*`: model/scenario accuracy weakness in local runs.
- `grounded_synthesis*`: model/scenario weakness. Compare it only against the
  selected published baseline mode; the direct `LS/N` and prompt `LS/P` rows
  have different behavior for these columns.
- `data_gap_recovery_extended*`: strict scorer/string-literal misses.

When investigating a local score drop, first compare the current JSONL against
the previous local JSONL by scenario and `proxy_failure_classification`. Treat
small run-to-run deltas in completed-but-inaccurate rows as stochastic model
accuracy changes unless the current run shows incomplete rows, missing required
steps, or `proxy_contract_mismatch` classifications.

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
- `inconsistent_api_recovery_stateful`

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
proxy_failure_classification, raw_response_on_failure, missing_required_steps,
required_step_mismatch
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
published comparison helper against the published LS/P row by default. If you
select `--published-mode LS/N`, the helper skips direct comparison for proxy
rows unless `--force-published-compare` is passed.

This default matters when reading warnings: an `LS/P` comparison is a
proxy/prompt-mode leaderboard comparison, while `LS/N` is a direct native-tool
baseline. Do not use the skipped `LS/N` behavior to explain an `LS/P` warning.

Run live evals manually against real backends. CI should stay deterministic
unless a job is explicitly marked as live-backend integration.

## ONNX Classifier Mode Runs

Download the local classifier artifact once:

```bash
cargo run --features classifier --bin download-classifier -- \
  --artifact tool-call \
  --classifier-model quantized
```

The downloader defaults to the pinned Hugging Face revision used by Rust's
`DEFAULT_CLASSIFIER_REVISION`, currently
`1c87eceea15ec42f755deafb0ac4166bd0bd51b0` for
`cowWhySo/toolcall-verifier-classifier-production`. It writes the runnable ONNX
artifact under `target/classifier-artifacts/onnx` and also downloads published
schema/report sidecars such as `input_schema_v1.json`,
`input_schema_v2.json`, and `serializer_fixture_v2.json` when they are present
in the model repo.

Then run the same local release eval with the classifier loaded by the proxy:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx
```

The shortcut below downloads the quantized artifact first when needed:

```bash
scripts/run_local_eval.sh --suite release --runs 10 --download-classifier
```

For a side-by-side comparison, keep output directories explicit and include
enforce mode when evaluating whether ONNX thresholds can safely change
behavior:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --output-dir target/local-eval/release-baseline

scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx \
  --output-dir target/local-eval/release-onnx-shadow

scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx \
  --classifier-mode enforce \
  --output-dir target/local-eval/release-onnx-enforce
```

The classifier default is `shadow`, but the supported modes are `disabled`,
`shadow`, `advisory`, and `enforce`. In shadow mode it should not change
completeness, success, retries, or nudges. In enforce mode it can retry/block
only labels whose artifact threshold is met; labels with thresholds above `1.0`
remain telemetry-only, and deterministic guardrails remain authoritative for
schema/protocol invalidity. `scripts/run_local_eval.sh` writes proxy classifier
telemetry to `proxy_classifier_<budget>.jsonl` whenever the classifier is
enabled, using the `FORGE_CLASSIFIER_LOG` environment variable. Use that JSONL
and Rust smoke JSONL classifier fields to inspect classifier scores; use the
Python oracle JSONL and reports to confirm behavior changes.

ONNX scorer throughput knobs are opt-in. Keep defaults for parity baselines.
For proxy throughput comparisons, set `FORGE_CLASSIFIER_SESSION_POOL_SIZE` and
`FORGE_CLASSIFIER_INTRA_THREADS`; final-response verifier variants use
`FORGE_FINAL_RESPONSE_CLASSIFIER_SESSION_POOL_SIZE` and
`FORGE_FINAL_RESPONSE_CLASSIFIER_INTRA_THREADS`. Session pools are bounded to
`1..=4`; each extra session loads another model copy, so measure RSS alongside
latency.

The Colab production notebook now exports the tool-call verifier and the
separate final-response verifier by default. Keep both in `shadow` for first
replay. Download the final-response artifact with:

```bash
cargo run --features classifier --bin download-classifier -- \
  --artifact final-response \
  --classifier-model quantized
```

That command writes the published files under
`target/final-response-classifier-artifacts/onnx` by default. The pinned
final-response revision is
`69d1a75d0fad25e3cf1333c7ea9c7cf0584614a4` for
`cowWhySo/final-response-verifier-classifier-production`.

Run final-response variants with the matching final-response classifier flags:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx \
  --classifier-mode shadow \
  --final-response-classifier-dir target/final-response-classifier-artifacts/onnx \
  --final-response-classifier-mode shadow
```

Final-response variants are required before expecting improvement on terminal
synthesis failures such as grounded-synthesis or data-gap recovery. They should
not be used to explain tool-call ordering or argument failures; inspect the
tool-call classifier rows for those.
