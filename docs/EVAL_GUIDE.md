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

## Proxy Resource Baseline

`scripts/run_local_eval.sh` can sample local CPU and resident memory while eval
requests are running:

```bash
scripts/run_local_eval.sh --suite smoke --runs 1 --resource-baseline
```

Use `--resource-interval SECONDS` to adjust the sample interval. The default is
`1.0`.

The launcher starts the sampler only after `/health` succeeds and stops it
before eval reports are generated. This excludes proxy and model startup CPU,
but includes loaded memory during the measured eval window. Output files are
written beside the eval JSONL:

```text
resource_samples_<label>.jsonl
resource_summary_<label>.json
resource_baseline_report.txt
```

The report treats the Rust proxy process as the headline. Managed
`llama-server` resource use is reported separately as backend context, followed
by the combined wrapper/proxy/backend process tree. CPU is the sampled `ps`
`%CPU` value and can exceed `100` for multithreaded processes. RSS is process
resident memory, not total system pressure, allocator high-water marks, VRAM, or
billing-grade usage. The v1 report is a baseline artifact only; it does not set
or enforce regression thresholds.

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

## Tool-Output Compression Eval

Use `eval-release-compression` to compare disabled compression against one
selected compression mode on the same release suite. The standard comparison is
disabled baseline vs `standard` mode:

```bash
make eval-release-compression \
  TOOL_OUTPUT_COMPRESSION=standard \
  TOOL_OUTPUT_COMPRESSION_METHOD=lzw \
  COMPRESSION_MIN_INPUT_TOKEN_SAVINGS=1
```

Other compression checks use the same target with a different mode:

```bash
make eval-release-compression TOOL_OUTPUT_COMPRESSION=safe
make eval-release-compression TOOL_OUTPUT_COMPRESSION=aggressive TOOL_OUTPUT_COMPRESSION_METHOD=auto
```

The target writes separate disabled and compressed runs under
`target/local-eval/compression-<mode>-<timestamp>/` unless `OUTPUT_DIR` is set.
It then runs `scripts/compare_compression_eval.py` against the two
`python_oracle.jsonl` files and writes `compression_report.txt`.
When compression is enabled, the launcher also writes
`proxy_tool_output_compression_<budget>.jsonl` with per-tool-result strategy
names, token estimates, byte and line counts, bounded request debug metadata,
argument/output fingerprints, and redaction/capping/dedup flags. These events
exclude raw tool output.

Controls:

- `TOOL_OUTPUT_COMPRESSION`: `safe`, `standard`, or `aggressive`; `disabled`
  is rejected because the target needs a compressed side.
- `TOOL_OUTPUT_COMPRESSION_METHOD`: `lzw`, `repair`, or `auto`; used only by
  aggressive dictionary compression.
- `COMPRESSION_MIN_INPUT_TOKEN_SAVINGS`: minimum aggregate prompt-token savings
  required by the comparator. When paired oracle rows do not contain
  `input_tokens`, the comparator falls back to compression telemetry estimated
  before/after token counts and labels that line as a telemetry estimate.

The available compression techniques are:

- `safe`: secret redaction, ANSI stripping, binary suppression, and oversized
  output capping.
- `standard`: `safe` plus JSON minification, table-whitespace cleanup,
  tool-family filters, repeated-line folding, and whitespace normalization.
- `aggressive`: `standard` plus dynamic log-noise normalization, scalar
  JSON-array table conversion, and dictionary compression.
- Aggressive dictionary methods: `lzw`, `repair`, or `auto`.

When request-level compression telemetry is available, the comparator fails on
success, completeness, or accuracy regressions only for rows touched by a
compression event. Behavior changes in rows without compression telemetry are
reported as warnings because live-backend variance can affect scenarios whose
tool outputs were not compressed. Without request-level telemetry, behavior
regression checks fall back to all paired rows. Pass
`--allow-behavior-regression` through direct script use only for exploratory
analysis. Treat this as a live-backend check, not a deterministic CI gate.

## ONNX Classifier Mode Runs

This section separates the user-cache classifier shortcut from the
`target/` artifact directories used for repeatable eval/training runs. For
normal proxy use, `forge-guardrails-proxy --classify` is the user-facing
shortcut and downloads the quantized tool-call artifact into the user cache.
The eval launcher also supports `--classify` for benchmark runs that should
use that user-cache shortcut and auto-download or validate the model before
the proxy starts.

Run without the classifier:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --output-dir target/local-eval/release-baseline
```

Run with the user-cache classifier shortcut:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode shadow \
  --output-dir target/local-eval/release-onnx-shadow
```

`--classify` uses `forge-guardrails-proxy --classify-download` under the hood,
prints the resolved `classifier_dir=...`, records it in
`local_eval_metadata.txt`, and then passes the artifact path to the proxy and
Rust smoke runner. Its default mode is `advisory`, matching proxy shortcut
behavior; pass `--classifier-mode shadow` for first replay baselines.

Download the local classifier artifact once:

```bash
cargo run --features classifier --bin download-classifier -- \
  --artifact tool-call \
  --classifier-model quantized
```

The downloader defaults to the pinned Hugging Face revision used by Rust's
`DEFAULT_CLASSIFIER_REVISION`, currently
`b8e292b4de5725250bd1698eb5c795ffcb1a4cde` for
`cowWhySo/toolcall-verifier-classifier-production`. It writes the runnable ONNX
artifact under `target/classifier-artifacts/onnx` and also downloads published
schema/report sidecars such as `input_schema_v1.json`,
`input_schema_v2.json`, and `serializer_fixture_v2.json` when they are present
in the model repo.

Then run the same local release eval with the target artifact loaded by the
proxy:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx
```

The target-artifact shortcut below downloads the quantized artifact first when
needed:

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
  --classify \
  --classifier-mode shadow \
  --output-dir target/local-eval/release-onnx-shadow

scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode shadow \
  --verify-final-response \
  --final-response-classifier-mode shadow \
  --output-dir target/local-eval/release-onnx-final-shadow

scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode enforce \
  --output-dir target/local-eval/release-onnx-enforce
```

The classifier default is `shadow`, but the supported modes are `disabled`,
`shadow`, `advisory`, and `enforce`. In shadow mode it should not change
completeness, success, retries, or nudges. Shadow runs show counterfactual
classifier decisions in telemetry only: inspect `action`, `label`, and
`confidence` in `proxy_classifier_<budget>.jsonl`, plus the sorted `top_k`
probability entries for calibration and label-collapse checks. Also inspect
`classifier_*` and `final_response_classifier_*` fields in `rust_smoke.jsonl`.
In enforce mode it can retry/block only labels whose artifact threshold is met;
labels with thresholds above `1.0` remain telemetry-only, and deterministic
guardrails remain authoritative for schema/protocol invalidity.
`scripts/run_local_eval.sh` writes proxy classifier telemetry to
`proxy_classifier_<budget>.jsonl` whenever a tool-call classifier or
final-response verifier is enabled, using the `FORGE_CLASSIFIER_LOG`
environment variable. Use that JSONL and Rust smoke JSONL classifier fields to
inspect classifier scores; use the Python oracle JSONL and reports to confirm
behavior changes. The proxy summary also compares classifier labels against
oracle requests when classifier JSONL is available, so threshold promotion
decisions should use oracle outcomes rather than confidence alone.

Classifier JSONL writes use a bounded async sink. Optional controls are
`FORGE_CLASSIFIER_LOG_QUEUE_CAPACITY`, `FORGE_CLASSIFIER_LOG_MAX_EVENT_BYTES`,
and `FORGE_CLASSIFIER_LOG_REDACT=true` for payload redaction.

ONNX scorer throughput knobs are opt-in. Keep defaults for parity baselines.
For proxy throughput comparisons, set `FORGE_CLASSIFIER_SESSION_POOL_SIZE` and
`FORGE_CLASSIFIER_INTRA_THREADS`; final-response verifier variants use
`FORGE_FINAL_RESPONSE_CLASSIFIER_SESSION_POOL_SIZE` and
`FORGE_FINAL_RESPONSE_CLASSIFIER_INTRA_THREADS`. Session pools are bounded to
`1..=4`; each extra session loads another model copy, so measure RSS alongside
latency. Each ONNX scorer also keeps a bounded in-process cache of serialized
scorer inputs to avoid repeated tokenization and model execution for identical
candidate/context pairs during replay.

The Colab production notebook now exports the tool-call verifier and the
separate final-response verifier by default. Keep both in `shadow` for first
replay. The eval shortcut below downloads the pinned quantized final-response
verifier into `target/final-response-classifier-artifacts/onnx` when needed,
enables it, and passes it to both the proxy and Rust smoke runner:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode shadow \
  --verify-final-response \
  --final-response-classifier-mode shadow \
  --output-dir target/local-eval/release-onnx-final-shadow
```

You can also download the final-response artifact explicitly with:

```bash
cargo run --features classifier --bin download-classifier -- \
  --artifact final-response \
  --classifier-model quantized
```

That command writes the published files under
`target/final-response-classifier-artifacts/onnx` by default. The pinned
final-response revision is
`bb11f0aaece9cae6f9b553e7522cb6d75d9cafbc` for
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
