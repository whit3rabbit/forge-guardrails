# Forge Verifier Model Artifacts

This directory documents the verifier model artifacts used by Forge local evals
and the user-facing proxy classifier shortcut.
Do not commit downloaded ONNX weights, model snapshots, Hugging Face caches, or
`target/classifier-artifacts` / `target/final-response-classifier-artifacts`
outputs. Keep eval model binaries under `target/`.

Latest checked Hub state: 2026-05-30.
Latest local eval review: 2026-05-30, documented in
[`local_eval_findings_2026-05-30.md`](local_eval_findings_2026-05-30.md).

## Artifact Repositories

| Artifact | Hugging Face repo | Latest checked revision | Runtime status |
|---|---|---|---|
| Tool-call verifier | [`cowWhySo/toolcall-verifier-classifier-production`](https://huggingface.co/cowWhySo/toolcall-verifier-classifier-production) | `b8e292b4de5725250bd1698eb5c795ffcb1a4cde` | Runnable by Rust ONNX scorer; pinned for reproducibility, shadow-only due metric regression |
| Final-response verifier | [`cowWhySo/final-response-verifier-classifier-production`](https://huggingface.co/cowWhySo/final-response-verifier-classifier-production) | `bb11f0aaece9cae6f9b553e7522cb6d75d9cafbc` | Runnable by Rust ONNX scorer; experimental/shadow-only |

Both artifacts are DeBERTa-v3-small text classifiers exported as FP32 ONNX and
quantized ONNX. Both should start in `shadow` mode. Deterministic Forge
validation remains authoritative.

Current deployment recommendation: keep both artifact families shadow-only.
For local active-mode policy, set every non-valid label's advisory and enforce
thresholds to `1.01` until release replay proves the label is safe. This is
stricter than the current downloaded threshold metadata for
`wrong_arguments_semantic`, `tool_not_needed`, and all final-response non-valid
labels.

The current tool-call pin is worse than the previous strong default revision
`b35b9734b6a3195e335ceb0a11b49d6782fec3b4`: macro F1 dropped from `0.9830` to
`0.6813`, and valid-call recall dropped to `0.41`. Do not promote it beyond
shadow mode without a new replay-backed reason.

## Download

For normal proxy use, prefer the `forge-guardrails-proxy` shortcut:

```bash
cargo run --features classifier --bin forge-guardrails-proxy -- --classify-download
```

That command downloads the pinned quantized tool-call verifier, prints
`classifier_dir=...`, and exits. The default user cache root is
`FORGE_CLASSIFIER_CACHE_DIR`, then `XDG_CACHE_HOME`, then
`$HOME/.cache/forge-guardrails/classifiers`. A repo-local cache such as
`.forge/classifiers` is ignored by git.

The commands below are eval/training-oriented and intentionally keep artifacts
under `target/`.

Tool-call classifier:

```bash
cargo run --features classifier --bin download-classifier -- \
  --artifact tool-call \
  --classifier-model quantized
```

Default output:

```text
target/classifier-artifacts/onnx
```

Final-response classifier:

```bash
cargo run --features classifier --bin download-classifier -- \
  --artifact final-response \
  --classifier-model quantized
```

Default output:

```text
target/final-response-classifier-artifacts/onnx
```

Download both artifact families:

```bash
cargo run --features classifier --bin download-classifier -- \
  --artifact both \
  --classifier-model quantized
```

Use `--classifier-model full` to also download `model.onnx`. The default
quantized path downloads `model_quantized.onnx` plus metadata, schemas,
thresholds, required tokenizer files, and published sidecars.

## Tool-Call Verifier

Purpose: score one candidate tool call after deterministic checks parse the
call, validate JSON shape, match tool names, enforce required steps, enforce
prerequisites, and reject malformed or unsafe batches.

Artifact contract:

| Field | Value |
|---|---|
| Artifact schema | `toolcall-verifier-artifact/v1` |
| Input schema | `toolcall-verifier-input/v1` |
| Serializer | `serialize_state_v1` |
| Max sequence length | `1280` |
| Base model | `microsoft/deberta-v3-small` |
| Default deployment | `shadow` |

Labels:

```text
valid
wrong_tool_semantic
wrong_arguments_semantic
tool_not_needed
needs_clarification
deterministic_invalid
```

Latest checked test metrics:

| Metric | Value |
|---|---:|
| Accuracy | `0.7996544197890142` |
| Macro precision | `0.7170414256557014` |
| Macro recall | `0.6878722779274437` |
| Macro F1 | `0.6813248632806018` |
| Test rows | `21992` |

Important deployment risks:

- `deterministic_invalid` is telemetry-only. Rust deterministic validation owns
  schema, unknown-tool, step, prerequisite, malformed-call, and unsafe-batch
  failures.
- `wrong_tool_semantic` remains conservative because previous Forge telemetry
  showed high-confidence false positives on valid terminal/summarize calls.
- `needs_clarification` has tiny held-out support and should not be enforced.
- The pinned default revision changes which artifact eval downloads, not the
  conservative runtime mode. Keep first replay runs in `shadow`.
- The 2026-05-30 enforce replay regressed `error_recovery*` because valid
  zero-padded numeric recovery calls were blocked as `wrong_arguments_semantic`.
  Keep that label action-disabled until numeric-string semantics are fixed.
- Valid-call false objections matter more than aggregate F1 for promotion.

## Final-Response Verifier

Purpose: score a candidate terminal response against the user request,
workflow state, required facts, tool trace, tool results, and scoring metadata.
This is a separate artifact family from the tool-call verifier.

Artifact contract:

| Field | Value |
|---|---|
| Artifact schema | `final-response-verifier-artifact/v1` |
| Input schema | `final-response-verifier-input/v1` |
| Serializer | `serialize_final_response_state_v1` |
| Max sequence length | `768` |
| Base model | `microsoft/deberta-v3-small` |
| Default deployment | `shadow` |

Labels:

```text
valid_final_response
missing_tool_fact
contradicts_tool_result
unsupported_claim
failed_to_acknowledge_data_gap
```

Latest checked test metrics:

| Metric | Value |
|---|---:|
| Accuracy | `0.09090909090909091` |
| Macro precision | `0.01818181818181818` |
| Macro recall | `0.2` |
| Macro F1 | `0.03333333333333333` |
| Test rows | `33` |

The current published final-response ONNX directory includes
`onnx/tokenizer.json`, so the Rust `OnnxFinalResponseScorer` can load the local
artifact directly.

The previous 2026-05-30 release replay labeled every final response as
`failed_to_acknowledge_data_gap` at roughly `0.23` confidence. This is not
useful enough for advisory or enforcement.

## Eval Usage

User-cache tool-call shadow run. This downloads or validates the quantized
tool-call artifact if it is missing, then passes the resolved artifact path to
the proxy and Rust smoke runner:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode shadow \
  --output-dir target/local-eval/release-toolcall-shadow
```

Target-artifact tool-call shadow run:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --download-classifier \
  --classifier-mode shadow \
  --output-dir target/local-eval/release-toolcall-shadow
```

Baseline comparison:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --output-dir target/local-eval/release-baseline

scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx \
  --classifier-mode shadow \
  --output-dir target/local-eval/release-toolcall-shadow
```

Minimum promotion matrix:

```text
no_classifier
classifier_fp32_onnx_shadow
classifier_quantized_onnx_shadow
classifier_fp32_onnx_advisory
classifier_quantized_onnx_advisory
```

Add final-response variants when evaluating terminal synthesis behavior.

Track:

- score and accuracy
- completeness regressions
- classifier disagreements
- valid-call false objections
- terminal-tool and summarize/report workflows
- `argument_transformation*` recovery
- `grounded_synthesis*` and data-gap recovery once final-response scoring is available
- p95/p99 classifier latency
- proxy RSS and CPU impact for each artifact family
- zero-padded numeric-string semantics, especially `0010` versus `10`

## Runtime Invariant

Forge owns interception and nudging. The verifier models may add telemetry,
advisory nudges, or eval-backed blocks, but they must not bypass deterministic
validation, execute tools, rewrite arguments, or relax workflow requirements.
