# Forge Verifier Model Artifacts

This directory documents the verifier model artifacts used by Forge local evals.
Do not commit downloaded ONNX weights, model snapshots, Hugging Face caches, or
`target/classifier-artifacts` / `target/final-response-classifier-artifacts`
outputs. Keep model binaries under `target/`.

Latest checked Hub state: 2026-05-27.

## Artifact Repositories

| Artifact | Hugging Face repo | Latest checked revision | Runtime status |
|---|---|---|---|
| Tool-call verifier | [`cowWhySo/toolcall-verifier-classifier-production`](https://huggingface.co/cowWhySo/toolcall-verifier-classifier-production) | `1c87eceea15ec42f755deafb0ac4166bd0bd51b0` | Runnable by Rust ONNX scorer |
| Final-response verifier | [`cowWhySo/final-response-verifier-classifier-production`](https://huggingface.co/cowWhySo/final-response-verifier-classifier-production) | `69d1a75d0fad25e3cf1333c7ea9c7cf0584614a4` | Runnable by Rust ONNX scorer |

Both artifacts are DeBERTa-v3-small text classifiers exported as FP32 ONNX and
quantized ONNX. Both should start in `shadow` mode. Deterministic Forge
validation remains authoritative.

## Download

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
| Accuracy | `0.9729105322763307` |
| Macro precision | `0.9801958127901861` |
| Macro recall | `0.9793045374079483` |
| Macro F1 | `0.9796981345691861` |
| Test rows | `22075` |

Important deployment risks:

- `deterministic_invalid` is telemetry-only. Rust deterministic validation owns
  schema, unknown-tool, step, prerequisite, malformed-call, and unsafe-batch
  failures.
- `wrong_tool_semantic` remains conservative because previous Forge telemetry
  showed high-confidence false positives on valid terminal/summarize calls.
- `needs_clarification` has tiny held-out support and should not be enforced.
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
| Accuracy | `0.2` |
| Macro precision | `0.04` |
| Macro recall | `0.2` |
| Macro F1 | `0.06666666666666667` |
| Test rows | `10` |

The current published final-response ONNX directory includes
`onnx/tokenizer.json`, so the Rust `OnnxFinalResponseScorer` can load the local
artifact directly.

## Eval Usage

Tool-call shadow run:

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

## Runtime Invariant

Forge owns interception and nudging. The verifier models may add telemetry,
advisory nudges, or eval-backed blocks, but they must not bypass deterministic
validation, execute tools, rewrite arguments, or relax workflow requirements.
