---
language:
  - en
license: other
base_model: microsoft/deberta-v3-small
library_name: transformers
pipeline_tag: text-classification
tags:
  - tool-use
  - function-calling
  - tool-calling
  - guardrails
  - verifier
  - text-classification
  - onnx
  - rust
  - shadow-mode
metrics:
  - accuracy
  - f1
  - precision
  - recall
---

# Tool-call Verifier Classifier

This document tracks the current tool-call verifier training state for Forge. It
is a recovery playbook, not a promotion record. The current published tool-call
artifact is telemetry-only and must stay in `shadow` mode until a replacement
passes notebook gates, ONNX parity, release shadow replay, and advisory replay.

The classifier is a DeBERTa sequence-classification sidecar over
`serialize_state_v1` tool-call contexts. It runs after deterministic validation:
syntax, JSON schema, unknown tools, required steps, prerequisites, unsafe
batches, and terminal-tool rules remain Rust-owned and authoritative.

## Current Status

| Field | Value |
|---|---|
| Base model | `microsoft/deberta-v3-small` |
| Notebook | `notebook/toolcall_verifier_training_production_colab_v4.ipynb` |
| Label mode | `production` |
| Input schema | `toolcall-verifier-input/v1` |
| Serializer | `serialize_state_v1` |
| Default runtime mode | `shadow` |
| Active non-valid thresholds | `1.01` |
| Current published tool-call pin | `b8e292b4de5725250bd1698eb5c795ffcb1a4cde` |
| Previous strong tool-call pin | `b35b9734b6a3195e335ceb0a11b49d6782fec3b4` |
| Current final-response pin | `bb11f0aaece9cae6f9b553e7522cb6d75d9cafbc` |

Do not promote the current published tool-call pin. It regressed from the
previous strong revision: held-out macro F1 dropped to about `0.681`, and
`valid` recall dropped to about `0.41`. The confusion matrix showed valid calls
being pushed into `wrong_tool_semantic`, so this was a training distribution
failure, not a threshold problem.

## Labels

Production mode uses six labels:

| Label | Meaning | Deployment guidance |
|---|---|---|
| `valid` | Candidate call appears appropriate for the request and workflow state. | Allow. |
| `wrong_tool_semantic` | Candidate uses the wrong tool for the request or workflow state. | Shadow-only until replay proves precision. |
| `wrong_arguments_semantic` | Candidate uses a plausible tool but semantically wrong arguments. | Shadow-only until numeric and recovery slices pass. |
| `tool_not_needed` | Candidate calls a tool when no tool call is needed. | Shadow-only until replay proves safety. |
| `needs_clarification` | Request is underspecified and should be clarified before tool use. | Ignore as a gate unless support is at least `50` rows. |
| `deterministic_invalid` | Collapsed bucket for deterministic failures. | Deterministic-only. Never enforce from ML. |

Raw deterministic labels collapse into `deterministic_invalid`:
`invalid_args_schema`, `missing_required_args`, `unknown_tool`,
`premature_terminal`, `missing_prerequisite`, `unsafe_parallel_batch`, and
`malformed_tool_call`.

## Current Notebook Settings

These are the current recovery defaults that should be preserved unless a new
run gives a concrete reason to change them.

### Dataset Mix

| Setting | Current value | Reason |
|---|---:|---|
| `FORGE_AGENT_HF_DATASET_WEIGHT` | `1` | Private rows tune Forge slices; they should not dominate. |
| `FORGE_AGENT_HF_TRAIN_FRACTION_TARGET` | `0.25` | Keep private rows in the `0.15` to `0.30` range. |
| `FORGE_AGENT_HF_PUBLIC_ONLY_TRAIN_CAP` | `0` | Preserve broad public coverage. |
| `FORGE_AGENT_HF_DOWNSAMPLE_PUBLIC_FOR_TARGET` | `False` | Do not shrink the public backbone to satisfy private fraction. |
| `PREFER_FORGE_AGENT_HF_DATASET` | `True` | Keep reviewed private rows when present. |
| `INCLUDE_PRIVATE_AGENT_LOGS` | `False` | Local agent logs remain opt-in. |

Use group-preserving sampling by `example_group_id`. If a hard negative is
included, keep the paired valid/corrected row in the same group so splitting and
sampling do not separate the contrastive pair.

### Uploaded Eval Files

Use this hard-negative glob:

```python
FORGE_HARD_NEGATIVE_GLOB = "/content/*hard_negatives.jsonl"
```

The previous glob, `/content/*.hard_negatives.jsonl`, did not match files named
`rust_smoke.tool_call_hard_negatives.jsonl` or
`rust_smoke.final_response_hard_negatives.jsonl`. A corrected T4 audit showed
the hard-negative loader working: `forge_hard_negative` rows were present, with
`7` corrected positives and `6` corrected error-recovery positives.

Telemetry files such as `proxy_classifier_budget_8192.jsonl` and
`rust_smoke.jsonl` are diagnostics only. Mine them for top-k failures, but do
not feed raw top-k telemetry into training or use it as promotion evidence.

### Train Rebalance

High-coverage and T4 profiles intentionally use different rebalance behavior.
The T4 profile is for cheap diagnosis; it is not promotion evidence.

| Setting | High-coverage default | T4/debug default |
|---|---:|---:|
| `VALID_TRAIN_FRACTION_TARGET` | `0.40` | `0.40` |
| `VALID_TRAIN_MAX_DUPLICATION_FACTOR` | `2` | `2` |
| `ENABLE_SEMANTIC_NEGATIVE_TRAIN_REBALANCE` | `False` | `False` |
| `WRONG_TOOL_TRAIN_TO_VALID_RATIO_TARGET` | `0.90` unused while disabled | `0.55` unused while disabled |
| `WRONG_ARGUMENTS_TRAIN_TO_VALID_RATIO_TARGET` | `0.75` unused while disabled | `0.70` unused while disabled |
| `MAX_SEMANTIC_NEGATIVE_DUPLICATION_FACTOR` | `4` | `2` unused while disabled |
| `ENABLE_VALID_PROTECTION_EXTRA_TRAIN_REBALANCE` | `True` | `False` |
| `VALID_PROTECTION_EXTRA_COPY_FACTOR` | `2` | disabled |
| `VALID_PROTECTION_EXTRA_COPY_ROWS_CAP` | `5000` | disabled |

Non-valid caps remain:

| Label | Max ratio to valid rows |
|---|---:|
| `deterministic_invalid` | `0.35` |
| `wrong_tool_semantic` | `0.75` |
| `wrong_arguments_semantic` | `0.90` |
| `tool_not_needed` | `0.30` |

### Valid-Protection Slices

Track these slices on validation and test. Apply valid recall and
false-objection gates when a slice has at least `25` valid rows.

- terminal-like tools: `respond`, `summarize`, `report`, `submit_*`, `present`,
  `recommend`, and `diagnose`,
- corrected error-recovery positives,
- fixed-width numeric string arguments, especially zero-padded values such as
  `0010`,
- no-op valid calls with empty argument objects.

## Promotion Gates

The immediate notebook gates are:

| Gate | Threshold |
|---|---:|
| `valid` recall | `>= 0.94` |
| `valid` false objection at confidence `0.90` | `<= 0.005` |
| `wrong_tool_semantic` precision | `>= 0.90` |
| `needs_clarification` | ignored unless support is at least `50` rows |
| valid-protection slices with at least `25` valid rows | same valid recall and false-objection gates |

Passing the notebook gates is necessary but not sufficient. Promotion also
requires FP32 ONNX parity, shadow release replay, false-objection mining, and a
later clean advisory replay.

## Lessons Learned

### Do Not Threshold Around A Bad Boundary

The current published pin learned a bad boundary: valid calls were pushed into
`wrong_tool_semantic`. Lowering or raising thresholds cannot fix that. Treat
that artifact as telemetry-only.

### Public Coverage Is The Backbone

The bad high-VRAM setup over-corrected toward private data: private fraction
`0.60`, private weight `4x`, and public-only caps around `6000` rows. That
shrunk broad valid/wrong-tool/wrong-argument coverage and collapsed valid-call
generalization. Current defaults restore public coverage and keep private rows
as a tuning slice.

### Hard Negatives Must Stay Paired

Hard negatives without their valid/corrected counterparts teach the classifier
to object broadly. Keep pairs together with `example_group_id`, and evaluate
their slices separately.

### Numeric Formatting Is Semantic

For the `error_recovery` smoke tool, `{"count":"0010"}` is valid and
`{"count":"10"}` is wrong for that schema. This must be trained and evaluated
as a semantic argument distinction, not treated as a harmless formatting issue.

### T4 Runs Are Diagnostics

T4 runs exposed data-path and balance issues but are not promotion candidates:

| Run | Useful finding | Failure |
|---|---|---|
| T4 valid-heavy run | `valid` recall reached `0.947` | `valid` false objection `0.0132`, `wrong_tool_semantic` precision `0.676`, `wrong_tool_semantic` recall `0.088` |
| T4 semantic-heavy run | `wrong_tool_semantic` recall recovered to `0.773` | `valid` recall collapsed to `0.628`, `wrong_tool_semantic` precision only `0.422` |
| T4 softened semantic run | `valid` recall recovered to `0.794` and `wrong_tool_semantic` precision improved to `0.528` | still failed `valid` recall, `valid` false objection, `wrong_tool_semantic` precision, and no-op valid slice gates |

The current T4-only rebalance backs off semantic-negative upsampling entirely
and disables extra protected-valid duplication. This is a diagnostic attempt to
separate the effects of global valid balance from semantic-negative pressure.
Use T4 to iterate on data flow, not to decide promotion. If this still fails,
prefer `t4_proven` or a high-coverage GPU run over more `t4_fast` ratio chasing.

### High-Coverage Recovery Is Closer

The best recovery signal so far came from a high-coverage run after public
downsampling was disabled:

| Metric | Value |
|---|---:|
| Test macro F1 | `0.9848` |
| `valid` recall | `0.9815` |
| `wrong_tool_semantic` precision | `0.9865` |
| `valid` false objection at `0.90` | `0.0077` |

That candidate still failed the `0.005` false-objection gate and was not
promoted. The next high-coverage run should keep public coverage, preserve
private rows at `0.25`, and focus on valid-protection false objections.

### Quantized ONNX Is A Separate Candidate

A prior quantized parity result had FP32/quantized top-label agreement around
`0.342`. Quantized output cannot be trusted just because PyTorch or FP32 ONNX
looks good. Calibrate thresholds against the artifact that will actually run.

Required parity gates:

| Check | Gate |
|---|---:|
| PyTorch vs FP32 ONNX top-label agreement | `>= 0.995` |
| Quantized ONNX vs FP32 ONNX top-label agreement | `>= 0.98` |

If quantized parity fails, write the parity report, stop packaging/upload, and
use FP32 ONNX for replay. Publish quantized only as shadow telemetry until
parity is fixed.

### Final-Response Verifier Is Separate

The final-response verifier is a separate artifact family and is not mature
enough for active behavior. A recent runtime replay labeled `302/302` final
responses as `failed_to_acknowledge_data_gap` at low confidence. Keep it
shadow-only and document/evaluate it separately.

## Threshold Policy

The exported default mode is `shadow`, with default action `allow`. Thresholds
are policy metadata, not proof that enforcement is safe.

Recommended local policy:

```json
{
  "schema_version": "toolcall-verifier-thresholds/v1",
  "mode": "shadow",
  "default_action": "allow",
  "labels": {
    "valid": {
      "action": "allow",
      "advisory_min_confidence": 0.0,
      "enforce_min_confidence": 1.01
    },
    "wrong_tool_semantic": {
      "action": "shadow_only_until_eval_proven",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    },
    "wrong_arguments_semantic": {
      "action": "shadow_only_until_eval_proven",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    },
    "tool_not_needed": {
      "action": "shadow_only_until_eval_proven",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    },
    "needs_clarification": {
      "action": "shadow_only_until_eval_proven",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    },
    "deterministic_invalid": {
      "action": "deterministic_only",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    }
  }
}
```

Candidate calibrated thresholds may be recorded for diagnostics, but non-valid
active thresholds should remain above `1.0` until shadow replay and advisory
replay both pass.

## Input Format

The classifier expects the canonical serialized format produced by
`serialize_state_v1`.

```text
SCHEMA_VERSION:
toolcall-verifier-input/v1

USER_REQUEST:
Generate a sales report from the Q4 2024 dataset.

WORKFLOW_STATE:
required_steps=['fetch_sales_data', 'analyze_sales']
completed_steps=[]
pending_steps=['fetch_sales_data', 'analyze_sales']
terminal_tools=['report']
recent_errors=[]

AVAILABLE_TOOLS:
report: Produce the final report from findings.
PARAMETERS: {"properties": {"summary": {"type": "string"}}, "required": ["summary"], "type": "object"}

fetch_sales_data: Fetch sales data for a given quarter and year.
PARAMETERS: {"properties": {"quarter": {"type": "integer"}, "year": {"type": "integer"}}, "required": ["quarter", "year"], "type": "object"}

analyze_sales: Analyze the loaded sales data and produce findings.
PARAMETERS: {"properties": {}, "type": "object"}

CANDIDATE_CALL:
{"arguments": {"summary": "Done."}, "name": "report"}
```

Runtime integrations should byte-compare serializer output against
`serializer_fixture.json` before trusting model scores.

## Runtime Files

Required artifact files:

```text
model.onnx
labels.json
thresholds.json
candidate_thresholds.json
artifact_manifest.json
input_schema.json
serializer_fixture.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
spm.model
config.json
training_run_summary.json
test_metrics.json
promotion_gate_report.json
valid_protection_slice_metrics.json
onnx_parity_report.json
```

`model_quantized.onnx` may be published only when quantized parity passes. If it
does not pass, treat it as telemetry-only and prefer FP32 ONNX for replay.

## Rust Deployment Guidance

Recommended integration order:

```text
1. Parse provider response.
2. Validate format, known tool names, and JSON-schema arguments.
3. Enforce required steps, prerequisites, terminal rules, and unsafe batches.
4. If the call is still valid-looking, run the classifier.
5. Shadow mode: log classifier verdict only.
6. Advisory mode: use classifier verdict to choose better nudges.
7. Enforce mode: block only high-confidence semantic labels after eval proof.
```

Loading failures should fail closed for strict deployment modes. Scoring
failures should fail open in `shadow` and `advisory` modes, with telemetry.

## Promotion Ladder

1. Train replacement.
2. Require good PyTorch validation/test metrics.
3. Require good FP32 ONNX parity.
4. Require good quantized parity, or skip quantized active use.
5. Run release eval in `shadow`.
6. Mine false objections and top-k disagreement rows.
7. Run advisory replay.
8. Consider enforcement only after advisory replay is clean.

Minimum replay matrix:

```text
no_classifier
classifier_fp32_onnx_shadow
classifier_quantized_onnx_shadow
classifier_fp32_onnx_advisory
classifier_quantized_onnx_advisory
```

Promotion must show:

- `valid` recall at least `0.94`,
- `valid` false objection at confidence `0.90` at most `0.005`,
- `wrong_tool_semantic` precision at least `0.90`,
- valid-protection slice gates for any slice with at least `25` valid rows,
- no regression in terminal-tool workflows,
- no regression in summarize/report workflows,
- no regression in fixed-width numeric strings or corrected error-recovery calls,
- acceptable p95/p99 latency and proxy RSS,
- stable behavior across real Forge tool schemas, not only public datasets.
