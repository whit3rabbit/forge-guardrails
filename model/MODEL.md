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
model-index:
  - name: toolcall-verifier-classifier-production
    results:
      - task:
          type: text-classification
          name: Tool-call verification
        dataset:
          name: toolcall-verifier-dataset
          type: cowWhySo/toolcall-verifier-dataset
        metrics:
          - name: Accuracy
            type: accuracy
            value: 0.9729105322763307
          - name: Macro F1
            type: f1
            value: 0.9796981345691861
          - name: Macro Precision
            type: precision
            value: 0.9801958127901861
          - name: Macro Recall
            type: recall
            value: 0.9793045374079483
---

# Tool-call Verifier Classifier Production

This repository contains a production-mode text-classification sidecar for tool-call guardrails. The model scores serialized tool-call candidates after deterministic validation has already handled syntax, JSON schema, unknown tool names, required-step enforcement, prerequisite checks, unsafe batches, and terminal-tool rules.

The intended deployment pattern is conservative: deterministic guardrails remain authoritative, while this classifier starts in `shadow` mode and is promoted only after repository-specific eval replay proves that it improves nudging or routing without introducing false blocks on valid tool calls.

## Model summary

| Field | Value |
|---|---|
| Base model | `microsoft/deberta-v3-small` |
| Model kind | Text-classification cross-encoder |
| Label mode | `production` |
| Input schema | `toolcall-verifier-input/v1` |
| Serializer | `serialize_state_v1` |
| Max sequence length | `1280` |
| Deployment default | `shadow` |
| Primary artifact | `model.onnx` |
| Quantized artifact | `model_quantized.onnx` |
| Required tokenizer files | `tokenizer_config.json`, `special_tokens_map.json`, `spm.model` |
| Threshold file | `thresholds.json` |
| Manifest file | `artifact_manifest.json` |

## Intended use

Use this model to classify a candidate tool call in the context of:

- the original user request,
- available tool definitions,
- required workflow steps,
- completed and pending steps,
- terminal tools,
- recent errors,
- and the candidate tool call.

It is meant to support:

- shadow telemetry for semantic tool-call quality,
- advisory nudges when the selected tool or arguments look semantically wrong,
- eval-backed enforcement for high-confidence semantic errors,
- Rust-side inference through ONNX Runtime.

It is not meant to replace deterministic guardrails. It should not accept malformed calls, override JSON-schema validation, rewrite arguments, execute tools, or relax required workflow rules.

## Labels

Production mode uses six labels:

| Label | Meaning | Deployment guidance |
|---|---|---|
| `valid` | Candidate call appears appropriate for the request and workflow state. | Allow. |
| `wrong_tool_semantic` | Candidate uses the wrong tool for the request or workflow state. | Conservative; currently disabled for advisory/enforcement by thresholds. |
| `wrong_arguments_semantic` | Candidate uses a plausible tool but semantically wrong arguments. | Advisory first; enforce only after eval proof. |
| `tool_not_needed` | Candidate calls a tool when no tool call is needed. | Advisory first; enforce only after eval proof. |
| `needs_clarification` | Request is underspecified and should be clarified before tool use. | Advisory first; enforce only after eval proof. |
| `deterministic_invalid` | Collapsed bucket for failures owned by deterministic validation. | Deterministic-only. Do not enforce from ML. |

In production mode, the following raw labels are collapsed into `deterministic_invalid`: `invalid_args_schema`, `missing_required_args`, `unknown_tool`, `premature_terminal`, `missing_prerequisite`, `unsafe_parallel_batch`, and `malformed_tool_call`.

## Training configuration

Latest production run:

| Field | Value |
|---|---:|
| GPU profile | `high_vram_quality` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 95.0 GB |
| Precision | bf16 + tf32 |
| Seed | `42` |
| Max per source | `40000` |
| Max sequence length | `1280` |
| Epochs requested | `5` |
| Per-device train batch | `64` |
| Eval batch | `128` |
| Gradient accumulation | `1` |
| Learning rate | `6e-6` |
| Warmup ratio | `0.08` |
| Early stopping patience | `2` |
| Optimizer | `adamw_torch_fused` |
| Gradient checkpointing | `false` |
| Class weights | disabled |
| Forge augmentation | enabled |
| Final-response verifier training | enabled in the notebook, but separate from this tool-call classifier |

Split sizes:

| Split | Rows |
|---|---:|
| Train | 176,705 |
| Validation | 10,819 |
| Test | 22,075 |

Best validation checkpoint:

| Metric | Value |
|---|---:|
| Best checkpoint | `/content/toolcall-verifier/model/checkpoint-15505` |
| Selection metric | `macro_f1` |
| Validation loss | 0.07549399137496948 |
| Validation accuracy | 0.973380164525372 |
| Validation macro precision | 0.980623200539544 |
| Validation macro recall | 0.9800955285359482 |
| Validation macro F1 | 0.9803291996618319 |

## Test metrics

Held-out test set metrics:

| Metric | Value |
|---|---:|
| Test loss | 0.0744357779622078 |
| Accuracy | 0.9729105322763307 |
| Macro precision | 0.9801958127901861 |
| Macro recall | 0.9793045374079483 |
| Macro F1 | 0.9796981345691861 |
| Samples/sec | 686.863 |
| Steps/sec | 5.383 |

Per-label test report:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `valid` | 0.94 | 0.97 | 0.95 | 4,955 |
| `wrong_tool_semantic` | 0.97 | 0.96 | 0.96 | 4,960 |
| `wrong_arguments_semantic` | 0.98 | 0.98 | 0.98 | 5,005 |
| `tool_not_needed` | 1.00 | 0.99 | 1.00 | 2,029 |
| `needs_clarification` | 1.00 | 1.00 | 1.00 | 10 |
| `deterministic_invalid` | 0.99 | 0.98 | 0.99 | 5,116 |
| **Macro avg** | **0.98** | **0.98** | **0.98** | **22,075** |
| **Weighted avg** | **0.97** | **0.97** | **0.97** | **22,075** |

Per-source test accuracy:

| Source | Rows | Accuracy | Avg confidence |
|---|---:|---:|---:|
| `Salesforce/xlam-function-calling-60k` | 14,234 | 0.973444 | 0.983573 |
| `glaiveai/glaive-function-calling-v2` | 5,449 | 0.971738 | 0.981286 |
| `Team-ACE/ToolACE` | 2,380 | 0.973109 | 0.982948 |
| `forge_augmented` | 12 | 1.000000 | 0.999698 |

Per-label test accuracy:

| True label | Rows | Accuracy | Avg confidence |
|---|---:|---:|---:|
| `deterministic_invalid` | 5,116 | 0.980258 | 0.992830 |
| `wrong_arguments_semantic` | 5,005 | 0.980819 | 0.989488 |
| `wrong_tool_semantic` | 4,960 | 0.956855 | 0.971546 |
| `valid` | 4,955 | 0.965691 | 0.972742 |
| `tool_not_needed` | 2,029 | 0.992607 | 0.994770 |
| `needs_clarification` | 10 | 1.000000 | 0.971681 |

## Confusion matrix

Rows are true labels. Columns are predicted labels.

| True \\ Predicted | `valid` | `wrong_tool_semantic` | `wrong_arguments_semantic` | `tool_not_needed` | `needs_clarification` | `deterministic_invalid` |
|---|---:|---:|---:|---:|---:|---:|
| `valid` | 4,785 | 76 | 74 | 0 | 0 | 20 |
| `wrong_tool_semantic` | 195 | 4,746 | 8 | 0 | 0 | 11 |
| `wrong_arguments_semantic` | 75 | 18 | 4,909 | 0 | 0 | 3 |
| `tool_not_needed` | 1 | 13 | 0 | 2,014 | 0 | 1 |
| `needs_clarification` | 0 | 0 | 0 | 0 | 10 | 0 |
| `deterministic_invalid` | 49 | 50 | 2 | 0 | 0 | 5,015 |

## Threshold policy

The exported default mode is `shadow`, with default action `allow`. These thresholds should be treated as deployment policy metadata, not as proof that enforcement is safe in a new environment.

```json
{
  "schema_version": "toolcall-verifier-thresholds/v1",
  "mode": "shadow",
  "default_action": "allow",
  "temperature": 1.1155287027359009,
  "notes": [
    "Deterministic guardrails remain authoritative.",
    "Use ML in shadow mode first, then advisory nudges, then high-confidence enforcement only after eval proof.",
    "deterministic_invalid is never enforced by ML in this default config.",
    "wrong_tool_semantic stays conservative because current Forge telemetry showed high-confidence false positives on valid terminal/summarize calls."
  ],
  "labels": {
    "valid": {
      "action": "allow",
      "advisory_min_confidence": 0.0,
      "enforce_min_confidence": 1.01
    },
    "wrong_tool_semantic": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    },
    "wrong_arguments_semantic": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.9,
      "enforce_min_confidence": 0.995
    },
    "tool_not_needed": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.8,
      "enforce_min_confidence": 0.95
    },
    "needs_clarification": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.8,
      "enforce_min_confidence": 0.95
    },
    "deterministic_invalid": {
      "action": "deterministic_only",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    }
  }
}
```

## Input format

The classifier expects the canonical serialized format produced by `serialize_state_v1`.

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

For this fixture, both PyTorch and ONNX selected `deterministic_invalid`, with a reported max absolute logit difference of `6.67572021484375e-06`.

## Inference

### Transformers pipeline

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

repo_id = "cowWhySo/toolcall-verifier-classifier-production"

tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    device=0,  # use -1 for CPU
)

scores = clf(serialized_tool_call, truncation=True, max_length=1280)[0]
scores = sorted(scores, key=lambda item: item["score"], reverse=True)
print(scores[:5])
```

### ONNX Runtime

The ONNX path is the recommended Rust/runtime deployment path. Load the model with the same tokenizer behavior and the same serialized input text used during training.

Required files:

```text
model.onnx
model_quantized.onnx
labels.json
thresholds.json
artifact_manifest.json
input_schema.json
serializer_fixture.json
tokenizer_config.json
special_tokens_map.json
spm.model
```

Runtime integrations should byte-compare their serializer output against `serializer_fixture.json` before trusting model scores. This catches train/inference drift.

## Rust deployment guidance

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

Suggested runtime flags:

```text
--classifier-dir <path>
--classifier-mode off|shadow|advisory|enforce
--classifier-max-latency-ms <n>
FORGE_CLASSIFIER_DIR
FORGE_CLASSIFIER_MODE
FORGE_CLASSIFIER_MAX_LATENCY_MS
```

Default should be `off` unless a classifier directory is explicitly provided. First rollout should use `shadow`.

Recommended artifact-loader checks:

```text
artifact_manifest.json exists and includes training_run_summary/test_metrics provenance
artifact_schema_version == "toolcall-verifier-artifact/v1"
input_schema_version == "toolcall-verifier-input/v1"
serializer == "serialize_state_v1"
labels.json labels match model config
thresholds.json has every deployed label
tokenizer files exist
ONNX file exists
```

Loading failures should fail closed for strict deployment modes. Scoring failures should fail open in `shadow` and `advisory` modes, with telemetry.

## Calibration and safety notes

- Keep the model in `shadow` mode until eval replay confirms behavior on your real traffic and workflow families.
- Do not use `deterministic_invalid` predictions to enforce blocks. Deterministic Rust guardrails own those decisions.
- `wrong_tool_semantic` is intentionally disabled by threshold values above `1.0` because the current telemetry showed high-confidence false positives on otherwise valid terminal/summarize calls.
- High-confidence mistakes were observed, including valid calls predicted as deterministic or wrong-argument failures. Use per-family replay, not only aggregate F1, before promotion.
- The `needs_clarification` test support is small (`10` rows), so treat that label as under-validated despite the perfect held-out score.
- Validate public dataset licenses and any Forge-derived traces before publishing derived artifacts broadly.

## Tokenizer notes

The training run emitted tokenizer warnings around slow-to-fast conversion and regex/tokenization behavior. For parity-sensitive deployment, prefer the tokenizer path used by the notebook and artifact tests, and keep `use_fast=False` unless you have separately verified byte-for-byte or score-level parity.

If your Transformers version emits a Mistral regex warning for the local artifact, load with the appropriate `fix_mistral_regex=True` setting where supported. For Rust deployment, verify whether `tokenizer.json` is present and equivalent. If tokenizer parity is uncertain, use a sidecar scorer process until the tokenizer path is proven.

## ONNX parity and latency smoke check

A smoke test from the latest run reported:

| Check | Value |
|---|---:|
| Example latency | 126.49 ms |
| PyTorch top label | `deterministic_invalid` |
| ONNX top label | `deterministic_invalid` |
| Max absolute difference | `6.67572021484375e-06` |

This is a single-fixture smoke check, not a full deployment benchmark. Run larger parity checks on the exported `onnx_parity_report.json` and real replay traces before using quantized artifacts in advisory or enforcement mode.

## Related final-response verifier

The notebook can also train a separate final-response verifier with labels such as `valid_final_response`, `missing_tool_fact`, `contradicts_tool_result`, `unsupported_claim`, and `failed_to_acknowledge_data_gap`. That verifier is a separate artifact family and should be documented, evaluated, and deployed independently from this tool-call verifier.

The latest final-response run was small: `90` total rows split into `70` train, `10` validation, and `10` test rows. Its validation macro F1 remained low in the shown run, so it should stay experimental/shadow-only until the dataset is materially expanded.

## Limitations

- This model was trained on serialized tool-call contexts, not arbitrary natural language.
- It assumes deterministic validation has already run.
- It is sensitive to serializer drift, tokenizer drift, and tool-list truncation.
- Aggregate metrics are strong, but valid-call false positives are more important than headline macro F1 for enforcement.
- The `forge_augmented` test slice shown in the run contains only `12` rows, so it is useful as a smoke signal, not as sufficient Forge coverage.
- The final-response verifier path in the notebook is not mature enough for enforcement based on the shown data.

## Recommended eval replay before promotion

Run at least these variants before changing deployment mode:

```text
no_classifier
classifier_fp32_onnx_shadow
classifier_quantized_onnx_shadow
classifier_fp32_onnx_advisory
classifier_quantized_onnx_advisory
```

Promotion criteria should include:

- zero or near-zero false objections on valid calls,
- no regression in terminal-tool workflows,
- no regression in summarize/report workflows,
- improved targeted scenario-family scores,
- acceptable p95/p99 latency,
- PyTorch/ONNX/quantized parity on replay traces,
- stable behavior across real tool schemas, not only public function-calling datasets.
