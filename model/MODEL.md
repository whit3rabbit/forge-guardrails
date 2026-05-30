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
            value: 0.9770491803278688
          - name: Macro F1
            type: f1
            value: 0.9830369261812494
          - name: Macro Precision
            type: precision
            value: 0.9832233976323463
          - name: Macro Recall
            type: recall
            value: 0.9828910846156007
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
| Required tokenizer files | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `spm.model` |
| Threshold file | `thresholds.json` |
| Manifest file | `artifact_manifest.json` |
| Promoted default/eval revision | `b35b9734b6a3195e335ceb0a11b49d6782fec3b4` |

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
| Train | 178,545 |
| Calibration | 11,075 |
| Validation | 11,082 |
| Test | 22,265 |

Best validation checkpoint:

| Metric | Value |
|---|---:|
| Best checkpoint | `/content/toolcall-verifier/model/checkpoint-15665` |
| Selection metric | `macro_f1` |
| Validation loss | 0.07397261261940002 |
| Validation accuracy | 0.9762678216928352 |
| Validation macro precision | 0.9824560833904975 |
| Validation macro recall | 0.9820312268109691 |
| Validation macro F1 | 0.9822315582439032 |

## Test metrics

Held-out test set metrics:

| Metric | Value |
|---|---:|
| Test loss | 0.061976924538612366 |
| Accuracy | 0.9770491803278688 |
| Macro precision | 0.9832233976323463 |
| Macro recall | 0.9828910846156007 |
| Macro F1 | 0.9830369261812494 |
| Samples/sec | 682.714 |
| Steps/sec | 5.335 |

Per-label test report:

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `valid` | 0.95 | 0.97 | 0.96 | 5,042 |
| `wrong_tool_semantic` | 0.98 | 0.96 | 0.97 | 5,106 |
| `wrong_arguments_semantic` | 0.98 | 0.98 | 0.98 | 5,033 |
| `tool_not_needed` | 1.00 | 1.00 | 1.00 | 2,049 |
| `needs_clarification` | 1.00 | 1.00 | 1.00 | 8 |
| `deterministic_invalid` | 0.99 | 0.99 | 0.99 | 5,027 |
| **Macro avg** | **0.98** | **0.98** | **0.98** | **22,265** |
| **Weighted avg** | **0.98** | **0.98** | **0.98** | **22,265** |

Per-source test accuracy:

| Source | Rows | Accuracy | Avg confidence |
|---|---:|---:|---:|
| `Salesforce/xlam-function-calling-60k` | 14,710 | 0.978110 | 0.986481 |
| `glaiveai/glaive-function-calling-v2` | 4,941 | 0.977130 | 0.985937 |
| `Team-ACE/ToolACE` | 2,299 | 0.970857 | 0.983688 |
| `agent_training_hf` | 274 | 0.978102 | 0.989761 |
| `forge_trace` | 30 | 0.933333 | 0.971542 |
| `forge_augmented` | 11 | 0.909091 | 0.971169 |

Per-label test accuracy:

| True label | Rows | Accuracy | Avg confidence |
|---|---:|---:|---:|
| `wrong_tool_semantic` | 5,106 | 0.961614 | 0.976973 |
| `valid` | 5,042 | 0.965093 | 0.976443 |
| `wrong_arguments_semantic` | 5,033 | 0.983310 | 0.991983 |
| `deterministic_invalid` | 5,027 | 0.990253 | 0.994170 |
| `tool_not_needed` | 2,049 | 0.997072 | 0.998249 |
| `needs_clarification` | 8 | 1.000000 | 0.972783 |

## Confusion matrix

Rows are true labels. Columns are predicted labels.

| True \\ Predicted | `valid` | `wrong_tool_semantic` | `wrong_arguments_semantic` | `tool_not_needed` | `needs_clarification` | `deterministic_invalid` |
|---|---:|---:|---:|---:|---:|---:|
| `valid` | 4,866 | 72 | 87 | 0 | 0 | 17 |
| `wrong_tool_semantic` | 167 | 4,910 | 13 | 0 | 0 | 16 |
| `wrong_arguments_semantic` | 65 | 13 | 4,949 | 0 | 0 | 6 |
| `tool_not_needed` | 1 | 4 | 0 | 2,043 | 0 | 1 |
| `needs_clarification` | 0 | 0 | 0 | 0 | 8 | 0 |
| `deterministic_invalid` | 18 | 31 | 0 | 0 | 0 | 4,978 |

## Threshold policy

The exported default mode is `shadow`, with default action `allow`. Thresholds
are deployment policy metadata, not proof that enforcement is safe.

The 2026-05-30 local release replay showed that the downloaded
`wrong_arguments_semantic` active thresholds are unsafe for Forge: valid
zero-padded numeric recovery calls such as `{"count":"0010"}` were blocked,
while invalid unpadded calls such as `{"count":"10"}` were allowed as valid.
The recommended local active-mode policy is therefore stricter than the
downloaded threshold file: keep every non-valid ML label action-disabled until
targeted replay proves the label safe.

```json
{
  "schema_version": "toolcall-verifier-thresholds/v1",
  "mode": "shadow",
  "default_action": "allow",
  "temperature": 1.1033822298049927,
  "notes": [
    "Deterministic guardrails remain authoritative.",
    "Use ML in shadow mode first, then advisory nudges, then high-confidence enforcement only after eval proof.",
    "All non-valid labels remain action-disabled for the current Forge deployment recommendation.",
    "wrong_arguments_semantic is action-disabled because the 2026-05-30 replay produced high-confidence false blocks on zero-padded numeric strings.",
    "wrong_tool_semantic stays action-disabled because Forge telemetry showed high-confidence false positives on valid terminal/summarize calls.",
    "deterministic_invalid is never enforced by ML."
  ],
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
      "action": "shadow_only_until_numeric_semantics_fixed",
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

For the exported parity report, PyTorch and FP32 ONNX agreed on the top label for every sampled row, with a reported max absolute logit difference of `9.655952453613281e-06`. Quantized ONNX was less exact on that sample: top-label agreement was `0.9299610894941635`, with `18` disagreements across `257` rows and a max absolute difference of `7.591222286224365`.

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
added_tokens.json
spm.model
config.json
test_metrics.json
training_metrics.json
training_run_summary.json
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
- `wrong_arguments_semantic` should also stay disabled for active action. The 2026-05-30 replay produced high-confidence false blocks on valid zero-padded numeric strings and high-confidence false allows on invalid unpadded numeric strings.
- High-confidence mistakes were observed, including valid calls predicted as deterministic or wrong-argument failures. Use per-family replay, not only aggregate F1, before promotion.
- Valid-call false block rates from the latest run were `70/5042 = 0.0139` at confidence `0.80`, `57/5042 = 0.0113` at `0.90`, `43/5042 = 0.0085` at `0.95`, `28/5042 = 0.0056` at `0.98`, and `16/5042 = 0.0032` at `0.99`.
- The `needs_clarification` test support is small (`8` rows), so treat that label as under-validated despite the perfect held-out score.
- For fixed-width numeric strings, train and evaluate both representation and value. A four-digit count field has a structural range of `0000` through `9999`, but a request for `10` records is semantically correct only as `0010` for that tool.
- The latest local eval/resource review is documented in [`local_eval_findings_2026-05-30.md`](local_eval_findings_2026-05-30.md).
- Validate public dataset licenses and any Forge-derived traces before publishing derived artifacts broadly.

## Tokenizer notes

The training run emitted tokenizer warnings around slow-to-fast conversion and regex/tokenization behavior. For parity-sensitive deployment, prefer the tokenizer path used by the notebook and artifact tests, and keep `use_fast=False` unless you have separately verified byte-for-byte or score-level parity.

If your Transformers version emits a Mistral regex warning for the local artifact, load with the appropriate `fix_mistral_regex=True` setting where supported. For Rust deployment, verify whether `tokenizer.json` is present and equivalent. If tokenizer parity is uncertain, use a sidecar scorer process until the tokenizer path is proven.

## ONNX parity check

The latest downloaded `onnx_parity_report.json` reported:

| Check | Value |
|---|---:|
| Rows | 257 |
| PyTorch/FP32 ONNX top-label agreement | 1.0 |
| PyTorch/FP32 ONNX max absolute difference | `9.655952453613281e-06` |
| Quantized artifact present | true |
| FP32/quantized top-label agreement | 0.9299610894941635 |
| FP32/quantized disagreements | 18 |
| FP32/quantized max absolute difference | `7.591222286224365` |

This is an artifact parity report, not a full deployment benchmark. The quantized disagreement rate is a concrete reason to keep quantized deployments in `shadow` until replay traces prove the thresholds are safe.

## Related final-response verifier

The notebook can also train a separate final-response verifier with labels such as `valid_final_response`, `missing_tool_fact`, `contradicts_tool_result`, `unsupported_claim`, and `failed_to_acknowledge_data_gap`. That verifier is a separate artifact family and should be documented, evaluated, and deployed independently from this tool-call verifier.

The latest final-response run was small: `128` total rows split into `97` train, `17` validation, and `14` test rows. Its held-out macro F1 was `0.05`, so it should stay experimental/shadow-only until the dataset is materially expanded.

The 2026-05-30 release replay also showed poor runtime separation: `302/302`
final responses were labeled `failed_to_acknowledge_data_gap` at roughly `0.23`
confidence. Keep all final-response non-valid labels action-disabled with
`advisory_min_confidence=1.01` and `enforce_min_confidence=1.01` until top-k
telemetry and expanded eval data show useful separation.

## Limitations

- This model was trained on serialized tool-call contexts, not arbitrary natural language.
- It assumes deterministic validation has already run.
- It is sensitive to serializer drift, tokenizer drift, and tool-list truncation.
- Aggregate metrics are strong, but valid-call false positives are more important than headline macro F1 for enforcement.
- The Forge-specific test slices are still small: `forge_trace` has `30` rows and `forge_augmented` has `11` rows. They are useful smoke signals, not sufficient Forge coverage.
- The final-response verifier path in the notebook is not mature enough for enforcement based on the shown data.
- Resource cost is not free: the local final-response shadow run raised proxy mean RSS from `416.82 MiB` in the tool-call-only enforce run to `906.51 MiB`, with proxy p95 RSS rising from `603.73 MiB` to `1276.23 MiB`.

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
- acceptable proxy RSS and CPU budgets with resource sampling enabled,
- PyTorch/ONNX/quantized parity on replay traces,
- stable behavior across real tool schemas, not only public function-calling datasets.
