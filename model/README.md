---
language:
  - en
library_name: transformers
base_model: microsoft/deberta-v3-small
pipeline_tag: text-classification
tags:
  - tool-calling
  - guardrails
  - verifier
  - classifier
  - onnx
  - rust
  - deberta-v3
  - semantic-routing
datasets:
  - glaiveai/glaive-function-calling-v2
  - Team-ACE/ToolACE
  - Salesforce/xlam-function-calling-60k
---

# Tool-Call Verifier Classifier Production Candidate

https://huggingface.co/cowWhySo/toolcall-verifier-classifier-production

This is a compact DeBERTa-v3 text-classification sidecar for Forge tool-call
guardrails. It scores one candidate tool call after deterministic checks have
parsed the call, matched the tool name, checked JSON shape, and enforced
workflow order.

This model must not replace deterministic validation. Schema checks, unknown
tool checks, required-step checks, prerequisite checks, malformed-call checks,
and protocol checks remain authoritative in Rust.

## Deployment Status

Use this artifact as a production candidate in `shadow` first.

The classifier has strong held-out metrics, but high-confidence valid-call false
objections still exist. Promotion to `advisory` or `enforce` requires eval
replay on Forge scenarios and application traces.

Recommended first runtime mode:

```json
{
  "mode": "shadow",
  "default_action": "allow"
}
```

## Labels

This run uses the six-label production order expected by Rust:

```json
[
  "valid",
  "wrong_tool_semantic",
  "wrong_arguments_semantic",
  "tool_not_needed",
  "needs_clarification",
  "deterministic_invalid"
]
```

| Label | Meaning | Deployment guidance |
|---|---|---|
| `valid` | Candidate tool call appears appropriate. | Allow only if deterministic checks also pass. |
| `wrong_tool_semantic` | Tool exists, but the selected tool is semantically wrong for the request or state. | Keep conservative; current telemetry shows high-confidence valid-call false positives. |
| `wrong_arguments_semantic` | Tool choice is plausible and schema-valid, but argument values do not match the request or workflow state. | Main new target label. Advisory at high confidence after eval replay. |
| `tool_not_needed` | A tool call is probably unnecessary. | Candidate for advisory/enforce after eval replay. |
| `needs_clarification` | The user request is too ambiguous for safe tool use. | Test support is tiny. Do not enforce yet. |
| `deterministic_invalid` | Collapsed class for failures owned by deterministic guardrails. | Never enforce from ML. Rust deterministic checks remain authoritative. |

## Training Run

The DeBERTa warning about uninitialized `classifier.*` and `pooler.dense.*`
weights is expected. Those layers are task-specific and are trained by this
run.

| Field | Value |
|---|---:|
| Base model | `microsoft/deberta-v3-small` |
| Label mode | `production` |
| Serializer | `serialize_state_v1` |
| Input schema | `toolcall-verifier-input/v1` |
| Training profile | `high_vram_quality` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 95 GB |
| Train rows | 176,705 |
| Validation rows | 10,819 |
| Test rows | 22,075 |
| Max sequence length | 1,280 |
| Epochs | 5 |
| Train batch size | 64 |
| Eval batch size | 128 |
| Gradient accumulation | 1 |
| Learning rate | `6e-6` |
| Warmup ratio | `0.08` |
| Optimizer | `adamw_torch_fused` |
| Gradient checkpointing | `false` |
| Precision | bf16 + tf32 |
| Training runtime | 3,605.703 seconds |
| Best checkpoint | `/content/toolcall-verifier/model/checkpoint-15505` |
| Best metric | macro F1 = 0.980263 |

### Validation Metrics

| Metric | Value |
|---|---:|
| Loss | 0.075595 |
| Accuracy | 0.973288 |
| Macro precision | 0.980559 |
| Macro recall | 0.980029 |
| Macro F1 | 0.980263 |
| Macro precision, all labels | 0.980559 |
| Macro recall, all labels | 0.980029 |
| Macro F1, all labels | 0.980263 |

### Test Metrics

| Metric | Value |
|---|---:|
| Loss | 0.074661 |
| Accuracy | 0.972911 |
| Macro precision | 0.980179 |
| Macro recall | 0.979304 |
| Macro F1 | 0.979694 |
| Macro precision, all labels | 0.980179 |
| Macro recall, all labels | 0.979304 |
| Macro F1, all labels | 0.979694 |

### Test Classification Report

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `valid` | 0.94 | 0.97 | 0.95 | 4,955 |
| `wrong_tool_semantic` | 0.97 | 0.96 | 0.96 | 4,960 |
| `wrong_arguments_semantic` | 0.98 | 0.98 | 0.98 | 5,005 |
| `tool_not_needed` | 1.00 | 0.99 | 1.00 | 2,029 |
| `needs_clarification` | 1.00 | 1.00 | 1.00 | 10 |
| `deterministic_invalid` | 0.99 | 0.98 | 0.99 | 5,116 |
| **Overall accuracy** |  |  | **0.97** | 22,075 |
| **Macro average** | 0.98 | 0.98 | 0.98 | 22,075 |
| **Weighted average** | 0.97 | 0.97 | 0.97 | 22,075 |

`needs_clarification` has only 10 held-out examples. Treat that metric as
insufficient for enforcement.

### Confusion Matrix

Rows are true labels. Columns are predicted labels.

| True \\ Predicted | valid | wrong_tool_semantic | wrong_arguments_semantic | tool_not_needed | needs_clarification | deterministic_invalid |
|---|---:|---:|---:|---:|---:|---:|
| valid | 4,784 | 76 | 74 | 0 | 0 | 21 |
| wrong_tool_semantic | 197 | 4,744 | 8 | 0 | 0 | 11 |
| wrong_arguments_semantic | 70 | 20 | 4,912 | 0 | 0 | 3 |
| tool_not_needed | 1 | 12 | 1 | 2,014 | 0 | 1 |
| needs_clarification | 0 | 0 | 0 | 0 | 10 | 0 |
| deterministic_invalid | 49 | 50 | 2 | 0 | 0 | 5,015 |

### Per-Source Accuracy

| Source | Rows | Accuracy | Avg. confidence |
|---|---:|---:|---:|
| `Salesforce/xlam-function-calling-60k` | 14,234 | 0.973233 | 0.983561 |
| `glaiveai/glaive-function-calling-v2` | 5,449 | 0.972105 | 0.981210 |
| `Team-ACE/ToolACE` | 2,380 | 0.973529 | 0.982985 |
| `forge_augmented` | 12 | 1.000000 | 0.999695 |

The `forge_augmented` held-out sample is too small to prove production behavior.
Use Forge eval replay and proxy traces before promotion.

### Per-Label Accuracy

| True label | Rows | Accuracy | Avg. confidence |
|---|---:|---:|---:|
| `deterministic_invalid` | 5,116 | 0.980258 | 0.992850 |
| `wrong_arguments_semantic` | 5,005 | 0.981419 | 0.989540 |
| `wrong_tool_semantic` | 4,960 | 0.956452 | 0.971436 |
| `valid` | 4,955 | 0.965489 | 0.972668 |
| `tool_not_needed` | 2,029 | 0.992607 | 0.994795 |
| `needs_clarification` | 10 | 1.000000 | 0.971631 |

### Valid-Call False-Objection Rates

These rates estimate how often valid calls would receive a non-`valid`
prediction if enforced at the listed confidence threshold.

| Confidence threshold | False objections | Rate |
|---:|---:|---:|
| 0.80 | 69 / 4,955 | 0.0139 |
| 0.90 | 53 / 4,955 | 0.0107 |
| 0.95 | 38 / 4,955 | 0.0077 |
| 0.98 | 25 / 4,955 | 0.0050 |
| 0.99 | 23 / 4,955 | 0.0046 |

These are not low enough for broad enforcement. The first safe deployment is
shadow telemetry.

## Rust Artifact Contract

Rust expects an artifact directory with:

```text
artifact_manifest.json
labels.json
thresholds.json
input_schema.json
tokenizer.json
model.onnx
model_quantized.onnx
serializer_fixture.json
```

Additional recommended files:

```text
input_schema_v1.json
input_schema_v2.json
serializer_fixture_v2.json
calibration_report.json
reliability_curves.jsonl
onnx_parity_report.json
training_run_summary.json
training_metrics.json
test_metrics.json
```

### `artifact_manifest.json`

```json
{
  "artifact_schema_version": "toolcall-verifier-artifact/v1",
  "model_kind": "text-classification-cross-encoder",
  "base_model": "microsoft/deberta-v3-small",
  "label_mode": "production",
  "input_schema_version": "toolcall-verifier-input/v1",
  "serializer": "serialize_state_v1",
  "max_length": 1280,
  "onnx_file": "model.onnx",
  "quantized_onnx_file": "model_quantized.onnx",
  "labels": [
    "valid",
    "wrong_tool_semantic",
    "wrong_arguments_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid"
  ],
  "deployment_default": "shadow",
  "shadow_first_reason": "production candidate; promote only after eval replay",
  "supports_legacy_five_labels": false
}
```

Rust also accepts legacy five-label artifacts, but this artifact is six-label.
The label order must match exactly.

### `labels.json`

```json
{
  "label_mode": "production",
  "labels": [
    "valid",
    "wrong_tool_semantic",
    "wrong_arguments_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid"
  ],
  "label2id": {
    "valid": 0,
    "wrong_tool_semantic": 1,
    "wrong_arguments_semantic": 2,
    "tool_not_needed": 3,
    "needs_clarification": 4,
    "deterministic_invalid": 5
  },
  "id2label": {
    "0": "valid",
    "1": "wrong_tool_semantic",
    "2": "wrong_arguments_semantic",
    "3": "tool_not_needed",
    "4": "needs_clarification",
    "5": "deterministic_invalid"
  }
}
```

### `thresholds.json`

Keep global runtime mode `shadow` for initial deployment. Labels with thresholds
above `1.0` are telemetry-only even if the runtime is later set to `enforce`.

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
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 1.01,
      "enforce_min_confidence": 1.01
    },
    "wrong_arguments_semantic": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.90,
      "enforce_min_confidence": 0.995
    },
    "tool_not_needed": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.80,
      "enforce_min_confidence": 0.95
    },
    "needs_clarification": {
      "action": "advisory_then_enforce_after_eval",
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

## Input Format

This artifact was trained with `serialize_state_v1`.

Canonical serialized input:

```text
SCHEMA_VERSION:
toolcall-verifier-input/v1

USER_REQUEST:
...

WORKFLOW_STATE:
required_steps=[...]
completed_steps=[...]
pending_steps=[...]
terminal_tools=[...]
recent_errors=[...]

AVAILABLE_TOOLS:
tool_name: description
PARAMETERS: {...}

CANDIDATE_CALL:
{"name":"...","arguments":{...}}
```

`serialize_state_v1` must be byte-stable. Use `serializer_fixture.json` as a
parity test for Rust and any non-Python integration. Do not use
`serialize_state_v2` with this artifact unless the manifest explicitly changes
to `input_schema_version: "toolcall-verifier-input/v2"` and
`serializer: "serialize_state_v2"`.

## Python Inference Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "cowWhySo/toolcall-verifier-classifier-production"
subfolder = "hf_model"

labels = [
    "valid",
    "wrong_tool_semantic",
    "wrong_arguments_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid",
]

tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
model = AutoModelForSequenceClassification.from_pretrained(repo, subfolder=subfolder)
model.eval()

text = """SCHEMA_VERSION:
toolcall-verifier-input/v1

USER_REQUEST:
What is the capital of France?

WORKFLOW_STATE:
required_steps=['get_country_info']
completed_steps=[]
pending_steps=['get_country_info']
terminal_tools=['respond']
recent_errors=[]

AVAILABLE_TOOLS:
get_country_info: Get country information.
PARAMETERS: {"type":"object","properties":{"country":{"type":"string"}},"required":["country"]}

CANDIDATE_CALL:
{"name":"get_country_info","arguments":{"country":"France"}}
"""

with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1280)
    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    best = int(torch.argmax(probs))

print(labels[best], float(probs[best]))
```

## ONNX / Rust Deployment Notes

Recommended Rust path:

1. Load `artifact_manifest.json`.
2. Reject unsupported `artifact_schema_version`, `input_schema_version`, or
   `serializer`.
3. Validate exact label order from `labels.json`.
4. Load `tokenizer.json` with Hugging Face `tokenizers`.
5. Load `model.onnx` or `model_quantized.onnx` with ONNX Runtime.
6. Run `serializer_fixture.json` as a byte-for-byte serializer parity test.
7. Run a logits-shape test. This artifact must output six logits.
8. Convert logits to probabilities with softmax.
9. Map label IDs using `labels.json`.
10. Apply runtime mode and per-label `thresholds.json`.

Suggested Rust-side behavior:

```text
deterministic guardrail reject -> reject; classifier cannot override
deterministic guardrail allow  -> run classifier in shadow first
classifier error              -> log and allow deterministic path
classifier deterministic_invalid -> log only; never enforce from ML
classifier wrong_tool_semantic -> conservative; advisory only after replay
classifier wrong_arguments_semantic -> target label; advisory after replay
classifier tool_not_needed    -> advisory/enforce only after replay
classifier needs_clarification -> disabled until more support exists
```

## Required Eval Replay Before Promotion

Minimum matrix:

```text
no_classifier
classifier_fp32_onnx_shadow
classifier_quantized_onnx_shadow
classifier_fp32_onnx_advisory
classifier_quantized_onnx_advisory
```

Track:

- score and accuracy
- completeness regressions
- classifier disagreements
- valid-call false objections
- latency
- `argument_transformation*` recovery
- `grounded_synthesis*` interaction, if a final-response verifier is also used

## Limitations

- This classifier is not a replacement for JSON parsing, schema validation,
  tool-name validation, step enforcement, prerequisite checks, policy checks,
  protocol validity, or sandboxing.
- Valid-call false objections remain non-zero even at high confidence.
- `wrong_tool_semantic` has enough high-confidence mistakes to stay
  conservative.
- `needs_clarification` has too little support for enforcement.
- Public tool-calling datasets contain noisy labels and may not match Forge
  production behavior.
- `forge_augmented` held-out support is small in this run.
- Thresholds must be calibrated and replayed in the host environment before
  blocking.

## Citation

If you use this model, cite this repository and pin the exact model revision.
Production deployments should pin labels, tokenizer, ONNX weights, thresholds,
serializer, and artifact manifest together.
