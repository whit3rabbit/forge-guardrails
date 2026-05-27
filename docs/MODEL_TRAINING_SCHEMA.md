# Model Training Schema

This document defines the training-data, artifact, and eval telemetry schemas for Forge semantic verifier models. It covers the tool-call verifier and the final-response verifier. These schemas are Rust-facing contracts; notebooks or training jobs should emit data that loads into the Rust types without lossy conversion.

The production training notebook is [`notebook/toolcall_verifier_training_production_colab_v4.ipynb`](../notebook/toolcall_verifier_training_production_colab_v4.ipynb). It is a Google Colab notebook. Do not treat local notebook execution as a verification gate. Validate local edits with JSON/static syntax checks, then run the notebook in Colab. `UPLOAD_TO_HUB`, `ENABLE_FORGE_AUGMENTATION`, and `ENABLE_FINAL_RESPONSE_VERIFIER` are intentionally enabled by default; artifacts are private by default and must carry shadow-first provenance until eval replay promotes them.

## Versioning Rules

- Tool-call verifier input v1 uses `schema_version: "toolcall-verifier-input/v1"` and `serialize_state_v1`.
- Tool-call verifier input v2 keeps v1 fields and adds generic `metadata`; it is serialized by `serialize_state_v2`.
- `serialize_state_v1` must remain byte-stable for legacy ONNX artifacts and remains the default deployable serializer.
- `serialize_state_v2` is an explicit ablation/export option. Do not silently publish metadata-dependent artifacts as v1.
- Current Rust accepts legacy five-label tool-call artifacts and six-label artifacts.
- New production tool-call artifacts should use the six-label order.
- Final-response artifacts use `schema_version: "final-response-verifier-input/v1"` and `serialize_final_response_state_v1`.
- Training rows may contain richer review metadata, but model input must be derivable from the schemas below.

## Notebook Inputs And Data Mining

The Colab notebook may load:

- Public tool-calling corpora.
- `python_oracle_colab_trace.jsonl`.
- `*.tool_call_hard_negatives.jsonl`.
- `*.final_response_hard_negatives.jsonl`.
- Synthetic Forge rows for deterministic validation, workflow ordering, semantic argument transformation, and grounded synthesis.

Failed eval outputs are not automatically positives. Create positives only from reviewed `corrected_positive` fields or canonical scenario validators/tests. Failed candidates may become negatives when the candidate is known bad. Group splits must keep related rows together by scenario, run, family, or explicit `example_group_id`.

The Colab hard-negative loader consumes the enriched eval envelope below, including `context.user_request`, `context.workflow_state`, `context.available_tools`, `context.required_facts`, candidate call lists, classifier scores, and corrected candidate calls/final responses. For `error_recovery`, label the failed `fetch` call as `wrong_arguments_semantic`, not `wrong_tool_semantic`; the tool choice is correct and the semantic argument value is wrong.

Tool-call hard negatives should cover:

- changed numeric units or totals
- missing converted values
- wrong threshold boundaries
- swapped source/target fields
- stale workflow state instead of current state
- wrong vendor alias handling

Final-response hard negatives should cover:

- omitted required facts
- contradictions of tool output
- unsupported claims
- missing data gaps treated as facts
- too-vague terminal summaries

## Shared Metadata

Tool-call v2 inputs and final-response inputs may carry this generic metadata block:

```json
{
  "scenario_family": "argument_transformation",
  "requires_transform": true,
  "requires_synthesis": false,
  "requires_all_tool_facts": true,
  "must_acknowledge_missing_data": false
}
```

These fields are training/scoring contracts, not public `_forge` message metadata. Keep names general. Do not encode leaderboard scenario names as a production-only shortcut.

## Tool-Call Training Row

Each tool-call training row contains a request/workflow context, one candidate tool call, the reviewed label, and optional positive correction data.

```json
{
  "schema_version": "toolcall-verifier-training/v1",
  "input": {
    "schema_version": "toolcall-verifier-input/v2",
    "user_request": "Generate a sales report from the Q4 2024 dataset.",
    "workflow_state": {
      "required_steps": ["fetch_sales_data", "analyze_sales"],
      "completed_steps": ["fetch_sales_data"],
      "pending_steps": ["analyze_sales"],
      "terminal_tools": ["report"],
      "recent_errors": []
    },
    "available_tools": [
      {
        "name": "report",
        "description": "Produce final report.",
        "parameters": {
          "type": "object",
          "properties": {
            "findings": { "type": "string" }
          },
          "required": ["findings"]
        }
      }
    ],
    "candidate_call": {
      "name": "report",
      "arguments": {
        "findings": "Done."
      }
    },
    "metadata": {
      "scenario_family": "argument_transformation",
      "requires_transform": true,
      "requires_synthesis": false,
      "requires_all_tool_facts": true,
      "must_acknowledge_missing_data": false
    }
  },
  "label": "wrong_arguments_semantic",
  "review": {
    "source": "forge-eval",
    "scenario": "argument_transformation",
    "run": 4,
    "failure_kind": "accuracy_false"
  },
  "corrected_positive": {
    "candidate_call": {
      "name": "report",
      "arguments": {
        "findings": "Revenue grew 23%; Widget Pro led sales; APAC was weakest."
      }
    }
  }
}
```

### Tool-Call Labels

Legacy five-label artifacts use this order:

```json
[
  "valid",
  "wrong_tool_semantic",
  "tool_not_needed",
  "needs_clarification",
  "deterministic_invalid"
]
```

Six-label artifacts use this order:

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

Label meanings:

| Label | Meaning |
| :--- | :--- |
| `valid` | The tool, schema, and argument values satisfy the user request and workflow state. |
| `wrong_tool_semantic` | The tool exists and the JSON may be valid, but the selected tool is semantically wrong. |
| `wrong_arguments_semantic` | The tool choice is plausible and schema-valid, but argument values do not match the request or workflow state. |
| `tool_not_needed` | The assistant should not call a tool for this turn. |
| `needs_clarification` | The assistant should ask for clarification before tool use. |
| `deterministic_invalid` | The candidate corresponds to deterministic validation failure; Rust deterministic checks remain authoritative. |

## Tool-Call Artifact Files

A tool-call verifier artifact directory contains:

```text
artifact_manifest.json
labels.json
thresholds.json
input_schema.json
input_schema_v1.json
input_schema_v2.json
serializer_fixture.json
serializer_fixture_v2.json
calibration_report.json
reliability_curves.jsonl
onnx_parity_report.json
training_run_summary.json
training_metrics.json
test_metrics.json
tokenizer.json
model.onnx
model_quantized.onnx
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
  "shadow_first_reason": "experimental verifier; promote only after eval replay",
  "supports_legacy_five_labels": false,
  "created_unix": 1779735826
}
```

### `thresholds.json`

`deterministic_invalid` must stay non-authoritative. For six-label artifacts, `wrong_arguments_semantic` should be calibrated separately from `wrong_tool_semantic`.

```json
{
  "schema_version": "toolcall-verifier-thresholds/v1",
  "mode": "enforce",
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

## Final-Response Training Row

The final-response verifier scores only terminal answers: `respond`, real terminal tools, or proxy text responses. It should not run on intermediate tool calls.

```json
{
  "schema_version": "final-response-verifier-training/v1",
  "input": {
    "schema_version": "final-response-verifier-input/v1",
    "user_request": "Summarize the Q4 2024 sales findings.",
    "workflow_state": {
      "required_steps": ["fetch_sales_data", "analyze_sales"],
      "completed_steps": ["fetch_sales_data", "analyze_sales"],
      "pending_steps": [],
      "terminal_tools": ["report"],
      "recent_errors": []
    },
    "required_facts": ["23% YoY growth", "Widget Pro", "APAC"],
    "tool_trace": ["fetch_sales_data", "analyze_sales", "report"],
    "tool_results": [
      {
        "tool_name": "analyze_sales",
        "content": "Revenue grew 23% YoY. Top product: Widget Pro. Weakest region: APAC."
      }
    ],
    "candidate_final_response": "Sales improved and the report is complete.",
    "metadata": {
      "scenario_family": "grounded_synthesis",
      "requires_transform": false,
      "requires_synthesis": true,
      "requires_all_tool_facts": true,
      "must_acknowledge_missing_data": false
    }
  },
  "label": "missing_tool_fact",
  "review": {
    "source": "forge-eval",
    "scenario": "grounded_synthesis",
    "run": 7,
    "failure_kind": "accuracy_false"
  },
  "corrected_positive": {
    "candidate_final_response": "Q4 revenue grew 23% year over year. Widget Pro was the top product, and APAC was the weakest region."
  }
}
```

### Final-Response Labels

```json
[
  "valid_final_response",
  "missing_tool_fact",
  "contradicts_tool_result",
  "unsupported_claim",
  "failed_to_acknowledge_data_gap"
]
```

## Final-Response Artifact Files

A final-response verifier artifact directory contains a separate model and metadata set:

```text
artifact_manifest.json
labels.json
thresholds.json
input_schema.json
training_provenance.json
tokenizer.json
model.onnx
model_quantized.onnx
```

### `artifact_manifest.json`

```json
{
  "artifact_schema_version": "final-response-verifier-artifact/v1",
  "model_kind": "text-classification-cross-encoder",
  "base_model": "microsoft/deberta-v3-small",
  "label_mode": "production",
  "input_schema_version": "final-response-verifier-input/v1",
  "serializer": "serialize_final_response_state_v1",
  "max_length": 768,
  "requested_gpu_profile": "auto",
  "run_profile": "high_vram_quality",
  "memory_profile": {
    "max_length": 768,
    "epochs": 5,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "grad_accum": 4,
    "max_per_label": 5000
  },
  "gpu_info": {
    "available": true,
    "name": "NVIDIA A100",
    "capability": [8, 0],
    "total_gb": 40.0
  },
  "onnx_file": "model.onnx",
  "quantized_onnx_file": "model_quantized.onnx",
  "labels": [
    "valid_final_response",
    "missing_tool_fact",
    "contradicts_tool_result",
    "unsupported_claim",
    "failed_to_acknowledge_data_gap"
  ],
  "deployment_default": "shadow",
  "shadow_first_reason": "experimental final-response verifier; promote only after eval replay",
  "created_unix": 1779735826
}
```

### `training_provenance.json`

The final-response training provenance records compact run details without
requiring large in-memory notebook objects to survive until upload:

```json
{
  "schema_version": "final-response-verifier-training-provenance/v1",
  "base_model": "microsoft/deberta-v3-small",
  "run_profile": "high_vram_quality",
  "gpu_info": {
    "available": true,
    "name": "NVIDIA A100",
    "capability": [8, 0],
    "total_gb": 40.0
  },
  "memory_profile": {
    "max_length": 768,
    "epochs": 5,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "grad_accum": 4,
    "max_per_label": 5000
  },
  "rows": 90,
  "train_rows": 70,
  "validation_rows": 10,
  "test_rows": 10,
  "label_counts": {
    "valid_final_response": 18,
    "missing_tool_fact": 18,
    "contradicts_tool_result": 18,
    "unsupported_claim": 18,
    "failed_to_acknowledge_data_gap": 18
  }
}
```

### `thresholds.json`

```json
{
  "schema_version": "final-response-verifier-thresholds/v1",
  "mode": "shadow",
  "default_action": "allow",
  "labels": {
    "valid_final_response": {
      "action": "allow",
      "advisory_min_confidence": 0.0,
      "enforce_min_confidence": 1.01
    },
    "missing_tool_fact": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.90,
      "enforce_min_confidence": 0.995
    },
    "contradicts_tool_result": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.90,
      "enforce_min_confidence": 0.995
    },
    "unsupported_claim": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.90,
      "enforce_min_confidence": 0.995
    },
    "failed_to_acknowledge_data_gap": {
      "action": "advisory_then_enforce_after_eval",
      "advisory_min_confidence": 0.90,
      "enforce_min_confidence": 0.995
    }
  }
}
```

## Calibration And ONNX Parity

Training jobs must write a calibration split separate from validation/test when enough groups exist. The notebook writes:

- `calibration.jsonl`
- `calibration_scored.jsonl`
- `calibration_report.json`
- `reliability_curves.jsonl`

Thresholds should be selected from calibrated probabilities and eval telemetry, not only classifier test accuracy. Temperature scaling and expected calibration error are required for new artifacts. `wrong_tool_semantic` should remain conservative until eval telemetry shows no valid-call false objections. `wrong_arguments_semantic` may use advisory `0.90` and enforce `0.995`, but runtime promotion beyond `shadow` requires eval replay.

ONNX parity reports must compare:

- PyTorch versus FP32 ONNX top-label agreement.
- FP32 ONNX versus quantized ONNX top-label/action disagreement.
- Per-label drift and valid false objections when eval rows are available.
- Latency for FP32 and quantized variants when eval replay is run.

If quantized ONNX drifts on the target scenarios, use FP32 for quality experiments and keep quantized in shadow telemetry only.

## Eval Hard-Negative Files

When `forge-eval --output path.jsonl` is used, Rust eval writes sibling files when a corrected positive is available:

```text
path.tool_call_hard_negatives.jsonl
path.final_response_hard_negatives.jsonl
```

Rows use this envelope:

```json
{
  "kind": "tool_call",
  "context": {
    "user_request": "Generate a sales report from the Q4 2024 dataset.",
    "workflow_state": {
      "required_steps": ["fetch_sales_data", "analyze_sales"],
      "completed_steps": ["fetch_sales_data"],
      "pending_steps": ["analyze_sales"],
      "terminal_tools": ["report"],
      "recent_errors": []
    },
    "available_tools": [
      {
        "name": "report",
        "description": "Produce final report.",
        "parameters": {
          "type": "object",
          "properties": {
            "findings": { "type": "string" }
          },
          "required": ["findings"]
        }
      }
    ],
    "required_facts": ["23% YoY growth", "Widget Pro", "APAC"]
  },
  "candidate": {
    "tool_sequence": ["fetch_sales_data", "report"],
    "tool_args": [{ "quarter": 4, "year": 2024 }, { "findings": "Done." }],
    "candidate_call": { "name": "report", "arguments": { "findings": "Done." } },
    "candidate_calls": [
      { "name": "fetch_sales_data", "arguments": { "quarter": 4, "year": 2024 } },
      { "name": "report", "arguments": { "findings": "Done." } }
    ]
  },
  "classifier_scores": [],
  "outcome": {
    "scenario": "sequential_3step",
    "scenario_family": "sequential_3step",
    "user_request": "Generate a sales report from the Q4 2024 dataset.",
    "run": 1,
    "failure_kind": "accuracy_false",
    "accuracy": false,
    "corrected_positive": {
      "final_text": "Revenue grew 23%; Widget Pro led sales; APAC was weakest."
    },
    "corrected_candidate_call": {
      "name": "report",
      "arguments": {
        "findings": "Revenue grew 23%; Widget Pro led sales; APAC was weakest."
      }
    },
    "corrected_candidate_calls": [
      {
        "name": "report",
        "arguments": {
          "findings": "Revenue grew 23%; Widget Pro led sales; APAC was weakest."
        }
      }
    ],
    "corrected_final_response": "Revenue grew 23%; Widget Pro led sales; APAC was weakest."
  }
}
```

Final-response hard negatives use the same context and outcome envelope with
`kind: "final_response"`, `candidate.final_text`, and
`final_response_classifier_scores`.

The notebook also emits reviewed corrected positives from
`outcome.corrected_candidate_calls` and `outcome.corrected_final_response`.
Those positives share the hard-negative group so train/validation/test splitting
does not leak a failed candidate and its correction across splits.

## Deployment Modes

Rust supports these verifier modes:

| Mode | Behavior |
| :--- | :--- |
| `disabled` | Do not run the verifier. |
| `shadow` | Score and log only. |
| `advisory` | Emit retry nudges when label confidence meets `advisory_min_confidence`. |
| `enforce` | Block/retry when label confidence meets `enforce_min_confidence`; otherwise advisory can still fire if its lower threshold is met. |

Enforcement is not globally enabled by a model file alone. Both runtime mode and per-label thresholds must request action. A label with thresholds above `1.0` is effectively telemetry-only even in `enforce` mode.

Promotion gates:

1. `shadow`: always allowed for telemetry.
2. `advisory`: only after eval replay shows no completeness regressions and no unacceptable valid-call false objections.
3. `enforce`: only after advisory replay proves the label-specific threshold is safe.

Required eval replay matrix:

```text
no_classifier
classifier_fp32_onnx_shadow
classifier_quantized_onnx_shadow
classifier_fp32_onnx_advisory
classifier_quantized_onnx_advisory
```

Add final-response verifier variants when final-response artifacts are being evaluated.
