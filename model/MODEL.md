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

The classifier is a DeBERTa sequence-classification sidecar over serialized
tool-call contexts. Current published artifacts use `serialize_state_v1`; new
replacement runs should use `toolcall-verifier-input/v2` with
`serialize_state_v2`. It runs after deterministic validation: syntax, JSON
schema, unknown tools, required steps, prerequisites, unsafe batches, and
terminal-tool rules remain Rust-owned and authoritative.

## Current Status

| Field | Value |
|---|---|
| Base model | `microsoft/deberta-v3-small` |
| Notebook | `notebook/toolcall_verifier_training_production_colab_v5.ipynb` |
| Label mode | `production` |
| Current published input schema | `toolcall-verifier-input/v1` |
| Current published serializer | `serialize_state_v1` |
| Replacement notebook input schema | `toolcall-verifier-input/v2` |
| Replacement notebook serializer | `serialize_state_v2` |
| Default runtime mode | `shadow` |
| Active non-valid thresholds | `1.01` |
| Current published tool-call pin | `f4f5cfe96aa93fd6b3bf028157895b7ec0113c89` |
| Previous strong tool-call pin | `b35b9734b6a3195e335ceb0a11b49d6782fec3b4` |
| Current final-response pin | `bb11f0aaece9cae6f9b553e7522cb6d75d9cafbc` |

The default published tool-call pin was updated to the June 11 high-coverage run (`f4f5cfe96aa93fd6b3bf028157895b7ec0113c89`). This resolves the training distribution failure/regression of `b8e292b4de5725250bd1698eb5c795ffcb1a4cde` (which had F1 `0.681` and `valid` recall `0.41`). The new candidate pin achieves a test macro F1 of `0.9014`, `valid` recall of `0.9824`, and `wrong_tool_semantic` precision of `0.9890`. However, it still fails the strict `0.005` false objection promotion gate (obtaining `0.0068`), so it should remain in `shadow` mode.

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
| `USE_SERIALIZER_V2` | `True` | Train/export the metadata-aware schema used by new Forge rows. |

Use group-preserving sampling by `example_group_id`. If a hard negative is
included, keep the paired valid/corrected row in the same group so splitting and
sampling do not separate the contrastive pair.

### Private Generated Dataset

The private generated dataset used for `agent_training_hf` in the latest run is `addenda/forge-eval-3k-v2/agent_training.notebook.jsonl` from the Hugging Face repo `cowWhySo/forge-toolcall-verifier-openrouter-2650-v1` (revision `01eedcb861324df5fe5b6584ed4f12995b103d0f`), containing `724` agent-derived rows:

| Label | Rows |
|---|---:|
| `valid` | `413` |
| `tool_not_needed` | `241` |
| `wrong_arguments_semantic` | `38` |
| `wrong_tool_semantic` | `32` |

This legacy dataset is useful as Forge-style valid-call coverage, but it is not
strong wrong-tool training evidence. In this run, `246/247` private wrong-tool
rows used a literal `synthetic_unrelated_tool` distractor, so the negative
boundary is mostly a name-level shortcut. The latest pasted evaluation showed
`agent_training_hf` accuracy around `0.975`, while the large wrong-tool
confusions still came from public datasets. Do not infer from that private score
that the classifier has learned real wrong-tool semantics.

For the next private addendum, use `forge-dataset` reviewed rows rather than the
legacy distractor dataset. The generator now creates targeted alternatives only
from verified-valid captures and reviewer/verifier-accepts them before training:

- prefer real competing tools from the same observed task group when available;
- include paired valid rows in the same `example_group_id`;
- keep schema-valid arguments for the distractor so the label remains semantic
  wrong-tool, not deterministic invalid or wrong-argument noise;
- include bounded repeated-tool (`tool_not_needed`) and underspecified-request
  (`needs_clarification`) alternatives;
- mine high-confidence reviewed quarantines, such as `uv lock` requested but
  `make build` executed, into paired wrong-argument or wrong-tool examples only
  after verification accepts them as training rows.

Recommended private capture-review mix for the next OpenRouter addendum:

```bash
--review-max-alternatives-per-group 4 \
--review-max-alternative-ratio 0.50
```

After generation, require `forge-dataset validate` and `split_manifest.json` to
show nonzero counts for `valid`, `wrong_tool_semantic`,
`wrong_arguments_semantic`, `tool_not_needed`, and `needs_clarification` before
using the addendum in a production notebook run.

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
| `MAX_NEEDS_CLARIFICATION_TO_VALID_TRAIN_RATIO` | `0.15` | `0.15` |
| `ENABLE_VALID_PROTECTION_EXTRA_TRAIN_REBALANCE` | `True` | `True` |
| `VALID_PROTECTION_EXTRA_COPY_FACTOR` | `2` | `2` |
| `VALID_PROTECTION_EXTRA_COPY_ROWS_CAP` | `5000` | `5000` |

Non-valid caps remain:

| Label | Max ratio to valid rows |
|---|---:|
| `deterministic_invalid` | `0.35` |
| `wrong_tool_semantic` | `0.75` |
| `wrong_arguments_semantic` | `0.90` |
| `tool_not_needed` | `0.30` |
| `needs_clarification` | `0.15` |

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

Since the v5e notebook patch, `promotion_gate_report.json` is the single source
of truth for notebook-side promotability. It carries `promotion_status`
(`blocked` or `promotable_pending_replay`, never plain `promotable`),
`blocked_reasons[]`, and `artifact_promotable`, which are mirrored into
`artifact_manifest.json`, `thresholds.json`, `candidate_thresholds.json`,
`test_metrics.json`, and `training_run_summary.json`; the notebook raises if
any exported file claims promotability while the gate report is blocked. The
per-eval `eval_checkpoint_constrained_promotable` flag is checkpoint-selection
telemetry only and uses the strict core gates, replacing the old ambiguous
`eval_constrained_promotable` key. The report also embeds diagnostic-only
`threshold_sweep`, `confidence_margin_diagnostics`, and
`per_source_diagnostics` keys that never block promotion, and the run exports
`high_confidence_mistakes.jsonl` for manual audit of confident wrong
predictions.

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
| T4 `auto`/`t4_proven` recovery run | macro F1 recovered to `0.7603` and `valid` recall to `0.9109` after the `t4_fast` collapse | still failed `valid` recall, `valid` false objection `0.0127`, `wrong_tool_semantic` precision `0.7273`, fixed-width/no-op slice gates, and showed `CANDIDATE_CALL` truncation around `12.5%` |
| T4 `openrouter-train-3k` run | test `valid` recall reached `0.9408`, `wrong_arguments_semantic` precision reached `0.9523`, and `agent_training_hf` accuracy reached about `0.975` | validation/test still failed promotion: test valid false objection `0.0128`, test `wrong_tool_semantic` precision `0.8462`, wrong-tool recall only about `0.30`, and protected valid slices still failed |

The current T4-only rebalance backs off semantic-negative upsampling entirely
and keeps extra protected-valid duplication enabled. This is a diagnostic
attempt to separate the effects of global valid balance and protected valid
support from semantic-negative pressure. Use T4 to iterate on data flow, not to
decide promotion. If T4 continues to fail after data-quality fixes, prefer a
high-coverage GPU run with a longer context window over more `t4_fast` ratio
chasing.

The `openrouter-train-3k` result changes the immediate diagnosis. It no longer
looks like the model primarily pushes valid calls into `wrong_tool_semantic`.
Instead, it is too permissive on public wrong-tool rows: `755/1139` test
`wrong_tool_semantic` rows were predicted `valid`, while private
`agent_training_hf` rows were already mostly correct. Fix generated and public
wrong-tool evidence before changing gates or thresholds.

The latest `auto`/`t4_proven` sidecars also exposed a reporting issue: split
balancing produced `25` corrected error-recovery valid rows in both validation
and test, but the evaluation slice mask reported zero rows. Slice diagnostics
must use the precomputed `valid_protection_*` columns when present, not only
metadata reparsing after JSON dataset reload.

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
promoted. The latest run on 2026-06-11 (detailed in [Latest Run Results](#latest-run-results-2026-06-11) below) further improved key metrics: `valid` recall reached `0.9824`, `wrong_tool_semantic` precision reached `0.9890`, and `valid` false objection at `0.90` was reduced to `0.0068`. However, it still fails the strict `0.005` false objection promotion gate on the test set.

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

## Latest Run Results (2026-06-11)

The latest high-coverage run on June 11, 2026, was executed with `enable_forge_augmentation=True` and `enable_final_response_verifier=True`. 

### Dataset Statistics

During preprocessing, `33,056` deterministic invalid rows were removed. In addition, `62` rows were quarantined due to source-quality flags (`forge_argument_semantic`, `forge_contrastive_wts`, `forge_hard_negative`, and `forge_synthetic`), leaving `290,019` rows after quarantine. 

After applying group-preserving label caps (max `50,000` per label), the dataset size was reduced to `226,599` rows, preserving all preferred private HF rows.

**Capped training rows by source and label:**
- **Salesforce/xlam-function-calling-60k**: 130,870 rows (valid: 47,237, wrong_arguments: 45,568, wrong_tool: 37,221, needs_clarification: 538, tool_not_needed: 12,844)
- **glaiveai/glaive-function-calling-v2**: 48,763 rows (valid: 19,398, wrong_arguments: 18,350, wrong_tool: 5,314, needs_clarification: 237, tool_not_needed: 5,414)
- **Team-ACE/ToolACE**: 27,713 rows (valid: 9,486, wrong_arguments: 8,926, wrong_tool: 7,184, needs_clarification: 120, tool_not_needed: 2,017)
- **agent_training_hf**: 724 rows (valid: 413, wrong_arguments: 38, wrong_tool: 32, tool_not_needed: 241)
- **forge_error_recovery_protected**: 2,559 rows (valid: 525, wrong_arguments: 1,509, wrong_tool: 525)
- **forge_fixed_width_numeric**: 1,874 rows (valid: 570, wrong_arguments: 1,304)
- **forge_trace**: 1,069 rows (valid: 1,051, wrong_arguments: 18)
- **forge_error_recovery_numeric**: 419 rows (valid: 60, wrong_arguments: 359)
- **forge_augmented**: 100 rows (needs_clarification: 100)

**Final split sizes:**
- **Train**: 190,692 rows (after valid rebalancing duplication factor of 2)
- **Validation**: 11,293 rows
- **Test**: 22,370 rows

### Training Profile

- **Device**: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB VRAM)
- **Batch Size**: 64 (gradient accumulation: 1)
- **Max Sequence Length**: 1,280
- **Optimizer**: `adamw_torch_fused`
- **Gradient Checkpointing**: Disabled
- **Epochs**: 5

### Training Progress & Best Checkpoint

The best model checkpoint was saved at step `13384` (end of Epoch 4) based on the `gate_deficit_score` metric.

| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation Macro F1 | Valid Recall | Valid False Objection at 0.90 | Wrong Tool Precision | Wrong Arguments Recall | Gate Deficit Score | Checkpoint Constrained Promotable |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1653 | 0.1621 | 0.9486 | 0.7662 | 0.9005 | 0.0290 | 0.9089 | 0.9786 | 0.8367 | False |
| 2 | 0.1033 | 0.0832 | 0.9730 | 0.8536 | 0.9684 | 0.0069 | 0.9810 | 0.9781 | 101.0584 | False |
| 3 | 0.0733 | 0.0845 | 0.9752 | 0.8716 | 0.9809 | 0.0079 | 0.9988 | 0.9820 | 101.0623 | False |
| **4** | **0.0526** | **0.0657** | **0.9792** | **0.9273** | **0.9817** | **0.0048** | **0.9873** | **0.9783** | **101.0911** | **True** |
| 5 | 0.0485 | 0.0624 | 0.9806 | 0.9422 | 0.9796 | 0.0051 | 0.9865 | 0.9799 | 101.0881 | False |

### Test Evaluation Results

Evaluated on the held-out test split of `22,370` rows:

| Metric | Value |
| :--- | :---: |
| **Test Accuracy** | `0.9780` |
| **Macro F1 (5 Active Labels)** | `0.9014` |
| **Macro F1 (All Labels)** | `0.7512` |
| **`valid` Recall** | `0.9824` |
| **`valid` Precision** | `0.9583` |
| **`valid` False Objection at 0.90** | `0.0068` (22 false objections / 7,836 valid rows) |
| **`wrong_tool_semantic` Precision** | `0.9890` |
| **`wrong_tool_semantic` Recall** | `0.9718` |
| **`wrong_arguments_semantic` Precision** | `0.9878` |
| **`wrong_arguments_semantic` Recall** | `0.9793` |
| **`valid` to `wrong_arguments_semantic` Error Rate** | `0.0103` |
| **`wrong_tool` to `wrong_arguments_semantic` Rate** | `0.0008` |
| **Gate Deficit Score** | `101.0743` |

#### Test Classification Report

```text
                          precision    recall  f1-score   support

                   valid       0.96      0.98      0.97      7836
     wrong_tool_semantic       0.99      0.97      0.98      4921
wrong_arguments_semantic       0.99      0.98      0.98      7543
         tool_not_needed       1.00      1.00      1.00      1969
     needs_clarification       0.85      0.44      0.58       101
   deterministic_invalid       0.00      0.00      0.00         0

                accuracy                           0.98     22370
               macro avg       0.80      0.73      0.75     22370
            weighted avg       0.98      0.98      0.98     22370
```

#### Test Confusion Matrix

| True \ Predicted | valid | wrong_tool_semantic | wrong_arguments_semantic | tool_not_needed | needs_clarification | deterministic_invalid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **valid** | 7698 | 48 | 81 | 3 | 6 | 0 |
| **wrong_tool_semantic** | 133 | 4783 | 4 | 1 | 0 | 0 |
| **wrong_arguments_semantic** | 149 | 2 | 7388 | 2 | 2 | 0 |
| **tool_not_needed** | 1 | 2 | 0 | 1966 | 0 | 0 |
| **needs_clarification** | 50 | 1 | 6 | 0 | 44 | 0 |
| **deterministic_invalid** | 0 | 0 | 0 | 0 | 0 | 0 |

#### Per-Source and Per-Label Accuracies

**Per-Source Accuracy:**
- `Salesforce/xlam-function-calling-60k`: 14,071 rows, Accuracy: `97.73%` (Avg Conf: `0.9882`)
- `glaiveai/glaive-function-calling-v2`: 4,775 rows, Accuracy: `99.52%` (Avg Conf: `0.9971`)
- `Team-ACE/ToolACE`: 2,822 rows, Accuracy: `95.11%` (Avg Conf: `0.9642`)
- `forge_error_recovery_protected`: 257 rows, Accuracy: `100.00%` (Avg Conf: `0.9993`)
- `forge_fixed_width_numeric`: 197 rows, Accuracy: `99.49%` (Avg Conf: `0.9977`)
- `forge_trace`: 152 rows, Accuracy: `99.34%` (Avg Conf: `0.9987`)
- `agent_training_hf`: 48 rows, Accuracy: `85.42%` (Avg Conf: `0.9578`)
- `forge_error_recovery_numeric`: 35 rows, Accuracy: `97.14%` (Avg Conf: `0.9783`)
- `forge_augmented`: 13 rows, Accuracy: `100.00%` (Avg Conf: `0.9612`)

**Per-Label Accuracy:**
- `valid`: 7,836 rows, Accuracy: `98.24%` (Avg Conf: `0.9805`)
- `wrong_arguments_semantic`: 7,543 rows, Accuracy: `97.95%` (Avg Conf: `0.9910`)
- `wrong_tool_semantic`: 4,921 rows, Accuracy: `97.20%` (Avg Conf: `0.9902`)
- `tool_not_needed`: 1,969 rows, Accuracy: `99.85%` (Avg Conf: `0.9996`)
- `needs_clarification`: 101 rows, Accuracy: `43.56%` (Avg Conf: `0.8513`)

#### Guarded-Objection Sweep Details

Valid-call false block rate at different logit thresholds on the test set:
- `@ 0.80`: `73 / 7836` = `0.0093`
- `@ 0.90`: `52 / 7836` = `0.0066`
- `@ 0.95`: `39 / 7836` = `0.0050`
- `@ 0.98`: `30 / 7836` = `0.0038`
- `@ 0.99`: `22 / 7836` = `0.0028`

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

The current published classifier expects the canonical serialized format
produced by `serialize_state_v1`. New replacement artifacts should use
`serialize_state_v2`, which keeps the v1 body and appends `SCORING_METADATA`.

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
