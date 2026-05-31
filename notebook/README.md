# Forge Verifier Training Notebooks

This directory contains the Colab-first verifier training notebook:

- `toolcall_verifier_training_production_colab_v4.ipynb`

The notebook trains the Forge tool-call verifier and, by default, the separate
final-response verifier. It is intended to run in Google Colab. Local notebook
execution is not a release gate; use local checks only for JSON/static syntax,
then run the notebook in Colab for training and artifact export.

## Current Private Tool-Call Dataset

The production notebook now defaults to the private Hugging Face dataset:

```text
cowWhySo/forge-toolcall-verifier-openrouter-2650-v1
```

Uploaded commit:

```text
https://huggingface.co/datasets/cowWhySo/forge-toolcall-verifier-openrouter-2650-v1/commit/95d668edb81130257b1d06f6175eb944aa8f3957
```

This dataset was generated with `notebook/generatetd` from sanitized local Codex
agent logs, then reviewed and verified with OpenRouter `openrouter/owl-alpha`.
It is private agent-derived data and must stay private unless a separate privacy
review explicitly approves public export.

Generation summary:

```text
schema_version: generatetd-manifest/v1
serializer: v1
provider: OpenRouterClient
review_model: openrouter/owl-alpha
verifier_model: openrouter/owl-alpha
tool_rows: 2650
final_response_rows: 0
quarantine: 851
conflicts: 0
train: 2323
validation: 195
test: 132
```

Tool-call labels:

```text
valid: 2077
wrong_tool_semantic: 247
wrong_arguments_semantic: 80
tool_not_needed: 246
```

Synthetic hard negatives in this dataset:

```text
missing_argument: 64
wrong_tool: 246
tool_not_needed: 246
```

Important limitations:

- It is a tool-call dataset only. `final_response_training.jsonl` is empty.
- It covers four labels. It does not add `needs_clarification` or
  `deterministic_invalid` examples.
- Most negative rows are synthetic, so treat it as preferred augmentation, not
  standalone production proof.
- Rows are marked `private_agent_log: true` and
  `public_export_allowed: false`.

## Notebook Defaults

The notebook loads the dataset from Hugging Face with these controls:

```python
ENABLE_FORGE_AGENT_HF_DATASET = True
FORGE_AGENT_HF_DATASET_REPO = "cowWhySo/forge-toolcall-verifier-openrouter-2650-v1"
FORGE_AGENT_HF_DATASET_FILE = "agent_training.notebook.jsonl"
FORGE_AGENT_HF_DATASET_WEIGHT = 2
PREFER_FORGE_AGENT_HF_DATASET = True
```

The loader uses `datasets.load_dataset(..., token=True)`, so Colab must have an
`HF_TOKEN` with access to the private dataset. The notebook keeps local agent
log ingestion opt-in via `INCLUDE_PRIVATE_AGENT_LOGS`; the private Hub dataset
is loaded through its own path.

`FORGE_AGENT_HF_DATASET_WEIGHT = 2` duplicates private Hub tool-call rows only
inside the training split, with stable weighted IDs and the original
`example_group_id`. Calibration, validation, and test splits stay unweighted.

`PREFER_FORGE_AGENT_HF_DATASET = True` preserves those private Hub rows before
sampling other sources during per-label class caps.

## Training Flow

The tool-call path combines:

1. Public function-calling sources.
2. Forge synthetic workflow rows.
3. Optional eval/proxy traces and hard negatives.
4. The private `generatetd` Hugging Face dataset above.

The notebook then:

1. Normalizes labels to the Forge six-label production schema.
2. Builds group-safe train/calibration/validation/test splits.
3. Trains a DeBERTa v3 sequence classifier.
4. Calibrates conservative per-label thresholds.
5. Exports PyTorch and ONNX artifacts.
6. Runs ONNX parity and optional quantization drift checks.
7. Trains and exports the final-response verifier if enabled.
8. Uploads private artifacts to the configured Hugging Face model repos.

Keep new verifier artifacts shadow-first until eval replay proves they are safe
to promote.
