# Forge Verifier Training Notebooks

This directory contains the Colab-first verifier training notebook:

- `toolcall_verifier_training_production_colab_v5.ipynb`

`toolcall_verifier_training_production_colab_v4.ipynb` is legacy. Keep new
training changes on v5 unless you are intentionally reproducing an older run.

The notebook trains the Forge tool-call verifier and, by default, the separate
final-response verifier. It is intended to run in Google Colab. Local notebook
execution is not a release gate; use local checks only for JSON/static syntax,
then run the notebook in Colab for training and artifact export.

## Current Private Tool-Call Dataset

The production v5 notebook defaults to this private Hugging Face dataset repo
and file:

```text
cowWhySo/forge-toolcall-verifier-openrouter-2650-v1
addenda/forge-eval-3k-v2/agent_training.notebook.jsonl
revision: 01eedcb861324df5fe5b6584ed4f12995b103d0f
```

Uploaded commit:

```text
https://huggingface.co/datasets/cowWhySo/forge-toolcall-verifier-openrouter-2650-v1/commit/01eedcb861324df5fe5b6584ed4f12995b103d0f
```

The addendum was generated with `forge-dataset` from local Forge eval proxy
captures, reviewed with OpenRouter `openrouter/owl-alpha` and MiniMax
`MiniMax-M2.7`, assembled with `--drop-conflicts`, and uploaded under a
versioned path so the older root dataset remains reproducible.

Addendum summary:

```text
source: target/dataset/forge-eval-3k/upload-clean
canonical rows: 724
notebook adapter rows: 724
duplicates removed: 2345
conflicts recorded: 33
conflicted inputs dropped: 16
quarantine: 0
```

Addendum labels:

```text
valid: 413
tool_not_needed: 241
wrong_arguments_semantic: 38
wrong_tool_semantic: 32
```

The older root file remains available:

```text
agent_training.notebook.jsonl
https://huggingface.co/datasets/cowWhySo/forge-toolcall-verifier-openrouter-2650-v1/commit/95d668edb81130257b1d06f6175eb944aa8f3957
```

Older root dataset details:

The root file was generated with `notebook/generatetd` from sanitized local
Codex agent logs, then reviewed and verified with OpenRouter
`openrouter/owl-alpha`. It is private agent-derived data and must stay private
unless a separate privacy review explicitly approves public export.

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
FORGE_AGENT_HF_DATASET_FILE = "addenda/forge-eval-3k-v2/agent_training.notebook.jsonl"
FORGE_AGENT_HF_DATASET_REVISION = "01eedcb861324df5fe5b6584ed4f12995b103d0f"
FORGE_AGENT_HF_DATASET_WEIGHT = 1
PREFER_FORGE_AGENT_HF_DATASET = True
```

The loader uses `datasets.load_dataset(..., token=True)`, so Colab must have an
`HF_TOKEN` with access to the private dataset. The notebook keeps local agent
log ingestion opt-in via `INCLUDE_PRIVATE_AGENT_LOGS`; the private Hub dataset
is loaded through its own path. `FORGE_AGENT_HF_DATASET_REVISION` pins the
uploaded addendum commit; clear it only when you intentionally want Colab to
follow the dataset repo's default revision.

`FORGE_AGENT_HF_DATASET_WEIGHT = 1` keeps private Hub tool-call rows unweighted.
If this is raised above `1`, only training rows are duplicated with stable
weighted IDs and the original `example_group_id`; calibration, validation, and
test splits stay unweighted.

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
