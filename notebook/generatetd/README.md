# Generate Training Data

`generatetd` mines local Codex and Claude Code logs for tool-use examples, sanitizes them, optionally asks MiniMax or OpenRouter to review ambiguous cases, and emits Forge-compatible verifier training rows.

The generated rows are private by default. Do not upload them to public Hugging Face repos.

## What This Tool Produces

Primary outputs are Forge verifier training rows:

- `toolcall-verifier-training/v1` rows for tool-call classifier training.
- `final-response-verifier-training/v1` rows for final-response verifier training.
- `agent_training.notebook.jsonl`, an adapter file for the production notebook loader.

All rows derived from local agent logs are marked:

```json
{
  "private_agent_log": true,
  "public_export_allowed": false
}
```

## Safety Contract

- Sanitize before any external API call.
- Never send raw logs, raw home paths, secrets, auth headers, emails, large code blocks, large diffs, or full file contents to providers.
- Quarantine rows with post-sanitize privacy findings.
- Keep generated outputs under `out/`; do not commit `out/`, `.cache/`, `.env`, `.venv/`, or local review dumps.
- Treat LLM labels as proposals. Local schema validation, privacy validation, dedupe, confidence thresholds, and optional verifier approval decide what reaches training output.

## Setup

```bash
cd notebook/generatetd
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Smoke Run

```bash
python -m generatetd generate --no-api --limit 25 --out out/smoke
```

This extracts and validates sanitized rows without sending anything to an external API. Successful tool calls become private `valid` positives. Failed or ambiguous calls are quarantined unless `--llm-review` is enabled.

Run tests before trusting changes:

```bash
python -m py_compile generatetd/*.py tests/*.py
pytest -q
```

## LLM Review

MiniMax is tried first when `MINIMAX_API_KEY` is set. OpenRouter is used when `OPENROUTER_API_KEY` is set. The CLI loads `notebook/generatetd/.env` before reading provider keys or default model names.

Default models:

- MiniMax: `MiniMax-M2.7`
- OpenRouter: `deepseek/deepseek-v4-flash:free`

Local `.env`:

```bash
MINIMAX_API_KEY=
OPENROUTER_API_KEY=
GENERATETD_MINIMAX_MODEL=MiniMax-M2.7
GENERATETD_OPENROUTER_MODEL=deepseek/deepseek-v4-flash:free
```

```bash
export MINIMAX_API_KEY=...
python -m generatetd generate --out out/minimax --provider minimax --llm-review

export OPENROUTER_API_KEY=...
python -m generatetd generate --out out/openrouter --provider openrouter --llm-review
```

External review always receives sanitized excerpts. Rows that still look private after sanitization go to `quarantine.jsonl`.
Non-valid LLM labels below `0.85` confidence are also quarantined to avoid training on weak false positives.
OpenRouter first requests strict JSON Schema output. If the selected model has no endpoint that supports those parameters, the client logs an `api fallback` message and retries without `response_format`, relying on the prompt and local validator.

Add `--verify-review` to run a second LLM call as a training-row gate after generation. The verifier sees the sanitized transcript and the proposed decision, then approves or rejects the row. It does not generate new labels. By default it reuses the review provider; use `--verifier-provider openrouter|minimax|auto|none` to override.

```bash
python -m generatetd generate \
  --out out/openrouter-verified \
  --provider openrouter \
  --llm-review \
  --verify-review \
  --limit 25
```

For higher-quality rows, prefer using different providers or models for generation and verification when practical. That reduces correlated mistakes.

## Synthetic Negatives

Valid tool-call rows can be deterministically mutated into bounded hard negatives. These rows stay private and keep the original task group for leakage-safe splitting.

```bash
python -m generatetd generate \
  --no-api \
  --limit 100 \
  --synthetic-balanced 75 \
  --out out/synthetic-balanced-smoke
```

This splits the requested synthetic total evenly across the current synthetic types. For `75`, that means up to `25` each. If the number is not divisible by three, the remainder is assigned deterministically in this order: `missing_argument`, `wrong_tool`, `tool_not_needed`. Final counts may be lower after schema validation and dedupe.

Use per-type counts when you need exact control:

```bash
python -m generatetd generate \
  --no-api \
  --limit 100 \
  --synthetic-missing-argument 25 \
  --synthetic-wrong-tool 25 \
  --synthetic-tool-not-needed 25 \
  --out out/synthetic-smoke
```

`--synthetic-balanced` cannot be combined with per-type synthetic count flags.

Synthetic types:

- `missing_argument`: removes one argument and labels `wrong_arguments_semantic`.
- `wrong_tool`: swaps the candidate call to a synthetic distractor tool and labels `wrong_tool_semantic`.
- `tool_not_needed`: changes the request to a direct no-tool request and labels `tool_not_needed`.

Synthetic negatives are useful for coverage, but keep them capped. They should supplement real hard negatives, not dominate the dataset.

## Outputs

- `tool_call_training.jsonl`: canonical `toolcall-verifier-training/v1` rows.
- `final_response_training.jsonl`: canonical `final-response-verifier-training/v1` rows.
- `agent_training.notebook.jsonl`: adapter format consumed by the training notebook.
- `manifest.json`: run counts, sources, provider, and split counts.
- `quarantine.jsonl`: skipped rows and reasons.
- `conflicts.jsonl`: duplicate scorer inputs with conflicting labels.
- `splits/*.jsonl`: stable group-hash train/validation/test splits.

## Recommended Workflows

Private extraction only:

```bash
python -m generatetd generate --no-api --limit 100 --out out/no-api-smoke
```

OpenRouter reviewed and verified rows:

```bash
python -m generatetd generate \
  --provider openrouter \
  --llm-review \
  --verify-review \
  --limit 100 \
  --out out/openrouter-reviewed
```

Claude-only review:

```bash
python -m generatetd generate \
  --provider openrouter \
  --llm-review \
  --no-codex \
  --limit 100 \
  --out out/claude-only
```

Balanced smoke with synthetic negatives:

```bash
python -m generatetd generate \
  --no-api \
  --limit 100 \
  --synthetic-balanced 75 \
  --out out/synthetic-smoke
```

## Useful Flags

```bash
python -m generatetd generate \
  --out out/run \
  --provider minimax \
  --llm-review \
  --serializer v1 \
  --since 2026-05-01 \
  --project forge-rs
```

- `--include-codex` / `--no-codex`: include or skip Codex session logs.
- `--include-claude` / `--no-claude`: include or skip Claude Code project logs.
- `--project`: substring filter against cwd/project/source path.
- `--no-api`: force extraction, sanitizer, validation, dedupe, and split only.
- `--serializer v1|v2`: choose the Forge scorer serialization used for dedupe keys.
- `--verify-review`: run a second LLM gate before accepting reviewed rows.
- `--verifier-provider`: provider for `--verify-review`, default `same`.
- `--synthetic-balanced`: total synthetic hard negatives split evenly across current synthetic types.
- `--synthetic-missing-argument`: number of valid rows to mutate into missing-argument negatives.
- `--synthetic-wrong-tool`: number of valid rows to mutate into wrong-tool negatives.
- `--synthetic-tool-not-needed`: number of valid rows to mutate into tool-not-needed negatives.
- `--api-max-attempts`: total attempts per review request, default `4`.
- `--api-backoff-seconds`: initial retry delay, doubled on each retry, default `1.0`.

## Interpreting Quarantine

Common quarantine reasons:

- `privacy_findings_after_sanitize`: sanitizer failed closed. Do not train this row.
- `llm_review_failed`: provider request, parsing, or schema failure.
- `review_not_training_tool_row`: reviewer intentionally chose quarantine or a non-tool-call label.
- `review_verifier_rejected`: generator proposed a row but the verifier rejected it.
- `low_confidence_non_valid_review`: non-valid label below the local confidence threshold.
- `needs_llm_review_for_failed_tool`: failed tool call needs LLM review before becoming training data.

## Prompt Contract

The reviewer system prompt is:

```text
You review sanitized agent tool-use transcripts for Forge verifier training. Output exactly one JSON object and no other text. Do not include hidden reasoning. Do not infer hidden facts. Prefer needs_human_review over guessing.
```

Tool-call review asks for a Forge tool-call label, confidence, rationale, optional corrected call, and privacy warnings. Final-response review asks whether the terminal response is valid, omits required facts, contradicts tool output, invents unsupported claims, or fails to acknowledge a data gap.
