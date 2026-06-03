# `generatetd` Backend

`generatetd` is a transitional Python backend for `forge-dataset agent-logs`.
Use `forge-dataset` for normal dataset creation, review, assembly, validation,
and notebook-ready output. Keep direct `python -m generatetd ...` runs for
debugging log parsing, sanitizer behavior, provider review, and final-response
row generation.

Generated rows are private by default. Do not upload them to public Hugging
Face repos.

## Canonical Entry Point

Run agent-log mining through `forge-dataset`:

```bash
cargo run --bin forge-dataset -- agent-logs \
  --out target/dataset/run/agent_logs \
  --provider openrouter \
  --verifier-provider minimax \
  --limit 1000 \
  --synthetic-balanced 250
```

Then assemble proxy-reviewed rows with agent-log rows:

```bash
cargo run --bin forge-dataset -- assemble \
  --input target/dataset/run/training.toolcall.jsonl \
  --input target/dataset/run/agent_logs/tool_call_training.jsonl \
  --out-dir target/dataset/run \
  --combined-output training.toolcall.combined.jsonl
```

See `../../src/bin/forge-dataset/README.md` and
`../../docs/DATASET_WORKFLOW.md` for the full workflow.

## What This Backend Produces

Direct backend outputs:

- `tool_call_training.jsonl`: `toolcall-verifier-training/v1` rows.
- `final_response_training.jsonl`: `final-response-verifier-training/v1` rows.
- `agent_training.notebook.jsonl`: legacy notebook adapter.
- `manifest.json`: source, provider, label, synthetic, and split counts.
- `quarantine.jsonl`: skipped rows and reasons.
- `conflicts.jsonl`: duplicate scorer inputs with conflicting labels.
- `splits/*.jsonl`: stable group-hash train/validation/test splits.

`forge-dataset agent-logs` calls this backend with `--tool-calls-only` and
`--serializer v2`, so final-response rows stay out of the merged default path.
Generate final-response rows directly only when working on that verifier.

## Safety Contract

- Sanitize before any external API call.
- Never send raw logs, raw home paths, secrets, auth headers, emails, large code
  blocks, large diffs, or full file contents to providers.
- Quarantine rows with post-sanitize privacy findings.
- Keep generated outputs under `out/` or `target/dataset/`; do not commit
  outputs, `.cache/`, `.env`, `.venv/`, provider responses, or review dumps.
- Treat LLM labels as proposals. Local schema validation, privacy validation,
  dedupe, confidence thresholds, and optional verifier approval decide what
  reaches training output.

Rows derived from local agent logs must remain marked:

```json
{
  "private_agent_log": true,
  "public_export_allowed": false
}
```

## Setup For Direct Debugging

```bash
cd notebook/generatetd
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run backend tests before trusting backend changes:

```bash
python -m py_compile generatetd/*.py tests/*.py
pytest -q
```

## Direct Backend Examples

Private extraction only, no external API:

```bash
python -m generatetd generate \
  --no-api \
  --limit 100 \
  --out out/no-api-smoke
```

Tool-call-only reviewed rows:

```bash
python -m generatetd generate \
  --provider openrouter \
  --llm-review \
  --verify-review \
  --verifier-provider minimax \
  --tool-calls-only \
  --serializer v2 \
  --limit 100 \
  --out out/tool-call-debug
```

Final-response debugging:

```bash
python -m generatetd generate \
  --provider openrouter \
  --llm-review \
  --verify-review \
  --limit 100 \
  --out out/final-response-debug
```

Synthetic tool-call negatives:

```bash
python -m generatetd generate \
  --no-api \
  --limit 100 \
  --synthetic-balanced 75 \
  --tool-calls-only \
  --out out/synthetic-smoke
```

`--synthetic-balanced` splits across the active synthetic types:
`missing_argument` and `tool_not_needed`. Synthetic wrong-tool generation is
disabled because the old unrelated-tool distractor was noisy. Generate
wrong-tool rows through reviewed multi-tool contexts in `forge-dataset`.

## Provider Configuration

The backend loads `notebook/generatetd/.env` before reading provider keys or
default model names.

```bash
MINIMAX_API_KEY=
OPENROUTER_API_KEY=
GENERATETD_MINIMAX_MODEL=MiniMax-M2.7
GENERATETD_OPENROUTER_MODEL=openrouter/free
```

Useful flags:

- `--provider auto|minimax|openrouter|none`
- `--llm-review`
- `--verify-review`
- `--verifier-provider same|auto|minimax|openrouter|none`
- `--serializer v1|v2`
- `--tool-calls-only`
- `--include-codex` / `--no-codex`
- `--include-claude` / `--no-claude`
- `--project TEXT`
- `--since YYYY-MM-DD`
- `--limit N`
- `--synthetic-balanced N`
- `--synthetic-missing-argument N`
- `--synthetic-tool-not-needed N`
- `--api-max-attempts N`
- `--api-backoff-seconds SECONDS`

OpenRouter first requests strict JSON Schema output. If the selected route does
not support those parameters, the client logs `api fallback` and retries with
plain JSON prompting plus local validation.

## Quarantine Reasons

Common reasons:

- `privacy_findings_after_sanitize`: sanitizer failed closed.
- `llm_review_failed`: provider request, parsing, or schema failure.
- `review_not_training_tool_row`: reviewer chose quarantine or a non-tool label.
- `review_verifier_rejected`: verifier rejected the proposed row.
- `low_confidence_non_valid_review`: non-valid label below `0.85`.
- `needs_llm_review_for_failed_tool`: failed tool call needs review.

Do not train quarantined rows without a separate review pass.
