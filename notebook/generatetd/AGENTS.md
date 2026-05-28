# AGENTS.md

## Scope

These instructions apply to `notebook/generatetd/`.

This folder contains the local training-data generator for Forge verifier models. It reads Codex and Claude Code logs, sanitizes examples, optionally asks external LLM providers to label or verify them, and emits Forge-compatible private training rows.

## Invariants

- Privacy first. Sanitize before every external API call.
- Never send raw logs, raw home paths, secrets, auth headers, emails, large code blocks, large diffs, or full file contents to providers.
- Log-derived rows must remain private:
  - `private_agent_log=true`
  - `public_export_allowed=false`
- Do not weaken privacy checks, schema validation, dedupe, split-by-session/task grouping, retry/backoff, or quarantine behavior without a concrete reason.
- Do not commit generated outputs, local caches, `.env`, `.venv`, provider responses, or local review dumps.
- Keep synthetic negatives bounded and explicit. They are hard-negative supplements, not replacements for real failures.

## Files

- `generatetd/cli.py`: CLI flags and option wiring.
- `generatetd/pipeline.py`: extraction orchestration, row building, verification gate, dedupe, split writing, synthetic rows.
- `generatetd/parsers.py`: Codex and Claude Code JSONL parsers.
- `generatetd/sanitizer.py`: privacy redaction and output bounding.
- `generatetd/providers.py`: MiniMax/OpenRouter clients, retry/backoff, JSON parsing.
- `generatetd/prompts.py`: reviewer and verifier prompt contracts.
- `generatetd/serialization.py`: Forge-compatible scorer input serialization and hashes.
- `generatetd/validator.py`: row and tool-argument validation.
- `tests/`: synthetic parser, sanitizer, provider, and pipeline tests.

## External Providers

- MiniMax uses `MINIMAX_API_KEY` and defaults to `MiniMax-M2.7`.
- OpenRouter uses `OPENROUTER_API_KEY` and defaults to `deepseek/deepseek-v4-flash:free`, unless overridden by `.env` or CLI.
- The CLI loads local `.env`; shell environment variables take precedence.
- OpenRouter first tries strict JSON Schema output. If the selected model cannot route those parameters, the client falls back to prompt-only JSON and local validation.
- Provider failures must quarantine rows. Do not accept rows from malformed provider output unless parsing and schema validation prove the decision object is usable.

## Review And Verification

- Reviewer calls propose labels.
- Verifier calls gate proposed rows when `--verify-review` is enabled.
- The verifier should approve or reject; it should not be treated as a second generator.
- Non-valid labels below the local confidence threshold must quarantine.
- Prefer `needs_human_review` or quarantine over speculative labels.

## Synthetic Rows

Synthetic invalids are generated only from accepted `valid` tool-call rows and only when requested:

- `--synthetic-balanced`: total synthetic hard negatives split evenly across the current synthetic types before schema validation and dedupe. Do not combine with per-type synthetic count flags.
- `--synthetic-missing-argument`: removes one candidate argument and labels `wrong_arguments_semantic`.
- `--synthetic-wrong-tool`: swaps in a synthetic distractor tool and labels `wrong_tool_semantic`.
- `--synthetic-tool-not-needed`: changes the request to a direct no-tool request and labels `tool_not_needed`.

Do not add broad or random mutations. Each synthetic type must be deterministic, explainable, schema-valid, and counted in the manifest.

## Local Setup

```bash
cd notebook/generatetd
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Verification

Run these after meaningful changes:

```bash
python -m py_compile generatetd/*.py tests/*.py
pytest -q
```

Useful smoke commands:

```bash
python -m generatetd generate --no-api --limit 25 --out out/smoke
python -m generatetd generate --provider openrouter --llm-review --verify-review --limit 5 --out out/openrouter-verified-smoke
python -m generatetd generate --no-api --limit 25 --synthetic-balanced 15 --out out/synthetic-smoke
```

## Change Policy

- Keep edits small and behavior-driven.
- Add or update tests for parser behavior, provider response handling, sanitizer redaction, verifier gating, dedupe conflicts, and synthetic row generation.
- Preserve compatibility with `docs/MODEL_TRAINING_SCHEMA.md` and the production notebook adapter.
- Do not change public CLI output or row schemas casually; update README and tests when flags or output contracts change.
