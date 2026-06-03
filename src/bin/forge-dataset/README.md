# `forge-dataset`

`forge-dataset` is the canonical private dataset tool for Forge tool-call
verifier training. It can:

- generate harmless tool-call prompt payloads,
- capture real local-model tool calls through the Forge proxy,
- review and verify captured candidates with MiniMax/OpenRouter,
- mine sanitized Codex/Claude agent logs through the temporary `generatetd`
  backend,
- assemble proxy and agent-log rows into one private notebook-ready dataset,
- validate generated JSONL files.

Do not use shell tools to create synthetic dataset scenarios. The capture path
uses deterministic in-memory stub tools with realistic schemas.

## Outputs Stay Private

Generated rows must remain private:

```json
{
  "private_agent_log": true,
  "public_export_allowed": false
}
```

Keep outputs under `target/dataset/` and do not commit generated datasets,
provider responses, review dumps, model artifacts, or local caches.

## One-Command Proxy Workflow

This starts managed `llama-server`, starts the Forge proxy, captures local model
tool calls, reviews them, and writes private training rows:

```bash
scripts/run_dataset_workflow.sh mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf \
  --provider openrouter \
  --verifier-provider minimax \
  --domains repo_docs,shopping,calendar,support,forge_eval \
  --runs 75 \
  --out-dir target/dataset/forge-eval-3k
```

Defaults are smoke-sized. With `repo_docs,shopping,calendar,support,forge_eval`,
there are 17 prompt contexts per run. `--runs 75` creates 1275 prompt contexts
and usually lands near a 2k-4k reviewed addendum after rejects.

Use capture-only when you want to inspect local model behavior before spending
reviewer tokens:

```bash
scripts/run_dataset_workflow.sh mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf \
  --capture-only \
  --domains support \
  --runs 1 \
  --out-dir target/dataset/capture-smoke
```

## Merged Proxy + Agent-Log Workflow

Opt in to sanitized local agent-log mining and assembly:

```bash
scripts/run_dataset_workflow.sh mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf \
  --provider openrouter \
  --verifier-provider minimax \
  --domains repo_docs,shopping,calendar,support,forge_eval \
  --runs 75 \
  --include-agent-logs \
  --agent-log-limit 1000 \
  --agent-log-synthetic-balanced 250 \
  --out-dir target/dataset/forge-eval-merged
```

This writes:

- `training.toolcall.jsonl`: proxy-reviewed rows.
- `agent_logs/tool_call_training.jsonl`: sanitized agent-log tool-call rows.
- `training.toolcall.combined.jsonl`: assembled proxy-first merged rows.
- `agent_training.notebook.jsonl`: notebook adapter.
- `dataset_manifest.json`: assembled run summary.
- `quarantine.jsonl`: invalid/private-policy rows.
- `conflicts.jsonl`: duplicate serialized inputs with conflicting labels.

Proxy rows are passed to `assemble` first, so real Forge proxy traffic remains
the backbone and wins exact duplicate precedence.

## Manual Commands

Generate prompt payloads only:

```bash
cargo run --bin forge-dataset -- prompts \
  --model test-model \
  --domains repo_docs,shopping,calendar,support,forge_eval \
  --runs 1 \
  --output target/dataset/run/tool_prompts.jsonl
```

Capture against an already running Forge proxy:

```bash
cargo run --bin forge-dataset -- capture \
  --proxy-base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --domains repo_docs,shopping,calendar,support,forge_eval \
  --runs 1 \
  --max-turns 4 \
  --output target/dataset/run/capture.jsonl
```

Review and verify captured rows:

```bash
cargo run --bin forge-dataset -- review \
  --input target/dataset/run/capture.jsonl \
  --output target/dataset/run/training.toolcall.jsonl \
  --provider openrouter \
  --openrouter-model openrouter/owl-alpha \
  --verifier-provider minimax \
  --concurrency 4
```

`review` appends accepted rows and rejects as it goes. If interrupted, resume
against the same output and sibling rejects file:

```bash
cargo run --bin forge-dataset -- review \
  --input target/dataset/run/capture.jsonl \
  --output target/dataset/run/training.toolcall.jsonl \
  --provider openrouter \
  --verifier-provider minimax \
  --resume
```

`--concurrency N` overlaps reviewer/verifier API calls for the main capture
pass. Start with `--concurrency 4`; raise it only if the providers do not
rate-limit. Targeted alternatives remain cap-ordered. If you only need reviewed
real model calls, use `--max-alternative-ratio 0` to skip the alternative pass.

Mine sanitized local agent logs:

```bash
cargo run --bin forge-dataset -- agent-logs \
  --out target/dataset/run/agent_logs \
  --provider openrouter \
  --verifier-provider minimax \
  --limit 1000 \
  --synthetic-balanced 250
```

No-API agent-log smoke:

```bash
cargo run --bin forge-dataset -- agent-logs \
  --no-api \
  --limit 25 \
  --project forge-rs \
  --no-claude \
  --out target/dataset/agent-logs-smoke
```

Assemble canonical tool-call inputs:

```bash
cargo run --bin forge-dataset -- assemble \
  --input target/dataset/run/training.toolcall.jsonl \
  --input target/dataset/run/agent_logs/tool_call_training.jsonl \
  --out-dir target/dataset/run \
  --combined-output training.toolcall.combined.jsonl
```

Validate outputs:

```bash
cargo run --bin forge-dataset -- validate \
  --input target/dataset/run/tool_prompts.jsonl \
  --input target/dataset/run/capture.jsonl \
  --input target/dataset/run/proxy_training_capture.jsonl \
  --input target/dataset/run/training.toolcall.combined.jsonl \
  --input target/dataset/run/quarantine.jsonl \
  --input target/dataset/run/conflicts.jsonl
```

## Providers And Models

`review` and `agent-logs` load `notebook/generatetd/.env` by default. Explicit
CLI flags win over shell environment, which wins over the env file.

```bash
MINIMAX_API_KEY=
OPENROUTER_API_KEY=
GENERATETD_MINIMAX_MODEL=MiniMax-M2.7
GENERATETD_OPENROUTER_MODEL=openrouter/free
```

Useful review flags:

- `--provider auto|minimax|openrouter`
- `--verifier-provider same|auto|minimax|openrouter`
- `--reviewer-model MODEL`
- `--verifier-model MODEL`
- `--openrouter-model MODEL`
- `--minimax-model MODEL`
- `--concurrency N`
- `--reviewer-api-key KEY`
- `--verifier-api-key KEY`

Useful agent-log flags:

- `--provider auto|minimax|openrouter|none`
- `--verifier-provider same|auto|minimax|openrouter|none`
- `--no-api`
- `--limit N`
- `--since YYYY-MM-DD`
- `--project TEXT`
- `--include-codex` / `--no-codex`
- `--include-claude` / `--no-claude`
- `--synthetic-balanced N`
- `--synthetic-missing-argument N`
- `--synthetic-tool-not-needed N`

For OpenRouter, prefer `openrouter/free` or a specific model whose metadata
supports structured outputs. Unsupported routes fall back to plain JSON
prompting plus local validation.

## Data Quality Rules

- Real captured calls are the backbone.
- Reviewer-corrected positives become `valid` only after verifier approval.
- Targeted alternatives stay capped by group and ratio.
- Same tool with wrong semantic values is `wrong_arguments_semantic`.
- Wrong competing tools must come from the same available tool set and be
  schema-valid for the wrong tool.
- Synthetic wrong-tool rows from unrelated fake tools stay disabled.
- Keep captured, corrected, and targeted alternatives in one
  `example_group_id`.

## Verification

Focused checks for this tool:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --bin forge-dataset
python3 -m py_compile notebook/generatetd/generatetd/*.py notebook/generatetd/tests/*.py
(cd notebook/generatetd && python3 -m pytest -q tests)
```

Smoke the merged path with existing local outputs:

```bash
cargo run --bin forge-dataset -- assemble \
  --input target/dataset/forge-eval-3k/training.toolcall.v2.jsonl \
  --input target/dataset/agent-logs-smoke/tool_call_training.jsonl \
  --out-dir target/dataset/merged-smoke \
  --combined-output training.toolcall.combined.jsonl
```
