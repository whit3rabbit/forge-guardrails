# Dataset Capture Workflow

This workflow generates private tool-call verifier rows from real local-model
tool calls through the Forge proxy.

```text
forge-dataset prompts
  -> OpenAI-compatible messages + harmless tool schemas
llama-server
  -> forge-guardrails-proxy
forge-dataset capture
  -> deterministic stub tool execution
forge-dataset review
  -> MiniMax/OpenRouter reviewer
  -> MiniMax/OpenRouter verifier
  -> streaming toolcall-verifier-training/v1 JSONL
forge-dataset validate
  -> JSONL/schema sanity checks
```

Generated rows are private by default:

```json
{
  "private_agent_log": true,
  "public_export_allowed": false
}
```

## One-Command Run

```bash
scripts/run_dataset_workflow.sh \
  /path/to/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf \
  --provider openrouter \
  --verifier-provider same \
  --runs 1 \
  --out-dir target/dataset/openrouter-run
```

The script:

- starts `scripts/start_llamaserver_proxy.sh` in managed llama-server mode,
- sets `FORGE_TRAINING_CAPTURE_LOG` for proxy-side private capture,
- waits for `http://127.0.0.1:8081/health`,
- writes tool prompt payloads to `tool_prompts.jsonl`,
- captures model tool calls to `capture.jsonl`,
- reviews and verifies rows to `training.toolcall.jsonl`.

Use `--capture-only` or `--provider none` to skip external review.

## Tool-Call Prompts

Generate inspectable prompt payloads without starting a model:

```bash
cargo run --bin forge-dataset -- prompts \
  --model test-model \
  --domains repo_docs,shopping,calendar,support,forge_eval \
  --runs 1 \
  --output target/dataset/tool_prompts.jsonl
```

Each row contains the exact OpenAI-compatible request shape used by capture:

- `messages`: system + user request,
- `tools`: function tools sent to the proxy/model,
- `available_tools`: training-row tool specs, including Forge `respond`.

This is how tool calls are fed to the local model: the proxy receives normal
OpenAI `tools` definitions, then Forge validates and nudges model calls before
returning accepted tool calls to the capture client.

Use `forge_eval` for Forge-specific recovery slices:

- `fetch_records(count)` uses a zero-padded 4-digit string such as `0010`;
- `inspect_workflow_state()` is a valid no-argument call;
- `run_smoke_eval(...)` and `run_release_eval(...)` are similar real competing
  tools for hard wrong-tool examples;
- `diagnose_failure(...)`, `summarize_records(...)`, and `report_result(...)`
  exercise terminal/report/diagnostic distinctions.

Generated wrong-tool alternatives are intentionally conservative. They are
created only from verified-valid captured rows, use curated real competing tools
from the same captured scenario, must be schema-valid for the wrong tool, and
must pass both reviewer and verifier review.

## Manual Run

Terminal 1:

```bash
FORGE_TRAINING_CAPTURE_LOG=target/dataset/proxy_training_capture.jsonl \
scripts/start_llamaserver_proxy.sh \
  /path/to/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf
```

Terminal 2:

```bash
cargo run --bin forge-dataset -- capture \
  --proxy-base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 1 \
  --output target/dataset/capture.jsonl
```

Review with OpenRouter:

```bash
OPENROUTER_API_KEY=... \
cargo run --bin forge-dataset -- review \
  --input target/dataset/capture.jsonl \
  --output target/dataset/training.toolcall.jsonl \
  --provider openrouter \
  --verifier-provider same
```

Review with MiniMax:

```bash
MINIMAX_API_KEY=... \
cargo run --bin forge-dataset -- review \
  --input target/dataset/capture.jsonl \
  --output target/dataset/training.toolcall.jsonl \
  --provider minimax \
  --verifier-provider same
```

`review` appends verifier-approved rows as soon as each row is accepted. It no
longer keeps all accepted rows in memory until the end of the run. If review is
interrupted, resume against the same capture/output/reject files:

```bash
cargo run --bin forge-dataset -- review \
  --input target/dataset/capture.jsonl \
  --output target/dataset/training.toolcall.jsonl \
  --provider openrouter \
  --verifier-provider minimax \
  --resume
```

`--resume` skips capture candidates already present in either
`training.toolcall.jsonl` or the sibling rejects file. To retry previously
rejected reviewer/verifier failures with a different provider, use a fresh
output path or move the rejects file aside.

Validate JSONL files:

```bash
cargo run --bin forge-dataset -- validate \
  --input target/dataset/tool_prompts.jsonl \
  --input target/dataset/capture.jsonl \
  --input target/dataset/proxy_training_capture.jsonl \
  --input target/dataset/training.toolcall.jsonl
```

## Provider Configuration

`forge-dataset review` loads `notebook/generatetd/.env` by default, matching
the Python `generatetd` workflow. Model precedence is role-specific
`--reviewer-model`/`--verifier-model`, then provider-specific
`--openrouter-model`/`--minimax-model`, then shell environment, then the env
file, then built-in defaults.

Supported keys:

```bash
MINIMAX_API_KEY=
OPENROUTER_API_KEY=
GENERATETD_MINIMAX_MODEL=MiniMax-M2.7
GENERATETD_OPENROUTER_MODEL=openrouter/free
```

Provider selection:

- `--provider auto`: MiniMax when `MINIMAX_API_KEY` is set, otherwise
  OpenRouter when `OPENROUTER_API_KEY` is set.
- `--provider minimax`: official MiniMax-compatible chat completions path used
  by `notebook/generatetd`.
- `--provider openrouter`: OpenRouter chat completions with strict JSON Schema
  first, then fallback without `response_format` if the selected route rejects
  strict structured output.
- `--verifier-provider same`: reuse the reviewer endpoint/model.
- For free OpenRouter review, prefer `openrouter/free` or a specific free model
  whose OpenRouter model metadata includes `structured_outputs`. Free models
  without `response_format`/`structured_outputs`, such as some Poolside routes,
  will reject strict schema routing and fall back to plain JSON prompting.

Manual OpenAI-compatible endpoints are still possible:

```bash
cargo run --bin forge-dataset -- review \
  --reviewer-base-url http://127.0.0.1:9000/v1 \
  --reviewer-model reviewer \
  --verifier-base-url http://127.0.0.1:9001/v1 \
  --verifier-model verifier
```

## Outputs

Default one-command output files under `target/dataset/run/`:

- `tool_prompts.jsonl`: prompt/tool payloads used for capture.
- `capture.jsonl`: client-side real model tool-call candidates and stub results.
- `proxy_training_capture.jsonl`: proxy-side accepted candidate telemetry.
- `training.toolcall.jsonl`: verifier-approved training rows.
- `training.toolcall.rejects.jsonl`: malformed/rejected reviewer or verifier rows.

The proxy capture log is useful for auditing what Forge accepted. The canonical
training input is `training.toolcall.jsonl`.

## Recommended Generation Size

Do not change notebook weights, public downsampling, thresholds, or gates for
this recovery pass. Generate a small reviewed addendum first:

- `300` to `500` valid rows for each protected slice:
  no-argument valid calls, fixed-width numeric strings, and corrected
  error-recovery positives.
- `500` to `1000` paired Forge-specific wrong-tool groups, where each group has
  a verified valid row and at most `2` reviewed negatives.
- Keep generated alternatives capped at the default `33%` of accepted rows.
- Keep `synthetic_unrelated_tool` at zero. Use only real competing tools from
  multi-tool contexts or reviewed quarantine rows.

For the current notebook mix, a `2k` to `4k` private addendum is the right first
target. It is large enough to make validation/test slices visible, but small
enough to keep public data as the backbone under the existing
`FORGE_AGENT_HF_DATASET_WEIGHT=1` and
`FORGE_AGENT_HF_TRAIN_FRACTION_TARGET=0.25` settings.

`--runs` repeats every selected scenario with a new `example_group_id` per
scenario/run. Defaults stay smoke-sized:

```text
default domains: 12 prompt rows per run
with forge_eval: 17 prompt rows per run
```

Training rows are model- and review-dependent. A practical estimate for all
five domains is `30` to `45` accepted training rows per run after review,
including real positives, real bad calls, corrected positives, and capped
reviewed alternatives. For roughly `3000` rows, start with:

```text
--runs 75
```

Then inspect `training.toolcall.jsonl` and rerun with a higher or lower count if
the verifier reject rate is unusual.

Recommended capture command:

```bash
scripts/run_dataset_workflow.sh /path/to/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf \
  --provider openrouter \
  --verifier-provider minimax \
  --domains repo_docs,shopping,calendar,support,forge_eval \
  --runs 75 \
  --out-dir target/dataset/forge-eval-reviewed
```

The emitted `training.toolcall.jsonl` already matches
`toolcall-verifier-training/v1`. Feed it to the same notebook path as other
private agent rows. Do not upload it publicly.
