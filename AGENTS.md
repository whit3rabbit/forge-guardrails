# AGENTS.md

## Purpose

This repo is `forge-guardrails`, a Rust implementation inspired by [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge), produced via the clean-room-skill workflow and subsequently verified for full behavioral parity with the Python reference.

It provides foundation types and runtime pieces for guarded LLM-agent workflows:
- workflow and step enforcement
- tool specs, tool execution, and terminal tools
- prompt rescue and tool-call parsing
- context tracking and compaction
- backend adapters for Anthropic, Llamafile, and Ollama
- anyllm runtime and sidecar client support for provider routing
- optional MLX eval support through OpenAI-compatible anyllm upstream routing
- Anthropic/OpenAI request translation through `anyllm_translate`
- OpenAI-compatible proxy/server surfaces

The reference Python implementation is available in the [forge](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/forge) git submodule. Use its `src/` directory as the gold standard for behavioral reference, structure, and details. Ensure that all benchmark matrix scenarios are implemented so we can guarantee complete alignment with the Python implementation.

## Core layout

```text
forge-guardrails (root)
├── docker/                             # Docker deployment & container supervisor configs
├── docs/                               # Developer manuals, architecture specs, user guides
├── forge/                              # Submodule pointing to Python Forge reference implementation
├── notebook/                           # Jupyter notebooks for classifier & verifier model training
├── scripts/                            # Release scripts, local run launchers, eval pipelines, setup
├── src/                                # Rust implementation source
│   ├── bin/                            # Binary entrypoints for proxy and CLI eval runner
│   │   ├── forge-eval/                 # Local benchmark/eval scenario execution suite
│   │   └── forge-guardrails-proxy/     # Production HTTP server daemon
│   ├── classifier_download/            # Helpers for checking, caching, and downloading verifier models
│   ├── clients/                        # Low-level multi-provider adapters (Anthropic, Llamafile, Ollama)
│   │   ├── anthropic/                  # Message payload mapper and cache control
│   │   ├── llamafile/                  # Llama.cpp backend adapter and stream parser
│   │   └── ollama/                     # Ollama local runtime adapter
│   ├── context/                        # Token budgets, memory manager, hardware probe, and compaction logic
│   ├── core/                           # Core guardrail loop orchestration, messages, slots, step definitions
│   ├── guardrails/                     # Schema validation, error budget tracking, policy enforcement, classifier scoring
│   ├── prompts/                        # Nudge generators, rescue text templates, and XML/JSON tool extraction logic
│   ├── proxy/                          # SSE streaming mapping, router routes, request/response translation
│   ├── server/                         # Proxy daemon daemon server controller lifecycle
│   ├── tool_output/                    # Standard & aggressive tool-result compaction, LZW compression
│   └── tools/                          # Default executor terminal tools (e.g. respond)
└── tests/                              # Integration & regression tests
    ├── fixtures/                       # Static mock files and target schema data
    ├── parity/                         # Generate scripts & fixtures comparisons for Python behavioral parity
    └── support/                        # Mock servers, test doubles, and shared assertions
```

### Directory Descriptions

- **`docker/`**: Contains Docker-related deployment artifacts, configuration files, and supervisor entrypoints (like `docker/entrypoint.sh`).
- **`docs/`**: Comprehensive developer and reference documentation, covering user guides, internal data contracts, schemas, verification strategies, compaction logic, and parity details.
- **`forge/`**: Git submodule pointing to the reference Python implementation (`antoinezambelli/forge`). Used as the gold standard for behavioral parity, validation test references, and benchmark matrices.
- **`notebook/`**: Google Colab-first Jupyter notebooks for verifier model training (e.g., tool-call classifiers).
- **`scripts/`**: DevOps scripts, release management, local runner tools, managed llama-server helpers, and performance evaluation tooling.
- **`src/`**: Main Rust source directory containing library components and binary entrypoints.
  - **`src/core/`**: Orchestrates runner loops and defines central data structures (messages, inference metadata, slots, workflows, steps, and tool specifications).
  - **`src/guardrails/`**: Coordinates workflow validations, step enforcement rules, response parsing, schema checks, retry loops, and classification/scoring executor pipelines.
  - **`src/clients/`**: Host adapters for individual LLM backends (Anthropic, Llamafile, Ollama, and anyllm router translator).
  - **`src/context/`**: Token usage budget tracking, hardware probes (M-series plist, NVIDIA CUDA memory checks), and phased/sliding-window memory compaction strategies.
  - **`src/prompts/`**: Formulates templates for agent retry/rescue nudges and implements XML/JSON extraction regexes/parsers.
  - **`src/proxy/`**: Server handlers translating OpenAI/Anthropic format payloads, managing SSE streaming, intercepts, and scoring.
  - **`src/server/`**: Server runner daemon, lifecycle orchestration, and health control.
  - **`src/tool_output/`**: Token compaction/compression utilities for prior tool results (Standard/Aggressive compression, LZW, and redaction logic).
  - **`src/tools/`**: Built-in runtime tool definitions executed inside the runner environment (such as `respond`).
  - **`src/classifier_download/`**: Controls download, verification, caching, and loading of verifier/classifier ONNX binaries.
  - **`src/bin/`**: Binaries, including the evaluation CLI runner (`forge-eval`) and the main daemon server (`forge-guardrails-proxy`).
- **`tests/`**: Integration, parity, and regression test suites.
  - **`tests/parity/`**: Fixture generation and test fixtures generated by the Python reference to enforce byte-for-byte behavior alignment.
  - **`tests/fixtures/`**: Static test files and mock message histories.
  - **`tests/support/`**: Shared test doubles, mock clients, and validation harnesses.

## Commands

Run before committing:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Use `cargo fmt --all` to apply formatting.

Makefile shortcuts:

```bash
make build
make build-release
make fmt-check
make clippy
make test
```

`make build`, `make build-release`, `make check`, `make test`, and
`make clippy` use `FEATURES=classifier` by default. Override with
`FEATURES=""` when a no-feature build is intentional.

Focused eval/parity checks:

```bash
cargo test --test parity_tests
cargo test --test classifier_tests
cargo test --test backend_streaming_tests
cargo test proxy::handler
cargo test server::tests
cargo test --bin forge-eval
python scripts/eval_openai_proxy.py --help
scripts/run_local_eval.sh --suite smoke --runs 1
make eval-smoke
make eval-smoke-classify
make eval-smoke-final-response
```

Verifier model training:

- The production notebook is
  `notebook/toolcall_verifier_training_production_colab_v4.ipynb`.
- The notebook is Google Colab-first. Do not use local notebook execution as a
  test or release gate. For local edits, validate only JSON/static syntax and
  Rust runtime compatibility, then run the notebook in Colab.
- `UPLOAD_TO_HUB=True` is the default. Keep uploads private unless explicitly
  told otherwise, and keep new verifier artifacts shadow-first until eval replay
  proves safety.
- `ENABLE_FORGE_AUGMENTATION=True` and
  `ENABLE_FINAL_RESPONSE_VERIFIER=True` are the production notebook defaults.
  Disable them only for smoke checks or controlled ablations.
- Keep the notebook's CUDA/object cleanup around the final-response verifier
  path. The final-response run loads a second classifier after the tool-call
  ONNX parity checks, so stale `Trainer`, model, tokenizer, tokenized dataset,
  logits, and scored dataframe objects must be released before it starts.
- Follow `docs/MODEL_TRAINING_SCHEMA.md` for row schemas, label order,
  artifact manifests, thresholds, calibration files, ONNX parity reports, and
  hard-negative envelopes.
- Tool-call artifacts may be legacy five-label or six-label. New production
  artifacts should use six labels in this exact order: `valid`,
  `wrong_tool_semantic`, `wrong_arguments_semantic`, `tool_not_needed`,
  `needs_clarification`, `deterministic_invalid`.
- `serialize_state_v1` must remain byte-stable and is the default deployable
  serializer. `serialize_state_v2` is an explicit metadata-aware ablation; do
  not silently publish v2 inputs as v1 artifacts.
- `deterministic_invalid` is telemetry-only. Rust deterministic validation and
  step enforcement remain authoritative.
- Final-response verifier artifacts are separate from tool-call classifier
  artifacts and use `final-response-verifier-artifact/v1` with
  `serialize_final_response_state_v1`.
- The notebook hard-negative loader consumes the enriched eval context
  envelope. `error_recovery` tool-call hard negatives are
  `wrong_arguments_semantic`, not `wrong_tool_semantic`, because the failed
  call uses the right tool with wrong semantic arguments.
- Do not commit ONNX classifier artifacts, downloaded model snapshots, Colab
  workdirs, Hugging Face caches, or generated `target/local-eval` outputs.

Dataset creation for tool-call verifier training:

- Use `src/bin/forge-dataset/` and `scripts/run_dataset_workflow.sh` to create
  private reviewed JSONL. Do not generate dataset rows through shell tools.
  The dataset tool uses harmless deterministic stub registries and real model
  tool calls through the Forge proxy.
- The one-command local workflow starts managed `llama-server`, starts the
  Forge proxy, writes prompt/capture JSONL, reviews with MiniMax/OpenRouter,
  and writes `toolcall-verifier-training/v1` rows:

```bash
scripts/run_dataset_workflow.sh mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf \
  --provider openrouter \
  --verifier-provider minimax \
  --domains repo_docs,shopping,calendar,support,forge_eval \
  --runs 75 \
  --out-dir target/dataset/forge-eval-3k
```

- `--runs` repeats every selected scenario. With
  `repo_docs,shopping,calendar,support,forge_eval`, there are 17 prompt rows
  per run; `--runs 75` creates 1275 prompt contexts and usually lands near a
  2k-4k reviewed private addendum after reviewer/verifier rejects.
- Reviewer and verifier keys are read from explicit flags first, then the shell
  environment, then `notebook/generatetd/.env`. Supported providers are
  `minimax` and `openrouter`; `--verifier-provider same` reuses the reviewer.
  For free OpenRouter review, prefer `openrouter/free` or a specific free model
  whose OpenRouter model metadata includes `structured_outputs`.
- `forge-dataset review` streams accepted rows as they are approved. If review
  is interrupted, resume against the same capture/output/reject files:

```bash
cargo run --bin forge-dataset -- review \
  --input target/dataset/forge-eval-3k/capture.jsonl \
  --output target/dataset/forge-eval-3k/training.toolcall.jsonl \
  --provider openrouter \
  --verifier-provider minimax \
  --resume
```

- Validate generated JSONL before using it for training:

```bash
cargo run --bin forge-dataset -- validate \
  --input target/dataset/forge-eval-3k/tool_prompts.jsonl \
  --input target/dataset/forge-eval-3k/capture.jsonl \
  --input target/dataset/forge-eval-3k/proxy_training_capture.jsonl \
  --input target/dataset/forge-eval-3k/training.toolcall.jsonl
```

- `training.toolcall.jsonl` is the canonical notebook input. The Colab
  notebook's private-HF default filename is `agent_training.notebook.jsonl`, so
  upload the generated file under that name or set
  `FORGE_AGENT_HF_DATASET_FILE`.
- Keep generated dataset outputs under `target/dataset/` private and do not
  commit them. Use `docs/DATASET_WORKFLOW.md` for the full contract, including
  capture-only mode, provider overrides, rejects, and validation details.

Local managed llama-server proxy launcher:

```bash
scripts/start_llamaserver_proxy.sh /path/to/mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf
```

Without an explicit path, the script looks for
`mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf` in `FORGE_MODELS_DIR`,
`MODELS_DIR`, the repo, nearby `models/` directories, `~/Models`,
`~/models`, and the HuggingFace cache. It verifies `llama-server` is on
`PATH`, checks that proxy port 8081 and backend port 8080 are available, then
starts `forge-guardrails-proxy --backend llamaserver --gguf <path>`. Override
ports with `FORGE_PROXY_PORT` and `FORGE_BACKEND_PORT`. The launcher must
reuse an existing proxy binary from `FORGE_PROXY_BIN`, `PATH`,
`CARGO_TARGET_DIR`, or `target/` before falling back to `cargo build`.

When changing `scripts/start_llamaserver_proxy.sh`, preserve Ctrl+C behavior:
the launcher must forward SIGINT/SIGTERM/SIGHUP to the proxy process and wait
for it to exit so the proxy can stop the managed `llama-server` backend.

Proxy tool-output compression:

```bash
forge-guardrails-proxy \
  --backend-url http://localhost:8080 \
  --tool-output-compression standard

forge-guardrails-proxy \
  --backend-url http://localhost:8080 \
  --tool-output-compression aggressive \
  --tool-output-compression-method auto
```

- `--tool-output-compression` accepts `disabled`, `safe`, `standard`, or
  `aggressive`. Default is `disabled`.
- `--tool-output-compression-method` accepts `lzw`, `repair`, or `auto`.
  Default is `lzw` and the method is used only by `aggressive`.
- Environment equivalents are `FORGE_TOOL_OUTPUT_COMPRESSION` and
  `FORGE_TOOL_OUTPUT_COMPRESSION_METHOD`.
- Request override lives under `_forge.tool_output_compression`; object form
  supports `mode`, `method`, `session_id`, `dedup`, `redact_secrets`, and
  `max_output_bytes`.
- Compression must remain opt-in and must mutate only prior tool-result
  content. Do not compress tool calls, tool IDs, tool names, tool arguments, or
  final responses. Preserve tool-call/tool-result pairing.
- Do not add runtime dependencies for compression without an explicit need.
  Current runtime compression is Rust/std plus existing dependencies.
- See `docs/COMPRESSION.md` for the full contract and examples.

Docker image and publish flow:

```bash
docker build -t forge-guardrails:local .
docker build -f Dockerfile.classifier -t forge-guardrails:classifier .
docker image inspect forge-guardrails:local --format '{{json .Config.ExposedPorts}}'
docker run --rm -p 8081:8081 \
  -e OPENAI_API_KEY=sk-... \
  -e FORGE_MODEL=gpt-4o-mini \
  forge-guardrails:local
```

The Docker image must expose only the Forge proxy port, `8081/tcp`. The
anyllm sidecar is an internal upstream hop and must not be published as a
client-facing port. Keep the entrypoint behavior aligned with that invariant.
The default `Dockerfile` must remain the normal no-classifier proxy image.
`Dockerfile.classifier` may preload the ONNX tool-call classifier artifact and
enable it by default, but it must still support `FORGE_CLASSIFIER_MODE=disabled`
at runtime. Do not bundle GGUF/provider LLM weights into either Docker image
unless explicitly asked.

Publish Docker Hub image `followthewhit3rabbit/forge-guardrails` only when
explicitly asked to publish:

```bash
docker login -u followthewhit3rabbit
scripts/publish_docker.sh
```

`scripts/publish_docker.sh` defaults to `VERSION=0.1.0`,
`IMAGE=followthewhit3rabbit/forge-guardrails`,
`PLATFORMS=linux/amd64,linux/arm64`, and
`BUILDER=forge-guardrails-builder`. Override those environment variables for a
different tag, registry, platform matrix, or buildx builder. The script pushes
both `${VERSION}` and `latest`.

Crates.io, GitHub Release, and Homebrew cask release flow:

```bash
git status --short --branch
git submodule status forge
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo package --locked
git push origin main
git tag v0.1.0
git push origin v0.1.0
```

- The release workflow runs on `v*.*.*` tags and verifies the tag matches
  `Cargo.toml`'s package version before publishing.
- Keep the `forge` submodule pinned to a commit that is fetchable from its
  configured remote. Do not release a parent commit that points at a local-only
  submodule commit.
- The workflow publishes the crate with `cargo publish --locked`, builds
  platform archives for `forge-guardrails-proxy`, publishes the GitHub release,
  and updates `whit3rabbit/homebrew-tap` when `HOMEBREW_TAP_TOKEN` is set.
- `HOMEBREW_TAP_TOKEN` must be an Actions secret on
  `whit3rabbit/forge-guardrails`, because this repo owns the release workflow.
  The token itself must have read/write `Contents` access to
  `whit3rabbit/homebrew-tap`. A token stored only on the tap repo is not visible
  to this workflow.
- Verify release secrets before tagging:

```bash
gh secret list --repo whit3rabbit/forge-guardrails
```

- Do not run `cargo publish` or manually edit the Homebrew cask unless the
  workflow fails and the user explicitly asks for manual recovery.
- If the tag release already published to crates.io, do not rerun the whole
  release workflow. Rerun only the `Update Homebrew Cask` job, otherwise
  `cargo publish` will fail because crates.io versions are immutable:

```bash
gh run view <release-run-id> --json jobs --jq '.jobs[] | {name,databaseId,status,conclusion}'
gh run rerun <release-run-id> --job <update-homebrew-cask-job-id>
```

- The Homebrew cask install command is:

```bash
brew install --cask whit3rabbit/tap/forge-guardrails-proxy
```

Regenerate Python parity fixtures after intentional reference-behavior changes:

```bash
uv run --project forge python tests/parity/generate_fixtures.py
```

The generated `tests/parity/fixtures/python_golden.json` file is checked in.
Normal Rust test runs consume that JSON and should not invoke Python.

## Python parity tests

The parity suite compares Rust behavior to synthetic golden outputs generated by
the Python reference submodule. The source of truth for fixture generation is
`tests/parity/generate_fixtures.py`; the checked-in output is
`tests/parity/fixtures/python_golden.json`; Rust assertions live in
`tests/parity_tests.rs`.

Use the parity suite for behavior that must match Python exactly, especially:
- Pydantic-style tool schema output and OpenAI tool formatting.
- Prompt-injected tool text.
- Unknown-tool ordering, retry nudges, and rescue history.
- Internal tool-call ID generation and tool-result pairing.
- Step and prerequisite nudge metadata.
- Tool resolution versus hard execution error budgets.
- Tiered compaction phases.
- Reasoning folding and provider request conversion.
- Native malformed tool arguments in backend adapters.
- Max-iteration diagnostics for pending steps.

When updating parity behavior:
1. Add or update the Python fixture first.
2. Regenerate `tests/parity/fixtures/python_golden.json`.
3. Add or update the matching Rust assertion in `tests/parity_tests.rs`.
4. Run `cargo test --test parity_tests` before broader repo gates.

Do not weaken parity assertions to make Rust pass. If a fixture fails, either
fix Rust to match Python or intentionally update the Python fixture because the
reference behavior changed. For byte-level JSON schema or tool output parity,
compare both normalized JSON and Python-style `json.dumps(...)` strings.

## Eval parity

Use the upstream Python eval scenarios as the live-backend oracle, but do not
port the Python dashboard/report platform into Rust unless explicitly asked.

Standard local smoke run:

```bash
scripts/run_local_eval.sh --suite smoke --runs 1
make eval-smoke
```

Standard local release benchmark:

```bash
scripts/run_local_eval.sh --suite release --runs 10
make eval-release
```

User-cache classifier benchmark, downloading the quantized tool-call artifact
if missing:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode shadow
make eval-release-classify
```

Final-response verifier shadow benchmark, downloading verifier artifacts if
missing:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode shadow \
  --verify-final-response \
  --final-response-classifier-mode shadow \
  --output-dir target/local-eval/release-onnx-final-shadow
make eval-release-final-response-shadow
```

ONNX classifier mode comparison:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --output-dir target/local-eval/release-baseline

scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode shadow \
  --output-dir target/local-eval/release-onnx-shadow

scripts/run_local_eval.sh --suite release --runs 10 \
  --classify \
  --classifier-mode enforce \
  --output-dir target/local-eval/release-onnx-enforce

make eval-release OUTPUT_DIR=target/local-eval/release-baseline
make eval-release-classify OUTPUT_DIR=target/local-eval/release-onnx-shadow
make eval-release-classify CLASSIFIER_MODE=enforce OUTPUT_DIR=target/local-eval/release-onnx-enforce
make eval-release-classify-shadow OUTPUT_DIR=target/local-eval/release-onnx-shadow
make eval-release-classify-advisory OUTPUT_DIR=target/local-eval/release-onnx-advisory
make eval-release-classify-enforce OUTPUT_DIR=target/local-eval/release-onnx-enforce
```

The Makefile eval targets enable `--resource-baseline` by default. Use
`RESOURCE_INTERVAL=...`, `RUNS=...`, `OUTPUT_DIR=...`,
`CLASSIFIER_MODE=...`, `FINAL_RESPONSE_CLASSIFIER_MODE=...`, and
`EVAL_ARGS="..."` to pass common overrides without editing the launcher.

Use `make eval-release-final-response` for release evals with both the
tool-call classifier and final-response verifier enabled. It uses
`CLASSIFIER_MODE=shadow` and `FINAL_RESPONSE_CLASSIFIER_MODE=shadow` by
default. Use `make eval-release-final-response-shadow` for the standard shadow
output directory `target/local-eval/release-onnx-final-shadow`.

For eval convenience, `scripts/run_local_eval.sh --classify` uses the same
user-facing cache as `forge-guardrails-proxy --classify-download`,
downloads/validates missing quantized tool-call files, records the resolved
`classifier_dir` in eval metadata, and passes that artifact to the proxy and
Rust smoke runner. `--classify` defaults to `advisory`; pass
`--classifier-mode shadow` for first replay baselines.

Use `--download-classifier` only when you intentionally want to populate
`target/classifier-artifacts/onnx` for eval/training artifact parity. The
default classifier mode is `shadow`, but ONNX evals should include `enforce`
when testing whether thresholds can safely improve behavior. Enforcement only
acts when the runtime mode is `enforce` and the artifact threshold for the
predicted label is met. Labels with thresholds above `1.0` remain
telemetry-only even in enforce mode, and `deterministic_invalid` must stay
non-authoritative. ONNX classifier artifacts are local test data and must not
be committed. Compare Python oracle reports for behavior, and inspect
`rust_smoke.jsonl` plus proxy logs for classifier scores.

Verifier promotion gates are intentionally conservative:

1. Start in `shadow`.
2. Move to `advisory` only after eval replay shows no completeness regression
   and no unacceptable valid-call false objections.
3. Move to `enforce` only after advisory replay proves the label-specific
   threshold is safe.

Minimum classifier replay matrix:

```text
no_classifier
classifier_fp32_onnx_shadow
classifier_quantized_onnx_shadow
classifier_fp32_onnx_advisory
classifier_quantized_onnx_advisory
```

Add final-response variants when evaluating grounded-synthesis recovery.

`scripts/run_local_eval.sh` starts
`scripts/start_llamaserver_proxy.sh mistralai_Ministral-3-8B-Instruct-2512-Q8_0.gguf`
on proxy port `8081`, waits for `/health`, runs the Rust `forge-eval` smoke
runner, runs the Python oracle wrapper, writes JSONL/report artifacts under
`target/local-eval/<timestamp>/`, and stops the proxy on exit or failure.

The release suite runs the published leaderboard scenarios with an `8192`
process budget and compares local results against the published
`Ministral-3-8B-Instruct-2512-Q8_0 LS/N [reforged]` row in
`forge/docs/results/raw/native-vs-prompt.md`. Compaction-chain scenarios are
not part of that published leaderboard; run them only when explicitly needed
with `--include-compaction-chain`.

Manual Python oracle against a running Rust proxy:

```bash
env -u VIRTUAL_ENV uv run --project forge python scripts/eval_openai_proxy.py \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 10 \
  --stream \
  --scenario basic_2step sequential_3step error_recovery \
  --budget-tokens 8192 \
  --output eval_results_rust_proxy.jsonl
```

Native Rust smoke runner:

```bash
cargo run --bin forge-eval -- \
  --backend openai-proxy \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 3 \
  --scenario basic_2step \
  --stream
```

The Rust smoke runner supports only the initial small scenario set:
`basic_2step`, `sequential_3step`, and `error_recovery`. It emits JSONL for
quick CI/smoke checks and should not grow into a reporting dashboard.

For release benchmarking, Python eval scenarios and Python scoring/reporting
are used to score Rust proxy output, then the result is compared with the
published Forge leaderboard. Do not require exact live JSONL or generated text
parity. Compare scenario coverage, aggregate score/completeness, and published
baseline tolerance. Latency, token counts, generated IDs, JSON key order outside
schema tests, provider metadata, and stochastic final wording are not release
gates.

Proxy parity must cover client-visible behavior: no-tools passthrough, empty
text for unexpected no-tools tool calls, retry-exhaustion raw text, rescue
success/failure, unknown-tool retry, `respond` stripping, mixed respond plus
real tool calls, streaming final chunk shape, and final chunk usage/call
metadata.

## Agent rules

Keep changes small and behavior-driven. Align design and behavior with the Python codebase by checking the reference implementation in the `forge` submodule.

Refer to the [Python forge src](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/forge/src/forge) as the gold standard for logic, defaults, and API shapes. Implement and verify all benchmark matrices/evaluation scenarios to ensure complete parity.

Preserve these invariants:
- Tool-call IDs and tool-result IDs must stay paired.
- Step enforcement must not leave invalid tool-call history behind.
- Compaction must not produce protocol-invalid transcripts.
- Backend adapters must keep their wire formats separate.
- Per-call sampling overrides must not mutate client defaults.
- Retry and rescue logic should nudge the model without hiding hard failures.
- Server/proxy code must clearly separate passthrough behavior from guarded workflow behavior.
- Forge owns interception and nudging. Do not route guarded traffic through `anyllm_proxy` HTTP handlers before forge validates it.
- Use `anyllm_translate` for Anthropic/OpenAI compatibility instead of hand-rolling request or response translation.
- Use `AnyLlmRuntimeClient` for in-process anyllm routing when possible. It must use `anyllm_proxy::runtime::ChatCompletionService`, not the axum router.
- Use `AnyLlmProxyClient` for a separate sidecar process when admin UI, cache, metrics, batch, or standalone provider config is needed.
- Treat MLX as an optional macOS eval path through an OpenAI-compatible server such as `mlx_lm.server`, routed by `AnyLlmRuntimeClient` or `AnyLlmProxyClient`.
- Do not treat MLX as a Python-parity backend or add it to `ServerManager` unless explicitly requested. Prefer llama-server/`LlamafileClient` for parity and top-eval reproduction.
- Do not assume GGUF-on-MLX behaves like llama.cpp GGUF. MLX GGUF support is model, architecture, and quantization dependent; document live smoke recipes instead of putting live MLX/GGUF requirements in CI.
- Unless a specific MLX server/model combination is qualified for native tool calls, evaluate MLX through Forge prompt/rescue guardrails rather than assuming llama-server `--jinja`-style native tool parity.
- Keep anyllm server-side tool execution out of Forge's guarded path. Forge must inspect tool calls before anything executes them.
- `LLMResponseEnvelope` and the final `StreamChunk` are the primary per-call metadata carriers. Treat `last_usage()`, `last_usage_details()`, and `last_call_info()` as compatibility and observability shims, not the primary source for accepted proxy or runner responses.
- Keep `TokenUsage` token-only. Surface anyllm provider metadata, rate limits, warnings, cache state, and estimated cost through `LLMResponseEnvelope::call_info`, `StreamChunk::call_info`, and compatibility `LLMClient::last_call_info()`.
- Async proxy and runner classifier paths must use `ScoringPipeline` / `ScoringExecutor`; do not call blocking scorer traits directly on Tokio worker threads.
- Treat anyllm pricing-derived cost as observability, not billing authority.

When changing workflow execution:
1. Add or update tests first.
2. Cover blocked steps, malformed tool calls, terminal tools, and mixed tool batches.
3. Cover classifier nudges through `ScoringPipeline` when touching scoring.
4. Check both non-streaming and streaming paths when touching backend clients.

When changing context or compaction:
1. Preserve system/user setup messages.
2. Keep recent workflow state coherent.
3. Do not drop one side of a tool-call/tool-result pair unless the whole group is summarized as inert text.

When changing backend clients:
1. Avoid shared assumptions between Anthropic, Ollama, and OpenAI-compatible backends.
2. Mock HTTP responses in tests.
3. Assert request bodies, not only parsed outputs.

When changing Python parity behavior:
1. Follow the Python parity test workflow above.
2. Keep fixture updates, regenerated JSON, and Rust assertions in the same change.
3. Run `cargo test --test parity_tests` before broader repo gates.

When changing eval/backend parity:
1. Update `docs/PARITY.md`, `docs/EVAL_GUIDE.md`, or `docs/BACKEND_SETUP.md`
   when the contract changes.
2. Keep the upstream `forge/` submodule source clean unless the task explicitly
   asks to patch upstream Python.
3. Prefer the Python proxy oracle for cross-language live checks and
   `forge-eval` for small Rust smoke checks.
4. Do not require exact parity for latency, generated OpenAI IDs, JSON key
   order outside explicit schema tests, provider metadata, or token estimates
   from backends that do not report usage.

## Current status notes

The initial clean-room run produced 8/8 units, 487 passing tests, and 0 contamination incidents. After that exercise, a full parity review was conducted against the Python reference to establish complete behavioral alignment. As the codebase evolved, the test suite expanded to 750 passing tests covering edge cases, client adapters, proxy handling, and compaction. See [`docs/CLEANROOM.md`](docs/CLEANROOM.md) for the full narrative.

Treat test counts as historical. Re-run the local test suite after any meaningful change.
