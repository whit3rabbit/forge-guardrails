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

- `src/`: Rust library and binary implementation
  - `core/`: Core types and runner orchestration loop
    - `inference.rs`: Low-level LLM request/response parsing and role/reasoning folding
    - `message.rs`: Unified message format, roles, and metadata models
    - `runner.rs`: Multi-turn stateful executor enforcing the guardrails loop
    - `slot_worker.rs`: Handles slot allocation and queuing for backend runtimes
    - `steps.rs`: Required step tracking and workflow iteration constraints
    - `tool_spec.rs`: ToolSpec schemas, properties, and parameter conversion
    - `workflow.rs`: Target workflows, required steps, and prerequisites
  - `guardrails/`: Multi-tiered validator and step enforcement layer
    - `error_tracker.rs`: Monitors soft resolution and hard execution error budgets
    - `guardrails.rs`: Main facade pipeline orchestrating checks and retries
    - `history.rs`: Events timeline tracking validation results and violations
    - `nudge.rs`: Manages nudge states for retrying/rescuing tools
    - `policy.rs`: Allowed/blocked tools based on sequence prerequisites
    - `response_validator.rs`: JSON Schema verification and malformed argument checking
    - `step_enforcer.rs`: Sequences verification and prerequisite rules
  - `clients/`: Multi-provider backend adapters and settings
    - `base.rs`: `LLMClient` trait, streaming, and tool formatting definitions
    - `sampling.rs`: Model-specific sampling presets for 69+ models
    - `anthropic/`: Messages format translations, thinking extraction, and cache controls
    - `llamafile/`: Llamafile/llama.cpp server adapter, NDJSON stream parsing, and reasoning folding
    - `ollama/`: Ollama local adapter, options validation, and stream boundaries
    - `anyllm_proxy.rs`: AnyLlm Proxy / Runtime Client for unified provider translation
  - `context/`: Memory budget tracking and token compaction logic
    - `hardware.rs`: Apple Silicon plist / NVIDIA system memory context probing
    - `manager.rs`: Active token counts, budget warnings, and compaction triggers
    - `strategies.rs`: Slidewindow, phased compaction, and summary fallback strategies
  - `prompts/`: Nudge generation and raw text extraction
    - `nudges.rs`: System template builders for step, unknown tool, and error retries
    - `parse_strategies.rs`: Robust extraction of JSON/XML tool calls from chat text
  - `proxy/`: OpenAI-compatible and Anthropic-compatible proxy server
    - `handler.rs`: Route handling, stream intercepts, and result mapping
    - `proxy.rs`: Request translations, SSE chunk shaping, and respond stripping
    - `server.rs`: Lifecycle of the proxy daemon
  - `tools/`: Built-in terminal commands
    - `respond.rs`: Built-in terminal tool for concluding workflows
  - `bin/`: Execute runners
    - `forge-eval/`: Benchmark suite evaluating model performance locally
    - `forge-guardrails-proxy/`: HTTP daemon server entrypoint
- `tests/`: Extensive integration and regression test coverage
  - `parity/`: Python golden fixtures and generate scripts to preserve behavioral parity
- `scripts/`: Local dev tooling, launchers, and eval summaries
  - `start_llamaserver_proxy.sh`: Launches local llama-server backend using GGUF
  - `run_local_eval.sh`: Automates smoke/release suites, starting the proxy and oracle
  - `eval_openai_proxy.py`: Python oracle evaluating scenario completeness
  - `publish_docker.sh`: Publishes multi-arch docker builds to Docker Hub
- `docs/`: Comprehensive manuals and guides
  - `USER_GUIDE.md`: Developer guide to library APIs, configurations, and use cases
  - `WORKFLOW.md`: Explanation of guardrails loop and sequential step enforcement
  - `SCHEMA.md`: Data contracts, JSON specs, and internal `_forge` metadata structures
  - `MODEL_TRAINING_SCHEMA.md`: Tool-call and final-response verifier training, artifact, threshold, telemetry, and hard-negative schemas
  - `PARITY.md`: Specifications for maintaining strict parity with Python Forge
  - `CLEANROOM.md`: Summary of clean-room implementation history
  - `BACKEND_SETUP.md`: Setup configurations for local and remote providers
  - `EVAL_GUIDE.md`: Guide to setting up, running, and analyzing evaluation benchmarks
- `Dockerfile`: Single-port deployment configuration for production use cases
- `docker/entrypoint.sh`: Process supervisor for private sidecar and public proxy services

## Commands

Run before committing:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Use `cargo fmt --all` to apply formatting.

Focused eval/parity checks:

```bash
cargo test --test parity_tests
cargo test proxy::handler
cargo test server::tests
cargo test --bin forge-eval
python scripts/eval_openai_proxy.py --help
scripts/run_local_eval.sh --suite smoke --runs 1
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

Docker image and publish flow:

```bash
docker build -t forge-guardrails:local .
docker image inspect forge-guardrails:local --format '{{json .Config.ExposedPorts}}'
docker run --rm -p 8081:8081 \
  -e OPENAI_API_KEY=sk-... \
  -e FORGE_MODEL=gpt-4o-mini \
  forge-guardrails:local
```

The Docker image must expose only the Forge proxy port, `8081/tcp`. The
anyllm sidecar is an internal upstream hop and must not be published as a
client-facing port. Keep the entrypoint behavior aligned with that invariant.

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
```

Standard local release benchmark:

```bash
scripts/run_local_eval.sh --suite release --runs 10
```

ONNX classifier mode comparison:

```bash
scripts/run_local_eval.sh --suite release --runs 10 \
  --output-dir target/local-eval/release-baseline

scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx \
  --output-dir target/local-eval/release-onnx-shadow

scripts/run_local_eval.sh --suite release --runs 10 \
  --classifier-dir target/classifier-artifacts/onnx \
  --classifier-mode enforce \
  --output-dir target/local-eval/release-onnx-enforce
```

When evaluating a final-response verifier, include the matching
`--final-response-classifier-dir`, `--final-response-classifier-mode`, and
`--final-response-classifier-model` flags.

Use `--download-classifier` to populate
`target/classifier-artifacts/onnx` before a classifier run. The default
classifier mode is `shadow`, but ONNX evals should include `enforce` when
testing whether thresholds can safely improve behavior. Enforcement only acts
when the runtime mode is `enforce` and the artifact threshold for the predicted
label is met. Labels with thresholds above `1.0` remain telemetry-only even in
enforce mode, and `deterministic_invalid` must stay non-authoritative. ONNX
classifier artifacts are local test data and must not be committed. Compare
Python oracle reports for behavior, and inspect `rust_smoke.jsonl` plus proxy
logs for classifier scores.

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
real tool calls, and streaming final chunk shape.

## Agent rules

Keep changes small and behavior-driven. Align design and behavior with the Python codebase by checking the reference implementation in the `forge` submodule.

Refer to the [Python forge src](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/forge/src) as the gold standard for logic, defaults, and API shapes. Implement and verify all benchmark matrices/evaluation scenarios to ensure complete parity.

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
- Keep `TokenUsage` token-only. Surface anyllm provider metadata, rate limits, warnings, cache state, and estimated cost through `LLMClient::last_call_info()`.
- Treat anyllm pricing-derived cost as observability, not billing authority.

When changing workflow execution:
1. Add or update tests first.
2. Cover blocked steps, malformed tool calls, terminal tools, and mixed tool batches.
3. Check both non-streaming and streaming paths when touching backend clients.

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
