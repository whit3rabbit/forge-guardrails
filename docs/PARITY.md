# Parity Contract

`forge-guardrails` is parity-tested against the upstream Python Forge behavior,
but Rust should not copy the whole Python eval platform.

## What Must Match

- Tool schema JSON and OpenAI tool formatting.
- Prompt-injected tool text and rescue parsing.
- Tool-call ID pairing with tool-result messages.
- Compacted transcripts must stay provider-valid. When historical Python
  behavior would preserve one side of a tool-call/tool-result pair, the fixture
  generator may apply a narrow, documented safety normalization instead.
- Retry, unknown-tool, step, and prerequisite nudge history.
- Proxy-visible behavior for no-tools passthrough, `respond` stripping, retry
  exhaustion text, and streaming final chunks.
- Backend wire separation for OpenAI-compatible, Ollama, and Anthropic clients.

Step nudge parity is covered at the workflow level. Proxy parity fixtures cover
client-visible handler behavior only and should not become a full guarded
workflow eval runner.

## What May Differ

- Latency and wall-clock timing.
- Token estimates when a backend does not report usage.
- Generated OpenAI response IDs.
- JSON object key order except where parity tests explicitly compare
  Python-style serialized schema strings.
- Provider metadata, rate limits, cache state, and cost estimates.

## Eval Roles

- Python evals are the live-backend oracle for large model/backend comparison.
- Rust parity tests are the deterministic CI gate.
- `forge-eval` is a small smoke runner for quick Rust-side checks.
- `forge-eval --num-ctx` keeps the smoke runner context budget explicit and
  mirrors that value into Ollama `num_ctx`; local server startup remains
  external.

The intended workflow is:

```bash
cargo test --test parity_tests
cargo test proxy::handler
cargo run --bin forge-eval -- --backend openai-proxy --base-url http://127.0.0.1:8081/v1 --model test-model --scenario basic_2step
python scripts/eval_openai_proxy.py --base-url http://127.0.0.1:8081/v1 --model test-model --scenario basic_2step --runs 1
```

Do not weaken parity assertions to make Rust pass. If Python behavior changes,
update `tests/parity/generate_fixtures.py`, regenerate
`tests/parity/fixtures/python_golden.json`, then update Rust assertions.
