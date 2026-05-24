# docs/CLEANROOM.md — Clean-room run summary

## Purpose

This document records the clean-room exercise that produced the initial Rust implementation of `forge-guardrails`, followed by the parity review that aligned it fully with the Python reference.

The clean-room phase was a deliberate test of the [`whit3rabbit/clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill) workflow: source-reading roles are kept separate from implementation roles, and durable behavioral specifications are produced before any code is written. No Python source was consulted directly during implementation.

---

## Phase 1 — Clean-room implementation

**Task:** `task-f5968adf` — forge-guardrails Python-to-Rust migration  
**Result:** `spec-slice-complete` (all 8 units, 0 blocked, 0 contamination incidents)

| Metric                   | Value                 |
|--------------------------|-----------------------|
| Units completed          | 8 / 8                 |
| Total tests              | 487, all passing      |
| Rust source files        | 27 (6,571 code lines) |
| Rust test files          | 13 (4,073 test lines) |
| Total Rust lines         | 20,886                |
| Iterations used          | 8 / 30                |
| Contamination incidents  | 0                     |

**Duration:** ~3 hours 14 seconds

### Crate structure produced

- `src/error.rs` — `ForgeError`, `ToolResolutionError`, `BackendError`, `StreamError`
- `src/message.rs` — `MessageRole`, `MessageType`, `MessageMeta`, `ToolCallInfo`, `Message`
- `src/tool_spec.rs`, `src/workflow.rs`, `src/steps.rs` — `ToolSpec`, `Workflow`, `StepTracker`
- `src/streaming.rs` — `ChunkType`, `StreamChunk`, `LLMResponse` union
- `src/nudges.rs`, `src/prompts/` — nudge generators, `extract_tool_call`, rescue parsing
- `src/respond.rs` — built-in `respond` tool
- `src/hardware.rs`, `src/compact.rs`, `src/context.rs` — hardware detection, compaction, context manager
- `src/guardrails/` — `ErrorTracker`, `ResponseValidator`, `StepEnforcer`, `Guardrails` facade
- `src/client.rs`, `src/sampling.rs` — `LLMClient` trait, sampling defaults (16 model families)
- `src/backends/` — `AnthropicClient`, `LlamafileClient`, `OllamaClient`
- `src/inference.rs`, `src/runner.rs`, `src/slot_worker.rs` — core orchestration engine
- `src/server.rs`, `src/proxy.rs`, `src/handler.rs`, `src/http_server.rs` — HTTP server and OpenAI-compatible proxy

Audit artifacts preserved under `~/Documents/CleanRoom/task-f5968adf/`: preflight goal, task manifest, source index, 8 draft + 8 sanitized behavior specs, clean-run context, 8 implementation reports, 8 QC reports, handoff package, and clean-room result.

---

## Phase 2 — Parity review

After the clean-room exercise concluded, a full parity review was conducted against the Python reference implementation ([`antoinezambelli/forge`](https://github.com/antoinezambelli/forge)), which is included in this repository as the `forge/` git submodule.

The review verified behavioral alignment across all areas covered by the clean-room specs and extended coverage to the following:

- Python-golden fixture suite (`tests/parity/fixtures/python_golden.json`) generated directly from the Python reference and consumed by `tests/parity_tests.rs`
- Pydantic-style tool schema output and OpenAI tool formatting
- Prompt-injected tool text and nudge text
- Unknown-tool ordering, retry nudges, and rescue history
- Internal tool-call ID generation and tool-result pairing
- Step and prerequisite nudge metadata
- Tool resolution versus hard execution error budgets
- Tiered compaction phases
- Reasoning folding and provider request conversion
- Native malformed tool arguments in backend adapters
- Max-iteration diagnostics for pending steps
- Proxy client-visible behavior: no-tools passthrough, retry-exhaustion raw text, rescue success/failure, unknown-tool retry, `respond` stripping, mixed respond + real tool calls, streaming final chunk shape

**Outcome:** Full behavioral parity with the Python reference established. The parity test suite (`cargo test --test parity_tests`) is the ongoing regression gate for any behavior that must match Python exactly.

See `docs/PARITY.md` for the parity test contract and `docs/EVAL_GUIDE.md` for live-backend eval procedures.

---

## Attribution

- **Original Python project:** [`antoinezambelli/forge`](https://github.com/antoinezambelli/forge) — MIT licensed
- **Clean-room workflow:** [`whit3rabbit/clean-room-skill`](https://github.com/whit3rabbit/clean-room-skill)
- **This repository** is a Rust implementation inspired by the Python project's behavior, not a direct source translation. Preserve upstream attribution and review license compatibility before redistribution.