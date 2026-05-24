# Workflow

Visual guide to the forge agentic tool-calling loop.

---

## Quick Reference

**Entry Point:** `WorkflowRunner::run` in [runner.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/runner.rs)

**Critical Files:**
- [inference.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/inference.rs) - `run_inference()` — shared "front half" (compact, fold, validate, retry)
- [runner.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/runner.rs) - Agentic loop "back half" (step enforcement, tool execution, terminal check)
- [workflow.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/workflow.rs) - `Workflow`, `ToolDef`, `ToolSpec`, `ToolCall`, `TextResponse`
- [message.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/message.rs) - `Message`, `MessageRole`, `MessageType`, `MessageMeta`
- [steps.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/steps.rs) - `StepTracker` (used internally by `StepEnforcer`)
- [guardrails/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/) - Composable middleware (`ResponseValidator`, `StepEnforcer`, `ErrorTracker`, `Nudge`)
- [manager.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context/manager.rs) - `ContextManager`, `CompactEvent`
- [strategies.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context/strategies.rs) - `CompactStrategy` (`TieredCompact` 3-phase compaction)
- [base.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/base.rs) - `LLMClient` trait and streaming types
- [clients/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/) - Ollama, Llamafile, Anthropic, and AnyLLM (proxy/runtime) clients
- [server/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/) - `ServerManager`, `BudgetMode`, `setup_backend()`
- [respond.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools/respond.rs) - Synthetic respond tool (`respond_tool()`, `respond_spec()`)
- [nudges.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/prompts/nudges.rs) - Retry, unknown-tool, and step nudges
- [mod.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/prompts/mod.rs) - Prompt-injected tool prompt templates
- [parse_strategies.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/prompts/parse_strategies.rs) - JSON extraction and tool call rescue parser

---

## Agentic Loop

The core of forge. The runner delegates inference to `run_inference()` (the shared "front half" — compaction, reasoning folding, serialization, sending, validation, and retry), then handles step enforcement, tool execution, and terminal checks (the "back half"). The proxy also consumes `run_inference()` directly, sharing the same validation logic.

![Agentic Loop](assets/agentic_loop.svg)

---

## Message Lifecycle

Every message flows through three stages: creation (with metadata), API serialization (metadata stripped, reasoning folded), and compaction eligibility (prioritized by type).

![Message Lifecycle](assets/message_lifecycle.svg)

### Message Types and Compaction Priority

Messages are tagged with `MessageType` metadata that determines their compaction priority:

| MessageType | Role | Created By | Cut Order |
|-------------|------|-----------|-----------|
| `SystemPrompt` | system | Runner init | Never cut |
| `UserInput` | user | Runner init | Never cut |
| `ToolCall` | assistant | After LLM response | Never cut (all phases) |
| `ToolResult` | tool | After tool execution | Truncated P1, dropped P2 |
| `Reasoning` | assistant | Thinking models | Preserved through P2, dropped P3 |
| `TextResponse` | assistant | Failed tool call attempt | Preserved through P2, dropped P3 |
| `StepNudge` | tool | Runner step enforcement (`[StepEnforcementError]` prefix) | Dropped P1 |
| `PrerequisiteNudge` | tool | Runner prereq enforcement (`[PrereqError]` prefix) | Dropped P1 |
| `RetryNudge` | user | Runner retry logic (bare-text rescue path) | Dropped P1 |
| `Summary` | system | Compaction output | Never cut |

---

## Compaction Phases

`TieredCompact` applies three escalating phases. Each phase fires only if the previous didn't reduce tokens below the threshold. All phases are deterministic text manipulation — no LLM calls.

![Compaction Phases](assets/compaction_phases.svg)

**Protected window:** The `keep_recent` most recent loop iterations (default 2) are never compacted, regardless of phase. Only older messages in the eligible window are affected.

---

## Client Adapter Flow

The `LLMClient` trait abstracts backend differences. The runner never sees raw HTTP — it gets `Result<LLMResponse, BackendError>`. All clients also expose `get_context_length()` for budget discovery.

![Client Adapter Flow](assets/client_adapter_flow.svg)

### Streaming Flow

![Streaming Flow](assets/streaming_flow.svg)

---

## Budget Resolution

`ServerManager` resolves context budgets before the agentic loop starts. The budget flows into `ContextManager`, which uses it as the compaction threshold.

![Budget Resolution](assets/budget_resolution.svg)

---

## Module Structure

The modular architecture of the Rust crate features a clear separation of core agentic loops, guardrail policies, context managers, and LLM backend clients:

![Module Structure](assets/module_structure.svg)

---

## Data Types

### Core Types (`src/core/workflow.rs`)

```rust
pub struct Workflow {
    pub name: String,
    pub description: String,
    pub tools: IndexMap<String, ToolDef>,          // keyed by tool name
    pub required_steps: Vec<String>,               // must be called before terminal
    pub terminal_tools: HashSet<String>,           // normalized (designates workflow end)
    pub system_prompt_template: String,            // may contain {placeholders}
}

pub struct ToolDef {
    pub spec: ToolSpec,
    pub callable: ToolCallable,
    pub prerequisites: Vec<PrerequisiteSpec>,      // conditional dependencies
}

pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: ParamModel,                    // dynamic parameter validation schema
    pub json_schema: Option<Value>,                // cached Pydantic-compatible JSON Schema
}

pub enum ParamModel {
    String { description: Option<String>, required: bool, default: Option<Value>, enum_values: Option<Vec<String>> },
    Number { description: Option<String>, required: bool, default: Option<Value> },
    Boolean { description: Option<String>, required: bool, default: Option<Value> },
    Integer { description: Option<String>, required: bool, default: Option<Value> },
    Object { description: Option<String>, required: bool, properties: IndexMap<String, ParamModel> },
    Array { description: Option<String>, required: bool, items: Box<ParamModel> },
    Unsupported { type_name: String },
}

pub struct ToolCall {
    pub name: String,
    pub args: IndexMap<String, Value>,
    pub call_id: String,
}

pub struct TextResponse {
    pub content: String,                           // non-tool-call output
}
```

### Message Types (`src/core/message.rs`)

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallInfo {
    pub name: String,
    pub args: Option<IndexMap<String, Value>>,
    pub call_id: String,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,                         // System, User, Assistant, Tool
    pub content: String,
    pub metadata: MessageMeta,
    pub tool_name: Option<String>,                 // for role=Tool results
    pub tool_call_id: Option<String>,              // OpenAI-format correlation
    pub tool_calls: Option<Vec<ToolCallInfo>>,     // for assistant tool calls
}

#[derive(Debug, Clone, PartialEq)]
pub struct MessageMeta {
    pub msg_type: MessageType,                     // Compaction priority tag
    pub step_index: Option<i64>,
    pub original_type: Option<MessageType>,
    pub token_estimate: Option<i64>,
}
```

### Streaming Types (`src/clients/base.rs`)

```rust
pub struct StreamChunk {
    pub chunk_type: ChunkType,                     // TextDelta, ToolCallDelta, Final, Retry
    pub content: String,                           // partial text for deltas
    pub response: Option<LLMResponse>,             // only set when chunk_type == Final
}
```

---

## Command Reference

```bash
# Run all cargo tests (no live backend needed for parity suite)
cargo test

# Run code format checking
cargo fmt --all --check

# Run clippy rules checking
cargo clippy --all-targets -- -D warnings

# Execute native Rust eval smoke runner
cargo run --bin forge-eval -- \
  --backend openai-proxy \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 3 \
  --scenario basic_2step \
  --stream

# Run upstream Python eval scenarios against Rust proxy server
python scripts/eval_openai_proxy.py \
  --base-url http://127.0.0.1:8081/v1 \
  --model test-model \
  --runs 10 \
  --stream \
  --scenario basic_2step sequential_3step error_recovery \
  --output eval_results_rust_proxy.jsonl

# Regenerate parity golden fixtures from the Python submodule reference
uv run --project forge python tests/parity/generate_fixtures.py
```

See [ARCHITECTURE.md](ARCHITECTURE.md) or the reference implementation for more design decisions.
