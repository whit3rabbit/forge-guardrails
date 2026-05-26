# User Guide

Practical usage patterns for forge — from single-turn tool calling to multi-turn conversations in Rust.

For model and backend selection, see [MODEL_GUIDE.md](MODEL_GUIDE.md). For backend installation, see [BACKEND_SETUP.md](BACKEND_SETUP.md).

---

## Integration Modes

Forge's guardrail stack (retry nudges, step enforcement, error recovery, context compaction, VRAM budgeting) can be consumed in three ways in Rust. All three share the same underlying guardrail logic.

### At a glance

Each mode trades control for convenience. `WorkflowRunner` handles everything; the proxy applies guardrails transparently but drops workflow-level features; the middleware gives you building blocks and nothing else.

| Feature | WorkflowRunner | Proxy | Middleware |
|---------|:-:|:-:|:-:|
| Validation + rescue parsing | Yes | Yes | Yes |
| Retry nudges | Yes | Yes | Yes |
| Respond tool | Caller adds | Auto-injected | Caller adds |
| Step enforcement | Yes | Limited (`_forge`) | Yes (caller wires) |
| Prerequisites | Yes | No | Yes (caller wires) |
| Max iterations | Yes | Bounded by max_retries | Caller's responsibility |
| Context compaction | Yes | Yes | Caller wires `ContextManager` |
| Context threshold warnings | Yes | No | Caller wires `ContextManager` |
| Cancellation | Between iterations | Between retries | Caller's responsibility |
| Streaming (token-by-token) | Yes | No-tools live; tools buffered SSE | Caller's responsibility |
| Tool execution | Yes | No (client executes) | No (caller executes) |
| Callbacks (on_message, on_compact) | Yes | No | No |

The proxy is intentionally bare-bones — it applies response-quality guardrails (validation, rescue, retry, respond tool) without requiring workflow knowledge. Features like step enforcement and prerequisites require workflow structure that doesn't exist in the OpenAI chat completions API. See [Proxy design boundaries](#proxy-design-boundaries) for details.

### Mode 1: Standalone Runner (batteries included)

Forge owns the full agentic loop — LLM communication, guardrail policy, tool execution, and orchestration. You provide tools and a task, forge handles everything.

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use forge_guardrails::WorkflowRunner;

// The runner owns the LLM client and context manager
let runner = WorkflowRunner::new(
    Arc::new(client),
    Arc::new(Mutex::new(context_manager)),
    15,    // max_iterations
    3,     // max_retries_per_step
    2,     // max_tool_errors
    true,  // stream
    None,  // on_chunk callback
    None,  // on_message callback
    true,  // rescue_enabled
    None,  // retry_nudge override
);

let result = runner.run(&workflow, "What's the weather in Paris?", None, None, None).await?;
```

**Best for:** Projects where forge is the primary framework. Command-line tools, backend tasks, and applications built around forge from the start. See [Single-Turn Workflow](#single-turn-workflow) and [Multi-Turn Conversations](#multi-turn-conversations) below.

### Mode 2: Proxy Server (drop-in, zero code changes)

Forge sits between any OpenAI-compatible client and your model server, intercepting requests and applying guardrails transparently. The client doesn't know forge is there.

```bash
# External mode — you manage the backend
cargo run --bin forge-guardrails-proxy -- --backend-url http://localhost:8080 --port 8081

# Managed mode — forge starts llama-server and the proxy together
cargo run --bin forge-guardrails-proxy -- --backend llamaserver --gguf path/to/model.gguf --port 8081
```

Then point any client at the forge proxy instead of the model server. For example, pointing `LlamafileClient` (or any OpenAI client) at the proxy:

```rust
use forge_guardrails::LlamafileClient;

let client = LlamafileClient::new("path/to/model.gguf")
    .with_base_url("http://localhost:8081/v1");
```

**Best for:** Adding guardrails to existing tools without modifying them. Works with any tool that speaks the OpenAI-compatible API — no per-client wrappers needed.

**Reliability note:** The proxy automatically injects a synthetic `respond` tool when tools are present in the request. The model calls `respond(message="...")` instead of producing bare text, keeping it in tool-calling mode where forge's full guardrail stack applies. The `respond` call is stripped from the outbound response — the client sees a normal text response and never knows the tool exists. Client tools named `respond` are rejected because the name is reserved by forge. Guiding the model to a tool is a must. See [ADR-013](decisions/013-text-response-intent.md) for the full analysis.

Guarded tool-call validation failures return an upstream error by default after retries are exhausted. For debug compatibility, a request can set `_forge.return_raw_on_guardrail_failure: true` to return the rejected raw model text as a normal assistant message.

#### Proxy design boundaries

The proxy is intentionally bare-bones: it applies response-quality guardrails without requiring workflow knowledge. The following features are available in `WorkflowRunner` but not in the proxy, by design:

- **Step enforcement.** The OpenAI chat completions API has no standard workflow contract, but the proxy accepts a forge extension: `_forge.required_steps` and `_forge.terminal_tools`. Required steps are unordered completion gates: every required tool must have a successful tool result before any terminal tool is allowed. Use tool prerequisites in `WorkflowRunner` or middleware when one non-terminal tool must happen before another. Prior proxy tool results are treated as successful when `_forge.tool_status` is `"ok"`; without that metadata, results are counted as success unless the content starts with an explicit tool error prefix (such as `[ToolError]`, `[ToolResolutionError]`, `[ToolExecutionError]`, or `[tool_error]`).

- **Prerequisites.** Tool dependencies require workflow structure that does not exist in the OpenAI chat completions API. If you need prerequisites, use `WorkflowRunner` or the middleware directly.

- **Max iterations.** The proxy calls `run_inference` once per request. Each call is bounded at `max_retries + 1` LLM attempts (default 4). There is no outer loop — a runaway model cannot loop indefinitely. This is sufficient for the proxy's single-request model.

- **Real streaming.** No-tools passthrough streams live from the backend. Tool-using guarded requests accept `stream=true` and return SSE events, but client output is buffered until validation completes. Token-by-token client streaming during guarded inference would require validating partial responses, which is incompatible with guardrails that need complete responses (rescue parsing, retry nudges). The guardrail-first design is the proxy's value proposition. Streaming usage is emitted only on the final chunk, and only when `stream_options.include_usage` is `true`.

- **Context threshold warnings.** The proxy is stateless — the client sends the full conversation history in every request and decides what to include. Context pressure is the client's concern. Compaction still fires when the budget is exceeded.

- **Cancellation on disconnect.** Client disconnects are detected but do not cancel in-flight inference. This is the same granularity as `WorkflowRunner`, which checks the cancel channel between loop iterations but does not interrupt a running LLM call. The worst case is `max_retries + 1` wasted calls (default 4) for a disconnected client.

### Mode 3: Middleware (composable guardrails)

Import forge's guardrail components directly into your own orchestration loop. You own the loop, forge provides the reliability logic.

**Simple API** (one struct wrapping validation, error tracking, and step enforcement):

```rust
use forge_guardrails::{
    Guardrails, TerminalTool, GuardAction, Message, MessageRole, MessageMeta, MessageType
};

let mut guardrails = Guardrails::new(
    vec!["search".to_string(), "lookup".to_string(), "answer".to_string()],
    TerminalTool::Single("answer".to_string()),
    Some(vec!["search".to_string(), "lookup".to_string()]),
    None,  // tool_prerequisites
    3,     // max_retries
    2,     // max_tool_errors
    true,  // rescue_enabled
    3,     // max_premature_attempts
    None,  // retry_nudge callback override
);

// Inside your loop, after each LLM response:
let result = guardrails.check(&response);

match result.action {
    GuardAction::Retry | GuardAction::StepBlocked => {
        let nudge = result.nudge.as_ref().unwrap();
        messages.push(Message::new(
            nudge.role,
            &nudge.content,
            MessageMeta::new(MessageType::RetryNudge)
        ));
        continue;
    }
    GuardAction::Fatal => {
        panic!("Fatal guardrail error: {:?}", result.reason);
    }
    GuardAction::Execute => {
        let tool_calls = result.tool_calls.as_ref().unwrap();
        // Run the tools, then tell forge what succeeded:
        let success = execute(tool_calls).await;
        
        let tool_names: Vec<&str> = tool_calls.iter().map(|tc| tc.tool.as_str()).collect();
        let done = guardrails.record(&tool_names);
    }
}
```

**Granular API** (individual components for custom control):

```rust
use std::collections::HashSet;
use indexmap::IndexSet;
use forge_guardrails::{
    ResponseValidator, StepEnforcer, ErrorTracker, Message, MessageRole, MessageMeta, MessageType
};

let mut validator = ResponseValidator::new(
    vec!["search".to_string(), "lookup".to_string(), "answer".to_string()],
    true, // rescue_enabled
    None, // custom retry nudge function
);

let mut terminal_tools = IndexSet::new();
terminal_tools.insert("answer".to_string());
let mut enforcer = StepEnforcer::new(
    vec!["search".to_string(), "lookup".to_string()],
    terminal_tools,
    None, // tool_prerequisites
    3,    // max_premature_attempts
    2,    // max_prereq_violations
);

let mut errors = ErrorTracker::new(3, 2);

// Inside your loop:
let validation_result = validator.validate(&response);
if validation_result.needs_retry {
    errors.record_retry();
    let nudge = validation_result.nudge.unwrap();
    messages.push(Message::new(
        nudge.role,
        &nudge.content,
        MessageMeta::new(MessageType::RetryNudge),
    ));
    continue;
}

let tool_calls = validation_result.tool_calls.unwrap();
let step_check = enforcer.check(&tool_calls);
if step_check.needs_nudge {
    let nudge = step_check.nudge.unwrap();
    messages.push(Message::new(
        nudge.role,
        &nudge.content,
        MessageMeta::new(MessageType::StepNudge),
    ));
    continue;
}

for tc in &tool_calls {
    let ok = execute(tc).await;
    enforcer.record(&tc.tool, Some(&tc.args));
    errors.record_result(ok, false); // (success, is_resolution_error)
}
```

**What you own:** The middleware provides validation, rescue parsing, retry nudges, and step enforcement. Your loop is responsible for: iteration caps, cancellation, context management (including compaction and threshold callbacks), and streaming. These are handled automatically by `WorkflowRunner` but are intentionally left to the caller in middleware mode — the middleware is an advisory layer, not an execution engine.

**Best for:** Framework developers embedding forge's guardrails inside a custom agent, a proprietary pipeline, or another open-source framework.

### How they relate

```
forge_guardrails::guardrails/  <-- extracted guardrail logic
    ^                 ^
proxy server       WorkflowRunner
(proxy mode)       (standalone mode)
```

The middleware layer is the foundation. Both the proxy server and the standalone runner compose the same guardrail components internally. The proxy wraps them behind an OpenAI-compatible API. The runner wraps them in a complete agentic loop. The middleware exposes them as building blocks.

| | Standalone | Proxy | Middleware |
|---|---|---|---|
| Who owns the loop? | Forge | Forge (transparent) | You |
| Code changes needed? | Build on forge | Change one URL | Import + integrate |
| Works with existing tools? | No | Yes | Depends on integration |
| Best for | New projects | Existing toolchains | Framework developers |

---

## Concepts

A forge workflow has four main pieces:

- **Tools** — Async closures or functions the LLM can call, each described by a `ToolSpec` with a JSON Schema parameters definition.
- **Workflow** — A named bundle of tools, with optional `required_steps` (tools the LLM *must* call) and a `terminal_tool` (the tool or tools that end the workflow — accepts single string or vector).
- **Client** — An LLM backend adapter implementing the `LLMClient` trait (`OllamaClient`, `LlamafileClient`, `AnthropicClient`, `AnyLlmRuntimeClient`).
- **Runner** — `WorkflowRunner` drives the agentic loop: send messages, parse tool calls, execute tools, enforce guardrails, manage compaction.

---

## Single-Turn Workflow

A two-step weather workflow: look up weather, then report it.

```rust
use std::sync::Arc;
use std::path::Path;
use tokio::sync::Mutex;
use indexmap::IndexMap;
use serde_json::Value;
use futures_util::future::{BoxFuture, FutureExt};
use forge_guardrails::{
    Workflow, ToolDef, ToolSpec, TerminalToolInput, WorkflowRunner, LlamafileClient,
    setup_backend, BudgetMode, error::ToolError
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define async tool callables
    let get_weather = Arc::new(|args: IndexMap<String, Value>| {
        async move {
            let city = args.get("city").and_then(|v| v.as_str()).ok_or_else(|| {
                ToolError::Execution("Missing city argument".to_string())
            })?;
            Ok(Value::String(format!("72°F and sunny in {}", city)))
        }.boxed()
    });

    let report_weather = Arc::new(|args: IndexMap<String, Value>| {
        async move {
            let city = args.get("city").and_then(|v| v.as_str()).unwrap_or("unknown");
            let weather = args.get("weather").and_then(|v| v.as_str()).unwrap_or("unknown");
            Ok(Value::String(format!("Weather report for {}: {}", city, weather)))
        }.boxed()
    });

    // 2. Build the workflow tools map
    let mut tools = IndexMap::new();
    tools.insert(
        "get_weather".to_string(),
        ToolDef::new(
            ToolSpec::from_json_schema(
                "get_weather",
                "Get current weather for a city",
                &serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name" }
                    },
                    "required": ["city"]
                })
            ).unwrap(),
            get_weather
        )
    );

    tools.insert(
        "report_weather".to_string(),
        ToolDef::new(
            ToolSpec::from_json_schema(
                "report_weather",
                "Report the weather",
                &serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name" },
                        "weather": { "type": "string", "description": "Weather description" }
                    },
                    "required": ["city", "weather"]
                })
            ).unwrap(),
            report_weather
        )
    );

    // 3. Create the workflow structure
    let workflow = Workflow::new(
        "weather",
        "Look up weather and report it.",
        tools,
        vec!["get_weather".to_string()],
        TerminalToolInput::Single("report_weather".to_string()),
        "You are a weather helper. Use the weather tools to answer.",
    ).unwrap();

    // 4. Setup the backend (starts server process if needed, returns ServerManager and ContextManager)
    let (server, context) = setup_backend(
        "llamaserver", // backend
        None, // model name (only for ollama)
        Some(Path::new("path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf")), // gguf path
        None, // llamafile runtime path
        BudgetMode::ForgeFull,
        None, // manual tokens
        8080, // port
        "native", // mode
        &[], // extra flags
        None, // cache type k
        None, // cache type v
        None, // n_slots
        false, // kv_unified
    ).unwrap();

    // 5. Create client and runner
    let client = LlamafileClient::new("path/to/Ministral-3-8B-Instruct-2512-Q8_0.gguf")
        .with_recommended_sampling(true);

    let runner = WorkflowRunner::new(
        Arc::new(client),
        Arc::new(Mutex::new(context)),
        15,    // max_iterations
        3,     // max_retries_per_step
        2,     // max_tool_errors
        true,  // stream
        None,  // on_chunk callback
        None,  // on_message callback
        true,  // rescue_enabled
        None,  // retry_nudge override
    );

    // 6. Run the workflow
    let result = runner.run(&workflow, "What's the weather in Paris?", None, None, None).await?;
    println!("Workflow Result: {:?}", result);

    // ServerManager implements Drop and will automatically stop the server when it goes out of scope,
    // or you can stop it manually:
    server.stop().unwrap();

    Ok(())
}
```

### What happens under the hood

1. `setup_backend()` starts the server, detects available VRAM, and calculates a context budget.
2. `WorkflowRunner.run()` builds a system prompt describing the available tools.
3. The LLM calls `get_weather(city="Paris")` — forge executes it and feeds the result back.
4. Step enforcement verifies `get_weather` was called (it's in `required_steps`).
5. The LLM calls `report_weather(...)` — forge executes it, sees it's the `terminal_tool`, and ends the loop.
6. If any step fails: retry nudges, rescue loops, and error recovery kick in automatically. Step-enforcement and prerequisite violations surface as tool-error responses on the tool channel (the same wire shape models are trained on for "tool call failed, try again"); bare-text retry nudges still arrive as user messages.

---

## Multi-Turn Conversations

`WorkflowRunner` accepts an optional `on_message` callback that fires each time a `Message` is appended to the conversation during `run()`. This is the primary observability hook — use it for logging, eval metric collection, or building conversation history for multi-turn flows.

- **Single-turn (default):** `on_message` fires for every message the runner creates — system prompt, user input, assistant responses, tool results, nudges.
- **Multi-turn (`initial_messages`):** `run()` accepts an optional `initial_messages` parameter that seeds the conversation with prior history. `on_message` fires **only for new messages created during this turn**, not for the replayed history.

`WorkflowRunner` does not manage server lifecycle or track conversation history across `run()` calls — both are the consumer's responsibility.

```rust
use std::sync::{Arc, Mutex};
use forge_guardrails::{
    Message, MessageRole, MessageMeta, MessageType, WorkflowRunner, OllamaClient,
    setup_backend, BudgetMode, OnMessageFn
};

// 1. Start server once — stays up for the lifetime of the consumer
let client = OllamaClient::new("ministral-3:8b-instruct-2512-q4_K_M")
    .with_recommended_sampling(true);

let (server, context) = setup_backend(
    "ollama",
    Some("ministral-3:8b-instruct-2512-q4_K_M"),
    None,
    None,
    BudgetMode::ForgeFull,
    None,
    11434,
    "native",
    &[],
    None,
    None,
    None,
    false,
).unwrap();

// 2. Consumer owns the conversation history
let conversation = Arc::new(Mutex::new(Vec::<Message>::new()));

// Turn 0 — normal run, on_message collects everything (system prompt, user input, etc.)
let conversation_cb = conversation.clone();
let on_message_cb = Box::new(move |msg: &Message| {
    conversation_cb.lock().unwrap().push(msg.clone());
}) as OnMessageFn;

let runner = WorkflowRunner::new(
    Arc::new(client),
    Arc::new(Mutex::new(context)),
    15, 3, 2, true, None,
    Some(on_message_cb),
    true, None,
);

runner.run(&workflow, "first question", None, None, None).await?;

// Turn 1+ — seed with full history, append new user message
let turn_messages = Arc::new(Mutex::new(Vec::<Message>::new()));
let turn_messages_cb = turn_messages.clone();
let on_message_turn = Box::new(move |msg: &Message| {
    turn_messages_cb.lock().unwrap().push(msg.clone());
}) as OnMessageFn;

let client_ref = OllamaClient::new("ministral-3:8b-instruct-2512-q4_K_M")
    .with_recommended_sampling(true);

// Re-create runner with turn-specific callback
let runner_turn = WorkflowRunner::new(
    Arc::new(client_ref),
    runner.context_manager.clone(), // reuse same context
    15, 3, 2, true, None,
    Some(on_message_turn),
    true, None,
);

let mut seed = conversation.lock().unwrap().clone();
seed.push(Message::new(
    MessageRole::User,
    "follow-up question",
    MessageMeta::new(MessageType::UserInput)
));

runner_turn.run(&workflow, "follow-up question", None, Some(seed), None).await?;

// Append new messages to primary log
conversation.lock().unwrap().extend(turn_messages.lock().unwrap().clone());
```

The system prompt lives in `conversation` from turn 0 — it is not rebuilt or duplicated on subsequent turns. `StepEnforcer` and the runner's tool call counter reset each `run()` call since they are per-turn state.

### Long-Running Sessions: Filtering Transient Messages

`on_message` emits everything the runner creates during a turn, including transient retry artifacts — failed bare text responses, retry nudges, step nudges, and prerequisite nudges. This is by design: consumers get full visibility for logging and debugging.

For long-running sessions where conversation history persists across turns, these transient messages accumulate. The model sees its own past failures and corrective nudges on every subsequent turn, polluting effective context and degrading coherence — especially on smaller models (8-14B).

**Who's affected:** Any consumer that appends all `on_message` outputs to a persistent message list and reuses it via `initial_messages` on subsequent turns.

**Not affected:** Single-shot workflows, eval scenarios, or consumers that rebuild the message list from scratch each turn.

**Fix:** Filter transient message types before persisting. The metadata already tags these:

```rust
use forge_guardrails::{Message, MessageType};

let transient_types = vec![
    MessageType::RetryNudge,
    MessageType::StepNudge,
    MessageType::PrerequisiteNudge,
    MessageType::TextResponse,
];

// Inside your on_message callback:
if !transient_types.contains(&msg.metadata.msg_type) {
    conversation.lock().unwrap().push(msg.clone());
}
```

`TextResponse` is included because in tool-calling workflows, bare text is always a failed attempt that triggered a retry — the successful response comes as a `ToolCall`. Consumers using the respond tool for conversational replies should keep `TextResponse` in their persist list.

**Why not fix this in forge?** The runner's job is to emit everything — within a turn, retry nudges are useful (the model needs to see the nudge to self-correct). The distinction between "within a turn" and "across turns" is a consumer concern. Compaction handles context overflow but doesn't proactively clean up transient messages — it fires based on token budget pressure, not session hygiene.

---

## Choosing a Backend

See [BACKEND_SETUP.md](BACKEND_SETUP.md) for the supported-backend table, boot commands, and client snippets. [MODEL_GUIDE.md](MODEL_GUIDE.md) covers which model to pick.

### Sampling Parameters

Each model family has its own recommended temperature / top_p / top_k — and those recommendations differ substantially across families. Running everything at a single default is a measurable handicap for most models. Forge ships a per-model recommendations map that consumers opt into explicitly via a builder flag:

```rust
use forge_guardrails::LlamafileClient;

let client = LlamafileClient::new("path/to/Qwen3.5-27B-Q4_K_M.gguf")
    .with_mode("native")
    .with_recommended_sampling(true);
```

For local-server backends, the GGUF (or llamafile) path is the canonical model identity — its filename stem (e.g. `Qwen3.5-27B-Q4_K_M`) is what forge uses for sampling-defaults lookup, the wire-format `model` field, and JSONL eval rows. Use Ollama-style strings only with `OllamaClient`.

The flag is opt-in. Default behavior (`recommended_sampling=false`) leaves sampling to backend defaults; if forge has opinions about the model, it logs a one-shot INFO message pointing the caller at the flag. With `with_recommended_sampling(true)`, an unknown model returns an `UnsupportedModelError` during the call.

#### Proxy mode

The proxy does not consult the recommendations map. It plumbs whatever sampling params the inbound request body carries (OpenAI-compatible fields: `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `seed`) through to the backend on a per-call basis. The proxy's pre-built client is treated as a "blank slate" — body fields are the only sampling source.

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:27b-q4_K_M",
    "messages": [{"role": "user", "content": "hi"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5
  }'
```

To get recommended sampling in proxy mode, the calling client looks up `forge_guardrails::get_sampling_defaults(model)` and includes the values in the request body — the proxy is intentionally pure pass-through.

See [MODEL_GUIDE.md#sampling-parameters](MODEL_GUIDE.md#sampling-parameters) for the supported-models table, source citations, and override patterns.

---

## Context Management

Forge automatically manages the context window. When the conversation approaches the budget limit, tiered compaction fires:

- **Phase 1** — Summarize older tool results, keep recent messages intact.
- **Phase 2** — Compress mid-conversation exchanges, preserve system prompt and recent context.
- **Phase 3** — Aggressive compression, retain only system prompt and last few exchanges.

You can configure this via the `ContextManager`:

```rust
use forge_guardrails::{ContextManager, TieredCompact, NoCompact};

// Default: tiered compaction protecting the last 2 workflow iterations
let ctx = ContextManager::new(
    Box::new(TieredCompact::new(2)),
    8192, // budget in tokens
    None, // on_compact callback
    None, // context thresholds
    None, // on_context_threshold callback
);

// No compaction (for short workflows that won't hit the limit)
let ctx_no_compact = ContextManager::new(
    Box::new(NoCompact),
    8192,
    None, None, None
);
```

Or let `setup_backend()` handle it — it detects your VRAM and calculates the budget automatically.

---

## Guardrails

Forge's guardrail stack runs automatically. Each layer can be independently disabled (e.g., via ablation configuration in evals):

| Guardrail | What it does |
|-----------|-------------|
| **Step enforcement** | Verifies required tools were called before the terminal tool fires |
| **Prerequisites** | Enforces conditional tool dependencies (e.g. must read before edit) |
| **Retry nudges** | Prompts the LLM to try again when a tool call fails validation |
| **Rescue loops** | Recovers malformed tool calls from the LLM's text output |
| **Error recovery** | Re-prompts after tool execution errors instead of crashing |
| **Compaction** | Prevents context overflow in long conversations |

The eval harness measures each guardrail's contribution — see [EVAL_GUIDE.md](EVAL_GUIDE.md) for details.

---

## Tool Prerequisites

Tools can declare conditional dependencies — "if you call this tool, you must have called tool X first." This is enforced at runtime via nudge-and-retry, the same pattern as step enforcement.

```rust
use forge_guardrails::{ToolDef, PrerequisiteSpec};

// Name-only prerequisite: any prior call to read_file satisfies it
let edit_def = ToolDef::new(edit_spec, edit_file)
    .with_prerequisites(vec![
        PrerequisiteSpec::NameOnly("read_file".to_string())
    ]);

// Arg-matched prerequisite: must have called read_file with the same "path" argument
let edit_def_matched = ToolDef::new(edit_spec, edit_file)
    .with_prerequisites(vec![
        PrerequisiteSpec::ArgMatched {
            tool: "read_file".to_string(),
            match_arg: "path".to_string(),
        }
    ]);
```

If the model calls a tool without satisfying its prerequisites, the runner blocks the batch and emits one tool-error response per blocked tool call (`[PrereqError] ...` on the tool channel, with `PrerequisiteNudge` message type for compaction prioritization). The model retries off the canonical "tool failed" wire shape rather than a trailing user message — which is friendlier to tool-trained models. After consecutive violations exceed the budget (default 2), a `PrerequisiteError` is returned.

Prerequisites are not included in the tool schema — the model discovers constraints via the tool-error reply, same as step enforcement.

---

## Multiple Terminal Tools

Workflows can have multiple valid exit points. Pass a vector to `TerminalToolInput`:

```rust
use forge_guardrails::{Workflow, TerminalToolInput};

let workflow = Workflow::new(
    "system_control",
    "Perform tasks and exit",
    tools,
    vec![],
    TerminalToolInput::Multiple(vec![
        "set_ac".to_string(),
        "no_action".to_string()
    ]),
    "You are a control helper."
).unwrap();
```

Internally this is normalized to a `HashSet` for O(1) membership checks.

---

## Cancellation

`WorkflowRunner.run()` accepts an optional `watch::Receiver<bool>` channel for cooperative cancellation:

```rust
use tokio::sync::watch;
use forge_guardrails::error::ForgeError;

let (cancel_tx, cancel_rx) = watch::channel(false);

// In another task/thread to cancel the execution:
cancel_tx.send(true).unwrap();

match runner.run(&workflow, "task", None, None, Some(cancel_rx)).await {
    Ok(result) => println!("Success: {:?}", result),
    Err(ForgeError::WorkflowCancelled(e)) => {
        println!("Cancelled at iteration {}", e.iteration);
        println!("Completed steps: {:?}", e.completed_steps);
        println!("Messages so far: {}", e.messages.len());
    }
    Err(other) => println!("Error: {:?}", other),
}
```

The runner checks the event once per iteration, before the inference call. This is cooperative — if the model is mid-inference, the runner waits for it to finish before checking. The `WorkflowCancelledError` includes the full conversation state (as a vector of string contents) and completed steps for the caller to resume, discard, or log.

---

## SlotWorker — Shared Slot Access

`SlotWorker` serializes workflow execution on a single inference slot with priority-based queuing and auto-preemption. Use it when multiple callers need to share a slot — for example, a home assistant's specialist workflows (calendar, AC management, escalation) all sharing slot 1 while the main conversation runs on slot 0.

### Basic usage (FIFO)

```rust
use std::sync::Arc;
use forge_guardrails::{SlotWorker, WorkflowRunner};

// One runner pinned to a slot, wrapped by one worker
let runner = Arc::new(WorkflowRunner::new(...));
let worker = Arc::new(SlotWorker::new(runner));

// Run the worker task loop in the background:
let worker_clone = worker.clone();
tokio::spawn(async move {
    worker_clone.run().await;
});

// From anywhere — multiple concurrent callers submit tasks and get serialized:
let result_rx = worker.submit(
    Arc::new(workflow),
    "do the thing".to_string(),
    0, // priority (default 0)
    None // optional prompt variables
).await;

let result = result_rx.await.unwrap()?;
```

### Priority

Priority is an `i32` — lower values run first. Forge imposes no semantics; the consumer defines what the levels mean:

```rust
// Consumer defines their own levels
const USER: i32 = 0;
const ESCALATED: i32 = 1;
const ROUTINE: i32 = 2;

// User-initiated request — highest priority
let rx = worker.submit(calendar_wf, "what's on my schedule?".to_string(), USER, None).await;

// Background cron — lowest priority, can be preempted
let rx = worker.submit(ac_wf, "check temperature".to_string(), ROUTINE, None).await;
```

Without an explicit priority, tasks default to 0.

### Auto-preemption

If a higher-priority task is submitted while a lower-priority task is running, the running task is automatically cancelled and the higher-priority task takes over. The cancelled task's receiver yields a `WorkflowCancelled` error.

```rust
// Routine AC check is running (priority=2)...
// User asks about calendar (priority=0) — AC check is auto-cancelled
let rx = worker.submit(calendar_wf, "what's next?".to_string(), USER, None).await;
```

You can also cancel manually:

```rust
worker.cancel_current().await;  // cancels whatever is running
```

### Multi-slot architecture

For multi-slot setups (e.g., with `--kv-unified`), create one `SlotWorker` per shared slot. The main conversation slot typically doesn't need a worker — it's dedicated to one persistent session.

```rust
// Slot 0: main conversation (no worker needed — dedicated)
let main_client = LlamafileClient::new("path/to/model.gguf").with_slot_id(0);
let main_runner = WorkflowRunner::new(Arc::new(main_client), context, ...);

// Slot 1: shared specialist slot (needs a worker)
let service_client = LlamafileClient::new("path/to/model.gguf").with_slot_id(1);
let service_runner = Arc::new(WorkflowRunner::new(Arc::new(service_client), context, ...));
let service_worker = Arc::new(SlotWorker::new(service_runner));

let worker_clone = service_worker.clone();
tokio::spawn(async move {
    worker_clone.run().await;
});

// Tools can then route requests through this background queue
```
