//! Core orchestration engine, workflow definitions, and parsing types.

// THREE of these modules DRIVE a live backend -- they hold an `LLMClient`,
// spawn tasks and await responses -- so they belong to the `proxy` feature
// alongside `clients` itself. The rest are vocabulary (messages, tool specs,
// tool calls, step state, workflow shape) that the pure guardrail stack reads
// and that costs nothing to compile. See the `guardrails` feature in
// Cargo.toml.
/// LLM inference runner helper functions.
#[cfg(feature = "proxy")]
pub mod inference;
/// Message roles, meta metadata, and base Message container.
pub mod message;
/// The response vocabulary every guardrail speaks: tool calls, text
/// responses, and the union of the two.
pub mod response;
/// Multi-turn workflow execution runner.
#[cfg(feature = "proxy")]
pub mod runner;
/// Concurrent runtime worker for managing workflows in execution slots.
#[cfg(feature = "proxy")]
pub mod slot_worker;
/// In-progress workflow step tracking.
pub mod steps;
/// Tool schema and callable definition models.
pub mod tool_spec;
// `workflow` is EXECUTION rather than vocabulary despite the name: its
// `ToolCallable` is an async tool implementation returning a `BoxFuture`,
// so it needs `futures_util`. Nothing in the pure guardrail stack refers to
// `Workflow` or `ToolDef` -- the guardrails take a `ToolSpec` (the schema)
// and never the callable behind it.
/// Workflow structural definitions.
#[cfg(feature = "proxy")]
pub mod workflow;

#[cfg(feature = "proxy")]
pub use inference::{
    fold_and_serialize, format_tool_call_id, run_inference, InferenceResult, OnChunkFn,
};
pub use message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
pub use response::{LLMResponse, TextResponse, ToolCall};
#[cfg(feature = "proxy")]
pub use runner::{FinalResponseScoreFn, OnMessageFn, ToolCallScoreFn, WorkflowRunner};
#[cfg(feature = "proxy")]
pub use slot_worker::SlotWorker;
pub use steps::{PrerequisiteCheck, StepTracker};
pub use tool_spec::{ParamModel, ToolSpec};
#[cfg(feature = "proxy")]
pub use workflow::{ToolDef, Workflow};
