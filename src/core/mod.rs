//! Core orchestration engine, workflow definitions, and parsing types.

/// LLM inference runner helper functions.
pub mod inference;
/// Message roles, meta metadata, and base Message container.
pub mod message;
/// Multi-turn workflow execution runner.
pub mod runner;
/// Concurrent runtime worker for managing workflows in execution slots.
pub mod slot_worker;
/// In-progress workflow step tracking.
pub mod steps;
/// Tool schema and callable definition models.
pub mod tool_spec;
/// Workflow structural definitions.
pub mod workflow;

pub use inference::{
    fold_and_serialize, format_tool_call_id, run_inference, InferenceResult, OnChunkFn,
};
pub use message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
pub use runner::{FinalResponseScoreFn, OnMessageFn, ToolCallScoreFn, WorkflowRunner};
pub use slot_worker::SlotWorker;
pub use steps::{PrerequisiteCheck, StepTracker};
pub use tool_spec::{ParamModel, ToolSpec};
pub use workflow::{ToolDef, Workflow};
