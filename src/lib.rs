pub mod clients;
pub mod context;
pub mod core;
pub mod error;
pub mod guardrails;
pub mod prompts;
pub mod proxy;
pub mod server;
pub mod tools;

// Legacy module-level re-exports for backwards compatibility and integration tests
pub use clients::base as client;
pub use clients::base as streaming;
pub use clients::sampling;
pub use context::hardware;
pub use context::strategies as compact;
pub use core::steps;
pub use core::workflow;
pub use prompts::nudges;
pub use tools::respond;

pub use clients::{
    apply_sampling_defaults, format_tool, get_sampling_defaults, AnthropicClient, ApiFormat,
    ChunkStream, ChunkType, LLMClient, LLMResponse, LlamafileClient, OllamaClient, SamplingParams,
    StreamChunk, TextResponse, TokenUsage, ToolCall, MODEL_SAMPLING_DEFAULTS,
};
pub use context::{
    default_context_warning, detect_hardware, CompactEvent, CompactStrategy, ContextManager,
    HardwareProfile, MemoryKind, NoCompact, SlidingWindowCompact, TieredCompact,
};
pub use core::{
    fold_and_serialize, format_tool_call_id, run_inference, InferenceResult, Message, MessageMeta,
    MessageRole, MessageType, OnChunkFn, OnMessageFn, ParamModel, PrerequisiteCheck, SlotWorker,
    StepTracker, ToolCallInfo, ToolDef, ToolSpec, Workflow, WorkflowRunner,
};
pub use error::{
    BackendError, BudgetResolutionError, ContextBudgetExceeded, ContextDiscoveryError, ForgeError,
    HardwareDetectionError, MaxIterationsError, PrerequisiteError, StepEnforcementError,
    StreamError, ThinkingNotSupportedError, ToolCallError, ToolExecutionError, ToolResolutionError,
    UnsupportedModelError, WorkflowCancelledError,
};
pub use guardrails::{
    CheckResult, ErrorTracker, GuardAction, Guardrails, Nudge, RetryNudgeFn, StepCheck,
    StepEnforcer, StepPrerequisite, TerminalTool, ValidationResult,
};
pub use prompts::{
    build_tool_prompt, extract_tool_call, prerequisite_nudge, rescue_tool_call, retry_nudge,
    step_nudge, unknown_tool_nudge,
};
pub use proxy::{
    extract_sampling, handle_chat_completions, has_respond_tool, openai_to_messages,
    respond_tool_openai, strip_respond_calls, text_response_to_openai, text_to_sse_events,
    tool_calls_to_openai, tool_calls_to_sse_events, HTTPServer, HandlerResult,
};
pub use server::{setup_backend, BudgetMode, ServerManager};
pub use tools::{respond_spec, respond_tool, RESPOND_TOOL_NAME};
