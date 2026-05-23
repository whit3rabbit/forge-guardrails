pub mod backends;
pub mod client;
pub mod compact;
pub mod context;
pub mod error;
pub mod guardrails;
pub mod handler;
pub mod hardware;
pub mod http_server;
pub mod inference;
pub mod message;
pub mod nudges;
pub mod prompts;
pub mod proxy;
pub mod respond;
pub mod runner;
pub mod sampling;
pub mod server;
pub mod slot_worker;
pub mod steps;
pub mod streaming;
pub mod tool_spec;
pub mod workflow;

pub use backends::{AnthropicClient, LlamafileClient, OllamaClient};
pub use client::{format_tool, ApiFormat, ChunkStream, LLMClient, SamplingParams, TokenUsage};
pub use compact::{CompactStrategy, NoCompact, SlidingWindowCompact, TieredCompact};
pub use context::{default_context_warning, CompactEvent, ContextManager};
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
pub use handler::{handle_chat_completions, HandlerResult};
pub use hardware::{detect_hardware, HardwareProfile, MemoryKind};
pub use http_server::HTTPServer;
pub use inference::{
    fold_and_serialize, format_tool_call_id, run_inference, InferenceResult, OnChunkFn,
};
pub use message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
pub use nudges::{prerequisite_nudge, retry_nudge, step_nudge, unknown_tool_nudge};
pub use prompts::{build_tool_prompt, extract_tool_call, rescue_tool_call};
pub use proxy::{
    extract_sampling, has_respond_tool, openai_to_messages, respond_tool_openai,
    strip_respond_calls, text_response_to_openai, text_to_sse_events, tool_calls_to_openai,
    tool_calls_to_sse_events,
};
pub use respond::{respond_spec, respond_tool, RESPOND_TOOL_NAME};
pub use runner::{OnMessageFn, WorkflowRunner};
pub use sampling::{apply_sampling_defaults, get_sampling_defaults, MODEL_SAMPLING_DEFAULTS};
pub use server::{setup_backend, BudgetMode, ServerManager};
pub use slot_worker::SlotWorker;
pub use steps::{PrerequisiteCheck, StepTracker};
pub use streaming::{ChunkType, LLMResponse, StreamChunk, TextResponse, ToolCall};
pub use tool_spec::{ParamModel, ToolSpec};
pub use workflow::{ToolDef, Workflow};
