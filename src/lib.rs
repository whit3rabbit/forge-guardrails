//! Guarded LLM-agent workflows with step and tool enforcement.

#![warn(missing_docs)]

/// Feature-gated Hugging Face classifier artifact download helpers.
#[cfg(feature = "classifier")]
pub mod classifier_download;
/// LLM backend client adapters and traits.
pub mod clients;
/// Context budget tracking and token compaction strategies.
pub mod context;
/// Core execution core components including message types and tool specifications.
pub mod core;
/// Custom error types for the framework.
pub mod error;
/// Response validation and step enforcement guardrails.
pub mod guardrails;
/// System and rescue prompt templates.
pub mod prompts;
/// HTTP and OpenAI-compatible proxy interface.
pub mod proxy;
/// In-process server backend manager.
pub mod server;
/// Tool-output compression for proxy-forwarded tool result messages.
pub mod tool_output;
/// Opt-in tool-call policy nudges for proxy-intercepted tool calls.
pub mod tool_policy;
/// Built-in tools like `respond`.
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

#[cfg(feature = "classifier")]
pub use classifier_download::{
    classifier_artifact_needs_download, default_cached_classifier_artifact_dir,
    default_classifier_cache_root, default_tool_call_classifier_artifact_dir,
    download_classifier_artifact_tree, ensure_classifier_artifact_dir, required_files_for_model,
    runtime_required_files, validate_classifier_artifact_dir, ClassifierArtifactKind,
    ClassifierDownloadPlan, ClassifierDownloadReport,
};
pub use clients::{
    apply_sampling_defaults, format_tool, get_sampling_defaults, AnthropicClient,
    AnyLlmProxyClient, AnyLlmRuntimeClient, ApiFormat, ChunkStream, ChunkType, LLMCallInfo,
    LLMClient, LLMRateLimitInfo, LLMRequestOptions, LLMResponse, LLMResponseEnvelope,
    LLMUsageDetails, LlamafileClient, OllamaClient, SamplingParams, StreamChunk, TextResponse,
    TokenUsage, ToolCall, MODEL_SAMPLING_DEFAULTS,
};
pub use context::{
    default_context_warning, detect_hardware, CompactEvent, CompactStrategy, ContextManager,
    HardwareProfile, MemoryKind, NoCompact, SlidingWindowCompact, TieredCompact,
};
pub use core::{
    fold_and_serialize, format_tool_call_id, run_inference, FinalResponseScoreFn, InferenceResult,
    Message, MessageMeta, MessageRole, MessageType, OnChunkFn, OnMessageFn, ParamModel,
    PrerequisiteCheck, SlotWorker, StepTracker, ToolCallInfo, ToolCallScoreFn, ToolDef, ToolSpec,
    Workflow, WorkflowRunner,
};
pub use error::{
    BackendError, BudgetResolutionError, ContextBudgetExceeded, ContextDiscoveryError, ForgeError,
    HardwareDetectionError, MaxIterationsError, PrerequisiteError, StepEnforcementError,
    StreamError, ThinkingNotSupportedError, ToolCallError, ToolExecutionError, ToolResolutionError,
    UnsupportedModelError, WorkflowCancelledError,
};
pub use guardrails::{
    final_response_top_k_from_logits, recent_errors_from_messages, score_final_response_async,
    score_tool_call_async, serialize_final_response_state_v1, serialize_state_v1,
    serialize_state_v2, tool_call_top_k_from_logits, validate_tool_arguments,
    validate_tool_call_batch, ArgValidationError, ArgValidationKind, ArtifactManifest,
    CandidateCallForScoring, CheckResult, ClassifierAction, ClassifierArtifact,
    ClassifierModelKind, ClassifierTopKEntry, ErrorTracker, FinalResponseClass,
    FinalResponseClassifierArtifact, FinalResponseContext, FinalResponseScore, FinalResponseScorer,
    FinalResponseToolResult, GuardAction, GuardrailDecision, GuardrailHistory, GuardrailState,
    GuardrailViolation, Guardrails, LabelThreshold, LabelsFile, NoopFinalResponseScorer,
    NoopToolCallScorer, Nudge, RetryNudgeFn, ScorerMode, ScoringContext, ScoringExecutor,
    ScoringMetadata, ScoringPipeline, StepCheck, StepEnforcer, StepPrerequisite, TerminalTool,
    Thresholds, ToolCallClass, ToolCallScore, ToolCallScorer, ToolSpecForScoring, ValidationResult,
    WorkflowStateForScoring, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_REVISION,
    DEFAULT_FINAL_RESPONSE_CLASSIFIER_REPO, DEFAULT_FINAL_RESPONSE_CLASSIFIER_REVISION,
    EXPECTED_LABELS, FINAL_RESPONSE_ARTIFACT_SCHEMA_VERSION, FINAL_RESPONSE_EXPECTED_LABELS,
    FINAL_RESPONSE_INPUT_SCHEMA_VERSION, FINAL_RESPONSE_SERIALIZER,
    FINAL_RESPONSE_THRESHOLDS_SCHEMA_VERSION, LEGACY_EXPECTED_LABELS, NEXT_INPUT_SCHEMA_VERSION,
    NEXT_SERIALIZER,
};
#[cfg(feature = "classifier")]
pub use guardrails::{
    OnnxFinalResponseScorer, OnnxScorerOptions, OnnxToolCallScorer, MAX_ONNX_SESSION_POOL_SIZE,
};
pub use prompts::{
    build_tool_prompt, classifier_nudge, extract_tool_call, prerequisite_nudge, rescue_tool_call,
    retry_nudge, step_nudge, unknown_tool_nudge, unsafe_batch_nudge,
};
pub use proxy::{
    extract_passthrough, extract_sampling, handle_anthropic_messages,
    handle_anthropic_messages_with_scorer, handle_anthropic_messages_with_scorers,
    handle_anthropic_messages_with_scorers_and_tool_controls,
    handle_anthropic_messages_with_scorers_and_tool_output_compression, handle_chat_completions,
    handle_chat_completions_with_scorer, handle_chat_completions_with_scorers,
    handle_chat_completions_with_scorers_and_tool_controls,
    handle_chat_completions_with_scorers_and_tool_output_compression, has_respond_tool,
    init_proxy_classifier_log_sink_from_env, openai_to_messages, respond_tool_openai,
    strip_respond_calls, text_response_to_openai, text_to_sse_events, tool_calls_to_openai,
    tool_calls_to_sse_events, AnthropicEventStream, AnthropicHandlerError, AnthropicHandlerResult,
    HTTPServer, HandlerError, HandlerResult, OpenAiEventStream, OpenAiMessageError,
};
pub use server::{setup_backend, BudgetMode, ServerManager};
pub use tool_output::{
    canonical_tool_name, compress_tool_output, detect_family, estimate_tokens,
    ToolOutputCompressionConfig, ToolOutputCompressionMethod, ToolOutputCompressionMode,
    ToolOutputCompressionResult, ToolOutputCompressionState, DEFAULT_MAX_DEDUP_ENTRIES_PER_SESSION,
    DEFAULT_MAX_DEDUP_SESSIONS, DEFAULT_MAX_OUTPUT_BYTES,
};
pub use tool_policy::{
    evaluate_tool_call_policy, ToolCallPolicyConfig, ToolCallPolicyMode, ToolCallPolicyNudge,
    ToolCallPolicyRequestState, DEFAULT_MAX_WRITE_PAYLOAD_BYTES,
};
pub use tools::{respond_spec, respond_tool, RESPOND_TOOL_NAME};
