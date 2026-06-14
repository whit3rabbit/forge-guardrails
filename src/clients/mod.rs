//! Adapters and clients for connecting to various LLM provider backends.

/// Anthropic client implementation.
pub mod anthropic;
/// AnyLLM proxy and runtime client implementation.
pub mod anyllm_proxy;
/// Base client traits and shared data structures.
pub mod base;
/// Llamafile client implementation.
pub mod llamafile;
/// Ollama client implementation.
pub mod ollama;
/// Shared retry policy for transient upstream HTTP failures.
pub mod retry;
/// Model sampling defaults management.
pub mod sampling;

pub use anthropic::AnthropicClient;
pub use anyllm_proxy::{AnyLlmProxyClient, AnyLlmRuntimeClient};
pub use base::{
    format_tool, ApiFormat, ChunkStream, ChunkType, LLMCallInfo, LLMClient, LLMRateLimitInfo,
    LLMRequestOptions, LLMResponse, LLMResponseEnvelope, LLMUsageDetails, SamplingParams,
    StreamChunk, TextResponse, TokenUsage, ToolCall,
};
pub use llamafile::LlamafileClient;
pub use ollama::OllamaClient;
pub use sampling::{apply_sampling_defaults, get_sampling_defaults, MODEL_SAMPLING_DEFAULTS};
