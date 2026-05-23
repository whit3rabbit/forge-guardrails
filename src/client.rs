//! LLM client trait and supporting types.
//!
//! The `LLMClient` trait defines the async interface for LLM backend adapters.
//! Implementations handle sending messages and parsing responses. The client
//! does NOT implement retry logic; retry is the caller's responsibility.

use std::pin::Pin;

use futures_core::Stream;

use crate::streaming::{LLMResponse, StreamChunk};
use crate::tool_spec::ToolSpec;

/// Token counts from a single LLM response.
///
/// Populated from the server's usage field when available. Backends that
/// don't report usage leave this at zero and callers fall back to heuristic
/// estimation. Immutable once constructed.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

impl TokenUsage {
    pub fn new(prompt_tokens: i64, completion_tokens: i64, total_tokens: i64) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }
    }

    pub fn empty() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        }
    }
}

/// Sampling parameters passed to an LLM call.
///
/// Values are optional; when `None`, the backend or client instance defaults
/// apply. Per-call values take precedence over instance state for that call
/// only, without mutating the client.
pub type SamplingParams = serde_json::Map<String, serde_json::Value>;

/// Wire format identifier for message serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApiFormat {
    OpenAI,
    Ollama,
}

impl ApiFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::Ollama => "ollama",
        }
    }
}

/// Type alias for a boxed stream of `StreamChunk` items.
pub type ChunkStream =
    Pin<Box<dyn Stream<Item = Result<StreamChunk, crate::error::StreamError>> + Send>>;

/// Trait defining the interface for LLM backend adapters.
///
/// Implementations handle sending messages, parsing responses, and optionally
/// streaming partial output. The client does NOT retry; retry logic lives
/// externally.
#[allow(async_fn_in_trait)]
pub trait LLMClient: Send + Sync {
    /// Wire format identifier for message serialization.
    fn api_format(&self) -> ApiFormat;

    /// Send messages to the LLM backend and return a parsed response.
    ///
    /// Returns `LLMResponse::ToolCalls` if the model produced valid tool
    /// invocations, or `LLMResponse::Text` for text output. Per-call
    /// sampling values take precedence over instance state for this call
    /// only; the client's instance fields are not mutated.
    async fn send(
        &self,
        messages: Vec<serde_json::Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError>;

    /// Send messages and yield `StreamChunk` objects progressively.
    ///
    /// Yields TEXT_DELTA or TOOL_CALL_DELTA chunks progressively. The final
    /// chunk has type FINAL with a resolved `LLMResponse`. Per-call sampling
    /// values win over instance state without mutating self.
    async fn send_stream(
        &self,
        messages: Vec<serde_json::Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError>;

    /// Query the backend for its configured context window size.
    ///
    /// Returns `None` if unavailable.
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError>;
}

/// Convert a `ToolSpec` into the OpenAI-compatible tool schema format.
///
/// Returns a JSON object with `"type" = "function"` and a `"function"` key
/// containing name, description, and parameters JSON schema. Shared across
/// backends that use the OpenAI wire format.
pub fn format_tool(spec: &ToolSpec) -> serde_json::Value {
    use serde_json::{Map, Value};

    let mut outer = Map::new();
    outer.insert("type".to_string(), Value::String("function".to_string()));

    let mut func = Map::new();
    func.insert("name".to_string(), Value::String(spec.name.clone()));
    func.insert(
        "description".to_string(),
        Value::String(spec.description.clone()),
    );
    func.insert("parameters".to_string(), spec.get_json_schema());

    outer.insert("function".to_string(), Value::Object(func));
    Value::Object(outer)
}
