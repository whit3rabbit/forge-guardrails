use futures_core::Stream;
use indexmap::IndexMap;
use serde_json::Value;
use std::fmt;
use std::pin::Pin;

use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

/// Type of streaming chunk from the LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkType {
    TextDelta,
    ToolCallDelta,
    Final,
    Retry,
}

impl ChunkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TextDelta => "text_delta",
            Self::ToolCallDelta => "tool_call_delta",
            Self::Final => "final",
            Self::Retry => "retry",
        }
    }
}

impl fmt::Display for ChunkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A validated tool call response from the LLM client.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: Option<String>,
    pub tool: String,
    pub args: IndexMap<String, Value>,
    pub reasoning: Option<String>,
}

impl ToolCall {
    pub fn new(tool: impl Into<String>, args: IndexMap<String, Value>) -> Self {
        Self {
            id: None,
            tool: tool.into(),
            args,
            reasoning: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }
}

/// A non-tool-call text response from the LLM.
#[derive(Debug, Clone, PartialEq)]
pub struct TextResponse {
    pub content: String,
}

impl TextResponse {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

/// Union type representing an LLM response: either a list of tool calls
/// or a text response.
#[derive(Debug, Clone, PartialEq)]
pub enum LLMResponse {
    ToolCalls(Vec<ToolCall>),
    Text(TextResponse),
}

/// An immutable streaming chunk from the LLM.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamChunk {
    pub chunk_type: ChunkType,
    pub content: String,
    pub response: Option<LLMResponse>,
}

impl StreamChunk {
    pub fn new(chunk_type: ChunkType) -> Self {
        Self {
            chunk_type,
            content: String::new(),
            response: None,
        }
    }

    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    pub fn with_response(mut self, response: LLMResponse) -> Self {
        self.response = Some(response);
        self
    }
}

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
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, StreamError>> + Send>>;

/// Trait defining the interface for LLM backend adapters.
///
/// Implementations handle sending messages, parsing responses, and optionally
/// streaming partial output. The client does NOT retry; retry logic lives
/// externally.
#[allow(async_fn_in_trait)]
pub trait LLMClient: Send + Sync {
    /// Wire format identifier for message serialization.
    fn api_format(&self) -> ApiFormat;

    /// Get the token usage of the last request.
    fn last_usage(&self) -> Option<TokenUsage> {
        None
    }

    /// Send messages to the LLM backend and return a parsed response.
    ///
    /// Returns `LLMResponse::ToolCalls` if the model produced valid tool
    /// invocations, or `LLMResponse::Text` for text output. Per-call
    /// sampling values take precedence over instance state for this call
    /// only; the client's instance fields are not mutated.
    fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> impl std::future::Future<Output = Result<LLMResponse, BackendError>> + Send;

    /// Send messages and yield `StreamChunk` objects progressively.
    ///
    /// Yields TEXT_DELTA or TOOL_CALL_DELTA chunks progressively. The final
    /// chunk has type FINAL with a resolved `LLMResponse`. Per-call sampling
    /// values win over instance state without mutating self.
    fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> impl std::future::Future<Output = Result<ChunkStream, StreamError>> + Send;

    /// Query the backend for its configured context window size.
    ///
    /// Returns `None` if unavailable.
    fn get_context_length(
        &self,
    ) -> impl std::future::Future<Output = Result<Option<i64>, ContextDiscoveryError>> + Send;
}

/// Convert a `ToolSpec` into the OpenAI-compatible tool schema format.
///
/// Returns a JSON object with `"type" = "function"` and a `"function"` key
/// containing name, description, and parameters JSON schema. Shared across
/// backends that use the OpenAI wire format.
pub fn format_tool(spec: &ToolSpec) -> Value {
    let mut outer = serde_json::Map::new();
    outer.insert("type".to_string(), Value::String("function".to_string()));

    let mut func = serde_json::Map::new();
    func.insert("name".to_string(), Value::String(spec.name.clone()));
    func.insert(
        "description".to_string(),
        Value::String(spec.description.clone()),
    );
    func.insert("parameters".to_string(), spec.get_json_schema());

    outer.insert("function".to_string(), Value::Object(func));
    Value::Object(outer)
}
