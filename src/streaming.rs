use indexmap::IndexMap;
use serde_json::Value;
use std::fmt;

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
