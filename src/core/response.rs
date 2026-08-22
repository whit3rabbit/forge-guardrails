//! The response vocabulary every guardrail speaks: `ToolCall`, `TextResponse`
//! and the `LLMResponse` union of the two.
//!
//! **These live in `core` rather than in `clients::base`, where they were
//! defined, so that the pure guardrail stack can compile without the transport
//! half of this crate.** `clients::base` also holds `LLMClient`, `ChunkStream`
//! and their `futures_core::Stream` import; `prompts`, `guardrails::policy`,
//! `guardrails::response_validator`, `guardrails::guardrails` and
//! `tool_policy` all speak these three types and none of them needs a client.
//! Leaving them there made the `guardrails` feature impossible to express,
//! because gating `clients` gated the vocabulary with it.
//!
//! The line between this module and `clients::base` is TRANSPORT, not
//! usefulness. `LLMResponseEnvelope` stays behind: it carries token usage,
//! rate-limit headers and the provider's raw response, which only a live
//! backend can produce.
//!
//! `clients::base` re-exports all three, so `forge_guardrails::ToolCall` and
//! `forge_guardrails::clients::base::ToolCall` both still resolve and no
//! caller had to change.

use indexmap::IndexMap;
use serde_json::Value;

/// A validated tool call response from the LLM client.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for the tool call.
    pub id: Option<String>,
    /// Name of the tool being called.
    pub tool: String,
    /// Arguments passed to the tool.
    pub args: IndexMap<String, Value>,
    /// Optional chain-of-thought or reasoning text.
    pub reasoning: Option<String>,
}

impl ToolCall {
    /// Creates a new `ToolCall`.
    pub fn new(tool: impl Into<String>, args: IndexMap<String, Value>) -> Self {
        Self {
            id: None,
            tool: tool.into(),
            args,
            reasoning: None,
        }
    }

    /// Sets the tool call ID.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the reasoning content.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }
}

/// A non-tool-call text response from the LLM.
#[derive(Debug, Clone, PartialEq)]
pub struct TextResponse {
    /// The generated text content.
    pub content: String,
}

impl TextResponse {
    /// Creates a new `TextResponse`.
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
    /// List of tool calls.
    ToolCalls(Vec<ToolCall>),
    /// Plain text response.
    Text(TextResponse),
}
