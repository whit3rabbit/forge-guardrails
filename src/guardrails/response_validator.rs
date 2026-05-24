use super::nudge::Nudge;
use crate::clients::base::{LLMResponse, TextResponse, ToolCall};
use crate::prompts;
use crate::prompts::nudges;
use indexmap::IndexSet;

/// Function type for generating custom retry nudge content.
pub type RetryNudgeFn = Box<dyn Fn(&str) -> String + Send + Sync>;

/// Result of ResponseValidator.validate(). Exactly one of tool_calls or nudge is set.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    pub tool_calls: Option<Vec<ToolCall>>,
    pub nudge: Option<Nudge>,
    pub needs_retry: bool,
}

impl ValidationResult {
    /// Valid response with tool calls that passed validation.
    pub fn valid(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            tool_calls: Some(tool_calls),
            nudge: None,
            needs_retry: false,
        }
    }

    /// Invalid response requiring a retry with the given nudge.
    pub fn invalid(nudge: Nudge) -> Self {
        Self {
            tool_calls: None,
            nudge: Some(nudge),
            needs_retry: true,
        }
    }
}

/// Stateless validator for LLM responses. Validates tool call lists against
/// the allowed tool set, attempts rescue parsing from text content, and
/// produces retry nudges for invalid responses.
pub struct ResponseValidator {
    tool_names: IndexSet<String>,
    rescue_enabled: bool,
    retry_nudge_fn: Option<RetryNudgeFn>,
}

impl ResponseValidator {
    pub fn new(
        tool_names: Vec<String>,
        rescue_enabled: bool,
        retry_nudge_fn: Option<RetryNudgeFn>,
    ) -> Self {
        Self {
            tool_names: tool_names.into_iter().collect(),
            rescue_enabled,
            retry_nudge_fn,
        }
    }

    /// Validate a response. Returns ValidationResult with either valid tool calls
    /// or a retry nudge.
    pub fn validate(&self, response: &LLMResponse) -> ValidationResult {
        match response {
            LLMResponse::ToolCalls(calls) => self.validate_tool_calls(calls),
            LLMResponse::Text(text) => self.validate_text(text),
        }
    }

    fn validate_tool_calls(&self, calls: &[ToolCall]) -> ValidationResult {
        if calls.is_empty() {
            return ValidationResult::valid(Vec::new());
        }
        let unknown: Vec<&str> = calls
            .iter()
            .filter(|c| !self.tool_names.contains(&c.tool))
            .map(|c| c.tool.as_str())
            .collect();
        if let Some(&first_unknown) = unknown.first() {
            let available: Vec<&str> = self.tool_names.iter().map(|s| s.as_str()).collect();
            let content = nudges::unknown_tool_nudge(first_unknown, &available);
            let nudge = Nudge::new("user", content, "unknown_tool");
            return ValidationResult::invalid(nudge);
        }
        ValidationResult::valid(calls.to_vec())
    }

    fn validate_text(&self, text: &TextResponse) -> ValidationResult {
        if self.rescue_enabled {
            let available: Vec<&str> = self.tool_names.iter().map(|s| s.as_str()).collect();
            let rescued = prompts::rescue_tool_call(&text.content, &available);
            if !rescued.is_empty() {
                return ValidationResult::valid(rescued);
            }
        }
        let content = match &self.retry_nudge_fn {
            Some(f) => f(&text.content),
            None => nudges::retry_nudge(&text.content),
        };
        let nudge = Nudge::new("user", content, "retry");
        ValidationResult::invalid(nudge)
    }
}
