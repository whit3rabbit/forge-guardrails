use super::nudge::Nudge;
use super::policy::{self, ArgValidationError};
use crate::clients::base::{LLMResponse, TextResponse, ToolCall};
use crate::core::tool_spec::ToolSpec;
use crate::prompts;
use crate::prompts::nudges;
use indexmap::{IndexMap, IndexSet};

/// Function type for generating custom retry nudge content.
pub type RetryNudgeFn = Box<dyn Fn(&str) -> String + Send + Sync>;

/// Result of ResponseValidator.validate(). Exactly one of tool_calls or nudge is set.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    /// Validated tool calls if the validation succeeded.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Instruction nudge if the validation failed and a retry is needed.
    pub nudge: Option<Nudge>,
    /// Whether a retry should be performed.
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
    tool_specs: IndexMap<String, ToolSpec>,
    rescue_enabled: bool,
    retry_nudge_fn: Option<RetryNudgeFn>,
}

impl ResponseValidator {
    /// Creates a new `ResponseValidator`.
    pub fn new(
        tool_names: Vec<String>,
        rescue_enabled: bool,
        retry_nudge_fn: Option<RetryNudgeFn>,
    ) -> Self {
        Self {
            tool_names: tool_names.into_iter().collect(),
            tool_specs: IndexMap::new(),
            rescue_enabled,
            retry_nudge_fn,
        }
    }

    /// Creates a schema-aware `ResponseValidator` from tool specs.
    pub fn from_tool_specs(
        tool_specs: Vec<ToolSpec>,
        rescue_enabled: bool,
        retry_nudge_fn: Option<RetryNudgeFn>,
    ) -> Self {
        let mut tool_names = IndexSet::new();
        let mut specs = IndexMap::new();
        for spec in tool_specs {
            tool_names.insert(spec.name.clone());
            specs.insert(spec.name.clone(), spec);
        }
        Self {
            tool_names,
            tool_specs: specs,
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
            if self.tool_names.is_empty() {
                return ValidationResult::valid(Vec::new());
            }
            let content = match &self.retry_nudge_fn {
                Some(f) => f(""),
                None => nudges::retry_nudge(""),
            };
            let nudge = Nudge::new("user", content, "retry");
            return ValidationResult::invalid(nudge);
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
        let arg_errors = policy::validate_tool_call_batch(calls, &self.tool_specs);
        if !arg_errors.is_empty() {
            let first_tool = arg_errors[0].tool.clone();
            let tool_errors: Vec<ArgValidationError> = arg_errors
                .into_iter()
                .filter(|error| error.tool == first_tool)
                .collect();
            let content = Self::invalid_arguments_nudge(&first_tool, &tool_errors);
            let nudge = Nudge::new("user", content, "invalid_arguments");
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

    fn invalid_arguments_nudge(tool_name: &str, errors: &[ArgValidationError]) -> String {
        let mut lines = Vec::with_capacity(errors.len() + 2);
        lines.push(format!("The call to {} has invalid arguments:", tool_name));
        for error in errors {
            lines.push(format!("- {}", error.message()));
        }
        lines.push("Retry with only this tool call and corrected arguments.".to_string());
        lines.join("\n")
    }
}
