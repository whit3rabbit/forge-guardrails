use std::fmt;

/// Base error type for the forge-guardrails framework.
///
/// All framework errors except `ToolResolutionError` are represented as
/// variants of this enum, preserving catch-as-base semantics.
#[derive(Debug, thiserror::Error)]
pub enum ForgeError {
    #[error(transparent)]
    UnsupportedModel(#[from] UnsupportedModelError),
    #[error(transparent)]
    ToolCall(#[from] ToolCallError),
    #[error(transparent)]
    ToolExecution(#[from] ToolExecutionError),
    #[error(transparent)]
    WorkflowCancelled(#[from] WorkflowCancelledError),
    #[error(transparent)]
    MaxIterations(#[from] MaxIterationsError),
    #[error(transparent)]
    StepEnforcement(#[from] StepEnforcementError),
    #[error(transparent)]
    Prerequisite(#[from] PrerequisiteError),
    #[error(transparent)]
    ContextBudgetExceeded(#[from] ContextBudgetExceeded),
    #[error(transparent)]
    HardwareDetection(#[from] HardwareDetectionError),
    #[error(transparent)]
    ContextDiscovery(#[from] ContextDiscoveryError),
    #[error(transparent)]
    BudgetResolution(#[from] BudgetResolutionError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Stream(#[from] StreamError),
}

#[derive(Debug, thiserror::Error)]
pub struct UnsupportedModelError {
    pub model: String,
}

impl UnsupportedModelError {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl fmt::Display for UnsupportedModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unsupported model '{}'. Add sampling defaults or use non-strict mode.",
            self.model
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub struct ToolCallError {
    pub message: String,
    pub raw_response: Option<String>,
    pub cause: Option<String>,
}

impl ToolCallError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            raw_response: None,
            cause: None,
        }
    }

    pub fn with_raw_response(mut self, raw: impl Into<String>) -> Self {
        self.raw_response = Some(raw.into());
        self
    }

    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }
}

impl fmt::Display for ToolCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[derive(Debug, thiserror::Error)]
pub struct ToolExecutionError {
    pub tool_name: String,
    pub cause: String,
}

impl ToolExecutionError {
    pub fn new(tool_name: impl Into<String>, cause: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            cause: cause.into(),
        }
    }
}

impl fmt::Display for ToolExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tool '{}' execution failed: {}",
            self.tool_name, self.cause
        )
    }
}

/// A standalone error type that is NOT a subtype of ForgeError.
/// Raised by tool callables to signal non-fatal resolution failure.
#[derive(Debug, thiserror::Error)]
pub struct ToolResolutionError {
    pub message: String,
    pub tool_name: Option<String>,
}

impl ToolResolutionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            tool_name: None,
        }
    }

    pub fn with_tool_name(mut self, tool_name: impl Into<String>) -> Self {
        self.tool_name = Some(tool_name.into());
        self
    }
}

impl fmt::Display for ToolResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

#[derive(Debug, thiserror::Error)]
pub struct WorkflowCancelledError {
    pub messages: Vec<String>,
    pub completed_steps: indexmap::IndexMap<String, ()>,
    pub iteration: i64,
}

impl WorkflowCancelledError {
    pub fn new(
        messages: Vec<String>,
        completed_steps: indexmap::IndexMap<String, ()>,
        iteration: i64,
    ) -> Self {
        Self {
            messages,
            completed_steps,
            iteration,
        }
    }
}

impl fmt::Display for WorkflowCancelledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let step_names: Vec<&str> = self.completed_steps.keys().map(|s| s.as_str()).collect();
        write!(
            f,
            "Workflow cancelled at iteration {} with completed steps: [{}]",
            self.iteration,
            step_names.join(", ")
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub struct MaxIterationsError {
    pub iterations: i64,
    pub completed_steps: indexmap::IndexMap<String, ()>,
    pub pending_steps: Vec<String>,
}

impl MaxIterationsError {
    pub fn new(
        iterations: i64,
        completed_steps: indexmap::IndexMap<String, ()>,
        pending_steps: Vec<String>,
    ) -> Self {
        Self {
            iterations,
            completed_steps,
            pending_steps,
        }
    }
}

impl fmt::Display for MaxIterationsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let completed: Vec<&str> = self.completed_steps.keys().map(|s| s.as_str()).collect();
        write!(
            f,
            "Max iterations ({}) reached. Completed: [{}]. Pending: [{}]",
            self.iterations,
            completed.join(", "),
            self.pending_steps.join(", ")
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub struct StepEnforcementError {
    pub terminal_tool: String,
    pub attempts: i64,
    pub pending_steps: Vec<String>,
}

impl StepEnforcementError {
    pub fn new(
        terminal_tool: impl Into<String>,
        attempts: i64,
        pending_steps: Vec<String>,
    ) -> Self {
        Self {
            terminal_tool: terminal_tool.into(),
            attempts,
            pending_steps,
        }
    }
}

impl fmt::Display for StepEnforcementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Terminal tool '{}' called prematurely (attempt {}), pending steps: [{}]",
            self.terminal_tool,
            self.attempts,
            self.pending_steps.join(", ")
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub struct PrerequisiteError {
    pub tool_name: String,
    pub violations: i64,
    pub missing_prereqs: Vec<String>,
}

impl PrerequisiteError {
    pub fn new(
        tool_name: impl Into<String>,
        violations: i64,
        missing_prereqs: Vec<String>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            violations,
            missing_prereqs,
        }
    }
}

impl fmt::Display for PrerequisiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Prerequisite violation for '{}' ({} violations), missing: [{}]",
            self.tool_name,
            self.violations,
            self.missing_prereqs.join(", ")
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub struct ContextBudgetExceeded {
    pub estimated_tokens: i64,
    pub budget_tokens: i64,
}

impl ContextBudgetExceeded {
    pub fn new(estimated_tokens: i64, budget_tokens: i64) -> Self {
        Self {
            estimated_tokens,
            budget_tokens,
        }
    }
}

impl fmt::Display for ContextBudgetExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Context budget exceeded: estimated {} tokens, budget {} tokens",
            self.estimated_tokens, self.budget_tokens
        )
    }
}

#[derive(Debug, thiserror::Error)]
pub struct HardwareDetectionError {
    pub cause: String,
}

impl HardwareDetectionError {
    pub fn new(cause: impl Into<String>) -> Self {
        Self {
            cause: cause.into(),
        }
    }
}

impl fmt::Display for HardwareDetectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hardware detection failed: {}", self.cause)
    }
}

#[derive(Debug, thiserror::Error)]
pub struct ContextDiscoveryError {
    pub cause: String,
}

impl ContextDiscoveryError {
    pub fn new(cause: impl Into<String>) -> Self {
        Self {
            cause: cause.into(),
        }
    }
}

impl fmt::Display for ContextDiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Context length discovery failed: {}", self.cause)
    }
}

#[derive(Debug, thiserror::Error)]
pub struct BudgetResolutionError {
    pub cause: Option<String>,
}

impl BudgetResolutionError {
    pub fn new() -> Self {
        Self { cause: None }
    }

    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }
}

impl Default for BudgetResolutionError {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BudgetResolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Some(c) => write!(f, "Could not determine context budget: {}", c),
            None => write!(f, "No GPU detected and no explicit budget provided"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Backend error (status {status_code}): {body}")]
    Generic { status_code: i64, body: String },
    #[error("Thinking mode not supported for model '{model}'")]
    ThinkingNotSupported {
        model: String,
        status_code: i64,
        body: String,
    },
}

impl BackendError {
    pub fn new(status_code: i64, body: impl Into<String>) -> Self {
        Self::Generic {
            status_code,
            body: body.into(),
        }
    }

    pub fn thinking_not_supported(model: impl Into<String>) -> Self {
        Self::ThinkingNotSupported {
            model: model.into(),
            status_code: 400,
            body: String::new(),
        }
    }
}

// ThinkingNotSupportedError is an alias that constructs the ThinkingNotSupported
// variant of BackendError. This preserves the catch-as-BackendError semantics.
pub type ThinkingNotSupportedError = BackendError;

#[derive(Debug, thiserror::Error)]
pub struct StreamError {
    pub message: String,
}

impl StreamError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Default for StreamError {
    fn default() -> Self {
        Self::new("Stream ended without a final chunk")
    }
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}
