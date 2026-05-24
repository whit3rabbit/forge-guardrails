use std::fmt;

/// Base error type for the forge-guardrails framework.
///
/// All framework errors except `ToolResolutionError` are represented as
/// variants of this enum, preserving catch-as-base semantics.
#[derive(Debug, thiserror::Error)]
pub enum ForgeError {
    /// The model is not supported.
    #[error(transparent)]
    UnsupportedModel(#[from] UnsupportedModelError),
    /// Failed to parse/construct a tool call.
    #[error(transparent)]
    ToolCall(#[from] ToolCallError),
    /// Tool execution failed.
    #[error(transparent)]
    ToolExecution(#[from] ToolExecutionError),
    /// Workflow cancelled.
    #[error(transparent)]
    WorkflowCancelled(#[from] WorkflowCancelledError),
    /// Max iterations reached.
    #[error(transparent)]
    MaxIterations(#[from] MaxIterationsError),
    /// Premature terminal tool call step violation.
    #[error(transparent)]
    StepEnforcement(#[from] StepEnforcementError),
    /// Prerequisite step check failed.
    #[error(transparent)]
    Prerequisite(#[from] PrerequisiteError),
    /// Context budget tokens exceeded.
    #[error(transparent)]
    ContextBudgetExceeded(#[from] ContextBudgetExceeded),
    /// Hardware detection failed.
    #[error(transparent)]
    HardwareDetection(#[from] HardwareDetectionError),
    /// Context length discovery failed.
    #[error(transparent)]
    ContextDiscovery(#[from] ContextDiscoveryError),
    /// Budget resolution failed.
    #[error(transparent)]
    BudgetResolution(#[from] BudgetResolutionError),
    /// Backend request failed.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Stream error.
    #[error(transparent)]
    Stream(#[from] StreamError),
}

/// Error indicating that a model is not supported because sampling defaults are missing and strict mode is active.
#[derive(Debug, thiserror::Error)]
pub struct UnsupportedModelError {
    /// The name of the unsupported model.
    pub model: String,
}

impl UnsupportedModelError {
    /// Creates a new `UnsupportedModelError` for the given model name.
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

/// Error indicating a failure to parse or construct a tool call from model output.
#[derive(Debug, thiserror::Error)]
pub struct ToolCallError {
    /// The error message.
    pub message: String,
    /// The raw response from the model, if available.
    pub raw_response: Option<String>,
    /// The underlying cause of the parsing/construction failure, if available.
    pub cause: Option<String>,
}

impl ToolCallError {
    /// Creates a new `ToolCallError` with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            raw_response: None,
            cause: None,
        }
    }

    /// Sets the raw response associated with this error.
    pub fn with_raw_response(mut self, raw: impl Into<String>) -> Self {
        self.raw_response = Some(raw.into());
        self
    }

    /// Sets the cause of this error.
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

/// Error indicating that a tool execution failed.
#[derive(Debug, thiserror::Error)]
pub struct ToolExecutionError {
    /// The name of the tool whose execution failed.
    pub tool_name: String,
    /// The detailed cause of the execution failure.
    pub cause: String,
}

impl ToolExecutionError {
    /// Creates a new `ToolExecutionError` for the given tool name and cause.
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
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub struct ToolResolutionError {
    /// Description of the resolution failure.
    pub message: String,
    /// Name of the tool, if available.
    pub tool_name: Option<String>,
}

impl ToolResolutionError {
    /// Creates a new `ToolResolutionError` with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            tool_name: None,
        }
    }

    /// Sets the tool name associated with this resolution failure.
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

/// Unified tool error returned by async tool callables.
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum ToolError {
    /// The tool could not be resolved or matched.
    #[error(transparent)]
    Resolution(#[from] ToolResolutionError),
    /// The tool resolved but failed during execution.
    #[error("Tool execution failed: {0}")]
    Execution(String),
}

/// Error indicating that a workflow was cancelled.
#[derive(Debug, thiserror::Error)]
pub struct WorkflowCancelledError {
    /// The conversation history messages prior to cancellation.
    pub messages: Vec<String>,
    /// Steps that were successfully completed before cancellation.
    pub completed_steps: indexmap::IndexMap<String, ()>,
    /// The workflow loop iteration count when cancellation occurred.
    pub iteration: i64,
}

impl WorkflowCancelledError {
    /// Creates a new `WorkflowCancelledError`.
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

/// Error indicating that a workflow reached its maximum allowed iteration limit.
#[derive(Debug, thiserror::Error)]
pub struct MaxIterationsError {
    /// The iteration limit that was reached.
    pub iterations: i64,
    /// Steps that were successfully completed before reaching the limit.
    pub completed_steps: indexmap::IndexMap<String, ()>,
    /// Steps that are still pending when execution was terminated.
    pub pending_steps: Vec<String>,
}

impl MaxIterationsError {
    /// Creates a new `MaxIterationsError`.
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

/// Error indicating that a terminal tool was called prematurely before all required steps were satisfied.
#[derive(Debug, thiserror::Error)]
pub struct StepEnforcementError {
    /// Name of the terminal tool that was called prematurely.
    pub terminal_tool: String,
    /// Number of premature attempts recorded.
    pub attempts: i64,
    /// The required workflow steps that remain pending.
    pub pending_steps: Vec<String>,
}

impl StepEnforcementError {
    /// Creates a new `StepEnforcementError`.
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

/// Error indicating that a tool prerequisite was violated.
#[derive(Debug, thiserror::Error)]
pub struct PrerequisiteError {
    /// Name of the tool whose prerequisite was violated.
    pub tool_name: String,
    /// Number of prerequisite violations recorded.
    pub violations: i64,
    /// Prerequisite step descriptions that were missing/unsatisfied.
    pub missing_prereqs: Vec<String>,
}

impl PrerequisiteError {
    /// Creates a new `PrerequisiteError`.
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

/// Error indicating that the context token limit has been exceeded.
#[derive(Debug, thiserror::Error)]
pub struct ContextBudgetExceeded {
    /// Estimated tokens required for the request.
    pub estimated_tokens: i64,
    /// The allocated budget of context tokens.
    pub budget_tokens: i64,
}

impl ContextBudgetExceeded {
    /// Creates a new `ContextBudgetExceeded` error.
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

/// Error indicating a failure to auto-detect hardware profile.
#[derive(Debug, thiserror::Error)]
pub struct HardwareDetectionError {
    /// Description of why hardware detection failed.
    pub cause: String,
}

impl HardwareDetectionError {
    /// Creates a new `HardwareDetectionError` with the given cause.
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

/// Error indicating a failure to query backend context limit support.
#[derive(Debug, thiserror::Error)]
pub struct ContextDiscoveryError {
    /// Description of why context length discovery failed.
    pub cause: String,
}

impl ContextDiscoveryError {
    /// Creates a new `ContextDiscoveryError` with the given cause.
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

/// Error indicating a failure to resolve context budget limits.
#[derive(Debug, thiserror::Error)]
pub struct BudgetResolutionError {
    /// Detailed cause of the budget resolution failure, if available.
    pub cause: Option<String>,
}

impl BudgetResolutionError {
    /// Creates a new `BudgetResolutionError` with no cause.
    pub fn new() -> Self {
        Self { cause: None }
    }

    /// Sets the cause of the budget resolution failure.
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

/// Error indicating that the backend request failed.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// A generic backend failure with status code and body description.
    #[error("Backend error (status {status_code}): {body}")]
    Generic {
        /// The HTTP or API status code returned by the backend.
        status_code: i64,
        /// The response body containing error details.
        body: String,
    },
    /// Error indicating that thinking mode is not supported by the model.
    #[error("Thinking mode not supported for model '{model}'")]
    ThinkingNotSupported {
        /// The model name.
        model: String,
        /// The status code.
        status_code: i64,
        /// The body.
        body: String,
    },
}

impl BackendError {
    /// Creates a new generic `BackendError`.
    pub fn new(status_code: i64, body: impl Into<String>) -> Self {
        Self::Generic {
            status_code,
            body: body.into(),
        }
    }

    /// Creates a `ThinkingNotSupported` backend error.
    pub fn thinking_not_supported(model: impl Into<String>) -> Self {
        Self::ThinkingNotSupported {
            model: model.into(),
            status_code: 400,
            body: String::new(),
        }
    }
}

/// ThinkingNotSupportedError is an alias that constructs the ThinkingNotSupported
/// variant of BackendError. This preserves the catch-as-BackendError semantics.
pub type ThinkingNotSupportedError = BackendError;

/// Error indicating that stream processing failed.
#[derive(Debug, thiserror::Error)]
pub struct StreamError {
    /// Description of the stream failure.
    pub message: String,
}

impl StreamError {
    /// Creates a new `StreamError`.
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
