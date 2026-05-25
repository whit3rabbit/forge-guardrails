use super::error_tracker::ErrorTracker;
use super::nudge::Nudge;
use super::response_validator::{ResponseValidator, RetryNudgeFn};
use super::scoring::{ToolCallScore, ToolCallScorer};
use super::scoring_context::ScoringContext;
use super::step_enforcer::{StepEnforcer, StepPrerequisite};
use crate::clients::base::{LLMResponse, ToolCall};
use indexmap::IndexSet;
use std::sync::Arc;

/// Action determined by Guardrails.check().
#[derive(Debug, Clone, PartialEq)]
pub enum GuardAction {
    /// Execute the tool call batch.
    Execute,
    /// Retry the inference step with a nudge.
    Retry,
    /// Step execution is blocked by dependencies/rules.
    StepBlocked,
    /// Execution terminated with a fatal guardrail error.
    Fatal,
}

/// Frozen result of Guardrails.check().
#[derive(Debug, Clone, PartialEq)]
pub struct CheckResult {
    /// Action designated by the guardrails check.
    pub action: GuardAction,
    /// Tool calls payload if validation succeeded.
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Escalating instruction nudge if blocked or retrying.
    pub nudge: Option<Nudge>,
    /// Underlying cause explanation.
    pub reason: Option<String>,
}

impl CheckResult {
    /// Creates a `CheckResult` that executes tool calls.
    pub fn execute(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            action: GuardAction::Execute,
            tool_calls: Some(tool_calls),
            nudge: None,
            reason: None,
        }
    }

    /// Creates a `CheckResult` requesting a retry with a nudge.
    pub fn retry(nudge: Nudge) -> Self {
        Self {
            action: GuardAction::Retry,
            tool_calls: None,
            nudge: Some(nudge),
            reason: None,
        }
    }

    /// Creates a `CheckResult` indicating step block.
    pub fn step_blocked(nudge: Nudge) -> Self {
        Self {
            action: GuardAction::StepBlocked,
            tool_calls: None,
            nudge: Some(nudge),
            reason: None,
        }
    }

    /// Creates a `CheckResult` indicating a fatal termination.
    pub fn fatal(reason: impl Into<String>) -> Self {
        Self {
            action: GuardAction::Fatal,
            tool_calls: None,
            nudge: None,
            reason: Some(reason.into()),
        }
    }
}

/// Terminal tool specification: either a single string or a set of strings.
pub enum TerminalTool {
    /// A single terminal tool name.
    Single(String),
    /// A set of multiple terminal tool names.
    Multiple(IndexSet<String>),
}

/// Facade wrapping ErrorTracker, ResponseValidator, and StepEnforcer into a
/// two-method check()/record() API. check() processes through two sequential
/// checkpoints: first validation, then step enforcement.
pub struct Guardrails {
    /// Internal error tracker.
    pub error_tracker: ErrorTracker,
    /// Internal response structure validator.
    pub validator: ResponseValidator,
    /// Internal step enforcement manager.
    pub step_enforcer: StepEnforcer,
    /// Designated set of terminal tools.
    pub terminal_tools: IndexSet<String>,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    last_scores: Vec<ToolCallScore>,
}

impl Guardrails {
    /// Creates a new `Guardrails` engine.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tool_names: Vec<String>,
        terminal_tool: TerminalTool,
        required_steps: Option<Vec<String>>,
        tool_prerequisites: Option<indexmap::IndexMap<String, Vec<StepPrerequisite>>>,
        max_retries: i32,
        max_tool_errors: i32,
        rescue_enabled: bool,
        max_premature_attempts: i32,
        retry_nudge: Option<RetryNudgeFn>,
    ) -> Self {
        let terminal_set: IndexSet<String> = match terminal_tool {
            TerminalTool::Single(name) => {
                let mut s = IndexSet::new();
                s.insert(name);
                s
            }
            TerminalTool::Multiple(names) => names,
        };
        let steps = required_steps.unwrap_or_default();
        Self {
            error_tracker: ErrorTracker::new(max_retries, max_tool_errors),
            validator: ResponseValidator::new(tool_names, rescue_enabled, retry_nudge),
            step_enforcer: StepEnforcer::new(
                steps,
                terminal_set.clone(),
                tool_prerequisites,
                max_premature_attempts,
                2,
            ),
            terminal_tools: terminal_set,
            scorer: None,
            last_scores: Vec::new(),
        }
    }

    /// Attach a tool-call scorer.
    pub fn with_scorer(mut self, scorer: Arc<dyn ToolCallScorer>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// Return classifier scores from the previous check.
    pub fn last_scores(&self) -> &[ToolCallScore] {
        &self.last_scores
    }

    /// Three-checkpoint pipeline: validation, step enforcement, then prerequisites.
    /// Returns CheckResult with action, tool_calls, nudge, and reason.
    pub fn check(&mut self, response: &LLMResponse) -> CheckResult {
        self.check_inner(response, None)
    }

    /// Run the normal check and shadow-score successful deterministic calls.
    pub fn check_with_scoring_context(
        &mut self,
        response: &LLMResponse,
        scoring_context: &ScoringContext,
    ) -> CheckResult {
        self.check_inner(response, Some(scoring_context))
    }

    fn check_inner(
        &mut self,
        response: &LLMResponse,
        scoring_context: Option<&ScoringContext>,
    ) -> CheckResult {
        self.last_scores.clear();

        // Checkpoint 1: Validation
        let validation = self.validator.validate(response);
        if validation.needs_retry {
            self.error_tracker.record_retry();
            if self.error_tracker.retries_exhausted() {
                return CheckResult::fatal("Too many bad responses");
            }
            return CheckResult::retry(validation.nudge.expect("needs_retry requires a nudge"));
        }

        // Validation passed: reset retry counter (ob-018).
        self.error_tracker.reset_retries();

        let tool_calls = validation
            .tool_calls
            .expect("valid response requires tool_calls");

        // Checkpoint 2: Step enforcement
        let step_check = self.step_enforcer.check(&tool_calls);
        if step_check.needs_nudge {
            if self.step_enforcer.premature_exhausted() {
                return CheckResult::fatal("Too many skipped required steps");
            }
            return CheckResult::step_blocked(
                step_check.nudge.expect("needs_nudge requires a nudge"),
            );
        }

        // Checkpoint 3: Prerequisites
        let prereq_check = self.step_enforcer.check_prerequisites(&tool_calls);
        if prereq_check.needs_nudge {
            if self.step_enforcer.prereq_exhausted() {
                return CheckResult::fatal("Too many prerequisite violations");
            }
            return CheckResult::step_blocked(
                prereq_check.nudge.expect("needs_nudge requires a nudge"),
            );
        }

        if let (Some(scorer), Some(ctx)) = (self.scorer.as_ref(), scoring_context) {
            for call in &tool_calls {
                match scorer.score(ctx, call) {
                    Ok(score) => {
                        tracing::info!(
                            target: "forge.classifier",
                            tool = %call.tool,
                            label = %score.label.as_label(),
                            confidence = score.confidence,
                            action = %score.action.as_str(),
                            latency_ms = score.latency_ms,
                            model_version = %score.model_version,
                            "tool-call classifier score"
                        );
                        self.last_scores.push(score);
                    }
                    Err(err) => {
                        tracing::warn!(
                            target: "forge.classifier",
                            tool = %call.tool,
                            error = %err,
                            "classifier scoring failed; allowing deterministic path"
                        );
                    }
                }
            }
        }

        CheckResult::execute(tool_calls)
    }

    /// Record executed tool names. Resets error counters and premature/prereq counters.
    /// Returns true only if the terminal tool is in the executed list AND all
    /// required steps are satisfied.
    pub fn record(&mut self, executed: &[&str]) -> bool {
        for name in executed {
            self.step_enforcer.record(name, None);
        }
        self.error_tracker.reset_retries();
        self.error_tracker.reset_errors();
        self.step_enforcer.reset_premature();
        self.step_enforcer.reset_prereq_violations();

        let has_terminal = executed
            .iter()
            .any(|name| self.terminal_tools.contains(*name));
        has_terminal && self.step_enforcer.is_satisfied()
    }

    /// Returns completed required steps as an IndexMap with unit values.
    pub fn completed_steps(&self) -> indexmap::IndexMap<String, ()> {
        self.step_enforcer.completed_steps()
    }

    /// Returns pending required steps.
    pub fn pending_steps(&self) -> Vec<String> {
        self.step_enforcer.pending()
    }

    /// Returns the premature attempt count.
    pub fn premature_attempts(&self) -> i32 {
        self.step_enforcer.premature_attempts()
    }
}
