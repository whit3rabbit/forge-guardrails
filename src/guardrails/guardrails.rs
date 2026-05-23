use super::error_tracker::ErrorTracker;
use super::nudge::Nudge;
use super::response_validator::{ResponseValidator, RetryNudgeFn};
use super::step_enforcer::{StepEnforcer, StepPrerequisite};
use crate::clients::base::{LLMResponse, ToolCall};
use indexmap::IndexSet;

/// Action determined by Guardrails.check().
#[derive(Debug, Clone, PartialEq)]
pub enum GuardAction {
    Execute,
    Retry,
    StepBlocked,
    Fatal,
}

/// Frozen result of Guardrails.check().
#[derive(Debug, Clone, PartialEq)]
pub struct CheckResult {
    pub action: GuardAction,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub nudge: Option<Nudge>,
    pub reason: Option<String>,
}

impl CheckResult {
    pub fn execute(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            action: GuardAction::Execute,
            tool_calls: Some(tool_calls),
            nudge: None,
            reason: None,
        }
    }

    pub fn retry(nudge: Nudge) -> Self {
        Self {
            action: GuardAction::Retry,
            tool_calls: None,
            nudge: Some(nudge),
            reason: None,
        }
    }

    pub fn step_blocked(nudge: Nudge) -> Self {
        Self {
            action: GuardAction::StepBlocked,
            tool_calls: None,
            nudge: Some(nudge),
            reason: None,
        }
    }

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
    Single(String),
    Multiple(IndexSet<String>),
}

/// Facade wrapping ErrorTracker, ResponseValidator, and StepEnforcer into a
/// two-method check()/record() API. check() processes through two sequential
/// checkpoints: first validation, then step enforcement.
pub struct Guardrails {
    pub error_tracker: ErrorTracker,
    pub validator: ResponseValidator,
    pub step_enforcer: StepEnforcer,
    pub terminal_tools: IndexSet<String>,
}

impl Guardrails {
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
        }
    }

    /// Three-checkpoint pipeline: validation, step enforcement, then prerequisites.
    /// Returns CheckResult with action, tool_calls, nudge, and reason.
    pub fn check(&mut self, response: &LLMResponse) -> CheckResult {
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
