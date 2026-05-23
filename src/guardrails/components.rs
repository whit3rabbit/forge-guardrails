//! Internal guardrail components: ErrorTracker, ResponseValidator, StepEnforcer.

use crate::nudges;
use crate::prompts;
use crate::steps::{Prerequisite, StepTracker};
use crate::streaming::ToolCall;
use indexmap::{IndexMap, IndexSet};
use std::collections::HashSet;

use super::{Nudge, StepCheck, ValidationResult};

// ---------------------------------------------------------------------------
// ErrorTracker
// ---------------------------------------------------------------------------

/// Tracks two independent counters: consecutive retries and consecutive tool errors.
///
/// Each counter has a configurable exhaustion threshold. Exhaustion is determined
/// by strict greater-than comparison (counter must exceed max, not just equal it).
/// Soft errors are excluded from the tool error counter.
/// Success does not auto-reset counters; only explicit reset methods clear them.
pub struct ErrorTracker {
    consecutive_retries: i32,
    consecutive_tool_errors: i32,
    max_retries: i32,
    max_tool_errors: i32,
}

impl ErrorTracker {
    pub fn new(max_retries: i32, max_tool_errors: i32) -> Self {
        Self {
            consecutive_retries: 0,
            consecutive_tool_errors: 0,
            max_retries,
            max_tool_errors,
        }
    }

    pub fn record_retry(&mut self) {
        self.consecutive_retries += 1;
    }

    pub fn reset_retries(&mut self) {
        self.consecutive_retries = 0;
    }

    /// Record a tool execution result. Soft errors are excluded from the counter.
    /// Success does not reset the counter (only reset_errors does).
    pub fn record_result(&mut self, success: bool, is_soft_error: bool) {
        if !success && !is_soft_error {
            self.consecutive_tool_errors += 1;
        }
    }

    pub fn reset_errors(&mut self) {
        self.consecutive_tool_errors = 0;
    }

    pub fn retries_exhausted(&self) -> bool {
        self.consecutive_retries > self.max_retries
    }

    pub fn tool_errors_exhausted(&self) -> bool {
        self.consecutive_tool_errors > self.max_tool_errors
    }

    pub fn consecutive_retries(&self) -> i32 {
        self.consecutive_retries
    }

    pub fn consecutive_tool_errors(&self) -> i32 {
        self.consecutive_tool_errors
    }
}

// ---------------------------------------------------------------------------
// ResponseValidator
// ---------------------------------------------------------------------------

/// Function type for generating custom retry nudge content.
pub type RetryNudgeFn = Box<dyn Fn(&str) -> String>;

/// Stateless validator for LLM responses. Validates tool call lists against
/// the allowed tool set, attempts rescue parsing from text content, and
/// produces retry nudges for invalid responses.
pub struct ResponseValidator {
    tool_names: HashSet<String>,
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
    pub fn validate(&self, response: &crate::streaming::LLMResponse) -> ValidationResult {
        match response {
            crate::streaming::LLMResponse::ToolCalls(calls) => self.validate_tool_calls(calls),
            crate::streaming::LLMResponse::Text(text) => self.validate_text(text),
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
            let nudge = Nudge::new("system", content, "unknown_tool");
            return ValidationResult::invalid(nudge);
        }
        ValidationResult::valid(calls.to_vec())
    }

    fn validate_text(&self, text: &crate::streaming::TextResponse) -> ValidationResult {
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
        let nudge = Nudge::new("system", content, "retry");
        ValidationResult::invalid(nudge)
    }
}

// ---------------------------------------------------------------------------
// StepEnforcer
// ---------------------------------------------------------------------------

/// Prerequisite specification for step enforcement.
/// Supports name-only and arg-matched variants.
#[derive(Debug, Clone, PartialEq)]
pub enum StepPrerequisite {
    NameOnly(String),
    ArgMatched { tool: String, match_arg: String },
}

impl From<&StepPrerequisite> for Prerequisite {
    fn from(sp: &StepPrerequisite) -> Self {
        match sp {
            StepPrerequisite::NameOnly(name) => Prerequisite::NameOnly(name.clone()),
            StepPrerequisite::ArgMatched { tool, match_arg } => Prerequisite::ArgMatched {
                tool: tool.clone(),
                match_arg: match_arg.clone(),
            },
        }
    }
}

/// Stateful step tracker that detects premature terminal tool calls and enforces
/// tool prerequisites, producing escalating nudges capped at tier 3.
pub struct StepEnforcer {
    tracker: StepTracker,
    terminal_tools: IndexSet<String>,
    tool_prerequisites: IndexMap<String, Vec<StepPrerequisite>>,
    max_premature_attempts: i32,
    max_prereq_violations: i32,
    premature_attempts: i32,
    prereq_violations: i32,
}

impl StepEnforcer {
    pub fn new(
        required_steps: Vec<String>,
        terminal_tools: IndexSet<String>,
        tool_prerequisites: Option<IndexMap<String, Vec<StepPrerequisite>>>,
        max_premature_attempts: i32,
        max_prereq_violations: i32,
    ) -> Self {
        Self {
            tracker: StepTracker::new(required_steps),
            terminal_tools,
            tool_prerequisites: tool_prerequisites.unwrap_or_default(),
            max_premature_attempts,
            max_prereq_violations,
            premature_attempts: 0,
            prereq_violations: 0,
        }
    }

    /// Check whether a batch of tool calls contains a premature terminal tool.
    pub fn check(&mut self, tool_calls: &[ToolCall]) -> StepCheck {
        if self.tracker.is_satisfied() {
            return StepCheck::ok();
        }
        let has_terminal = tool_calls
            .iter()
            .any(|c| self.terminal_tools.contains(&c.tool));
        if !has_terminal {
            return StepCheck::ok();
        }
        self.premature_attempts += 1;
        let tier = std::cmp::min(self.premature_attempts, 3);
        let pending = self.tracker.pending();
        let pending_refs: Vec<&str> = pending.iter().map(|s| s.as_str()).collect();
        let terminal_name = tool_calls
            .iter()
            .find(|c| self.terminal_tools.contains(&c.tool))
            .map(|c| c.tool.as_str())
            .unwrap_or("terminal");
        let content = nudges::step_nudge(terminal_name, &pending_refs, tier);
        let nudge = Nudge::new("system", content, "step").with_tier(tier);
        StepCheck::blocked(nudge)
    }

    /// Check tool prerequisites against pre-batch state.
    pub fn check_prerequisites(&self, tool_calls: &[ToolCall]) -> StepCheck {
        for tc in tool_calls {
            if let Some(prereqs) = self.tool_prerequisites.get(&tc.tool) {
                let rust_prereqs: Vec<Prerequisite> = prereqs.iter().map(|p| p.into()).collect();
                let result = self
                    .tracker
                    .check_prerequisites(&tc.tool, &tc.args, &rust_prereqs);
                if !result.satisfied {
                    let missing_refs: Vec<&str> =
                        result.missing.iter().map(|s| s.as_str()).collect();
                    let content = nudges::prerequisite_nudge(&tc.tool, &missing_refs);
                    let nudge = Nudge::new("system", content, "prerequisite");
                    return StepCheck::blocked(nudge);
                }
            }
        }
        StepCheck::ok()
    }

    pub fn record(&mut self, tool_name: &str, args: Option<&IndexMap<String, serde_json::Value>>) {
        self.tracker.record(tool_name, args);
    }

    pub fn is_satisfied(&self) -> bool {
        self.tracker.is_satisfied()
    }

    pub fn pending(&self) -> Vec<String> {
        self.tracker.pending()
    }

    pub fn terminal_reached(&self, tool_calls: &[ToolCall]) -> bool {
        tool_calls
            .iter()
            .any(|c| self.terminal_tools.contains(&c.tool))
    }

    pub fn reset_premature(&mut self) {
        self.premature_attempts = 0;
    }

    pub fn reset_prereq_violations(&mut self) {
        self.prereq_violations = 0;
    }

    pub fn summary_hint(&self) -> String {
        self.tracker.summary_hint()
    }

    pub fn premature_attempts(&self) -> i32 {
        self.premature_attempts
    }

    pub fn premature_exhausted(&self) -> bool {
        self.premature_attempts > self.max_premature_attempts
    }

    pub fn prereq_violations(&self) -> i32 {
        self.prereq_violations
    }

    pub fn prereq_exhausted(&self) -> bool {
        self.prereq_violations > self.max_prereq_violations
    }

    /// Returns completed steps as an IndexMap with string keys and unit values.
    pub fn completed_steps(&self) -> IndexMap<String, ()> {
        let all_required = self.tracker.required_steps();
        let pending = self.tracker.pending();
        all_required
            .iter()
            .filter(|s| !pending.contains(s))
            .map(|s| (s.clone(), ()))
            .collect()
    }
}
