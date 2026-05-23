use super::nudge::Nudge;
use crate::clients::base::ToolCall;
use crate::core::steps::{Prerequisite, StepTracker};
use crate::prompts::nudges;
use indexmap::{IndexMap, IndexSet};

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

/// Result of StepEnforcer.check().
#[derive(Debug, Clone, PartialEq)]
pub struct StepCheck {
    pub nudge: Option<Nudge>,
    pub needs_nudge: bool,
}

impl StepCheck {
    pub fn ok() -> Self {
        Self {
            nudge: None,
            needs_nudge: false,
        }
    }

    pub fn blocked(nudge: Nudge) -> Self {
        Self {
            nudge: Some(nudge),
            needs_nudge: true,
        }
    }
}

/// Stateful step tracker that detects premature terminal tool calls and enforces
/// tool prerequisites, producing escalating nudges capped at tier 3.
pub struct StepEnforcer {
    pub tracker: StepTracker,
    pub terminal_tools: IndexSet<String>,
    pub tool_prerequisites: IndexMap<String, Vec<StepPrerequisite>>,
    pub max_premature_attempts: i32,
    pub max_prereq_violations: i32,
    pub premature_attempts: i32,
    pub prereq_violations: i32,
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
        let nudge = Nudge::new("user", content, "step").with_tier(tier);
        StepCheck::blocked(nudge)
    }

    /// Check tool prerequisites against pre-batch state.
    pub fn check_prerequisites(&mut self, tool_calls: &[ToolCall]) -> StepCheck {
        for tc in tool_calls {
            if let Some(prereqs) = self.tool_prerequisites.get(&tc.tool) {
                let rust_prereqs: Vec<Prerequisite> = prereqs.iter().map(|p| p.into()).collect();
                let result = self
                    .tracker
                    .check_prerequisites(&tc.tool, &tc.args, &rust_prereqs);
                if !result.satisfied {
                    self.prereq_violations += 1;
                    let missing_refs: Vec<&str> =
                        result.missing.iter().map(|s| s.as_str()).collect();
                    let content = nudges::prerequisite_nudge(&tc.tool, &missing_refs);
                    let nudge = Nudge::new("user", content, "prerequisite");
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
        let has_terminal = tool_calls
            .iter()
            .any(|c| self.terminal_tools.contains(&c.tool));
        has_terminal && self.tracker.is_satisfied()
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
