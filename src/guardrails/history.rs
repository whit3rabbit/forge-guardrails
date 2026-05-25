//! Bounded guardrail failure memory for future nudge and scoring improvements.

use crate::guardrails::policy::GuardrailViolation;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::VecDeque;

/// Compact bounded history of recent guardrail failures.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardrailHistory {
    max_entries: usize,
    events: VecDeque<GuardrailViolation>,
    /// Last tool the model attempted to call.
    pub last_called_tool: Option<String>,
    /// Last invalid argument payload observed.
    pub last_invalid_args: Option<Value>,
    /// Counts repeated unknown-tool attempts by tool name.
    pub repeated_unknown_tool: IndexMap<String, usize>,
    /// Counts repeated invalid argument paths.
    pub repeated_invalid_arg_path: IndexMap<String, usize>,
    /// Last hard tool error content observed by the caller.
    pub last_tool_error: Option<String>,
}

impl GuardrailHistory {
    /// Create a history with a bounded event and counter capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            events: VecDeque::new(),
            last_called_tool: None,
            last_invalid_args: None,
            repeated_unknown_tool: IndexMap::new(),
            repeated_invalid_arg_path: IndexMap::new(),
            last_tool_error: None,
        }
    }

    /// Record a structured guardrail violation.
    pub fn record_violation(&mut self, violation: GuardrailViolation) {
        match &violation {
            GuardrailViolation::UnknownTool { called, .. } => {
                self.last_called_tool = Some(called.clone());
                bump_bounded(
                    &mut self.repeated_unknown_tool,
                    called.clone(),
                    self.max_entries,
                );
            }
            GuardrailViolation::InvalidArguments { tool, errors } => {
                self.last_called_tool = Some(tool.clone());
                for error in errors {
                    bump_bounded(
                        &mut self.repeated_invalid_arg_path,
                        error.path.clone(),
                        self.max_entries,
                    );
                }
            }
            GuardrailViolation::PrematureTerminal { terminal, .. } => {
                self.last_called_tool = Some(terminal.clone());
            }
            GuardrailViolation::MissingPrerequisite { tool, .. } => {
                self.last_called_tool = Some(tool.clone());
            }
            GuardrailViolation::NoToolCall
            | GuardrailViolation::UnsafeBatch { .. }
            | GuardrailViolation::RepeatedFailure { .. }
            | GuardrailViolation::WrongToolLikely { .. } => {}
        }

        self.events.push_back(violation);
        while self.events.len() > self.max_entries {
            self.events.pop_front();
        }
    }

    /// Record invalid arguments with the original candidate argument payload.
    pub fn record_invalid_arguments(
        &mut self,
        tool: impl Into<String>,
        args: Value,
        errors: Vec<crate::guardrails::policy::ArgValidationError>,
    ) {
        let tool = tool.into();
        self.last_invalid_args = Some(args);
        self.record_violation(GuardrailViolation::InvalidArguments { tool, errors });
    }

    /// Record a tool execution error for future repair hints.
    pub fn record_tool_error(&mut self, error: impl Into<String>) {
        self.last_tool_error = Some(error.into());
    }

    /// Return recent violations in oldest-to-newest order.
    pub fn recent(&self) -> Vec<GuardrailViolation> {
        self.events.iter().cloned().collect()
    }

    /// Maximum number of entries retained.
    pub fn capacity(&self) -> usize {
        self.max_entries
    }
}

fn bump_bounded(map: &mut IndexMap<String, usize>, key: String, max_entries: usize) {
    if let Some(count) = map.get_mut(&key) {
        *count += 1;
        return;
    }
    while map.len() >= max_entries {
        map.shift_remove_index(0);
    }
    map.insert(key, 1);
}
