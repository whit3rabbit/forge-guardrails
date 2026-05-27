use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

/// Result of checking prerequisites against prior tool executions.
#[derive(Debug, Clone, PartialEq)]
pub struct PrerequisiteCheck {
    /// Whether the prerequisites are satisfied.
    pub satisfied: bool,
    /// List of missing prerequisite step names or identifiers.
    pub missing: Vec<String>,
}

impl PrerequisiteCheck {
    /// Returns a satisfied prerequisite check.
    pub fn satisfied() -> Self {
        Self {
            satisfied: true,
            missing: Vec::new(),
        }
    }

    /// Returns an unsatisfied prerequisite check containing the missing steps.
    pub fn unsatisfied(missing: Vec<String>) -> Self {
        Self {
            satisfied: false,
            missing,
        }
    }
}

/// A prerequisite specification that can be checked against prior executions.
#[derive(Debug, Clone, PartialEq)]
pub enum Prerequisite {
    /// Prerequisite satisfied solely by the occurrence of a tool call.
    NameOnly(String),
    /// Prerequisite satisfied by a tool call only when specific arguments match.
    ArgMatched {
        /// Name of the tool.
        tool: String,
        /// Description or key matching parameter criteria.
        match_arg: String,
    },
}

/// Tracks required steps and executed tools in a workflow run.
pub struct StepTracker {
    required_steps: Vec<String>,
    completed_steps: IndexSet<String>,
    executed_tools: IndexMap<String, Vec<IndexMap<String, Value>>>,
}

impl StepTracker {
    /// Creates a new `StepTracker` with the given required steps.
    pub fn new(required_steps: Vec<String>) -> Self {
        Self {
            required_steps,
            completed_steps: IndexSet::new(),
            executed_tools: IndexMap::new(),
        }
    }

    /// Record that a tool was executed with the given arguments.
    ///
    /// completed_steps is idempotent: recording the same tool twice does not
    /// add a duplicate. executed_tools accumulates all invocations.
    pub fn record(&mut self, tool_name: &str, args: Option<&IndexMap<String, Value>>) {
        self.completed_steps.insert(tool_name.to_string());
        let empty = IndexMap::new();
        let args = args.unwrap_or(&empty);
        self.executed_tools
            .entry(tool_name.to_string())
            .or_default()
            .push(args.clone());
    }

    /// Returns true when all required steps have been recorded.
    /// An empty required_steps list is always satisfied.
    pub fn is_satisfied(&self) -> bool {
        self.required_steps
            .iter()
            .all(|step| self.completed_steps.contains(step))
    }

    /// Returns required steps not yet completed, in declaration order.
    pub fn pending(&self) -> Vec<String> {
        self.required_steps
            .iter()
            .filter(|step| !self.completed_steps.contains(step.as_str()))
            .cloned()
            .collect()
    }

    /// Check whether the prerequisites for a tool are satisfied given current args.
    pub fn check_prerequisites(
        &self,
        _tool_name: &str,
        args: &IndexMap<String, Value>,
        prerequisites: &[Prerequisite],
    ) -> PrerequisiteCheck {
        let mut missing = Vec::new();

        for prereq in prerequisites {
            match prereq {
                Prerequisite::NameOnly(tool) => {
                    if !self.completed_steps.contains(tool.as_str()) {
                        missing.push(tool.clone());
                    }
                }
                Prerequisite::ArgMatched { tool, match_arg } => {
                    let current_val = args.get(match_arg).cloned().unwrap_or(Value::Null);
                    let found = self
                        .executed_tools
                        .get(tool.as_str())
                        .map(|invocations| {
                            invocations.iter().any(|inv| {
                                inv.get(match_arg).cloned().unwrap_or(Value::Null) == current_val
                            })
                        })
                        .unwrap_or(false);
                    if !found {
                        missing.push(tool.clone());
                    }
                }
            }
        }

        if missing.is_empty() {
            PrerequisiteCheck::satisfied()
        } else {
            PrerequisiteCheck::unsatisfied(missing)
        }
    }

    /// Returns a human-readable summary of completed steps.
    ///
    /// If no steps are completed, returns `"[No steps completed yet]"`.
    /// Otherwise returns `"[Steps completed: names]"` in execution order.
    pub fn summary_hint(&self) -> String {
        if self.completed_steps.is_empty() {
            "[No steps completed yet]".to_string()
        } else {
            let names: Vec<&str> = self.completed_steps.iter().map(|s| s.as_str()).collect();
            format!("[Steps completed: {}]", names.join(", "))
        }
    }

    /// Number of completed steps (unique tool names).
    pub fn completed_count(&self) -> usize {
        self.completed_steps.len()
    }

    /// Returns a reference to the required steps list.
    pub fn required_steps(&self) -> &[String] {
        &self.required_steps
    }
}
