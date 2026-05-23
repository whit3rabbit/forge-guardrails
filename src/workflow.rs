use crate::error::ToolResolutionError;
use crate::tool_spec::ToolSpec;
use indexmap::IndexMap;
use std::collections::HashSet;
use std::fmt;

/// Callable signature for tool implementations.
pub type ToolCallable = fn(Vec<String>) -> Result<String, ToolResolutionError>;

/// Re-export ParamModel from the tool_spec module.
pub use crate::tool_spec::ParamModel;

/// A prerequisite specification: either name-only or arg-matched.
#[derive(Debug, Clone, PartialEq)]
pub enum PrerequisiteSpec {
    NameOnly(String),
    ArgMatched { tool: String, match_arg: String },
}

/// Binds a tool spec to a callable with optional prerequisites.
pub struct ToolDef {
    pub spec: ToolSpec,
    pub callable: ToolCallable,
    pub prerequisites: Vec<PrerequisiteSpec>,
}

impl ToolDef {
    pub fn new(spec: ToolSpec, callable: ToolCallable) -> Self {
        Self {
            spec,
            callable,
            prerequisites: Vec::new(),
        }
    }

    pub fn with_prerequisites(mut self, prereqs: Vec<PrerequisiteSpec>) -> Self {
        self.prerequisites = prereqs;
        self
    }

    pub fn name(&self) -> &str {
        &self.spec.name
    }
}

impl fmt::Debug for ToolDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolDef")
            .field("name", &self.spec.name)
            .field("prerequisites", &self.prerequisites)
            .finish()
    }
}

/// A validated workflow definition.
pub struct Workflow {
    pub name: String,
    pub description: String,
    pub tools: IndexMap<String, ToolDef>,
    pub required_steps: Vec<String>,
    pub terminal_tools: HashSet<String>,
    pub system_prompt_template: String,
}

impl Workflow {
    /// Construct and validate a Workflow.
    ///
    /// Validates:
    /// - Each tool dict key matches the tool definition's name.
    /// - Every required step exists in the tools map.
    /// - Every terminal tool exists in the tools map.
    /// - No terminal tool is also a required step.
    /// - Every prerequisite references a tool that exists in the tools map.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        tools: IndexMap<String, ToolDef>,
        required_steps: Vec<String>,
        terminal_tool: TerminalToolInput,
        system_prompt_template: impl Into<String>,
    ) -> Result<Self, String> {
        let name = name.into();
        let description = description.into();
        let system_prompt_template = system_prompt_template.into();

        // Validate tool dict keys match tool def names.
        for (key, def) in &tools {
            if key != &def.spec.name {
                return Err(format!(
                    "Tool dict key '{}' does not match tool definition name '{}'",
                    key, def.spec.name
                ));
            }
        }

        let tool_names: HashSet<&str> = tools.keys().map(|s| s.as_str()).collect();

        // Validate required steps exist in tools.
        for step in &required_steps {
            if !tool_names.contains(step.as_str()) {
                return Err(format!("Required step '{}' not found in tools", step));
            }
        }

        // Normalize terminal_tool to a HashSet.
        let terminal_set: HashSet<String> = match terminal_tool {
            TerminalToolInput::Single(s) => {
                let mut set = HashSet::new();
                set.insert(s);
                set
            }
            TerminalToolInput::Multiple(v) => v.into_iter().collect(),
        };

        // Validate terminal tools exist in tools.
        for t in &terminal_set {
            if !tool_names.contains(t.as_str()) {
                return Err(format!("Terminal tool '{}' not found in tools", t));
            }
        }

        // Validate terminal tools are not also required steps.
        let required_set: HashSet<&str> = required_steps.iter().map(|s| s.as_str()).collect();
        for t in &terminal_set {
            if required_set.contains(t.as_str()) {
                return Err(format!(
                    "Terminal tool '{}' cannot also be a required step",
                    t
                ));
            }
        }

        // Validate prerequisites reference existing tools.
        for (_, def) in &tools {
            for prereq in &def.prerequisites {
                let prereq_tool = match prereq {
                    PrerequisiteSpec::NameOnly(name) => name.as_str(),
                    PrerequisiteSpec::ArgMatched { tool, .. } => tool.as_str(),
                };
                if !tool_names.contains(prereq_tool) {
                    return Err(format!(
                        "Prerequisite references tool '{}' which is not in the tools map",
                        prereq_tool
                    ));
                }
            }
        }

        Ok(Self {
            name,
            description,
            tools,
            required_steps,
            terminal_tools: terminal_set,
            system_prompt_template,
        })
    }

    /// Render the system prompt template with the provided variables.
    pub fn build_system_prompt(&self, vars: &IndexMap<String, String>) -> String {
        let mut result = self.system_prompt_template.clone();
        for (key, value) in vars {
            let pattern = format!("{{{}}}", key);
            result = result.replace(&pattern, value);
        }
        result
    }

    /// Get all tool specs in insertion order.
    pub fn get_tool_specs(&self) -> Vec<&ToolSpec> {
        self.tools.values().map(|def| &def.spec).collect()
    }

    /// Get the callable for a tool by name.
    ///
    /// Returns an error if the tool name is not found.
    pub fn get_callable(&self, tool_name: &str) -> Result<ToolCallable, String> {
        match self.tools.get(tool_name) {
            Some(def) => Ok(def.callable),
            None => Err(format!("Tool '{}' not found", tool_name)),
        }
    }
}

impl fmt::Debug for Workflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Workflow")
            .field("name", &self.name)
            .field("required_steps", &self.required_steps)
            .field("terminal_tools", &self.terminal_tools)
            .finish()
    }
}

/// Input type for terminal_tool: either a single string or a list.
#[derive(Debug, Clone)]
pub enum TerminalToolInput {
    Single(String),
    Multiple(Vec<String>),
}

impl From<String> for TerminalToolInput {
    fn from(s: String) -> Self {
        Self::Single(s)
    }
}

impl From<Vec<String>> for TerminalToolInput {
    fn from(v: Vec<String>) -> Self {
        Self::Multiple(v)
    }
}
