use crate::core::tool_spec::ToolSpec;
use crate::error::{ToolError, ToolResolutionError};
use futures_util::future::BoxFuture;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

/// Callable signature for tool implementations: takes JSON arguments, returns a future yielding a JSON value.
pub type ToolCallable = Arc<
    dyn Fn(IndexMap<String, Value>) -> BoxFuture<'static, Result<Value, ToolError>> + Send + Sync,
>;

/// Trait to automatically convert sync or async tools into a standard ToolCallable.
pub trait IntoToolCallable {
    /// Converts this type into a boxed `ToolCallable` future wrapper.
    fn into_callable(self) -> ToolCallable;
}

impl IntoToolCallable for ToolCallable {
    fn into_callable(self) -> Self {
        self
    }
}

// Convert the old sync signature Fn(Vec<String>) -> Result<String, ToolResolutionError>
impl<F> IntoToolCallable for F
where
    F: Fn(Vec<String>) -> Result<String, ToolResolutionError> + Send + Sync + 'static,
{
    fn into_callable(self) -> ToolCallable {
        let func_arc = Arc::new(self);
        Arc::new(move |args| {
            let func = func_arc.clone();
            Box::pin(async move {
                let mut vec_args = Vec::new();
                for (k, v) in args {
                    let val_str = match v {
                        Value::String(s) => s,
                        other => other.to_string(),
                    };
                    vec_args.push(format!("{}={}", k, val_str));
                }
                (*func)(vec_args)
                    .map(Value::String)
                    .map_err(ToolError::Resolution)
            })
        })
    }
}

/// Re-export ParamModel from the tool_spec module.
pub use crate::core::tool_spec::ParamModel;

/// A prerequisite specification: either name-only or arg-matched.
#[derive(Debug, Clone, PartialEq)]
pub enum PrerequisiteSpec {
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

/// Binds a tool spec to a callable with optional prerequisites.
#[derive(Clone)]
pub struct ToolDef {
    /// The tool schema/specification.
    pub spec: ToolSpec,
    /// The asynchronous function pointer executing the tool.
    pub callable: ToolCallable,
    /// Optional dependencies/prerequisites for this tool.
    pub prerequisites: Vec<PrerequisiteSpec>,
}

impl ToolDef {
    /// Creates a new `ToolDef` linking a spec to a callable.
    pub fn new<C>(spec: ToolSpec, callable: C) -> Self
    where
        C: IntoToolCallable,
    {
        Self {
            spec,
            callable: callable.into_callable(),
            prerequisites: Vec::new(),
        }
    }

    /// Appends prerequisites to the tool definition.
    pub fn with_prerequisites(mut self, prereqs: Vec<PrerequisiteSpec>) -> Self {
        self.prerequisites = prereqs;
        self
    }

    /// Returns the name of the tool.
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
    /// Name of the workflow.
    pub name: String,
    /// Description of the workflow.
    pub description: String,
    /// Map of tool names to their definition.
    pub tools: IndexMap<String, ToolDef>,
    /// List of step names that must be completed.
    pub required_steps: Vec<String>,
    /// Set of tools designated as terminal (success terminates loop).
    pub terminal_tools: HashSet<String>,
    /// System prompt template containing variable interpolation placeholders.
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
            Some(def) => Ok(def.callable.clone()),
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
    /// A single terminal tool name.
    Single(String),
    /// Multiple terminal tool names.
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
