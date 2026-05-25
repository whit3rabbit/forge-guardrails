//! Structured guardrail policy types and deterministic argument validation.

use crate::clients::base::ToolCall;
use crate::core::tool_spec::{ParamModel, ToolSpec};
use crate::guardrails::nudge::Nudge;
use indexmap::{IndexMap, IndexSet};
use serde_json::Value;

/// Structured category for a tool-argument validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArgValidationKind {
    /// A required argument was omitted.
    MissingRequired,
    /// The argument has the wrong JSON type.
    WrongType {
        /// Expected JSON type or schema type.
        expected: String,
        /// Actual JSON type observed in the candidate call.
        actual: String,
    },
    /// The argument is not allowed by an explicit `additionalProperties: false`.
    ExtraArgument,
    /// A string argument was not one of the declared enum values.
    EnumMismatch {
        /// Valid enum values, in schema order.
        allowed: Vec<String>,
    },
}

/// A single argument-validation error for one candidate tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgValidationError {
    /// Tool that received invalid arguments.
    pub tool: String,
    /// Dot/bracket path to the invalid argument.
    pub path: String,
    /// Specific validation failure.
    pub kind: ArgValidationKind,
}

impl ArgValidationError {
    /// Create a new argument-validation error.
    pub fn new(tool: impl Into<String>, path: impl Into<String>, kind: ArgValidationKind) -> Self {
        Self {
            tool: tool.into(),
            path: path.into(),
            kind,
        }
    }

    /// Human-readable repair hint for this validation error.
    pub fn message(&self) -> String {
        match &self.kind {
            ArgValidationKind::MissingRequired => {
                format!("{} is required", self.path)
            }
            ArgValidationKind::WrongType { expected, actual } => {
                format!("{} must be {}, got {}", self.path, expected, actual)
            }
            ArgValidationKind::ExtraArgument => {
                format!("{} is not allowed", self.path)
            }
            ArgValidationKind::EnumMismatch { allowed } => {
                format!("{} must be one of: {}", self.path, allowed.join(", "))
            }
        }
    }
}

/// Structured explanation for a guardrail decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailViolation {
    /// The model produced text or an empty tool-call batch when a tool call was required.
    NoToolCall,
    /// The model called a tool that is not available.
    UnknownTool {
        /// Tool name from the candidate call.
        called: String,
        /// Available tool names, in declaration order.
        available: Vec<String>,
    },
    /// A terminal tool was called before required steps were complete.
    PrematureTerminal {
        /// Terminal tool that was attempted.
        terminal: String,
        /// Required steps still pending.
        pending: Vec<String>,
    },
    /// A tool was called before its prerequisites were complete.
    MissingPrerequisite {
        /// Tool that was blocked.
        tool: String,
        /// Missing prerequisite identifiers.
        missing: Vec<String>,
    },
    /// One or more tool-call arguments failed deterministic schema validation.
    InvalidArguments {
        /// Tool whose arguments failed validation.
        tool: String,
        /// Argument validation failures.
        errors: Vec<ArgValidationError>,
    },
    /// A terminal tool was combined with non-terminal work in one pending-step batch.
    UnsafeBatch {
        /// Brief reason for blocking the batch.
        reason: String,
    },
    /// Repeated failure of the same guardrail category.
    RepeatedFailure {
        /// Failure kind identifier.
        kind: String,
        /// Repeated count observed by the caller.
        count: usize,
    },
    /// Reserved for future semantic scoring; deterministic policy does not emit this.
    WrongToolLikely {
        /// Tool selected by the model.
        called: String,
        /// Suggested alternatives.
        suggested: Vec<String>,
    },
}

/// Snapshot of the current deterministic guardrail state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailState {
    /// Required steps already completed.
    pub completed_steps: Vec<String>,
    /// Required steps still pending.
    pub pending_steps: Vec<String>,
    /// Tools the model should call next under the current step state.
    pub allowed_next_tools: Vec<String>,
    /// Tools blocked until the current required steps are complete.
    pub blocked_tools: Vec<String>,
    /// Terminal tools that can end the workflow once requirements are met.
    pub terminal_tools: Vec<String>,
}

impl GuardrailState {
    /// Build state from completed/pending step lists plus known tools.
    pub fn from_parts(
        completed_steps: Vec<String>,
        pending_steps: Vec<String>,
        tool_names: &[String],
        terminal_tools: &IndexSet<String>,
    ) -> Self {
        let terminal_list: Vec<String> = terminal_tools.iter().cloned().collect();
        let blocked_tools = if pending_steps.is_empty() {
            Vec::new()
        } else {
            terminal_list.clone()
        };
        let allowed_next_tools = if pending_steps.is_empty() {
            tool_names.to_vec()
        } else {
            pending_steps.clone()
        };

        Self {
            completed_steps,
            pending_steps,
            allowed_next_tools,
            blocked_tools,
            terminal_tools: terminal_list,
        }
    }
}

/// Deterministic policy decision for a candidate tool-call batch.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardrailDecision {
    /// The tool calls are safe to continue through the existing execution path.
    Allow(Vec<ToolCall>),
    /// The model should be nudged and retried.
    Nudge {
        /// Structured violation that caused the nudge.
        violation: GuardrailViolation,
        /// User-facing nudge message.
        nudge: Nudge,
    },
    /// The guardrail budget is exhausted or the request must be rejected.
    Reject {
        /// Structured violation that caused rejection.
        violation: GuardrailViolation,
        /// Human-readable rejection message.
        message: String,
    },
}

/// Validate one tool call against its `ToolSpec`.
pub fn validate_tool_arguments(call: &ToolCall, spec: &ToolSpec) -> Vec<ArgValidationError> {
    let mut errors = Vec::new();
    let root_schema = spec.json_schema.as_ref();
    if let ParamModel::Object { properties, .. } = &spec.parameters {
        validate_object(
            &call.tool,
            "",
            &call.args,
            properties,
            root_schema,
            root_schema,
            &mut errors,
        );
    }
    errors
}

/// Validate a batch of calls against known tool specs.
pub fn validate_tool_call_batch(
    calls: &[ToolCall],
    specs: &IndexMap<String, ToolSpec>,
) -> Vec<ArgValidationError> {
    let mut errors = Vec::new();
    for call in calls {
        if let Some(spec) = specs.get(&call.tool) {
            errors.extend(validate_tool_arguments(call, spec));
        }
    }
    errors
}

fn validate_object(
    tool: &str,
    base_path: &str,
    args: &IndexMap<String, Value>,
    properties: &IndexMap<String, ParamModel>,
    schema: Option<&Value>,
    root_schema: Option<&Value>,
    errors: &mut Vec<ArgValidationError>,
) {
    let schema = resolve_schema(schema, root_schema);
    for (name, model) in properties {
        let path = join_path(base_path, name);
        match args.get(name) {
            Some(value) => validate_value(
                tool,
                &path,
                value,
                model,
                property_schema(schema, name, root_schema),
                root_schema,
                errors,
            ),
            None if model.is_required() => {
                errors.push(ArgValidationError::new(
                    tool,
                    path,
                    ArgValidationKind::MissingRequired,
                ));
            }
            None => {}
        }
    }

    if additional_properties_false(schema) {
        for key in args.keys() {
            if !properties.contains_key(key) {
                errors.push(ArgValidationError::new(
                    tool,
                    join_path(base_path, key),
                    ArgValidationKind::ExtraArgument,
                ));
            }
        }
    }
}

fn validate_value(
    tool: &str,
    path: &str,
    value: &Value,
    model: &ParamModel,
    schema: Option<&Value>,
    root_schema: Option<&Value>,
    errors: &mut Vec<ArgValidationError>,
) {
    match model {
        ParamModel::String { enum_values, .. } => {
            if let Some(actual) = wrong_type(value, "string") {
                errors.push(wrong_type_error(tool, path, "string", actual));
                return;
            }
            if let (Some(allowed), Some(actual)) = (enum_values, value.as_str()) {
                if !allowed.iter().any(|item| item == actual) {
                    errors.push(ArgValidationError::new(
                        tool,
                        path,
                        ArgValidationKind::EnumMismatch {
                            allowed: allowed.clone(),
                        },
                    ));
                }
            }
        }
        ParamModel::Number { .. } => {
            if let Some(actual) = wrong_type(value, "number") {
                errors.push(wrong_type_error(tool, path, "number", actual));
            }
        }
        ParamModel::Boolean { .. } => {
            if let Some(actual) = wrong_type(value, "boolean") {
                errors.push(wrong_type_error(tool, path, "boolean", actual));
            }
        }
        ParamModel::Integer { .. } => {
            if !(value.as_i64().is_some() || value.as_u64().is_some()) {
                errors.push(wrong_type_error(tool, path, "integer", actual_type(value)));
            }
        }
        ParamModel::Object { properties, .. } => {
            let Some(obj) = value.as_object() else {
                errors.push(wrong_type_error(tool, path, "object", actual_type(value)));
                return;
            };
            let nested_args: IndexMap<String, Value> = obj
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            validate_object(
                tool,
                path,
                &nested_args,
                properties,
                schema,
                root_schema,
                errors,
            );
        }
        ParamModel::Array { items, .. } => {
            let Some(values) = value.as_array() else {
                errors.push(wrong_type_error(tool, path, "array", actual_type(value)));
                return;
            };
            let item_schema = item_schema(schema, root_schema);
            for (idx, item) in values.iter().enumerate() {
                validate_value(
                    tool,
                    &join_index(path, idx),
                    item,
                    items,
                    item_schema,
                    root_schema,
                    errors,
                );
            }
        }
        ParamModel::Unsupported { .. } => {}
    }
}

fn wrong_type(value: &Value, expected: &str) -> Option<&'static str> {
    match expected {
        "string" if value.is_string() => None,
        "number" if value.is_number() => None,
        "boolean" if value.is_boolean() => None,
        _ => Some(actual_type(value)),
    }
}

fn wrong_type_error(
    tool: &str,
    path: &str,
    expected: impl Into<String>,
    actual: impl Into<String>,
) -> ArgValidationError {
    ArgValidationError::new(
        tool,
        path,
        ArgValidationKind::WrongType {
            expected: expected.into(),
            actual: actual.into(),
        },
    )
}

fn actual_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn additional_properties_false(schema: Option<&Value>) -> bool {
    schema
        .and_then(|schema| schema.get("additionalProperties"))
        .and_then(Value::as_bool)
        == Some(false)
}

fn property_schema<'a>(
    schema: Option<&'a Value>,
    name: &str,
    root_schema: Option<&'a Value>,
) -> Option<&'a Value> {
    let schema = resolve_schema(schema, root_schema)?;
    schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(name))
}

fn item_schema<'a>(schema: Option<&'a Value>, root_schema: Option<&'a Value>) -> Option<&'a Value> {
    let schema = resolve_schema(schema, root_schema)?;
    schema.get("items")
}

fn resolve_schema<'a>(
    schema: Option<&'a Value>,
    root_schema: Option<&'a Value>,
) -> Option<&'a Value> {
    let schema = schema?;
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        return resolve_ref(reference, root_schema);
    }
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        return any_of
            .iter()
            .find(|candidate| candidate.get("type").and_then(Value::as_str) != Some("null"))
            .and_then(|candidate| resolve_schema(Some(candidate), root_schema));
    }
    Some(schema)
}

fn resolve_ref<'a>(reference: &str, root_schema: Option<&'a Value>) -> Option<&'a Value> {
    let name = reference.strip_prefix("#/$defs/")?;
    root_schema?
        .get("$defs")
        .and_then(Value::as_object)?
        .get(name)
}

fn join_path(base: &str, field: &str) -> String {
    if base.is_empty() {
        field.to_string()
    } else {
        format!("{base}.{field}")
    }
}

fn join_index(base: &str, index: usize) -> String {
    format!("{base}[{index}]")
}
