//! Classifier input context and canonical serializer.

use crate::clients::base::ToolCall;
use crate::core::message::{Message, MessageType};
use crate::core::tool_spec::ToolSpec;
use crate::guardrails::step_enforcer::StepEnforcer;
use indexmap::IndexSet;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Workflow state included in classifier input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkflowStateForScoring {
    /// Required workflow step tool names.
    pub required_steps: Vec<String>,
    /// Completed workflow step tool names.
    pub completed_steps: Vec<String>,
    /// Pending workflow step tool names.
    pub pending_steps: Vec<String>,
    /// Terminal tool names.
    pub terminal_tools: Vec<String>,
    /// Recent guardrail or tool errors.
    pub recent_errors: Vec<String>,
}

/// Tool specification included in classifier input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSpecForScoring {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Tool parameter JSON schema.
    pub parameters: Value,
}

/// Candidate tool call shape used by fixture parsing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateCallForScoring {
    /// Candidate tool name.
    pub name: String,
    /// Candidate arguments.
    pub arguments: Value,
}

/// Complete classifier input context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoringContext {
    /// Classifier input schema version.
    pub schema_version: String,
    /// User request being satisfied.
    pub user_request: String,
    /// Current workflow state.
    pub workflow_state: WorkflowStateForScoring,
    /// Tools available to the model.
    pub available_tools: Vec<ToolSpecForScoring>,
    /// Optional generic eval or workflow contracts for semantic scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ScoringMetadata>,
}

/// Optional generic contracts included in v2 classifier input.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ScoringMetadata {
    /// Broad scenario family, not an exact scenario name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario_family: Option<String>,
    /// Whether the task requires semantic argument transformation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_transform: Option<bool>,
    /// Whether the task requires final synthesis from tool results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_synthesis: Option<bool>,
    /// Whether all relevant tool facts must be carried into the answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_all_tool_facts: Option<bool>,
    /// Whether missing data must be explicitly acknowledged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub must_acknowledge_missing_data: Option<bool>,
}

impl ScoringContext {
    /// Build a scoring context from explicit workflow pieces.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        user_request: impl Into<String>,
        required_steps: Vec<String>,
        completed_steps: Vec<String>,
        pending_steps: Vec<String>,
        terminal_tools: Vec<String>,
        recent_errors: Vec<String>,
        tool_specs: &[ToolSpec],
    ) -> Self {
        let available_tools = tool_specs
            .iter()
            .map(|spec| ToolSpecForScoring {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: spec
                    .json_schema
                    .clone()
                    .unwrap_or_else(default_object_schema),
            })
            .collect();

        Self {
            schema_version: "toolcall-verifier-input/v1".to_string(),
            user_request: user_request.into(),
            workflow_state: WorkflowStateForScoring {
                required_steps,
                completed_steps,
                pending_steps,
                terminal_tools,
                recent_errors,
            },
            available_tools,
            metadata: None,
        }
    }

    /// Return a copy of this context with scoring metadata attached.
    pub fn with_metadata(mut self, metadata: ScoringMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Build a scoring context from the current step enforcer.
    pub fn from_step_enforcer(
        user_request: impl Into<String>,
        step_enforcer: &StepEnforcer,
        terminal_tools: &IndexSet<String>,
        recent_errors: Vec<String>,
        tool_specs: &[ToolSpec],
    ) -> Self {
        Self::new(
            user_request,
            step_enforcer.tracker.required_steps().to_vec(),
            step_enforcer.completed_steps().keys().cloned().collect(),
            step_enforcer.pending(),
            terminal_tools.iter().cloned().collect(),
            recent_errors,
            tool_specs,
        )
    }
}

/// Extract recent guardrail and tool errors from a transcript.
pub fn recent_errors_from_messages(messages: &[Message], limit: usize) -> Vec<String> {
    let mut errors = messages
        .iter()
        .rev()
        .filter_map(|message| {
            let is_error = matches!(
                message.metadata.msg_type,
                MessageType::RetryNudge | MessageType::StepNudge | MessageType::PrerequisiteNudge
            ) || (message.metadata.msg_type == MessageType::ToolResult
                && has_error_prefix(&message.content));

            if is_error && !message.content.trim().is_empty() {
                Some(message.content.clone())
            } else {
                None
            }
        })
        .take(limit)
        .collect::<Vec<_>>();
    errors.reverse();
    errors
}

fn has_error_prefix(content: &str) -> bool {
    let normalized = content.trim_start().to_ascii_lowercase();
    normalized.starts_with("[toolerror]")
        || normalized.starts_with("[toolresolutionerror]")
        || normalized.starts_with("[toolexecutionerror]")
        || normalized.starts_with("[tool_error]")
        || normalized.starts_with("[unknown_tool]")
        || normalized.starts_with("[invalidarguments]")
        || normalized.starts_with("[guardrail]")
}

fn default_object_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {}
    })
}

fn py_list(values: &[String]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }

    let body = values
        .iter()
        .map(|value| format!("'{}'", py_single_quote_escape(value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{body}]")
}

fn py_single_quote_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}

fn value_from_index_map(values: &indexmap::IndexMap<String, Value>) -> Value {
    Value::Object(
        values
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

fn python_json_dumps_sort_keys(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => python_json_string(value),
        Value::Array(values) => {
            let body = values
                .iter()
                .map(python_json_dumps_sort_keys)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{body}]")
        }
        Value::Object(values) => python_json_object(values),
    }
}

fn python_json_object(values: &Map<String, Value>) -> String {
    let mut keys = values.keys().collect::<Vec<_>>();
    keys.sort();
    let body = keys
        .into_iter()
        .map(|key| {
            format!(
                "{}: {}",
                python_json_string(key),
                python_json_dumps_sort_keys(&values[key])
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{body}}}")
}

fn python_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch <= '\u{1f}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch if ch as u32 <= 0x7f => out.push(ch),
            ch if ch as u32 <= 0xffff => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => {
                let code = ch as u32 - 0x1_0000;
                let high = 0xd800 + ((code >> 10) & 0x3ff);
                let low = 0xdc00 + (code & 0x3ff);
                out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    out.push('"');
    out
}

fn optional_json_string(value: Option<&str>) -> String {
    value
        .map(python_json_string)
        .unwrap_or_else(|| "null".to_string())
}

fn optional_json_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    }
}

/// Serialize classifier input with the artifact's `serialize_state_v1` format.
pub fn serialize_state_v1(ctx: &ScoringContext, candidate: &ToolCall) -> String {
    let mut out = String::new();

    out.push_str("SCHEMA_VERSION:\n");
    out.push_str(&ctx.schema_version);
    out.push_str("\n\nUSER_REQUEST:\n");
    out.push_str(&ctx.user_request);

    out.push_str("\n\nWORKFLOW_STATE:\n");
    out.push_str(&format!(
        "required_steps={}\n",
        py_list(&ctx.workflow_state.required_steps)
    ));
    out.push_str(&format!(
        "completed_steps={}\n",
        py_list(&ctx.workflow_state.completed_steps)
    ));
    out.push_str(&format!(
        "pending_steps={}\n",
        py_list(&ctx.workflow_state.pending_steps)
    ));
    out.push_str(&format!(
        "terminal_tools={}\n",
        py_list(&ctx.workflow_state.terminal_tools)
    ));
    out.push_str(&format!(
        "recent_errors={}",
        py_list(&ctx.workflow_state.recent_errors)
    ));

    out.push_str("\n\nAVAILABLE_TOOLS:\n");
    let tool_text = ctx
        .available_tools
        .iter()
        .map(|tool| {
            format!(
                "{}: {}\nPARAMETERS: {}",
                tool.name,
                tool.description,
                python_json_dumps_sort_keys(&tool.parameters)
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    out.push_str(&tool_text);

    let candidate_json = Value::Object(
        [
            (
                "arguments".to_string(),
                value_from_index_map(&candidate.args),
            ),
            ("name".to_string(), Value::String(candidate.tool.clone())),
        ]
        .into_iter()
        .collect(),
    );

    out.push_str("\n\nCANDIDATE_CALL:\n");
    out.push_str(&python_json_dumps_sort_keys(&candidate_json));

    out
}

/// Serialize classifier input with the metadata-aware `serialize_state_v2` format.
pub fn serialize_state_v2(ctx: &ScoringContext, candidate: &ToolCall) -> String {
    let mut out = serialize_state_v1(ctx, candidate);

    out.push_str("\n\nSCORING_METADATA:\n");
    let metadata = ctx.metadata.as_ref();
    out.push_str(&format!(
        "scenario_family={}\n",
        optional_json_string(metadata.and_then(|value| value.scenario_family.as_deref()))
    ));
    out.push_str(&format!(
        "requires_transform={}\n",
        optional_json_bool(metadata.and_then(|value| value.requires_transform))
    ));
    out.push_str(&format!(
        "requires_synthesis={}\n",
        optional_json_bool(metadata.and_then(|value| value.requires_synthesis))
    ));
    out.push_str(&format!(
        "requires_all_tool_facts={}\n",
        optional_json_bool(metadata.and_then(|value| value.requires_all_tool_facts))
    ));
    out.push_str(&format!(
        "must_acknowledge_missing_data={}",
        optional_json_bool(metadata.and_then(|value| value.must_acknowledge_missing_data))
    ));

    out
}

/// Serialize candidate tool parameter schema only.
pub fn serialize_candidate_tool_schema(
    tools: &[ToolSpecForScoring],
    candidate: &ToolCall,
) -> String {
    let target = &candidate.tool;
    for t in tools {
        if &t.name == target {
            return format!(
                "{}: {}\nPARAMETERS: {}",
                t.name,
                t.description,
                compact_json(&t.parameters, 2400)
            );
        }
    }
    String::new()
}

/// Serialize name + description only (no parameters) for non-candidate tools.
pub fn serialize_competing_tool_signatures(
    tools: &[ToolSpecForScoring],
    candidate: &ToolCall,
    max_competing: usize,
) -> String {
    let target = &candidate.tool;
    let mut lines = Vec::new();
    for t in tools {
        if &t.name != target {
            lines.push(format!("{}: {}", t.name, t.description));
            if lines.len() >= max_competing {
                break;
            }
        }
    }
    lines.join("\n")
}

fn compact_json(value: &Value, max_chars: usize) -> String {
    let s = python_json_dumps_sort_keys(value);
    if s.chars().count() > max_chars {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...<truncated>", truncated)
    } else {
        s
    }
}

/// Serialize classifier input with the metadata-aware candidate-first `serialize_state_v3` format.
pub fn serialize_state_v3(ctx: &ScoringContext, candidate: &ToolCall) -> String {
    let ws = &ctx.workflow_state;
    let metadata = ctx.metadata.as_ref();

    let candidate_schema = serialize_candidate_tool_schema(&ctx.available_tools, candidate);
    let competing_sigs = serialize_competing_tool_signatures(&ctx.available_tools, candidate, 12);

    let candidate_json = Value::Object(
        [
            (
                "arguments".to_string(),
                value_from_index_map(&candidate.args),
            ),
            ("name".to_string(), Value::String(candidate.tool.clone())),
        ]
        .into_iter()
        .collect(),
    );

    let front = format!(
        "SCHEMA_VERSION:\n{}\n\nUSER_REQUEST:\n{}\n\nCANDIDATE_CALL:\n{}\n\nCANDIDATE_TOOL_SCHEMA:\n{}\n\nWORKFLOW_STATE:\nrequired_steps={}\ncompleted_steps={}\npending_steps={}\nterminal_tools={}\nrecent_errors={}",
        ctx.schema_version,
        ctx.user_request,
        compact_json(&candidate_json, 2400),
        candidate_schema,
        py_list(&ws.required_steps),
        py_list(&ws.completed_steps),
        py_list(&ws.pending_steps),
        py_list(&ws.terminal_tools),
        py_list(&ws.recent_errors),
    );

    let tail = format!(
        "SCORING_METADATA:\nscenario_family={}\nrequires_transform={}\nrequires_synthesis={}\nrequires_all_tool_facts={}\nmust_acknowledge_missing_data={}",
        optional_json_string(metadata.and_then(|value| value.scenario_family.as_deref())),
        optional_json_bool(metadata.and_then(|value| value.requires_transform)),
        optional_json_bool(metadata.and_then(|value| value.requires_synthesis)),
        optional_json_bool(metadata.and_then(|value| value.requires_all_tool_facts)),
        optional_json_bool(metadata.and_then(|value| value.must_acknowledge_missing_data)),
    );

    let other_header = "\n\nOTHER_AVAILABLE_TOOLS:\n";
    let max_length = 1280usize;
    let char_budget = ((max_length as f64) * 3.5) as usize;

    let front_len = front.chars().count();
    let tail_len = tail.chars().count();
    let header_len = other_header.chars().count();

    let mut competing_block = String::new();
    if char_budget > front_len + tail_len + header_len + 2 {
        let remaining = char_budget - front_len - tail_len - header_len - 2;
        let sig_lines: Vec<&str> = competing_sigs
            .split('\n')
            .filter(|s| !s.is_empty())
            .collect();
        let mut kept_lines: Vec<String> = Vec::new();
        let mut used = 0;
        for line in &sig_lines {
            let cost = line.chars().count() + 1;
            if used + cost > remaining {
                break;
            }
            kept_lines.push(line.to_string());
            used += cost;
        }
        let dropped = sig_lines.len() - kept_lines.len();
        if dropped > 0 {
            kept_lines.push(format!("...<+{} more tools>", dropped));
        }
        competing_block = kept_lines.join("\n");
    }

    format!("{}{}{}\n\n{}", front, other_header, competing_block, tail)
        .trim()
        .to_string()
}
