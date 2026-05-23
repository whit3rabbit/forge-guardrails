//! Prompt template system for tool descriptions and tool call extraction.
//!
//! Provides three public functions:
//! - build_tool_prompt: generates structured tool descriptions from ToolSpecs
//! - extract_tool_call: parses tool calls from freeform model output
//! - rescue_tool_call: fallback parsing with multiple strategies

pub mod nudges;
mod parse_strategies;

use crate::clients::base::ToolCall;
use crate::core::tool_spec::{ParamModel, ToolSpec};
use indexmap::IndexMap;
pub use nudges::{prerequisite_nudge, retry_nudge, step_nudge, unknown_tool_nudge};
use serde_json::Value;

/// Build a structured prompt describing available tools and the expected
/// JSON call format.
pub fn build_tool_prompt(tools: &[ToolSpec]) -> String {
    let mut parts = Vec::new();
    parts.push("# Available Tools\n".to_string());
    for tool in tools {
        parts.push(format!("## {}\n", tool.name));
        parts.push(format!("{}\n", tool.description));
        parts.push("Parameters:\n".to_string());
        render_parameters(&tool.parameters, &mut parts);
        parts.push(String::new());
    }
    let example_args = build_example_args(tools);
    parts.push(format!(
        "Respond with a JSON tool call in this format:\n\
         {{\"tool\": \"<tool_name>\", \"args\": {{<arguments>}}}}\n\
         Example: {{\"tool\": \"<tool_name>\", \"args\": {{{}}}}}\n\
         Respond with only the JSON tool call.",
        example_args
    ));
    parts.join("\n")
}

fn render_parameters(param: &ParamModel, parts: &mut Vec<String>) {
    let properties = match param {
        ParamModel::Object { properties, .. } => properties,
        _ => return,
    };
    for (name, model) in properties {
        let (type_str, required_str, desc, enum_vals) = model_info(model);
        let mut line = format!("- {} ({}): {}", name, type_str, required_str);
        if let Some(d) = desc {
            line.push_str(&format!(". {}", d));
        }
        parts.push(line);
        if let Some(evs) = enum_vals {
            parts.push(format!("  Allowed values: {}", evs.join(", ")));
        }
    }
}

fn model_info(
    model: &ParamModel,
) -> (
    &'static str,
    &'static str,
    Option<&str>,
    Option<Vec<String>>,
) {
    match model {
        ParamModel::String {
            description,
            required,
            enum_values,
            ..
        } => (
            "string",
            if *required { "required" } else { "optional" },
            description.as_deref(),
            enum_values.clone(),
        ),
        ParamModel::Number {
            description,
            required,
            ..
        } => (
            "number",
            if *required { "required" } else { "optional" },
            description.as_deref(),
            None,
        ),
        ParamModel::Boolean {
            description,
            required,
            ..
        } => (
            "boolean",
            if *required { "required" } else { "optional" },
            description.as_deref(),
            None,
        ),
        ParamModel::Integer {
            description,
            required,
            ..
        } => (
            "integer",
            if *required { "required" } else { "optional" },
            description.as_deref(),
            None,
        ),
        ParamModel::Object {
            description,
            required,
            ..
        } => (
            "object",
            if *required { "required" } else { "optional" },
            description.as_deref(),
            None,
        ),
        ParamModel::Array {
            description,
            required,
            ..
        } => (
            "array",
            if *required { "required" } else { "optional" },
            description.as_deref(),
            None,
        ),
        ParamModel::Unsupported { type_name } => {
            let leaked: &'static str = Box::leak(type_name.clone().into_boxed_str());
            (leaked, "optional", None, None)
        }
    }
}

fn build_example_args(tools: &[ToolSpec]) -> String {
    let first = match tools.first() {
        Some(t) => t,
        None => return String::new(),
    };
    match &first.parameters {
        ParamModel::Object { properties, .. } => properties
            .keys()
            .map(|k| format!("\"{}\": ...", k))
            .collect::<Vec<_>>()
            .join(", "),
        _ => String::new(),
    }
}

/// Extract tool calls from model output text.
///
/// Strips markdown code fences, scans for balanced JSON object boundaries,
/// and recognizes forge format {"tool": "name", "args": {...}} and
/// OpenAI format {"name": "name", "arguments": {...}}.
pub fn extract_tool_call(text: &str, available_tools: &[&str]) -> Vec<ToolCall> {
    let text = strip_code_fences(text);
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(end) = find_balanced_brace(&text[i..]) {
                let candidate = &text[i..i + end + 1];
                if let Some(tc) = parse_tool_json(candidate, available_tools) {
                    results.push(tc);
                }
                i += end + 1;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    results
}

fn strip_code_fences(text: &str) -> String {
    let mut result = text.to_string();
    result = strip_fence(&result, "```json");
    result = strip_fence(&result, "```");
    result
}

fn strip_fence(text: &str, fence: &str) -> String {
    let mut result = text.to_string();
    if let Some(rest) = result.strip_prefix(fence) {
        if let Some(inner) = rest.strip_suffix("```") {
            return inner.trim().to_string();
        }
        result = rest.to_string();
    }
    if let Some(rest) = result.strip_suffix("```") {
        if let Some(idx) = rest.rfind("```") {
            let inner = &rest[idx + fence.len() - 3..];
            return inner.trim().to_string();
        }
    }
    result
}

/// Find the end index of a balanced brace sequence starting at position 0.
/// Handles braces inside string literals and escaped quotes.
fn find_balanced_brace(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || bytes[0] != b'{' {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut i = 0;
    while i < bytes.len() {
        let ch = bytes[i];
        if in_string {
            if ch == b'\\' {
                i += 2;
                continue;
            }
            if ch == b'"' {
                in_string = false;
            }
        } else {
            match ch {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn parse_tool_json(json_str: &str, available_tools: &[&str]) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(json_str).ok()?;
    let obj = v.as_object()?;
    let tool_name = if let Some(name) = obj.get("tool").and_then(|v| v.as_str()) {
        name.to_string()
    } else if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
        name.to_string()
    } else {
        return None;
    };
    if !available_tools.contains(&tool_name.as_str()) {
        return None;
    }
    let args_val = obj
        .get("args")
        .or_else(|| obj.get("arguments"))
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));
    let args_map = args_val.as_object()?;
    let args: IndexMap<String, Value> = args_map
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Some(ToolCall::new(tool_name, args))
}

/// Fallback parser for tool calls hidden in malformed model output.
///
/// Strips thinking-block tags, then tries strategies in strict priority order:
/// 1. JSON extraction via extract_tool_call
/// 2. Rehearsal syntax: tool_name[ARGS]{json_args}
/// 3. Qwen XML format: <function=name>...</function>
/// 4. Mistral bracket-tag: [TOOL_CALLS]name{...}
pub fn rescue_tool_call(text: &str, available_tools: &[&str]) -> Vec<ToolCall> {
    let text = parse_strategies::strip_think_tags(text);
    if text.trim().is_empty() {
        return Vec::new();
    }
    // Strategy 1: JSON extraction
    let result = extract_tool_call(&text, available_tools);
    if !result.is_empty() {
        return result;
    }
    // Strategy 2: Rehearsal syntax
    let result = parse_strategies::parse_rehearsal(&text, available_tools);
    if !result.is_empty() {
        return result;
    }
    // Strategy 3: Qwen XML
    let result = parse_strategies::parse_qwen_xml(&text, available_tools);
    if !result.is_empty() {
        return result;
    }
    // Strategy 4: Mistral bracket-tag
    parse_strategies::parse_mistral_bracket(&text, available_tools)
}
