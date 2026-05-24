//! Prompt template system for tool descriptions and tool call extraction.
//!
//! Provides three public functions:
//! - build_tool_prompt: generates structured tool descriptions from ToolSpecs
//! - extract_tool_call: parses tool calls from freeform model output
//! - rescue_tool_call: fallback parsing with multiple strategies

pub mod nudges;
mod parse_strategies;

use crate::clients::base::ToolCall;
use crate::core::tool_spec::ToolSpec;
use indexmap::IndexMap;
pub use nudges::{prerequisite_nudge, retry_nudge, step_nudge, unknown_tool_nudge};
use serde_json::Value;

/// Build a structured prompt describing available tools and the expected
/// JSON call format.
pub fn build_tool_prompt(tools: &[ToolSpec]) -> String {
    let mut lines = vec![
        "You have access to the following tools:".to_string(),
        String::new(),
    ];
    for tool in tools {
        let schema = tool.get_json_schema();
        let properties = schema.get("properties").and_then(Value::as_object);
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .collect::<std::collections::HashSet<_>>()
            })
            .unwrap_or_default();

        lines.push(format!("## {}", tool.name));
        lines.push(format!("Description: {}", tool.description));
        if let Some(properties) = properties.filter(|props| !props.is_empty()) {
            lines.push("Parameters:".to_string());
            for (name, prop) in properties {
                let req = if required.contains(name.as_str()) {
                    " (required)"
                } else {
                    " (optional)"
                };
                let ptype = prop.get("type").and_then(Value::as_str).unwrap_or("any");
                let desc = prop
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                lines.push(format!("  - {} ({}{}): {}", name, ptype, req, desc));
                if let Some(enum_values) = prop.get("enum").and_then(Value::as_array) {
                    let allowed = enum_values
                        .iter()
                        .map(|v| match v.as_str() {
                            Some(s) => s.to_string(),
                            None => v.to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    lines.push(format!("    Allowed values: {}", allowed));
                }
            }
        }
        lines.push(String::new());
    }
    lines.push("To call a tool, respond with ONLY a JSON object in this exact format:".to_string());
    lines.push("{\"tool\": \"<tool_name>\", \"args\": {<arguments>}}".to_string());
    lines.push(String::new());
    lines.push("Example:".to_string());
    if let Some(example_tool) = tools.first() {
        lines.push(example_tool_call(example_tool));
    }
    lines.push(String::new());
    lines.push("Respond with ONLY the JSON tool call. Do not include any other text.".to_string());

    lines.join("\n")
}

fn example_tool_call(tool: &ToolSpec) -> String {
    let schema = tool.get_json_schema();
    let args = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| {
            properties
                .keys()
                .map(|name| {
                    format!(
                        "{}: {}",
                        json_string(name),
                        json_string(&format!("<{}>", name))
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!(
        "{{\"tool\": {}, \"args\": {{{}}}}}",
        json_string(&tool.name),
        args
    )
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).expect("string serialization cannot fail")
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
