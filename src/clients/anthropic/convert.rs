//! Message and response format conversion for Anthropic API.

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::clients::base::{LLMResponse, TextResponse, ToolCall};
use crate::core::tool_spec::ToolSpec;

/// Convert tool specs to Anthropic tool definitions.
pub fn convert_tools(tools: &[ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.get_json_schema(),
            })
        })
        .collect()
}

/// Convert standardized messages to Anthropic format.
///
/// Extracts system messages separately. Converts tool_calls to tool_use
/// content blocks, tool results to tool_result blocks. Injects synthetic
/// error results for unpaired tool_use. Merges consecutive same-role
/// messages.
pub fn convert_messages(messages: &[Value]) -> (Option<Value>, Vec<Value>) {
    let mut system = None;
    let mut converted = Vec::new();
    let mut pending_tool_use_ids: Vec<String> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").cloned().unwrap_or(Value::Null);

        if role == "system" {
            system = Some(content);
            continue;
        }

        if let Some(tool_calls) = msg.get("tool_calls").and_then(|tc| tc.as_array()) {
            let mut blocks: Vec<Value> = Vec::new();
            if let Some(text) = content.as_str() {
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
            }
            for tc in tool_calls {
                let name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("");
                let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
                let arguments = tc.get("function").and_then(|f| f.get("arguments"));
                let input = match arguments {
                    Some(Value::String(s)) => {
                        serde_json::from_str::<Value>(s).unwrap_or(Value::Object(Map::new()))
                    }
                    Some(v @ Value::Object(_)) => v.clone(),
                    _ => Value::Object(Map::new()),
                };
                let fallback_id = format!("toolu_{}", converted.len());
                blocks.push(json!({
                    "type": "tool_use",
                    "id": if id.is_empty() { fallback_id.as_str() } else { id },
                    "name": name,
                    "input": input,
                }));
                pending_tool_use_ids.push(if id.is_empty() {
                    fallback_id
                } else {
                    id.to_string()
                });
            }
            converted.push(json!({"role": "assistant", "content": blocks}));
            continue;
        }

        if role == "tool" {
            let tool_call_id = msg
                .get("tool_call_id")
                .and_then(|i| i.as_str())
                .unwrap_or("unknown");
            pending_tool_use_ids.retain(|id| id != tool_call_id);
            converted.push(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }],
            }));
            continue;
        }

        if role == "user" && !pending_tool_use_ids.is_empty() {
            let mut blocks: Vec<Value> = pending_tool_use_ids
                .drain(..)
                .map(|id| {
                    json!({
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": "Not executed.",
                        "is_error": true,
                    })
                })
                .collect();
            blocks.extend(content_to_blocks(&content));
            converted.push(json!({"role": "user", "content": blocks}));
            continue;
        }

        converted.push(json!({"role": role, "content": content}));
    }

    merge_consecutive_roles(&mut converted);
    (system, converted)
}

/// Parse an Anthropic API response into LLMResponse.
pub fn parse_response(response: &Value) -> LLMResponse {
    let blocks = match response.get("content").and_then(|c| c.as_array()) {
        Some(b) => b,
        None => return LLMResponse::Text(TextResponse::new("")),
    };

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_uses: Vec<Value> = Vec::new();

    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "text" => {
                text_parts.push(
                    block
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                );
            }
            "tool_use" => {
                tool_uses.push(block.clone());
            }
            _ => {}
        }
    }

    if tool_uses.is_empty() {
        return LLMResponse::Text(TextResponse::new(text_parts.join("")));
    }

    let reasoning_text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for (i, tu) in tool_uses.iter().enumerate() {
        let name = tu.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let id = tu
            .get("id")
            .and_then(|id| id.as_str())
            .map(|s| s.to_string());
        let input = tu
            .get("input")
            .cloned()
            .unwrap_or(Value::Object(Map::new()));
        let args = match input.as_object() {
            Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            None => IndexMap::new(),
        };
        let mut tc = ToolCall::new(name, args);
        if let Some(id_str) = id {
            tc = tc.with_id(id_str);
        }
        if i == 0 {
            if let Some(r) = reasoning_text.as_ref() {
                tc = tc.with_reasoning(r);
            }
        }
        tool_calls.push(tc);
    }

    LLMResponse::ToolCalls(tool_calls)
}

/// Build request body for Anthropic API.
pub fn build_request_body(
    model: &str,
    messages: &[Value],
    max_tokens: i64,
    tools: Option<&[ToolSpec]>,
    tool_choice: Option<&str>,
) -> (Option<Value>, Value) {
    let (system, converted_msgs) = convert_messages(messages);

    let mut body = json!({
        "model": model,
        "messages": converted_msgs,
        "max_tokens": max_tokens,
    });

    if let Some(sys) = &system {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("system".to_string(), sys.clone());
        }
    }

    if let Some(tool_list) = tools {
        if !tool_list.is_empty() {
            let anthropic_tools = convert_tools(tool_list);
            if let Some(obj) = body.as_object_mut() {
                obj.insert("tools".to_string(), json!(anthropic_tools));
            }
            if let Some(tc) = tool_choice {
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("tool_choice".to_string(), json!({"type": tc}));
                }
            }
        }
    }

    (system, body)
}

/// Merge consecutive same-role messages for Anthropic API compatibility.
fn merge_consecutive_roles(messages: &mut Vec<Value>) {
    if messages.len() <= 1 {
        return;
    }
    let mut merged: Vec<Value> = Vec::new();
    for msg in messages.iter() {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if let Some(last) = merged.last_mut() {
            let last_role = last.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if last_role == role {
                let mut blocks = content_to_blocks(last.get("content").unwrap_or(&Value::Null));
                blocks.extend(content_to_blocks(
                    msg.get("content").unwrap_or(&Value::Null),
                ));
                if let Some(obj) = last.as_object_mut() {
                    obj.insert("content".to_string(), Value::Array(blocks));
                }
                continue;
            }
        }

        merged.push(msg.clone());
    }
    *messages = merged;
}

fn content_to_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::Array(blocks) => blocks.clone(),
        Value::String(text) => vec![json!({"type": "text", "text": text})],
        Value::Null => Vec::new(),
        other => vec![json!({"type": "text", "text": other.to_string()})],
    }
}
