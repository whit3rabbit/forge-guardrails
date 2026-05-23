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
                blocks.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input,
                }));
            }
            converted.push(json!({"role": "assistant", "content": blocks}));
            continue;
        }

        if role == "tool" {
            let tool_call_id = msg
                .get("tool_call_id")
                .and_then(|i| i.as_str())
                .unwrap_or("");
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

        converted.push(json!({"role": role, "content": content}));
    }

    inject_synthetic_tool_results(&mut converted);
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

/// Inject synthetic error tool_results for unpaired tool_use blocks.
fn inject_synthetic_tool_results(messages: &mut Vec<Value>) {
    let (used_ids, tool_use_ids): (Vec<String>, Vec<String>) = {
        let mut used: Vec<String> = Vec::new();
        let mut tool_uses: Vec<String> = Vec::new();
        for msg in messages.iter() {
            if let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) {
                for block in blocks {
                    let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match block_type {
                        "tool_result" => {
                            if let Some(id) = block.get("tool_use_id").and_then(|i| i.as_str()) {
                                used.push(id.to_string());
                            }
                        }
                        "tool_use" => {
                            if let Some(id) = block.get("id").and_then(|i| i.as_str()) {
                                tool_uses.push(id.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        (used, tool_uses)
    };

    let unpaired_ids: Vec<String> = tool_use_ids
        .into_iter()
        .filter(|id| !used_ids.contains(id))
        .collect();

    for id in unpaired_ids {
        messages.push(json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": id,
                "content": "Error: tool result not provided",
                "is_error": true,
            }],
        }));
    }
}

/// Merge consecutive same-role messages for Anthropic API compatibility.
fn merge_consecutive_roles(messages: &mut Vec<Value>) {
    if messages.len() <= 1 {
        return;
    }
    let mut merged: Vec<Value> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let current = &messages[i];
        let current_role = current.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let has_array_content = current
            .get("content")
            .map(|c| c.is_array())
            .unwrap_or(false);

        if has_array_content || i + 1 >= messages.len() {
            merged.push(current.clone());
            i += 1;
            continue;
        }

        let mut group: Vec<Value> = vec![current.clone()];
        let mut j = i + 1;
        while j < messages.len() {
            let next = &messages[j];
            let next_role = next.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let next_has_array = next.get("content").map(|c| c.is_array()).unwrap_or(false);
            if next_role == current_role && !next_has_array {
                group.push(next.clone());
                j += 1;
            } else {
                break;
            }
        }

        if group.len() == 1 {
            merged.push(group.into_iter().next().expect("one"));
        } else {
            let texts: Vec<String> = group
                .iter()
                .filter_map(|m| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string())
                })
                .collect();
            merged.push(json!({"role": current_role, "content": texts.join("\n")}));
        }
        i = j;
    }
    *messages = merged;
}
