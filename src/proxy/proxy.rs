//! OpenAI-compatible proxy conversion functions.
//!
//! Converts between OpenAI wire format and internal message/response types.
//! Includes SSE event generation for streaming responses.

use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::clients::base::{LLMUsageDetails, TextResponse, TokenUsage, ToolCall};
use crate::core::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use crate::tools::respond::RESPOND_TOOL_NAME;

static CALL_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a random-ish completion ID.
fn generate_completion_id() -> String {
    format!("chatcmpl-{}", uuid_prefix())
}

/// Generate a random-ish call ID.
fn generate_call_id() -> String {
    let id = CALL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{id:016x}")
}

pub(crate) fn openai_stream_completion_id() -> String {
    generate_completion_id()
}

pub(crate) fn text_delta_sse_event(
    completion_id: &str,
    model: &str,
    content: &str,
    include_role: bool,
    usage: Option<&Value>,
) -> Value {
    let mut delta = Map::new();
    if include_role {
        delta.insert("role".into(), json!("assistant"));
    }
    delta.insert("content".into(), json!(content));

    let mut event = json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": Value::Object(delta),
            "finish_reason": Value::Null
        }]
    });
    if let Some(usage_value) = usage {
        event["usage"] = usage_value.clone();
    }
    event
}

pub(crate) fn final_sse_event(
    completion_id: &str,
    model: &str,
    finish_reason: &str,
    usage: Option<&Value>,
) -> Value {
    let mut event = json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    });
    if let Some(usage_value) = usage {
        event["usage"] = usage_value.clone();
    }
    event
}

/// Short hex prefix for IDs (8 chars).
fn uuid_prefix() -> String {
    use std::time::SystemTime;
    let t = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:08x}", (t as u32).wrapping_mul(2654435761))
}

/// Convert OpenAI-format chat messages to internal Message objects.
///
/// Handles system, user, assistant (with optional tool_calls), and tool roles.
/// Unknown roles map to user. List content blocks are joined with newlines.
/// Null/empty content becomes empty string. Empty tool_calls list is treated
/// as a text response.
pub fn openai_to_messages(input: &[Value]) -> Vec<Message> {
    let mut messages = Vec::new();
    for item in input {
        let role_str = item.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = extract_content(item);
        let role = match role_str {
            "system" => MessageRole::System,
            "user" => MessageRole::User,
            "assistant" => MessageRole::Assistant,
            "tool" => MessageRole::Tool,
            _ => MessageRole::User,
        };
        let msg_type = match role {
            MessageRole::System => MessageType::SystemPrompt,
            MessageRole::User => MessageType::UserInput,
            MessageRole::Assistant => MessageType::TextResponse,
            MessageRole::Tool => MessageType::ToolResult,
        };

        let mut msg = Message::new(role, content, MessageMeta::new(msg_type));

        // Handle tool_calls on assistant messages.
        if role == MessageRole::Assistant {
            if let Some(tcs) = item.get("tool_calls").and_then(|t| t.as_array()) {
                if !tcs.is_empty() {
                    let mut infos = Vec::new();
                    for tc in tcs {
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("");
                        let args_raw = tc.get("function").and_then(|f| f.get("arguments"));
                        let args = parse_args_value(args_raw);
                        let call_id = tc
                            .get("id")
                            .and_then(|i| i.as_str())
                            .filter(|id| !id.is_empty())
                            .map(str::to_string)
                            .unwrap_or_else(generate_call_id);
                        infos.push(ToolCallInfo::new(name, Some(args), call_id));
                    }
                    msg = msg.with_tool_calls(infos);
                }
            }
        }

        // Handle tool result fields.
        if role == MessageRole::Tool {
            if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                msg = msg.with_tool_name(name);
            }
            if let Some(id) = item.get("tool_call_id").and_then(|i| i.as_str()) {
                msg = msg.with_tool_call_id(id);
            }
        }

        messages.push(msg);
    }
    messages
}

/// Extract text content from an OpenAI message, handling null, string, and list.
fn extract_content(item: &Value) -> String {
    match item.get("content") {
        None => String::new(),
        Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let texts: Vec<String> = parts
                .iter()
                .filter_map(|p| match p {
                    Value::String(s) => Some(s.clone()),
                    Value::Object(_) => {
                        if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                            p.get("text")
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string())
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect();
            texts.join("\n")
        }
        _ => String::new(),
    }
}

/// Parse arguments from various JSON shapes.
fn parse_args_value(args_raw: Option<&Value>) -> IndexMap<String, Value> {
    match args_raw {
        None => IndexMap::new(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(obj)) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            _ => IndexMap::new(),
        },
        Some(Value::Object(obj)) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => IndexMap::new(),
    }
}

/// Convert internal tool calls to OpenAI chat completion response format.
///
/// Includes reasoning text as message content (or None if no reasoning).
/// Generates unique call and completion IDs.
pub fn tool_calls_to_openai(calls: &[ToolCall], model: &str) -> Value {
    tool_calls_to_openai_with_usage(calls, model, None)
}

/// Convert internal tool calls to OpenAI chat completion response format with usage.
pub fn tool_calls_to_openai_with_usage(
    calls: &[ToolCall],
    model: &str,
    usage: Option<&TokenUsage>,
) -> Value {
    tool_calls_to_openai_with_usage_details(calls, model, usage, None)
}

pub(crate) fn tool_calls_to_openai_with_usage_details(
    calls: &[ToolCall],
    model: &str,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> Value {
    let completion_id = generate_completion_id();
    let mut tool_calls_out = Vec::new();

    let reasoning = calls.first().and_then(|c| c.reasoning.clone());
    let content: Option<String> = reasoning;

    for (i, tc) in calls.iter().enumerate() {
        let call_id = tc
            .id
            .as_deref()
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(generate_call_id);
        let args_json = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());
        let mut func_map = Map::new();
        func_map.insert("name".into(), json!(tc.tool));
        func_map.insert("arguments".into(), json!(args_json));

        let mut tc_map = Map::new();
        tc_map.insert("index".into(), json!(i));
        tc_map.insert("id".into(), json!(call_id));
        tc_map.insert("type".into(), json!("function"));
        tc_map.insert("function".into(), Value::Object(func_map));
        tool_calls_out.push(Value::Object(tc_map));
    }

    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert(
        "content".into(),
        match content {
            Some(c) => json!(c),
            None => Value::Null,
        },
    );
    message.insert("tool_calls".into(), json!(tool_calls_out));

    let mut choice = Map::new();
    choice.insert("index".into(), json!(0));
    choice.insert("message".into(), Value::Object(message));
    choice.insert("finish_reason".into(), json!("tool_calls"));

    json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [Value::Object(choice)],
        "usage": usage_to_openai_json_with_details(usage, usage_details)
    })
}

/// Convert internal TextResponse to OpenAI chat completion response format.
pub fn text_response_to_openai(response: &TextResponse, model: &str) -> Value {
    text_response_to_openai_with_usage(response, model, None)
}

/// Convert internal TextResponse to OpenAI chat completion response format with usage.
pub fn text_response_to_openai_with_usage(
    response: &TextResponse,
    model: &str,
    usage: Option<&TokenUsage>,
) -> Value {
    text_response_to_openai_with_usage_details(response, model, usage, None)
}

pub(crate) fn text_response_to_openai_with_usage_details(
    response: &TextResponse,
    model: &str,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> Value {
    let completion_id = generate_completion_id();

    json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.content},
            "finish_reason": "stop"
        }],
        "usage": usage_to_openai_json_with_details(usage, usage_details)
    })
}

/// Convert tool calls to SSE event chunks for streaming.
///
/// Events: reasoning as content delta first, then tool call deltas with
/// indexed entries, then a final chunk with finish_reason=tool_calls.
/// All events share the same completion ID.
pub fn tool_calls_to_sse_events(calls: &[ToolCall], model: &str) -> Vec<Value> {
    tool_calls_to_sse_events_with_usage(calls, model, None)
}

/// Convert tool calls to SSE event chunks with usage.
pub fn tool_calls_to_sse_events_with_usage(
    calls: &[ToolCall],
    model: &str,
    usage: Option<&TokenUsage>,
) -> Vec<Value> {
    tool_calls_to_sse_event_iter_with_usage(calls, model, usage).collect()
}

/// Convert text to SSE event chunks for streaming.
///
/// First chunk includes role=assistant. Text is split by chunk_size if >0.
/// Final chunk has finish_reason=stop. All events share the same completion ID.
pub fn text_to_sse_events(text: &str, model: &str, chunk_size: usize) -> Vec<Value> {
    text_to_sse_events_with_usage(text, model, chunk_size, None)
}

/// Convert text to SSE event chunks with usage.
pub fn text_to_sse_events_with_usage(
    text: &str,
    model: &str,
    chunk_size: usize,
    usage: Option<&TokenUsage>,
) -> Vec<Value> {
    text_to_sse_event_iter_with_usage(text, model, chunk_size, usage).collect()
}

pub(crate) fn text_to_sse_event_iter_with_usage<'a>(
    text: &'a str,
    model: &'a str,
    chunk_size: usize,
    usage: Option<&'a TokenUsage>,
) -> impl Iterator<Item = Value> + 'a {
    text_to_sse_event_iter_with_usage_details(text, model, chunk_size, usage, None)
}

pub(crate) fn text_to_sse_event_iter_with_usage_details<'a>(
    text: &'a str,
    model: &'a str,
    chunk_size: usize,
    usage: Option<&'a TokenUsage>,
    usage_details: Option<&'a LLMUsageDetails>,
) -> impl Iterator<Item = Value> + 'a {
    let completion_id = generate_completion_id();
    let usage_json = usage.map(|u| usage_to_openai_json_with_details(Some(u), usage_details));
    let mut chunks = text_chunks(text, chunk_size);
    let mut emitted_content = false;
    let mut emitted_final = false;

    std::iter::from_fn(move || {
        if let Some(chunk) = chunks.next() {
            let include_role = !emitted_content;
            emitted_content = true;
            let usage = if include_role {
                usage_json.as_ref()
            } else {
                None
            };
            return Some(text_delta_sse_event(
                &completion_id,
                model,
                chunk,
                include_role,
                usage,
            ));
        }

        if emitted_final {
            return None;
        }
        emitted_final = true;
        Some(final_sse_event(
            &completion_id,
            model,
            "stop",
            usage_json.as_ref(),
        ))
    })
}

pub(crate) fn tool_calls_to_sse_event_iter_with_usage<'a>(
    calls: &'a [ToolCall],
    model: &'a str,
    usage: Option<&'a TokenUsage>,
) -> impl Iterator<Item = Value> + 'a {
    tool_calls_to_sse_event_iter_with_usage_details(calls, model, usage, None)
}

pub(crate) fn tool_calls_to_sse_event_iter_with_usage_details<'a>(
    calls: &'a [ToolCall],
    model: &'a str,
    usage: Option<&'a TokenUsage>,
    usage_details: Option<&'a LLMUsageDetails>,
) -> impl Iterator<Item = Value> + 'a {
    let completion_id = generate_completion_id();
    let usage_json = usage.map(|u| usage_to_openai_json_with_details(Some(u), usage_details));
    let reasoning = calls.first().and_then(|c| c.reasoning.clone());
    let mut step = 0;

    std::iter::from_fn(move || {
        if step == 0 {
            step = 1;
            if let Some(ref r) = reasoning {
                return Some(text_delta_sse_event(
                    &completion_id,
                    model,
                    r,
                    true,
                    usage_json.as_ref(),
                ));
            }
        }

        if step == 1 {
            step = 2;
            let mut tc_deltas = Vec::new();
            for (i, tc) in calls.iter().enumerate() {
                let call_id = tc
                    .id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(generate_call_id);
                let args_json =
                    serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());
                let mut func_map = Map::new();
                func_map.insert("name".into(), json!(tc.tool));
                func_map.insert("arguments".into(), json!(args_json));

                let mut tc_map = Map::new();
                tc_map.insert("index".into(), json!(i));
                tc_map.insert("id".into(), json!(call_id));
                tc_map.insert("type".into(), json!("function"));
                tc_map.insert("function".into(), Value::Object(func_map));
                tc_deltas.push(Value::Object(tc_map));
            }

            let mut delta = Map::new();
            delta.insert("tool_calls".into(), json!(tc_deltas));

            let mut event = json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": Value::Object(delta),
                    "finish_reason": Value::Null
                }]
            });
            if reasoning.is_none() {
                if let Some(ref usage_value) = usage_json {
                    event["usage"] = usage_value.clone();
                }
            }
            return Some(event);
        }

        if step == 2 {
            step = 3;
            return Some(final_sse_event(
                &completion_id,
                model,
                "tool_calls",
                usage_json.as_ref(),
            ));
        }

        None
    })
}

struct TextChunks<'a> {
    text: &'a str,
    chunk_size: usize,
    offset: usize,
    emitted_empty: bool,
    emitted_full: bool,
}

fn text_chunks(text: &str, chunk_size: usize) -> TextChunks<'_> {
    TextChunks {
        text,
        chunk_size,
        offset: 0,
        emitted_empty: false,
        emitted_full: false,
    }
}

impl<'a> Iterator for TextChunks<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk_size == 0 {
            if self.emitted_full {
                return None;
            }
            self.emitted_full = true;
            return Some(self.text);
        }

        if self.text.is_empty() {
            if self.emitted_empty {
                return None;
            }
            self.emitted_empty = true;
            return Some("");
        }

        if self.offset >= self.text.len() {
            return None;
        }

        let start = self.offset;
        let mut end = self.text.len();
        for (count, (idx, _)) in self.text[start..].char_indices().enumerate() {
            if count == self.chunk_size {
                end = start + idx;
                break;
            }
        }
        self.offset = end;
        Some(&self.text[start..end])
    }
}

pub(crate) fn usage_to_openai_json_with_details(
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> Value {
    let usage = usage.cloned().unwrap_or_else(TokenUsage::empty);
    let mut value = json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    });

    if let Some(details) = usage_details {
        if let Some(cached) = details.cached_prompt_tokens {
            value["prompt_tokens_details"] = json!({"cached_tokens": cached});
        }
        if let Some(hit) = details.prompt_cache_hit_tokens {
            value["prompt_cache_hit_tokens"] = json!(hit);
        }
        if let Some(miss) = details.prompt_cache_miss_tokens {
            value["prompt_cache_miss_tokens"] = json!(miss);
        }
    }

    value
}

/// Build a respond ToolSpec in OpenAI wire format for injection.
pub fn respond_tool_openai() -> Value {
    let spec = crate::tools::respond::respond_spec();
    crate::clients::base::format_tool(&spec)
}

/// Check if a list of tool specs already contains the respond tool.
pub fn has_respond_tool(tools: &[Value]) -> bool {
    tools.iter().any(|t| {
        t.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            == Some(RESPOND_TOOL_NAME)
    })
}

/// Extract per-request sampling parameters from a request body.
/// Returns a Map of recognized fields, or None if none found.
pub fn extract_sampling(body: &Value) -> Option<Map<String, Value>> {
    let mut map = Map::new();
    let recognized = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "seed",
        "chat_template_kwargs",
    ];

    for key in &recognized {
        if let Some(v) = body.get(key) {
            map.insert((*key).to_string(), v.clone());
        }
    }

    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Extract non-forge-owned request fields for client passthrough.
pub fn extract_passthrough(body: &Value) -> Option<Map<String, Value>> {
    let obj = body.as_object()?;
    let mut map = Map::new();
    let forge_owned = ["messages", "tools", "stream", "system"];
    let sampling_fields = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "seed",
        "chat_template_kwargs",
    ];

    for (key, value) in obj {
        if forge_owned.contains(&key.as_str()) || sampling_fields.contains(&key.as_str()) {
            continue;
        }
        map.insert(key.clone(), value.clone());
    }

    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Strip respond() tool calls from a response, returning the remaining
/// tool calls and/or extracted respond text.
pub fn strip_respond_calls(calls: &[ToolCall]) -> (Vec<ToolCall>, Option<String>) {
    let mut respond_text = None;
    let mut real_calls = Vec::new();

    for tc in calls {
        if tc.tool == RESPOND_TOOL_NAME {
            let msg = tc
                .args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if respond_text.is_none() {
                respond_text = Some(msg.to_string());
            }
        } else {
            real_calls.push(tc.clone());
        }
    }

    (real_calls, respond_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_to_messages_system_user() {
        let input = vec![
            json!({"role": "system", "content": "You are helpful"}),
            json!({"role": "user", "content": "Hello"}),
        ];
        let msgs = openai_to_messages(&input);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::System);
        assert_eq!(msgs[0].content, "You are helpful");
        assert_eq!(msgs[1].role, MessageRole::User);
        assert_eq!(msgs[1].content, "Hello");
    }

    #[test]
    fn openai_to_messages_assistant_with_tool_calls() {
        let input = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"p\": 1}"}}]
        })];
        let msgs = openai_to_messages(&input);
        assert_eq!(msgs.len(), 1);
        assert!(msgs[0].tool_calls.is_some());
        let calls = msgs[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].call_id, "c1");
    }

    #[test]
    fn openai_to_messages_assistant_tool_call_missing_id_gets_fallback() {
        let input = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": "{}"}}]
        })];
        let msgs = openai_to_messages(&input);
        let calls = msgs[0].tool_calls.as_ref().unwrap();
        assert!(calls[0].call_id.starts_with("call_"));
        assert!(calls[0].call_id.len() > "call_".len());
    }

    #[test]
    fn openai_to_messages_assistant_tool_call_empty_id_gets_fallback() {
        let input = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
        })];
        let msgs = openai_to_messages(&input);
        let calls = msgs[0].tool_calls.as_ref().unwrap();
        assert!(calls[0].call_id.starts_with("call_"));
        assert!(calls[0].call_id.len() > "call_".len());
    }

    #[test]
    fn openai_to_messages_empty_tool_calls_is_text() {
        let input = vec![json!({
            "role": "assistant",
            "content": "Just text",
            "tool_calls": []
        })];
        let msgs = openai_to_messages(&input);
        assert!(msgs[0].tool_calls.is_none());
        assert_eq!(msgs[0].content, "Just text");
    }

    #[test]
    fn openai_to_messages_tool_role() {
        let input = vec![json!({
            "role": "tool",
            "content": "result data",
            "name": "search",
            "tool_call_id": "c1"
        })];
        let msgs = openai_to_messages(&input);
        assert_eq!(msgs[0].role, MessageRole::Tool);
        assert_eq!(msgs[0].tool_name.as_deref(), Some("search"));
        assert_eq!(msgs[0].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn openai_to_messages_unknown_role_maps_to_user() {
        let input = vec![json!({"role": "function", "content": "test"})];
        let msgs = openai_to_messages(&input);
        assert_eq!(msgs[0].role, MessageRole::User);
    }

    #[test]
    fn openai_to_messages_null_content() {
        let input = vec![json!({"role": "user", "content": null})];
        let msgs = openai_to_messages(&input);
        assert_eq!(msgs[0].content, "");
    }

    #[test]
    fn openai_to_messages_list_content() {
        let input = vec![json!({
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
        })];
        let msgs = openai_to_messages(&input);
        assert_eq!(msgs[0].content, "Hello\nWorld");
    }

    #[test]
    fn openai_to_messages_list_content_keeps_string_blocks() {
        let input = vec![json!({
            "role": "user",
            "content": ["Hello", {"type": "text", "text": "World"}]
        })];
        let msgs = openai_to_messages(&input);
        assert_eq!(msgs[0].content, "Hello\nWorld");
    }

    #[test]
    fn tool_calls_to_openai_with_reasoning() {
        let calls = vec![ToolCall::new("search", IndexMap::new()).with_reasoning("thinking...")];
        let result = tool_calls_to_openai(&calls, "test-model");
        let content = result["choices"][0]["message"]["content"].as_str();
        assert_eq!(content, Some("thinking..."));
        assert_eq!(result["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn tool_calls_to_openai_no_reasoning() {
        let calls = vec![ToolCall::new("search", IndexMap::new())];
        let result = tool_calls_to_openai(&calls, "test-model");
        assert!(result["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn tool_calls_to_openai_preserves_existing_call_id() {
        let calls = vec![ToolCall::new("search", IndexMap::new()).with_id("call_keep")];
        let result = tool_calls_to_openai(&calls, "test-model");
        assert_eq!(
            result["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_keep"
        );
    }

    #[test]
    fn tool_calls_to_openai_fallback_ids_are_unique() {
        let calls: Vec<ToolCall> = (0..256)
            .map(|_| ToolCall::new("search", IndexMap::new()))
            .collect();
        let result = tool_calls_to_openai(&calls, "test-model");
        let ids: std::collections::HashSet<String> = result["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|call| call["id"].as_str().unwrap().to_string())
            .collect();

        assert_eq!(ids.len(), calls.len());
    }

    #[test]
    fn text_response_to_openai_basic() {
        let resp = TextResponse::new("Hello world");
        let result = text_response_to_openai(&resp, "test-model");
        assert_eq!(result["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn sse_tool_calls_with_reasoning() {
        let calls = vec![ToolCall::new("search", IndexMap::new()).with_reasoning("hmm")];
        let events = tool_calls_to_sse_events(&calls, "model");
        assert_eq!(events.len(), 3);
        // First: reasoning content delta.
        assert_eq!(events[0]["choices"][0]["delta"]["content"], "hmm");
        // Second: tool_calls delta.
        assert!(events[1]["choices"][0]["delta"]["tool_calls"].is_array());
        // Third: finish_reason = tool_calls.
        assert_eq!(events[2]["choices"][0]["finish_reason"], "tool_calls");
        // All share the same completion ID.
        assert_eq!(events[0]["id"], events[1]["id"]);
        assert_eq!(events[1]["id"], events[2]["id"]);
    }

    #[test]
    fn sse_tool_calls_preserves_existing_call_id() {
        let calls = vec![ToolCall::new("search", IndexMap::new()).with_id("call_keep")];
        let events = tool_calls_to_sse_events(&calls, "model");
        assert_eq!(
            events[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_keep"
        );
    }

    #[test]
    fn sse_text_chunking() {
        let events = text_to_sse_events("abcdefghijkl", "model", 5);
        // 12 chars / 5 chunk_size = 3 content chunks + 1 final = 4.
        assert_eq!(events.len(), 4);
        // Verify content joins to original.
        let content: String = (0..3)
            .map(|i| {
                events[i]["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(content, "abcdefghijkl");
        // Last event is finish.
        assert_eq!(events[3]["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn sse_text_chunking_preserves_multibyte_boundaries() {
        let text = "åßçд🙂z";
        let events = text_to_sse_events(text, "model", 2);
        let content: String = events
            .iter()
            .take(events.len() - 1)
            .map(|event| {
                event["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(content, text);
        assert_eq!(
            events.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    }

    #[test]
    fn sse_text_chunk_size_zero() {
        let events = text_to_sse_events("hello", "model", 0);
        // 1 content chunk + 1 final = 2.
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["choices"][0]["delta"]["content"], "hello");
        assert!(events[0]["choices"][0]["delta"]["role"].is_string());
    }

    #[test]
    fn sse_text_empty_string() {
        let events = text_to_sse_events("", "model", 5);
        // 1 content chunk + 1 final = 2.
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn extract_sampling_basic() {
        let body = json!({"temperature": 0.7, "top_p": 0.9});
        let sampling = extract_sampling(&body).unwrap();
        assert_eq!(sampling["temperature"], 0.7);
        assert_eq!(sampling["top_p"], 0.9);
    }

    #[test]
    fn extract_sampling_none_when_empty() {
        let body = json!({"messages": []});
        assert!(extract_sampling(&body).is_none());
    }

    #[test]
    fn extract_sampling_with_chat_template_kwargs() {
        let body = json!({"temperature": 0.5, "chat_template_kwargs": {"enable_thinking": true}});
        let sampling = extract_sampling(&body).unwrap();
        assert!(sampling.contains_key("chat_template_kwargs"));
    }

    #[test]
    fn extract_sampling_ignores_openai_only_fields() {
        let body = json!({"temperature": 0.5, "frequency_penalty": 1.0, "response_format": {"type": "json_object"}});
        let sampling = extract_sampling(&body).unwrap();
        assert!(sampling.contains_key("temperature"));
        assert!(!sampling.contains_key("frequency_penalty"));
        assert!(!sampling.contains_key("response_format"));
    }

    #[test]
    fn extract_passthrough_keeps_provider_fields() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "stream": true,
            "model": "request-model",
            "max_tokens": 128,
            "stop": ["done"],
            "tool_choice": {"type": "function", "function": {"name": "search"}},
            "response_format": {"type": "json_object"},
            "temperature": 0.5
        });
        let passthrough = extract_passthrough(&body).unwrap();

        assert_eq!(passthrough["model"], "request-model");
        assert_eq!(passthrough["max_tokens"], 128);
        assert_eq!(passthrough["stop"], json!(["done"]));
        assert_eq!(
            passthrough["tool_choice"],
            json!({"type": "function", "function": {"name": "search"}})
        );
        assert_eq!(
            passthrough["response_format"],
            json!({"type": "json_object"})
        );
        assert!(!passthrough.contains_key("messages"));
        assert!(!passthrough.contains_key("tools"));
        assert!(!passthrough.contains_key("stream"));
        assert!(!passthrough.contains_key("temperature"));
    }

    #[test]
    fn response_helpers_emit_nonzero_usage() {
        let usage = TokenUsage::new(11, 5, 16);
        let response = text_response_to_openai_with_usage(
            &TextResponse::new("counted"),
            "model",
            Some(&usage),
        );
        assert_eq!(response["usage"]["prompt_tokens"], 11);
        assert_eq!(response["usage"]["completion_tokens"], 5);
        assert_eq!(response["usage"]["total_tokens"], 16);

        let events = text_to_sse_events_with_usage("counted", "model", 0, Some(&usage));
        assert_eq!(events[0]["usage"]["prompt_tokens"], 11);
        assert_eq!(events.last().unwrap()["usage"]["total_tokens"], 16);
    }

    #[test]
    fn strip_respond_only_respond() {
        let calls = vec![ToolCall::new("respond", {
            let mut m = IndexMap::new();
            m.insert("message".into(), json!("hello"));
            m
        })];
        let (real, text) = strip_respond_calls(&calls);
        assert!(real.is_empty());
        assert_eq!(text, Some("hello".to_string()));
    }

    #[test]
    fn strip_respond_mixed() {
        let calls = vec![
            ToolCall::new("respond", {
                let mut m = IndexMap::new();
                m.insert("message".into(), json!("hi"));
                m
            }),
            ToolCall::new("search", IndexMap::new()),
        ];
        let (real, text) = strip_respond_calls(&calls);
        assert_eq!(real.len(), 1);
        assert_eq!(real[0].tool, "search");
        assert_eq!(text, Some("hi".to_string()));
    }

    #[test]
    fn strip_respond_preserves_first_respond_text() {
        let calls = vec![
            ToolCall::new("respond", {
                let mut m = IndexMap::new();
                m.insert("message".into(), json!("first"));
                m
            }),
            ToolCall::new("respond", {
                let mut m = IndexMap::new();
                m.insert("message".into(), json!("second"));
                m
            }),
        ];
        let (real, text) = strip_respond_calls(&calls);
        assert!(real.is_empty());
        assert_eq!(text, Some("first".to_string()));
    }

    #[test]
    fn has_respond_tool_true() {
        let tools = vec![json!({"function": {"name": "respond"}})];
        assert!(has_respond_tool(&tools));
    }

    #[test]
    fn has_respond_tool_false() {
        let tools = vec![json!({"function": {"name": "search"}})];
        assert!(!has_respond_tool(&tools));
    }
}
