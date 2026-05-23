//! OpenAI-compatible proxy conversion functions.
//!
//! Converts between OpenAI wire format and internal message/response types.
//! Includes SSE event generation for streaming responses.

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::clients::base::{TextResponse, ToolCall};
use crate::core::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use crate::tools::respond::RESPOND_TOOL_NAME;

/// Generate a random-ish completion ID.
fn generate_completion_id() -> String {
    format!("chatcmpl-{}", uuid_prefix())
}

/// Generate a random-ish call ID.
fn generate_call_id() -> String {
    format!("call_{}", uuid_prefix())
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
                            .unwrap_or("")
                            .to_string();
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
                .filter_map(|p| {
                    if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                        p.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
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
/// Generates unique call and completion IDs. Usage fields are zeroed placeholders.
pub fn tool_calls_to_openai(calls: &[ToolCall], model: &str) -> Value {
    let completion_id = generate_completion_id();
    let mut tool_calls_out = Vec::new();

    let reasoning = calls.first().and_then(|c| c.reasoning.clone());
    let content: Option<String> = reasoning;

    for (i, tc) in calls.iter().enumerate() {
        let call_id = generate_call_id();
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
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })
}

/// Convert internal TextResponse to OpenAI chat completion response format.
pub fn text_response_to_openai(response: &TextResponse, model: &str) -> Value {
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
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    })
}

/// Convert tool calls to SSE event chunks for streaming.
///
/// Events: reasoning as content delta first, then tool call deltas with
/// indexed entries, then a final chunk with finish_reason=tool_calls.
/// All events share the same completion ID.
pub fn tool_calls_to_sse_events(calls: &[ToolCall], model: &str) -> Vec<Value> {
    let completion_id = generate_completion_id();
    let mut events = Vec::new();

    // Reasoning delta first.
    let reasoning = calls.first().and_then(|c| c.reasoning.clone());
    if let Some(ref r) = reasoning {
        let mut delta = Map::new();
        delta.insert("role".into(), json!("assistant"));
        delta.insert("content".into(), json!(r));

        events.push(json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": Value::Object(delta),
                "finish_reason": Value::Null
            }]
        }));
    }

    // Tool call deltas.
    let mut tc_deltas = Vec::new();
    for (i, tc) in calls.iter().enumerate() {
        let call_id = generate_call_id();
        let args_json = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());
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

    events.push(json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": Value::Object(delta),
            "finish_reason": Value::Null
        }]
    }));

    // Final chunk.
    events.push(json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    }));

    events
}

/// Convert text to SSE event chunks for streaming.
///
/// First chunk includes role=assistant. Text is split by chunk_size if >0.
/// Final chunk has finish_reason=stop. All events share the same completion ID.
pub fn text_to_sse_events(text: &str, model: &str, chunk_size: usize) -> Vec<Value> {
    let completion_id = generate_completion_id();
    let mut events = Vec::new();

    if chunk_size == 0 {
        // Single content chunk.
        let mut delta = Map::new();
        delta.insert("role".into(), json!("assistant"));
        delta.insert("content".into(), json!(text));

        events.push(json!({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": Value::Object(delta),
                "finish_reason": Value::Null
            }]
        }));
    } else {
        // Split text into chunks.
        let chars: Vec<char> = text.chars().collect();
        let mut first = true;
        for chunk in chars.chunks(chunk_size) {
            let mut delta = Map::new();
            if first {
                delta.insert("role".into(), json!("assistant"));
                first = false;
            }
            delta.insert("content".into(), json!(chunk.iter().collect::<String>()));

            events.push(json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": Value::Object(delta),
                    "finish_reason": Value::Null
                }]
            }));
        }

        // If text was empty, still produce one chunk with role.
        if text.is_empty() {
            let mut delta = Map::new();
            delta.insert("role".into(), json!("assistant"));
            delta.insert("content".into(), json!(""));

            events.push(json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": Value::Object(delta),
                    "finish_reason": Value::Null
                }]
            }));
        }
    }

    // Final chunk with finish_reason=stop.
    events.push(json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }));

    events
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
    ];

    for key in &recognized {
        if let Some(v) = body.get(key) {
            map.insert((*key).to_string(), v.clone());
        }
    }

    // Handle chat_template_kwargs as a nested object.
    if let Some(kwargs) = body.get("chat_template_kwargs") {
        map.insert("chat_template_kwargs".to_string(), kwargs.clone());
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
            respond_text = Some(msg.to_string());
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
