use super::*;
use crate::clients::base::{TextResponse, TokenUsage, ToolCall};
use crate::core::message::MessageRole;
use indexmap::IndexMap;
use serde_json::{json, Value};

#[test]
fn openai_to_messages_system_user() {
    let input = vec![
        json!({"role": "system", "content": "You are helpful"}),
        json!({"role": "user", "content": "Hello"}),
    ];
    let msgs = openai_to_messages(&input).expect("messages");
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
    let msgs = openai_to_messages(&input).expect("messages");
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
    let msgs = openai_to_messages(&input).expect("messages");
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
    let msgs = openai_to_messages(&input).expect("messages");
    let calls = msgs[0].tool_calls.as_ref().unwrap();
    assert!(calls[0].call_id.starts_with("call_"));
    assert!(calls[0].call_id.len() > "call_".len());
}

#[test]
fn openai_to_messages_rewrites_duplicate_tool_call_ids_and_results() {
    let input = vec![
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "dup", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "dup", "type": "function", "function": {"name": "write", "arguments": "{}"}}
            ]
        }),
        json!({
            "role": "tool",
            "name": "read",
            "tool_call_id": "dup",
            "content": "read result"
        }),
        json!({
            "role": "tool",
            "name": "write",
            "tool_call_id": "dup",
            "content": "write result"
        }),
    ];

    let msgs = openai_to_messages(&input).expect("messages");
    let calls = msgs[0].tool_calls.as_ref().unwrap();
    let first_id = calls[0].call_id.as_str();
    let second_id = calls[1].call_id.as_str();

    assert_eq!(first_id, "dup");
    assert_ne!(second_id, "dup");
    assert!(second_id.starts_with("call_"));
    assert_eq!(msgs[1].tool_call_id.as_deref(), Some(first_id));
    assert_eq!(msgs[2].tool_call_id.as_deref(), Some(second_id));
}

#[test]
fn openai_to_messages_empty_tool_calls_is_text() {
    let input = vec![json!({
        "role": "assistant",
        "content": "Just text",
        "tool_calls": []
    })];
    let msgs = openai_to_messages(&input).expect("messages");
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
    let msgs = openai_to_messages(&input).expect("messages");
    assert_eq!(msgs[0].role, MessageRole::Tool);
    assert_eq!(msgs[0].tool_name.as_deref(), Some("search"));
    assert_eq!(msgs[0].tool_call_id.as_deref(), Some("c1"));
}

#[test]
fn openai_to_messages_unknown_role_rejected() {
    let input = vec![json!({"role": "function", "content": "test"})];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("role must be one of"));
}

#[test]
fn openai_to_messages_null_content() {
    let input = vec![json!({"role": "user", "content": null})];
    let msgs = openai_to_messages(&input).expect("messages");
    assert_eq!(msgs[0].content, "");
}

#[test]
fn openai_to_messages_list_content() {
    let input = vec![json!({
        "role": "user",
        "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
    })];
    let msgs = openai_to_messages(&input).expect("messages");
    assert_eq!(msgs[0].content, "Hello\nWorld");
}

#[test]
fn openai_to_messages_list_content_keeps_string_blocks() {
    let input = vec![json!({
        "role": "user",
        "content": ["Hello", {"type": "text", "text": "World"}]
    })];
    let msgs = openai_to_messages(&input).expect("messages");
    assert_eq!(msgs[0].content, "Hello\nWorld");
}

#[test]
fn openai_to_messages_rejects_image_content_part() {
    let input = vec![json!({
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {"url": "https://example.test/image.png"}
        }]
    })];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("type 'image_url' is unsupported"));
}

#[test]
fn openai_to_messages_rejects_audio_content_part() {
    let input = vec![json!({
        "role": "user",
        "content": [{
            "type": "input_audio",
            "input_audio": {"data": "abcd", "format": "wav"}
        }]
    })];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err
        .to_string()
        .contains("type 'input_audio' is unsupported"));
}

#[test]
fn openai_to_messages_rejects_malformed_text_content_part() {
    let input = vec![json!({
        "role": "user",
        "content": [{"type": "text", "text": 7}]
    })];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("content[0].text must be a string"));
}

#[test]
fn openai_to_messages_missing_role_rejected() {
    let input = vec![json!({"content": "test"})];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("role is required"));
}

#[test]
fn openai_to_messages_non_string_role_rejected() {
    let input = vec![json!({"role": 7, "content": "test"})];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("role must be a string"));
}

#[test]
fn openai_to_messages_malformed_tool_arguments_rejected() {
    let input = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{broken"}}]
    })];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("arguments must be valid JSON"));
}

#[test]
fn openai_to_messages_tool_arguments_string_non_object_rejected() {
    let input = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "[]"}}]
    })];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("must decode to a JSON object"));
}

#[test]
fn openai_to_messages_tool_arguments_value_non_object_rejected() {
    let input = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": null}}]
    })];
    let err = openai_to_messages(&input).unwrap_err();
    assert!(err.to_string().contains("must be an object"));
}

#[test]
fn openai_to_messages_absent_tool_arguments_is_empty_object() {
    let input = vec![json!({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read"}}]
    })];
    let msgs = openai_to_messages(&input).expect("messages");
    let calls = msgs[0].tool_calls.as_ref().unwrap();
    assert!(calls[0].args.as_ref().unwrap().is_empty());
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
fn tool_calls_to_openai_rewrites_duplicate_existing_call_ids() {
    let calls = vec![
        ToolCall::new("search", IndexMap::new()).with_id("dup"),
        ToolCall::new("read", IndexMap::new()).with_id("dup"),
        ToolCall::new("write", IndexMap::new()).with_id("keep"),
    ];
    let result = tool_calls_to_openai(&calls, "test-model");
    let ids: Vec<&str> = result["choices"][0]["message"]["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|call| call["id"].as_str().unwrap())
        .collect();

    assert_eq!(ids[0], "dup");
    assert_ne!(ids[1], "dup");
    assert_eq!(ids[2], "keep");
    assert_eq!(
        ids.iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        ids.len()
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
fn sse_tool_calls_rewrites_duplicate_existing_call_ids() {
    let calls = vec![
        ToolCall::new("search", IndexMap::new()).with_id("dup"),
        ToolCall::new("read", IndexMap::new()).with_id("dup"),
    ];
    let events = tool_calls_to_sse_events(&calls, "model");
    let ids: Vec<&str> = events[0]["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|call| call["id"].as_str().unwrap())
        .collect();

    assert_eq!(ids[0], "dup");
    assert_ne!(ids[1], "dup");
}

#[test]
fn sse_tool_calls_without_reasoning_include_assistant_role() {
    let calls = vec![ToolCall::new("search", IndexMap::new())];
    let events = tool_calls_to_sse_events(&calls, "model");

    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(events[0]["choices"][0]["delta"]["tool_calls"].is_array());
    assert_eq!(events[1]["choices"][0]["finish_reason"], "tool_calls");
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
    let response =
        text_response_to_openai_with_usage(&TextResponse::new("counted"), "model", Some(&usage));
    assert_eq!(response["usage"]["prompt_tokens"], 11);
    assert_eq!(response["usage"]["completion_tokens"], 5);
    assert_eq!(response["usage"]["total_tokens"], 16);

    let events = text_to_sse_events_with_usage("counted", "model", 0, Some(&usage));
    assert!(events[0].get("usage").is_none());
    assert_eq!(events.last().unwrap()["usage"]["total_tokens"], 16);
    let usage_events = events
        .iter()
        .filter(|event| event.get("usage").is_some())
        .count();
    assert_eq!(usage_events, 1);
}

#[test]
fn sse_tool_calls_emit_usage_only_on_final_chunk() {
    let usage = TokenUsage::new(11, 5, 16);
    let calls = vec![ToolCall::new("search", IndexMap::new()).with_reasoning("hmm")];
    let events: Vec<Value> = tool_calls_to_sse_events_with_usage(&calls, "model", Some(&usage));

    assert_eq!(events.len(), 3);
    assert!(events[0].get("usage").is_none());
    assert!(events[1].get("usage").is_none());
    assert_eq!(events[2]["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(events[2]["usage"]["total_tokens"], 16);
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
