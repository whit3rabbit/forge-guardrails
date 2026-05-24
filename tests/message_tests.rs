//! Integration tests for message tests.

use forge_guardrails::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use indexmap::IndexMap;
use serde_json::Value;

#[test]
fn role_values() {
    assert_eq!(MessageRole::System.as_str(), "system");
    assert_eq!(MessageRole::User.as_str(), "user");
    assert_eq!(MessageRole::Assistant.as_str(), "assistant");
    assert_eq!(MessageRole::Tool.as_str(), "tool");
}

#[test]
fn role_display() {
    assert_eq!(format!("{}", MessageRole::System), "system");
    assert_eq!(format!("{}", MessageRole::User), "user");
}

#[test]
fn role_serde() {
    let json = serde_json::to_string(&MessageRole::Assistant).expect("serialize");
    assert_eq!(json, "\"assistant\"");
}

#[test]
fn type_values() {
    assert_eq!(MessageType::SystemPrompt.as_str(), "system_prompt");
    assert_eq!(MessageType::UserInput.as_str(), "user_input");
    assert_eq!(MessageType::ToolCall.as_str(), "tool_call");
    assert_eq!(MessageType::ToolResult.as_str(), "tool_result");
    assert_eq!(MessageType::Reasoning.as_str(), "reasoning");
    assert_eq!(MessageType::TextResponse.as_str(), "text_response");
    assert_eq!(MessageType::StepNudge.as_str(), "step_nudge");
    assert_eq!(
        MessageType::PrerequisiteNudge.as_str(),
        "prerequisite_nudge"
    );
    assert_eq!(MessageType::RetryNudge.as_str(), "retry_nudge");
    assert_eq!(MessageType::ContextWarning.as_str(), "context_warning");
    assert_eq!(MessageType::Summary.as_str(), "summary");
}

#[test]
fn type_display() {
    assert_eq!(format!("{}", MessageType::ToolCall), "tool_call");
}

#[test]
fn meta_defaults() {
    let meta = MessageMeta::new(MessageType::UserInput);
    assert_eq!(meta.msg_type, MessageType::UserInput);
    assert!(meta.step_index.is_none());
    assert!(meta.original_type.is_none());
    assert!(meta.token_estimate.is_none());
}

#[test]
fn meta_with_all_fields() {
    let meta = MessageMeta::new(MessageType::UserInput)
        .with_step_index(3)
        .with_original_type(MessageType::StepNudge)
        .with_token_estimate(100);
    assert_eq!(meta.step_index, Some(3));
    assert_eq!(meta.original_type, Some(MessageType::StepNudge));
    assert_eq!(meta.token_estimate, Some(100));
}

#[test]
fn plain_message_ollama() {
    let msg = Message::new(
        MessageRole::User,
        "hello",
        MessageMeta::new(MessageType::UserInput),
    );
    let out = msg.serialize("ollama");
    let obj = out.as_object().expect("expected object");
    assert_eq!(obj.len(), 2);
    assert_eq!(obj["role"], "user");
    assert_eq!(obj["content"], "hello");
}

#[test]
fn plain_message_openai() {
    let msg = Message::new(
        MessageRole::User,
        "hello",
        MessageMeta::new(MessageType::UserInput),
    );
    let out = msg.serialize("openai");
    let obj = out.as_object().expect("expected object");
    assert_eq!(obj.len(), 2);
    assert_eq!(obj["role"], "user");
    assert_eq!(obj["content"], "hello");
}

#[test]
fn tool_call_ollama_format() {
    let mut args = IndexMap::new();
    args.insert("q".to_string(), Value::String("test".to_string()));
    let tc = ToolCallInfo::new("search", Some(args), "call-123");
    let msg = Message::new(
        MessageRole::Assistant,
        "",
        MessageMeta::new(MessageType::ToolCall),
    )
    .with_tool_calls(vec![tc]);
    let out = msg.serialize("ollama");
    let obj = out.as_object().expect("expected object");
    let tool_calls = obj["tool_calls"].as_array().expect("expected array");
    assert_eq!(tool_calls.len(), 1);
    let entry = tool_calls[0].as_object().expect("expected object");
    assert!(!entry.contains_key("id"));
    assert!(!entry.contains_key("type"));
    let func = entry["function"].as_object().expect("expected object");
    assert_eq!(func["name"], "search");
    assert!(func["arguments"].is_object());
}

#[test]
fn tool_call_openai_format() {
    let mut args = IndexMap::new();
    args.insert("q".to_string(), Value::String("test".to_string()));
    let tc = ToolCallInfo::new("search", Some(args), "call-456");
    let msg = Message::new(
        MessageRole::Assistant,
        "",
        MessageMeta::new(MessageType::ToolCall),
    )
    .with_tool_calls(vec![tc]);
    let out = msg.serialize("openai");
    let obj = out.as_object().expect("expected object");
    let tool_calls = obj["tool_calls"].as_array().expect("expected array");
    let entry = tool_calls[0].as_object().expect("expected object");
    assert_eq!(entry["id"], "call-456");
    assert_eq!(entry["type"], "function");
    let func = entry["function"].as_object().expect("expected object");
    assert_eq!(func["name"], "search");
    assert!(func["arguments"].is_string());
    let args_str = func["arguments"].as_str().expect("expected string");
    let parsed: Value = serde_json::from_str(args_str).expect("valid json");
    assert_eq!(parsed["q"], "test");
}

#[test]
fn tool_result_ollama() {
    let msg = Message::new(
        MessageRole::Tool,
        "result data",
        MessageMeta::new(MessageType::ToolResult),
    )
    .with_tool_name("search")
    .with_tool_call_id("call-789");
    let out = msg.serialize("ollama");
    let obj = out.as_object().expect("expected object");
    assert_eq!(obj["role"], "tool");
    assert!(obj.contains_key("tool_name"));
    assert_eq!(obj["tool_name"], "search");
    assert!(!obj.contains_key("name"));
    assert!(!obj.contains_key("tool_call_id"));
}

#[test]
fn tool_result_openai_with_call_id() {
    let msg = Message::new(
        MessageRole::Tool,
        "result data",
        MessageMeta::new(MessageType::ToolResult),
    )
    .with_tool_name("search")
    .with_tool_call_id("call-789");
    let out = msg.serialize("openai");
    let obj = out.as_object().expect("expected object");
    assert_eq!(obj["role"], "tool");
    assert!(obj.contains_key("name"));
    assert_eq!(obj["name"], "search");
    assert!(!obj.contains_key("tool_name"));
    assert!(obj.contains_key("tool_call_id"));
    assert_eq!(obj["tool_call_id"], "call-789");
}

#[test]
fn tool_result_openai_no_call_id() {
    let msg = Message::new(
        MessageRole::Tool,
        "result",
        MessageMeta::new(MessageType::ToolResult),
    )
    .with_tool_name("search");
    let out = msg.serialize("openai");
    let obj = out.as_object().expect("expected object");
    assert!(obj.contains_key("name"));
    assert!(!obj.contains_key("tool_call_id"));
}

#[test]
fn multiple_tool_calls_preserve_order() {
    let tc1 = ToolCallInfo::new("alpha", None, "c1");
    let tc2 = ToolCallInfo::new("beta", None, "c2");
    let msg = Message::new(
        MessageRole::Assistant,
        "",
        MessageMeta::new(MessageType::ToolCall),
    )
    .with_tool_calls(vec![tc1, tc2]);

    for fmt in &["ollama", "openai"] {
        let out = msg.serialize(fmt);
        let tool_calls = out["tool_calls"].as_array().expect("expected array");
        assert_eq!(tool_calls.len(), 2);
    }
}

#[test]
fn null_args_default_to_empty_map() {
    let tc = ToolCallInfo::new("tool", None, "call-1");
    let msg = Message::new(
        MessageRole::Assistant,
        "",
        MessageMeta::new(MessageType::ToolCall),
    )
    .with_tool_calls(vec![tc]);

    let ollama = msg.serialize("ollama");
    let args = &ollama["tool_calls"][0]["function"]["arguments"];
    assert!(args.is_object());
    assert_eq!(args.as_object().map(|o| o.len()), Some(0));

    let openai = msg.serialize("openai");
    let args_str = openai["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("expected string");
    let parsed: Value = serde_json::from_str(args_str).expect("valid json");
    assert!(parsed.is_object());
    assert_eq!(parsed.as_object().map(|o| o.len()), Some(0));
}

#[test]
fn all_roles_round_trip() {
    for (role, expected) in [
        (MessageRole::System, "system"),
        (MessageRole::User, "user"),
        (MessageRole::Assistant, "assistant"),
        (MessageRole::Tool, "tool"),
    ] {
        let msg = Message::new(role, "test", MessageMeta::new(MessageType::UserInput));
        for fmt in &["ollama", "openai"] {
            let out = msg.serialize(fmt);
            assert_eq!(out["role"], expected);
        }
    }
}

#[test]
fn no_metadata_in_output() {
    let msg = Message::new(
        MessageRole::User,
        "test",
        MessageMeta::new(MessageType::UserInput).with_step_index(5),
    );
    for fmt in &["ollama", "openai"] {
        let out = msg.serialize(fmt);
        let obj = out.as_object().expect("expected object");
        assert!(!obj.contains_key("metadata"));
        assert!(!obj.contains_key("step_index"));
    }
}

#[test]
fn serialize_default_format_is_ollama() {
    let msg = Message::new(
        MessageRole::User,
        "test",
        MessageMeta::new(MessageType::UserInput),
    );
    let out = msg.serialize("ollama");
    assert_eq!(out["role"], "user");
    assert_eq!(out["content"], "test");
}
