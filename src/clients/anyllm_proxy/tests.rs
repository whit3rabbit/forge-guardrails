use super::call_info::estimate_cost_usd;
use super::request::{build_openai_request_body, normalize_chat_completions_url};
use super::response::parse_args_string;
use super::streaming::{
    checked_stream_tool_call_index, sse_data_value, take_sse_line, MAX_STREAM_TOOL_CALLS,
};
use crate::clients::base::LLMRequestOptions;
use serde_json::json;

#[test]
fn normalize_full_endpoint() {
    assert_eq!(
        normalize_chat_completions_url("http://localhost:3000/v1/chat/completions"),
        "http://localhost:3000/v1/chat/completions"
    );
}

#[test]
fn normalize_v1_base() {
    assert_eq!(
        normalize_chat_completions_url("http://localhost:3000/v1"),
        "http://localhost:3000/v1/chat/completions"
    );
}

#[test]
fn normalize_server_base() {
    assert_eq!(
        normalize_chat_completions_url("http://localhost:3000"),
        "http://localhost:3000/v1/chat/completions"
    );
}

#[test]
fn request_body_rewrites_duplicate_tool_call_ids_across_transcript() {
    let body = build_openai_request_body(
        "model",
        vec![
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "dup",
                    "type": "function",
                    "function": {"name": "first", "arguments": "{}"}
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "dup",
                "name": "first",
                "content": "first result"
            }),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "dup",
                    "type": "function",
                    "function": {"name": "second", "arguments": "{}"}
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "dup",
                "name": "second",
                "content": "second result"
            }),
        ],
        None,
        LLMRequestOptions::default(),
        true,
    );
    let messages = body["messages"].as_array().unwrap();
    let first_id = messages[0]["tool_calls"][0]["id"].as_str().unwrap();
    let second_id = messages[2]["tool_calls"][0]["id"].as_str().unwrap();

    // Duplicate incoming ids are rewritten to distinct Mistral-safe ids and the
    // matching tool results are remapped to keep pairing intact.
    assert!(is_mistral_safe_id(first_id), "{first_id}");
    assert!(is_mistral_safe_id(second_id), "{second_id}");
    assert_ne!(first_id, second_id);
    assert_eq!(messages[1]["tool_call_id"].as_str(), Some(first_id));
    assert_eq!(messages[3]["tool_call_id"].as_str(), Some(second_id));
}

#[test]
fn request_body_generates_missing_tool_call_ids() {
    let body = build_openai_request_body(
        "model",
        vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "first", "arguments": "{}"}
            }]
        })],
        None,
        LLMRequestOptions::default(),
        false,
    );
    let id = body["messages"][0]["tool_calls"][0]["id"].as_str().unwrap();

    assert!(is_mistral_safe_id(id), "{id}");
}

/// Mistral requires every tool_call_id to match `^[a-zA-Z0-9]{9}$`.
fn is_mistral_safe_id(id: &str) -> bool {
    id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric())
}

#[test]
fn request_body_rewrites_forge_internal_tool_call_ids_for_mistral() {
    // Forge's internal ids are `call_000000000` (14 chars, underscore), which
    // Mistral rejects. Every outbound id must become 9-char alphanumeric, stay
    // unique, and keep each tool result paired with its assistant tool call.
    let body = build_openai_request_body(
        "mistralai/ministral-8b",
        vec![
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_000000000",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"}
                    },
                    {
                        "id": "call_000000001",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"}
                    }
                ]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_000000000",
                "name": "a",
                "content": "a result"
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_000000001",
                "name": "b",
                "content": "b result"
            }),
        ],
        None,
        LLMRequestOptions::default(),
        false,
    );
    let messages = body["messages"].as_array().unwrap();
    let id_a = messages[0]["tool_calls"][0]["id"].as_str().unwrap();
    let id_b = messages[0]["tool_calls"][1]["id"].as_str().unwrap();

    assert!(is_mistral_safe_id(id_a), "{id_a}");
    assert!(is_mistral_safe_id(id_b), "{id_b}");
    assert_ne!(id_a, id_b);
    assert_eq!(messages[1]["tool_call_id"].as_str(), Some(id_a));
    assert_eq!(messages[2]["tool_call_id"].as_str(), Some(id_b));
}

#[test]
fn request_body_rewrites_orphan_tool_result_id() {
    // A tool result whose tool_call_id has no matching assistant tool call (e.g.
    // its tool-call message was dropped by compaction) must still be rewritten to
    // a Mistral-safe id, otherwise a `call_*` id would leak to the upstream.
    let body = build_openai_request_body(
        "mistralai/ministral-8b",
        vec![json!({
            "role": "tool",
            "tool_call_id": "call_000000000",
            "name": "a",
            "content": "orphan result"
        })],
        None,
        LLMRequestOptions::default(),
        false,
    );
    let id = body["messages"][0]["tool_call_id"].as_str().unwrap();

    assert!(is_mistral_safe_id(id), "{id}");
    assert_ne!(id, "call_000000000");
}

#[test]
fn parse_args_rejects_non_object() {
    assert!(parse_args_string("[]").is_empty());
}

#[test]
fn parse_args_accepts_object() {
    let args = parse_args_string(r#"{"q":"rust"}"#);
    assert_eq!(args.get("q"), Some(&json!("rust")));
}

#[test]
fn sse_data_value_accepts_optional_single_space() {
    assert_eq!(sse_data_value("data:{\"ok\":true}"), Some("{\"ok\":true}"));
    assert_eq!(sse_data_value("data: {\"ok\":true}"), Some("{\"ok\":true}"));
    assert_eq!(
        sse_data_value("data:  {\"ok\":true}"),
        Some("{\"ok\":true}")
    );
    assert_eq!(sse_data_value("event: message"), None);
}

#[test]
fn take_sse_line_keeps_buffered_tail() {
    let mut line_buf = "data: first\r\ndata: second".to_string();
    assert_eq!(
        take_sse_line(&mut line_buf),
        Some("data: first".to_string())
    );
    assert_eq!(line_buf, "data: second");
    assert_eq!(take_sse_line(&mut line_buf), None);
}

#[test]
fn stream_tool_call_index_allows_existing_or_next_slot() {
    assert_eq!(checked_stream_tool_call_index(0, 0).unwrap(), 0);
    assert_eq!(checked_stream_tool_call_index(0, 1).unwrap(), 0);
    assert_eq!(checked_stream_tool_call_index(1, 1).unwrap(), 1);
}

#[test]
fn stream_tool_call_index_rejects_sparse_or_oversized_index() {
    let sparse = checked_stream_tool_call_index(2, 0).unwrap_err();
    assert!(sparse.to_string().contains("non-contiguous"));

    let oversized = checked_stream_tool_call_index(MAX_STREAM_TOOL_CALLS as u32, 0).unwrap_err();
    assert!(oversized.to_string().contains("exceeds"));
}

#[test]
fn estimate_cost_usd_delegates_to_anyllm_pricing() {
    let usage = anyllm_translate::openai::ChatUsage {
        prompt_tokens: 12_345,
        completion_tokens: 678,
        total_tokens: 13_023,
        ..Default::default()
    };
    let model = "claude-3-haiku-20240307";
    let pricing = ::anyllm_proxy::cost::pricing();

    let cost = estimate_cost_usd(Some(model), Some(&usage)).expect("known model has pricing");
    let expected = pricing.cost_for_usage(
        model,
        usage.prompt_tokens as u64,
        usage.completion_tokens as u64,
    );

    assert_eq!(cost, expected);
}
