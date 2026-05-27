use super::call_info::estimate_cost_usd;
use super::request::normalize_chat_completions_url;
use super::response::parse_args_string;
use super::streaming::{
    checked_stream_tool_call_index, sse_data_value, take_sse_line, MAX_STREAM_TOOL_CALLS,
};
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
fn estimate_cost_usd_uses_per_token_pricing_units() {
    let usage = anyllm_translate::openai::ChatUsage {
        prompt_tokens: 1_000_000,
        completion_tokens: 1_000_000,
        total_tokens: 2_000_000,
        ..Default::default()
    };

    let cost = estimate_cost_usd(Some("claude-3-haiku-20240307"), Some(&usage))
        .expect("known model has pricing");

    // anyllm_proxy::cost::price_for_model returns USD per token. For
    // Claude 3 Haiku this is $0.25/M input and $1.25/M output.
    assert!((cost - 1.50).abs() < 1e-10);
}
