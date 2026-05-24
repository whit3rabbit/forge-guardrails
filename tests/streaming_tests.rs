//! Integration tests for streaming tests.

use forge_guardrails::{ChunkType, LLMResponse, StreamChunk, TextResponse, ToolCall};
use indexmap::IndexMap;

#[test]
fn chunk_type_values() {
    assert_eq!(ChunkType::TextDelta.as_str(), "text_delta");
    assert_eq!(ChunkType::ToolCallDelta.as_str(), "tool_call_delta");
    assert_eq!(ChunkType::Final.as_str(), "final");
    assert_eq!(ChunkType::Retry.as_str(), "retry");
}

#[test]
fn chunk_type_display() {
    assert_eq!(format!("{}", ChunkType::TextDelta), "text_delta");
    assert_eq!(format!("{}", ChunkType::Final), "final");
}

#[test]
fn stream_chunk_defaults() {
    let chunk = StreamChunk::new(ChunkType::TextDelta);
    assert_eq!(chunk.content, "");
    assert!(chunk.response.is_none());
    assert_eq!(chunk.chunk_type, ChunkType::TextDelta);
}

#[test]
fn stream_chunk_with_fields() {
    let chunk = StreamChunk::new(ChunkType::Final)
        .with_content("done")
        .with_response(LLMResponse::Text(TextResponse::new("result")));
    assert_eq!(chunk.content, "done");
    assert!(chunk.response.is_some());
}

#[test]
fn tool_call_basic() {
    let args = IndexMap::new();
    let tc = ToolCall::new("search", args);
    assert_eq!(tc.tool, "search");
    assert!(tc.reasoning.is_none());
}

#[test]
fn tool_call_with_reasoning() {
    let args = IndexMap::new();
    let tc = ToolCall::new("search", args).with_reasoning("needed more info");
    assert_eq!(tc.reasoning.as_deref(), Some("needed more info"));
}

#[test]
fn text_response() {
    let tr = TextResponse::new("hello");
    assert_eq!(tr.content, "hello");
}

#[test]
fn llm_response_tool_calls() {
    let args = IndexMap::new();
    let tc = ToolCall::new("search", args);
    let resp = LLMResponse::ToolCalls(vec![tc]);
    assert!(matches!(resp, LLMResponse::ToolCalls(_)));
    if let LLMResponse::ToolCalls(calls) = resp {
        assert_eq!(calls.len(), 1);
    }
}

#[test]
fn llm_response_text() {
    let tr = TextResponse::new("hello");
    let resp = LLMResponse::Text(tr);
    assert!(matches!(resp, LLMResponse::Text(_)));
    if let LLMResponse::Text(t) = resp {
        assert_eq!(t.content, "hello");
    }
}

#[test]
fn stream_chunk_immutability() {
    // StreamChunk is immutable: fields are pub but there are no setter methods.
    // Construction uses builder pattern, so once built the values are fixed.
    let chunk = StreamChunk::new(ChunkType::Retry).with_content("retrying");
    assert_eq!(chunk.chunk_type, ChunkType::Retry);
    assert_eq!(chunk.content, "retrying");
}

#[test]
fn tool_call_with_args() {
    let mut args = IndexMap::new();
    args.insert(
        "query".to_string(),
        serde_json::Value::String("test".to_string()),
    );
    let tc = ToolCall::new("search", args.clone());
    assert_eq!(tc.args, args);
}
