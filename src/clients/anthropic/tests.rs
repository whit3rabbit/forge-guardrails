use super::*;
use crate::clients::base::{
    ApiFormat, ChunkStream, ChunkType, LLMClient, LLMRequestOptions, LLMResponse, SamplingParams,
    StreamChunk,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};
use futures_util::StreamExt;
use serde_json::json;
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn make_tool_spec(name: &str, desc: &str, props: &[(&str, &str)]) -> ToolSpec {
    let mut properties = json!({});
    let prop_map = properties.as_object_mut().expect("object");
    for (pname, ptype) in props {
        prop_map.insert(pname.to_string(), json!({"type": ptype}));
    }
    let schema = json!({"type": "object", "properties": properties});
    ToolSpec::from_json_schema(name, desc, &schema).expect("valid spec")
}

async fn collect_chunks(mut stream: ChunkStream) -> Result<Vec<StreamChunk>, StreamError> {
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk?);
    }
    Ok(chunks)
}

async fn split_sse_server(chunks: Vec<&'static str>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut request_buf = [0_u8; 4096];
        let _ = socket.read(&mut request_buf).await;
        let content_len: usize = chunks.iter().map(|chunk| chunk.len()).sum();
        let headers = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {content_len}\r\n\r\n"
        );
        socket
            .write_all(headers.as_bytes())
            .await
            .expect("write headers");
        for chunk in chunks {
            socket
                .write_all(chunk.as_bytes())
                .await
                .expect("write chunk");
            socket.flush().await.expect("flush chunk");
            tokio::task::yield_now().await;
        }
    });
    format!("http://{addr}")
}

#[test]
fn convert_tools_basic() {
    let spec = make_tool_spec("read_file", "Read a file", &[("path", "string")]);
    let tools = convert::convert_tools(&[spec]);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "read_file");
}

#[test]
fn convert_tools_enum() {
    let mut properties = json!({});
    if let Some(m) = properties.as_object_mut() {
        m.insert(
            "mode".to_string(),
            json!({"type": "string", "enum": ["fast", "slow"]}),
        );
    }
    let schema = json!({"type": "object", "properties": properties});
    let spec = ToolSpec::from_json_schema("run", "Run", &schema).expect("ok");
    let tools = convert::convert_tools(&[spec]);
    assert!(tools[0].get("input_schema").is_some());
}

#[test]
fn convert_tools_optional_params() {
    let schema = json!({
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
        "required": ["query"],
    });
    let spec = ToolSpec::from_json_schema("search", "Search", &schema).expect("ok");
    assert_eq!(convert::convert_tools(&[spec]).len(), 1);
}

#[test]
fn convert_tools_multiple() {
    let specs = vec![
        make_tool_spec("a", "A", &[("x", "string")]),
        make_tool_spec("b", "B", &[("y", "number")]),
    ];
    assert_eq!(convert::convert_tools(&specs).len(), 2);
}

#[test]
fn convert_messages_system_extraction() {
    let msgs = vec![
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hello"}),
    ];
    let (sys, conv) = convert::convert_messages(&msgs);
    assert_eq!(sys, Some(json!("You are helpful.")));
    assert_eq!(conv.len(), 1);
}

#[test]
fn convert_messages_user_assistant() {
    let msgs = vec![
        json!({"role": "user", "content": "Hi"}),
        json!({"role": "assistant", "content": "Hello!"}),
    ];
    let (sys, conv) = convert::convert_messages(&msgs);
    assert!(sys.is_none());
    assert_eq!(conv.len(), 2);
}

#[test]
fn convert_messages_tool_call() {
    let msgs = vec![
        json!({"role": "user", "content": "Read"}),
        json!({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_123",
            "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/a\"}"},
        }]}),
        json!({"role": "tool", "tool_call_id": "call_123", "content": "data"}),
    ];
    let (_, conv) = convert::convert_messages(&msgs);
    assert_eq!(conv.len(), 3);
    let content = conv[1]["content"].as_array().expect("array");
    let tu = content
        .iter()
        .find(|b| b["type"] == "tool_use")
        .expect("found");
    assert_eq!(tu["name"], "read_file");
    assert_eq!(tu["input"]["path"], "/tmp/a");
}

#[test]
fn convert_messages_tool_result() {
    let msgs = vec![json!({"role": "tool", "tool_call_id": "c1", "content": "result"})];
    let (_, conv) = convert::convert_messages(&msgs);
    assert_eq!(conv[0]["role"], "user");
    let blocks = conv[0]["content"].as_array().expect("array");
    assert_eq!(blocks[0]["type"], "tool_result");
}

#[test]
fn convert_messages_unpaired_tool_use() {
    let msgs = vec![
        json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "abc", "function": {"name": "run", "arguments": "{}"}}],
        }),
        json!({"role": "user", "content": "next"}),
    ];
    let (_, conv) = convert::convert_messages(&msgs);
    let blocks = conv[1]["content"].as_array().expect("array");
    assert_eq!(blocks[0]["type"], "tool_result");
    assert_eq!(blocks[0]["tool_use_id"], "abc");
    assert_eq!(blocks[0]["is_error"], true);
    assert_eq!(blocks[1]["type"], "text");
    assert_eq!(blocks[1]["text"], "next");
}

#[test]
fn convert_messages_consecutive_merging() {
    let msgs = vec![
        json!({"role": "user", "content": "First"}),
        json!({"role": "user", "content": "Second"}),
    ];
    let (_, conv) = convert::convert_messages(&msgs);
    assert_eq!(conv.len(), 1);
    let blocks = conv[0]["content"].as_array().expect("array");
    assert_eq!(blocks[0]["text"], "First");
    assert_eq!(blocks[1]["text"], "Second");
}

#[test]
fn convert_messages_multi_step() {
    let msgs = vec![
        json!({"role": "system", "content": "Sys"}),
        json!({"role": "user", "content": "Do"}),
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "read", "arguments": "{}"}},
        ]}),
        json!({"role": "tool", "tool_call_id": "c1", "content": "data"}),
        json!({"role": "assistant", "content": "Done"}),
    ];
    let (sys, conv) = convert::convert_messages(&msgs);
    assert_eq!(sys, Some(json!("Sys")));
    assert_eq!(conv.len(), 4);
}

#[test]
fn convert_messages_arguments_as_dict() {
    let msgs = vec![json!({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "run", "arguments": {"path": "/tmp/a"}}}],
    })];
    let (_, conv) = convert::convert_messages(&msgs);
    let content = conv[0]["content"].as_array().expect("array");
    let tu = content
        .iter()
        .find(|b| b["type"] == "tool_use")
        .expect("found");
    assert_eq!(tu["input"]["path"], "/tmp/a");
}

#[test]
fn convert_messages_missing_tool_id_uses_non_empty_fallback() {
    let msgs = vec![
        json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "run", "arguments": "{}"}}],
        }),
        json!({"role": "user", "content": "next"}),
    ];
    let (_, conv) = convert::convert_messages(&msgs);
    let tool_use = conv[0]["content"].as_array().expect("array")[0].clone();
    assert_eq!(tool_use["id"], "toolu_0");
    let synthetic = conv[1]["content"].as_array().expect("array")[0].clone();
    assert_eq!(synthetic["tool_use_id"], "toolu_0");
}

#[test]
fn convert_messages_merges_array_and_text_same_role() {
    let msgs = vec![
        json!({"role": "tool", "tool_call_id": "c1", "content": "result"}),
        json!({"role": "user", "content": "follow up"}),
    ];
    let (_, conv) = convert::convert_messages(&msgs);
    assert_eq!(conv.len(), 1);
    let blocks = conv[0]["content"].as_array().expect("array");
    assert_eq!(blocks[0]["type"], "tool_result");
    assert_eq!(blocks[1]["type"], "text");
    assert_eq!(blocks[1]["text"], "follow up");
}

#[test]
fn parse_response_text() {
    let r = json!({"content": [{"type": "text", "text": "Hello"}]});
    match convert::parse_response(&r) {
        LLMResponse::Text(tr) => assert_eq!(tr.content, "Hello"),
        _ => panic!("expected text"),
    }
}

#[test]
fn parse_response_tool_use() {
    let r = json!({"content": [
        {"type": "tool_use", "id": "tu1", "name": "read", "input": {"path": "/x"}},
    ]});
    match convert::parse_response(&r) {
        LLMResponse::ToolCalls(c) => {
            assert_eq!(c[0].tool, "read");
            assert_eq!(c[0].args["path"], "/x");
        }
        _ => panic!("expected tool calls"),
    }
}

#[test]
fn parse_response_tool_use_with_reasoning() {
    let r = json!({"content": [
        {"type": "text", "text": "Thinking..."},
        {"type": "tool_use", "id": "tu1", "name": "run", "input": {}},
    ]});
    match convert::parse_response(&r) {
        LLMResponse::ToolCalls(c) => {
            assert_eq!(c[0].reasoning, Some("Thinking...".to_string()));
        }
        _ => panic!("expected tool calls"),
    }
}

#[test]
fn parse_response_empty_content() {
    let r = json!({"content": []});
    match convert::parse_response(&r) {
        LLMResponse::Text(tr) => assert_eq!(tr.content, ""),
        _ => panic!("expected text"),
    }
}

#[tokio::test]
async fn get_context_length_returns_200k() {
    let client = AnthropicClient::new("claude-3", None);
    assert_eq!(
        client.get_context_length().await.expect("ok"),
        Some(200_000)
    );
}

#[test]
fn record_usage_extracts_tokens() {
    let client = AnthropicClient::new("claude-3", None);
    client.record_usage(&json!({"usage": {"input_tokens": 42, "output_tokens": 7}}));
    let u = client.get_last_usage().expect("set");
    assert_eq!(u.prompt_tokens, 42);
    assert_eq!(u.total_tokens, 49);
    assert!(client.get_last_usage_details().is_none());
}

#[test]
fn record_usage_includes_cache_tokens() {
    let client = AnthropicClient::new("claude-3", None);
    client.record_usage(&json!({
        "usage": {
            "input_tokens": 42,
            "cache_creation_input_tokens": 11,
            "cache_read_input_tokens": 17,
            "output_tokens": 7
        }
    }));
    let u = client.get_last_usage().expect("set");
    assert_eq!(u.prompt_tokens, 70);
    assert_eq!(u.completion_tokens, 7);
    assert_eq!(u.total_tokens, 77);
    let details = client.get_last_usage_details().expect("details");
    assert_eq!(details.cached_prompt_tokens, Some(17));
    assert_eq!(details.cache_creation_prompt_tokens, Some(11));
    assert_eq!(details.cache_read_input_tokens, Some(17));
    assert_eq!(details.cache_creation_input_tokens, Some(11));
}

#[test]
fn record_usage_includes_thinking_output_tokens() {
    let client = AnthropicClient::new("claude-3", None);
    client.record_usage(&json!({
        "usage": {
            "input_tokens": 42,
            "output_tokens": 7,
            "output_tokens_details": {"thinking_tokens": 5}
        }
    }));
    let details = client.get_last_usage_details().expect("details");
    assert_eq!(details.anthropic_thinking_output_tokens, Some(5));
}

#[tokio::test]
async fn stream_usage_includes_cache_tokens() {
    let mut server = mockito::Server::new_async().await;
    let sse = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":42,\"cache_creation_input_tokens\":11,\"cache_read_input_tokens\":17,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":7}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let mock = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;

    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_timeout(5.0);
    let mut stream = client
        .send_stream_with_options(
            vec![json!({"role": "user", "content": "hi"})],
            None,
            LLMRequestOptions::default(),
        )
        .await
        .expect("stream starts");
    let mut final_chunk = None;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("chunk");
        if chunk.chunk_type == ChunkType::Final {
            final_chunk = Some(chunk);
        }
    }

    let final_chunk = final_chunk.expect("final chunk");
    let final_usage = final_chunk.usage.expect("final usage");
    assert_eq!(final_usage.prompt_tokens, 70);
    assert_eq!(final_usage.completion_tokens, 7);
    assert_eq!(final_usage.total_tokens, 77);
    let final_details = final_chunk.usage_details.expect("final details");
    assert_eq!(final_details.cached_prompt_tokens, Some(17));
    assert_eq!(final_details.cache_creation_prompt_tokens, Some(11));
    let u = client.get_last_usage().expect("usage");
    assert_eq!(u.prompt_tokens, 70);
    assert_eq!(u.completion_tokens, 7);
    assert_eq!(u.total_tokens, 77);
    let details = client.get_last_usage_details().expect("details");
    assert_eq!(details.cached_prompt_tokens, Some(17));
    assert_eq!(details.cache_creation_prompt_tokens, Some(11));
    mock.assert_async().await;
}

#[tokio::test]
async fn stream_processes_final_message_stop_without_trailing_newline() {
    let mut server = mockito::Server::new_async().await;
    let sse = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1}}\n\n",
        "data: {\"type\":\"message_stop\"}"
    );
    let mock = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;

    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_timeout(5.0);
    let chunks = collect_chunks(
        client
            .send_stream_with_options(
                vec![json!({"role": "user", "content": "hi"})],
                None,
                LLMRequestOptions::default(),
            )
            .await
            .expect("stream starts"),
    )
    .await
    .expect("chunks");

    assert!(chunks
        .iter()
        .any(|chunk| chunk.chunk_type == ChunkType::Final));
    mock.assert_async().await;
}

#[tokio::test]
async fn stream_processes_final_line_split_across_byte_boundaries() {
    let base_url = split_sse_server(vec![
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1}}\n\n",
        "data: {\"type\":\"message_",
        "stop\"}",
    ])
    .await;
    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(base_url)
        .with_timeout(5.0);
    let chunks = collect_chunks(
        client
            .send_stream_with_options(
                vec![json!({"role": "user", "content": "hi"})],
                None,
                LLMRequestOptions::default(),
            )
            .await
            .expect("stream starts"),
    )
    .await
    .expect("chunks");

    assert_eq!(
        chunks.last().and_then(|chunk| chunk.response.as_ref()),
        Some(&LLMResponse::Text(crate::clients::base::TextResponse::new(
            "ok"
        )))
    );
}

#[tokio::test]
async fn stream_malformed_final_sse_line_returns_error() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body("data: {broken")
        .create_async()
        .await;
    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_timeout(5.0);
    let err = collect_chunks(
        client
            .send_stream_with_options(
                vec![json!({"role": "user", "content": "hi"})],
                None,
                LLMRequestOptions::default(),
            )
            .await
            .expect("stream starts"),
    )
    .await
    .expect_err("malformed data should fail");

    assert!(err.to_string().contains("Malformed Anthropic SSE data"));
    mock.assert_async().await;
}

#[tokio::test]
async fn stream_tool_use_preserves_provider_id() {
    let mut server = mockito::Server::new_async().await;
    let sse = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_keep\",\"name\":\"search\",\"input\":{}}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1}}\n\n",
        "data: {\"type\":\"message_stop\"}"
    );
    let mock = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;
    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_timeout(5.0);
    let chunks = collect_chunks(
        client
            .send_stream_with_options(
                vec![json!({"role": "user", "content": "hi"})],
                Some(vec![make_tool_spec("search", "Search", &[])]),
                LLMRequestOptions::default(),
            )
            .await
            .expect("stream starts"),
    )
    .await
    .expect("chunks");

    let final_response = chunks
        .last()
        .and_then(|chunk| chunk.response.as_ref())
        .expect("final response");
    match final_response {
        LLMResponse::ToolCalls(calls) => {
            assert_eq!(calls[0].id.as_deref(), Some("toolu_keep"));
            assert_eq!(calls[0].tool, "search");
        }
        other => panic!("expected tool calls, got {other:?}"),
    }
    mock.assert_async().await;
}

#[test]
fn rebuilt_body_maps_passthrough_to_anthropic_fields() {
    let client = AnthropicClient::new("fallback-model", None);
    let mut passthrough = serde_json::Map::new();
    passthrough.insert("model".to_string(), json!("request-model"));
    passthrough.insert("max_tokens".to_string(), json!(128));
    passthrough.insert("stop".to_string(), json!(["done"]));
    passthrough.insert(
        "tool_choice".to_string(),
        json!({"type": "function", "function": {"name": "search"}}),
    );
    passthrough.insert("user".to_string(), json!("user-123"));
    passthrough.insert(
        "thinking".to_string(),
        json!({"type": "adaptive", "display": "summarized"}),
    );
    passthrough.insert("output_config".to_string(), json!({"effort": "xhigh"}));

    let body = client.build_body_with_options(
        vec![json!({"role": "user", "content": "hi"})],
        Some(vec![make_tool_spec(
            "search",
            "Search",
            &[("query", "string")],
        )]),
        LLMRequestOptions {
            passthrough: Some(passthrough),
            ..Default::default()
        },
        false,
    );

    assert_eq!(body["model"], "request-model");
    assert_eq!(body["max_tokens"], 128);
    assert_eq!(body["stop_sequences"], json!(["done"]));
    assert_eq!(
        body["tool_choice"],
        json!({"type": "tool", "name": "search"})
    );
    assert_eq!(body["metadata"]["user_id"], "user-123");
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "xhigh");
}

#[tokio::test]
async fn raw_anthropic_body_is_sent_verbatim_for_clean_path() {
    let mut server = mockito::Server::new_async().await;
    let raw = json!({
        "model": "claude-request",
        "max_tokens": 128,
        "system": [{
            "type": "text",
            "text": "system",
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": {"type": "ephemeral"}
            }]
        }],
        "metadata": {"user_id": "user-123"},
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "xhigh"}
    });
    let mock = server
        .mock("POST", "/messages")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::Json(raw.clone()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 10, "output_tokens": 2}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = AnthropicClient::new("fallback-model", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_timeout(5.0);
    let result = client
        .send_with_options(
            vec![json!({"role": "user", "content": "mutated"})],
            None,
            LLMRequestOptions {
                inbound_anthropic_body: Some(Arc::new(raw)),
                ..Default::default()
            },
        )
        .await
        .expect("accepted");

    match result {
        LLMResponse::Text(text) => assert_eq!(text.content, "ok"),
        _ => panic!("expected text"),
    }
    let usage = client.last_usage().expect("usage");
    assert_eq!(usage.total_tokens, 12);
    mock.assert_async().await;
}

#[tokio::test]
async fn anthropic_extra_headers_are_forwarded() {
    let mut server = mockito::Server::new_async().await;
    let raw = json!({
        "model": "claude-request",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let mock = server
        .mock("POST", "/messages")
        .match_header("anthropic-beta", "thinking-2024-12-20")
        .match_header("x-claude-code-session-id", "session-1")
        .match_body(mockito::Matcher::Json(raw.clone()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 10, "output_tokens": 2}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = AnthropicClient::new("fallback-model", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_timeout(5.0);
    client
        .send_with_options(
            vec![json!({"role": "user", "content": "mutated"})],
            None,
            LLMRequestOptions {
                inbound_anthropic_body: Some(Arc::new(raw)),
                anthropic_headers: Some(vec![
                    (
                        "anthropic-beta".to_string(),
                        "thinking-2024-12-20".to_string(),
                    ),
                    (
                        "x-claude-code-session-id".to_string(),
                        "session-1".to_string(),
                    ),
                ]),
                ..Default::default()
            },
        )
        .await
        .expect("accepted");

    mock.assert_async().await;
}

struct RetryBodyRecordingAnthropicClient {
    inner: AnthropicClient,
    bodies: std::sync::Mutex<Vec<Value>>,
}

impl RetryBodyRecordingAnthropicClient {
    fn new() -> Self {
        Self {
            inner: AnthropicClient::new("fallback-model", None),
            bodies: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl LLMClient for RetryBodyRecordingAnthropicClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        Ok(LLMResponse::Text(crate::clients::base::TextResponse::new(
            "unused",
        )))
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, BackendError> {
        let body = self
            .inner
            .build_body_with_options(messages, tools, options, false);
        let mut bodies = self.bodies.lock().unwrap();
        let attempt = bodies.len();
        bodies.push(body);
        drop(bodies);

        if attempt == 0 {
            Ok(LLMResponse::Text(crate::clients::base::TextResponse::new(
                "not a tool call",
            )))
        } else {
            let mut args = indexmap::IndexMap::new();
            args.insert("message".to_string(), json!("ok"));
            Ok(LLMResponse::ToolCalls(vec![
                crate::clients::base::ToolCall::new("respond", args),
            ]))
        }
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        Err(StreamError::new("not implemented"))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn raw_anthropic_body_rebuilds_after_retry_without_cache_control() {
    let raw = json!({
        "model": "claude-request",
        "max_tokens": 128,
        "system": [{
            "type": "text",
            "text": "system",
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": {"type": "ephemeral"}
            }]
        }],
        "metadata": {"user_id": "user-123"},
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "xhigh"}
    });
    let client = RetryBodyRecordingAnthropicClient::new();
    let mut messages = vec![crate::core::message::Message::new(
        crate::core::message::MessageRole::User,
        "hi",
        crate::core::message::MessageMeta::new(crate::core::message::MessageType::UserInput),
    )];
    let initial_wire: Arc<[Value]> = Arc::from(
        crate::core::inference::fold_and_serialize(&messages, "openai").into_boxed_slice(),
    );
    let mut context = crate::context::manager::ContextManager::new(
        Box::new(crate::context::strategies::NoCompact),
        4096,
        None,
        None,
        None,
    );
    let validator =
        crate::guardrails::ResponseValidator::new(vec!["respond".to_string()], false, None);
    let mut tracker = crate::guardrails::ErrorTracker::new(3, 2);
    let mut counter = 0;
    let tools = vec![crate::tools::respond::respond_spec()];
    let mut passthrough = serde_json::Map::new();
    passthrough.insert("thinking".to_string(), raw["thinking"].clone());
    passthrough.insert("output_config".to_string(), raw["output_config"].clone());

    let result = crate::core::inference::run_inference_with_options(
        &mut messages,
        &client,
        &mut context,
        &validator,
        &mut tracker,
        &tools,
        &mut counter,
        0,
        "",
        Some(3),
        false,
        None,
        LLMRequestOptions {
            inbound_anthropic_body: Some(Arc::new(raw.clone())),
            initial_openai_messages: Some(initial_wire),
            passthrough: Some(passthrough),
            ..Default::default()
        },
    )
    .await
    .expect("inference")
    .expect("result");

    assert_eq!(result.attempts, 2);
    let bodies = client.bodies.lock().unwrap().clone();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["metadata"]["user_id"], "user-123");
    assert_eq!(bodies[0]["system"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(
        bodies[0]["messages"][0]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert!(bodies[0]["tools"]
        .as_array()
        .is_some_and(|tools| !tools.is_empty()));
    assert_eq!(bodies[1]["model"], "fallback-model");
    assert!(bodies[1].get("metadata").is_none());
    assert!(bodies[1].get("system").is_none());
    assert_eq!(bodies[1]["thinking"]["type"], "adaptive");
    assert_eq!(bodies[1]["output_config"]["effort"], "xhigh");
    assert!(!serde_json::to_string(&bodies[1])
        .expect("body json")
        .contains("cache_control"));
    assert!(bodies[1]["tools"]
        .as_array()
        .is_some_and(|tools| !tools.is_empty()));
    assert!(
        bodies[1]["messages"].as_array().expect("messages").len()
            > raw["messages"].as_array().expect("raw messages").len()
    );
}

#[tokio::test]
async fn send_retries_transient_429_then_succeeds() {
    let mut server = mockito::Server::new_async().await;
    // First call is rate-limited; the retry succeeds. Mockito serves the 429
    // mock once (exact expectation), then the 200 handles the retry.
    let rate_limited = server
        .mock("POST", "/messages")
        .with_status(429)
        .with_header("retry-after", "0")
        .with_body("rate limited")
        .expect(1)
        .create_async()
        .await;
    let ok = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "msg_retry",
                "type": "message",
                "role": "assistant",
                "model": "claude-3",
                "content": [{"type": "text", "text": "recovered"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_max_retries(1)
        .with_timeout(5.0);
    let response = client
        .send_with_options(
            vec![json!({"role": "user", "content": "hi"})],
            None,
            LLMRequestOptions::default(),
        )
        .await
        .expect("retry recovers from transient 429");
    match response {
        LLMResponse::Text(text) => assert_eq!(text.content, "recovered"),
        other => panic!("expected text response, got {other:?}"),
    }
    rate_limited.assert_async().await;
    ok.assert_async().await;
}

#[tokio::test]
async fn send_does_not_retry_quota_exhausted_429() {
    let mut server = mockito::Server::new_async().await;
    // A 429 carrying OpenAI's `insufficient_quota` code is a hard quota/credit
    // exhaustion that will not clear by waiting. The shared retry loop must
    // fast-fail it after exactly one attempt even with retries configured;
    // `expect(1)` proves no retry was issued.
    let mock = server
        .mock("POST", "/messages")
        .with_status(429)
        .with_header("retry-after", "0")
        .with_body(
            json!({
                "error": {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                    "code": "insufficient_quota"
                }
            })
            .to_string(),
        )
        .expect(1)
        .create_async()
        .await;

    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_max_retries(3)
        .with_timeout(5.0);
    let err = client
        .send_with_options(
            vec![json!({"role": "user", "content": "hi"})],
            None,
            LLMRequestOptions::default(),
        )
        .await
        .expect_err("quota exhaustion 429 should fail immediately");
    assert!(err.to_string().contains("status 429"));
    mock.assert_async().await;
}

#[tokio::test]
async fn send_does_not_retry_non_retryable_status() {
    let mut server = mockito::Server::new_async().await;
    // A 400 is a client error and must be surfaced after exactly one attempt,
    // even with retries configured.
    let mock = server
        .mock("POST", "/messages")
        .with_status(400)
        .with_body("bad request")
        .expect(1)
        .create_async()
        .await;

    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_max_retries(3)
        .with_timeout(5.0);
    let err = client
        .send_with_options(
            vec![json!({"role": "user", "content": "hi"})],
            None,
            LLMRequestOptions::default(),
        )
        .await
        .expect_err("non-retryable status should fail immediately");
    assert!(err.to_string().contains("status 400"));
    mock.assert_async().await;
}

#[tokio::test]
async fn send_stream_retries_transient_429_before_stream() {
    let mut server = mockito::Server::new_async().await;
    let sse = concat!(
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let rate_limited = server
        .mock("POST", "/messages")
        .with_status(429)
        .with_header("retry-after", "0")
        .with_body("rate limited")
        .expect(1)
        .create_async()
        .await;
    let ok = server
        .mock("POST", "/messages")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .expect(1)
        .create_async()
        .await;

    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(server.url())
        .with_max_retries(1)
        .with_timeout(5.0);
    let stream = client
        .send_stream_with_options(
            vec![json!({"role": "user", "content": "hi"})],
            None,
            LLMRequestOptions::default(),
        )
        .await
        .expect("stream starts after retry");
    let chunks = collect_chunks(stream).await.expect("chunks");
    assert!(chunks
        .iter()
        .any(|c| c.chunk_type == ChunkType::TextDelta && c.content == "hi"));
    rate_limited.assert_async().await;
    ok.assert_async().await;
}
