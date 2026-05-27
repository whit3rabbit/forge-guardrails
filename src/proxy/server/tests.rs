use super::*;
use crate::clients::base::{
    ApiFormat, ChunkStream, ChunkType, LLMRequestOptions, LLMResponse, SamplingParams, StreamChunk,
    TextResponse,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};
use indexmap::IndexMap;
use serde_json::Value;

// Dummy client for testing HTTP routing without a real backend.
struct DummyClient;

impl LLMClient for DummyClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<crate::clients::base::LLMResponse, BackendError> {
        Ok(crate::clients::base::LLMResponse::Text(
            crate::clients::base::TextResponse::new("test"),
        ))
    }
    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(
            StreamChunk::new(ChunkType::Final)
                .with_response(LLMResponse::Text(TextResponse::new("test"))),
        )])))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct RespondClient;

impl LLMClient for RespondClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<crate::clients::base::LLMResponse, BackendError> {
        let mut args = IndexMap::new();
        args.insert("message".to_string(), json!("responded"));
        Ok(crate::clients::base::LLMResponse::ToolCalls(vec![
            crate::clients::base::ToolCall::new("respond", args),
        ]))
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

fn dummy_ctx() -> ContextManager {
    ContextManager::new(
        Box::new(crate::context::strategies::NoCompact),
        4096,
        None,
        None,
        None,
    )
}

struct ChannelStreamClient {
    receiver:
        std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Result<StreamChunk, StreamError>>>>,
}

impl ChannelStreamClient {
    fn new(receiver: tokio::sync::mpsc::Receiver<Result<StreamChunk, StreamError>>) -> Self {
        Self {
            receiver: std::sync::Mutex::new(Some(receiver)),
        }
    }
}

impl LLMClient for ChannelStreamClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        Err(BackendError::new(500, "send should not be used"))
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        Err(StreamError::new("use send_stream_with_options"))
    }

    async fn send_stream_with_options(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _options: LLMRequestOptions,
    ) -> Result<ChunkStream, StreamError> {
        let mut receiver = self
            .receiver
            .lock()
            .unwrap()
            .take()
            .expect("receiver used once");
        Ok(Box::pin(async_stream::stream! {
            while let Some(chunk) = receiver.recv().await {
                yield chunk;
            }
        }))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[test]
fn http_server_new() {
    let srv = HTTPServer::new("127.0.0.1", 8081, true, 3, true, "test-model");
    assert_eq!(srv.host, "127.0.0.1");
    assert_eq!(srv.port, 8081);
    assert!(srv.serialize_requests);
    assert_eq!(srv.max_retries, 3);
}

#[tokio::test]
async fn health_endpoint() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let (status, _ct, _headers, body) = srv
        .handle_request(
            "GET",
            "/health",
            &[],
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "ok");
}

#[tokio::test]
async fn models_endpoint() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "my-model");
    let (status, _ct, _headers, body) = srv
        .handle_request(
            "GET",
            "/v1/models",
            &[],
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "my-model");
}

#[tokio::test]
async fn cors_preflight() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let (status, _ct, _headers, _body) = srv
        .handle_request(
            "OPTIONS",
            "/v1/chat/completions",
            &[],
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 204);
}

#[tokio::test]
async fn invalid_json_returns_400() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let (status, _ct, _headers, _body) = srv
        .handle_request(
            "POST",
            "/v1/chat/completions",
            b"not json",
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn oversized_body_returns_413() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let big_body = vec![b'x'; 17 * 1024 * 1024];
    let (status, _ct, _headers, _body) = srv
        .handle_request(
            "POST",
            "/v1/chat/completions",
            &big_body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 413);
}

#[tokio::test]
async fn unknown_route_returns_404() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let (status, _ct, _headers, _body) = srv
        .handle_request(
            "GET",
            "/unknown",
            &[],
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn chat_completions_valid_request() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test"
    }))
    .unwrap();
    let (status, _ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/chat/completions",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "test");
}

#[tokio::test]
async fn chat_completions_unknown_role_returns_400() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "messages": [{"role": "function", "content": "hi"}],
        "model": "test"
    }))
    .unwrap();
    let (status, _ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/chat/completions",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;

    assert_eq!(status, 400);
    let v: Value = serde_json::from_str(&body_str).unwrap();
    assert!(v["error"].as_str().unwrap().contains("role must be one of"));
}

#[tokio::test]
async fn chat_completions_malformed_tool_arguments_returns_400() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "messages": [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": "{broken"}
            }]
        }],
        "model": "test"
    }))
    .unwrap();
    let (status, _ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/chat/completions",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;

    assert_eq!(status, 400);
    let v: Value = serde_json::from_str(&body_str).unwrap();
    assert!(v["error"]
        .as_str()
        .unwrap()
        .contains("arguments must be valid JSON"));
}

#[tokio::test]
async fn anthropic_messages_valid_request() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let (status, ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/messages",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(ct, "application/json");
    let v: Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(v["type"], "message");
    assert_eq!(v["model"], "claude-test");
    assert_eq!(v["content"][0]["text"], "test");
    assert_eq!(v["stop_reason"], "end_turn");
}

#[tokio::test]
async fn anthropic_messages_route_ignores_query_string() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let (status, ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/messages?beta=true",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;

    assert_eq!(status, 200);
    assert_eq!(ct, "application/json");
    let v: Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(v["content"][0]["text"], "test");
}

#[tokio::test]
async fn anthropic_messages_with_tools_strips_respond() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "name": "search",
            "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }]
    }))
    .unwrap();
    let (status, _ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/messages",
            &body,
            &Arc::new(RespondClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(v["content"][0]["text"], "responded");
    assert_eq!(v["stop_reason"], "end_turn");
}

#[tokio::test]
async fn anthropic_messages_invalid_json_returns_400() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let (status, _ct, _headers, _body) = srv
        .handle_request(
            "POST",
            "/v1/messages",
            b"not json",
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn anthropic_messages_streaming_request() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    }))
    .unwrap();
    let (status, ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/messages",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(ct, "text/event-stream");
    assert!(body_str.contains("event: message_start"));
    assert!(body_str.contains("event: content_block_delta"));
    assert!(body_str.contains("event: message_stop"));
    assert!(body_str.contains("test"));
    assert!(!body_str.contains("[DONE]"));
}

#[tokio::test]
async fn live_anthropic_response_yields_body_chunk_before_backend_final() {
    use futures_util::StreamExt;
    use tokio::time::{timeout, Duration};

    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let client = Arc::new(ChannelStreamClient::new(rx));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let body = serde_json::to_vec(&json!({
        "model": "claude-test",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    }))
    .unwrap();

    let response = srv
        .handle_anthropic_messages_response(&body, &client, &ctx)
        .await;
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");
    assert_eq!(response.headers().get("x-accel-buffering").unwrap(), "no");

    let mut body_stream = response.into_body().into_data_stream();
    tx.send(Ok(
        StreamChunk::new(ChunkType::TextDelta).with_content("first")
    ))
    .await
    .unwrap();
    let first = timeout(Duration::from_millis(100), body_stream.next())
        .await
        .expect("first body chunk before final")
        .expect("body chunk")
        .expect("body ok");
    let first = std::str::from_utf8(&first).unwrap();
    assert!(first.starts_with("event: "));
    assert!(!first.contains("[DONE]"));
}

#[tokio::test]
async fn streaming_request() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test",
        "stream": true
    }))
    .unwrap();
    let (status, ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/chat/completions",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 200);
    assert_eq!(ct, "text/event-stream");
    assert!(body_str.contains("data: [DONE]"));
}

#[tokio::test]
async fn chat_completion_malformed_tool_returns_400() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test",
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "array"}}}]
        }))
        .unwrap();
    let (status, _ct, _headers, body_str) = srv
        .handle_request(
            "POST",
            "/v1/chat/completions",
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(status, 400);
    assert!(body_str.contains("function.parameters must have type 'object'"));
}

#[tokio::test]
async fn chat_completion_bad_forge_contract_returns_400() {
    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let body = serde_json::to_vec(&json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test",
        "tools": [{
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"type": "object", "properties": {}}
            }
        }],
        "_forge": {"required_steps": ["missing"]}
    }))
    .unwrap();
    let response = srv
        .handle_chat_completions_response(
            &body,
            &Arc::new(DummyClient),
            &Arc::new(Mutex::new(dummy_ctx())),
        )
        .await;
    assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn live_chat_response_yields_body_chunk_before_backend_final() {
    use futures_util::StreamExt;
    use tokio::time::{timeout, Duration};

    let srv = HTTPServer::new("127.0.0.1", 8081, false, 3, true, "test");
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let client = Arc::new(ChannelStreamClient::new(rx));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let body = serde_json::to_vec(&json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test",
        "stream": true
    }))
    .unwrap();

    let response = srv
        .handle_chat_completions_response(&body, &client, &ctx)
        .await;
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    assert_eq!(response.headers().get("cache-control").unwrap(), "no-cache");
    assert_eq!(response.headers().get("x-accel-buffering").unwrap(), "no");
    let mut body_stream = response.into_body().into_data_stream();

    tx.send(Ok(
        StreamChunk::new(ChunkType::TextDelta).with_content("first")
    ))
    .await
    .unwrap();
    let first = timeout(Duration::from_millis(100), body_stream.next())
        .await
        .expect("first body chunk before final")
        .expect("body chunk")
        .expect("body ok");
    let first = std::str::from_utf8(&first).unwrap();
    assert!(first.contains("first"));
    assert!(!first.contains("[DONE]"));

    assert!(timeout(Duration::from_millis(50), body_stream.next())
        .await
        .is_err());

    tx.send(Ok(StreamChunk::new(ChunkType::Final)
        .with_response(LLMResponse::Text(TextResponse::new("first")))))
        .await
        .unwrap();
    let final_event = timeout(Duration::from_millis(100), body_stream.next())
        .await
        .expect("final body chunk")
        .expect("final event")
        .expect("body ok");
    assert!(std::str::from_utf8(&final_event)
        .unwrap()
        .contains("\"finish_reason\":\"stop\""));
}

#[test]
fn cors_headers_content() {
    let headers = HTTPServer::cors_headers();
    assert!(headers
        .iter()
        .any(|(k, _)| *k == "Access-Control-Allow-Origin"));
    assert!(headers
        .iter()
        .any(|(k, v)| *k == "Access-Control-Allow-Headers" && v.contains("Content-Type")));
}

#[test]
fn format_sse_body_structure() {
    let events = vec![json!({"test": 1})];
    let body = format_sse_body(&events);
    assert!(body.starts_with("data: "));
    assert!(body.contains("data: [DONE]"));
}

#[test]
fn parse_http_request_basic() {
    let raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"test\": true}";
    let (method, path, headers, body) = parse_http_request(raw).unwrap();
    assert_eq!(method, "POST");
    assert_eq!(path, "/v1/chat/completions");
    assert_eq!(headers.len(), 1);
    assert!(body.starts_with(b"{"));
}

#[test]
fn parse_http_request_invalid() {
    assert!(parse_http_request(b"").is_none());
}

#[test]
fn max_body_size_is_16mb() {
    assert_eq!(MAX_BODY_SIZE, 16 * 1024 * 1024);
}
