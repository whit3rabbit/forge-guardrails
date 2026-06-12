//! Proxy routes unit tests.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::Mutex as TokioMutex;

use forge_guardrails::{
    ApiFormat, BackendError, ChunkStream, ChunkType, ClassifierModelKind, ContextDiscoveryError,
    LLMRequestOptions, LLMResponse, SamplingParams, SchemaCompressionMode, ScorerMode, StreamChunk,
    StreamError, TextResponse, ToolCallPolicyConfig, ToolOutputCompressionConfig, ToolSpec,
};

use super::handlers::{
    anthropic_messages, anthropic_messages_with_client, chat_completions, extract_anthropic_model,
    extract_openai_model, models,
};
use super::AppState;
use crate::client::ClientFactory;
use crate::config::ProxyConfig;

fn test_config() -> Arc<ProxyConfig> {
    Arc::new(ProxyConfig {
        host: "127.0.0.1".to_string(),
        port: 8081,
        default_model: "default".to_string(),
        default_model_explicit: true,
        context_tokens: 8192,
        max_retries: 0,
        rescue_enabled: true,
        serialize_requests: false,
        verbose: false,
        classifier_dir: None,
        classifier_mode: ScorerMode::Shadow,
        classifier_model: ClassifierModelKind::Quantized,
        classifier_auto_download: false,
        classifier_max_latency_ms: None,
        final_response_classifier_dir: None,
        final_response_classifier_mode: ScorerMode::Shadow,
        final_response_classifier_model: ClassifierModelKind::Quantized,
        final_response_classifier_max_latency_ms: None,
        tool_output_compression: ToolOutputCompressionConfig::disabled(),
        tool_call_policy: ToolCallPolicyConfig::disabled(),
        schema_compression: SchemaCompressionMode::Disabled,
    })
}

fn test_config_without_default_model() -> Arc<ProxyConfig> {
    let mut config = (*test_config()).clone();
    config.default_model = "forge-guardrails-unset".to_string();
    config.default_model_explicit = false;
    Arc::new(config)
}

fn test_state() -> AppState {
    AppState {
        config: test_config(),
        client_factory: Arc::new(ClientFactory::DirectOpenAi {
            base_url: "http://127.0.0.1:9".to_string(),
            api_key: None,
            http_client: reqwest::Client::new(),
            context_tokens: 8192,
        }),
        request_mutex: Arc::new(TokioMutex::new(())),
        scorer: None,
        final_response_scorer: None,
        tool_output_state: Arc::new(forge_guardrails::ToolOutputCompressionState::new()),
    }
}

fn test_state_without_default_model() -> AppState {
    AppState {
        config: test_config_without_default_model(),
        client_factory: Arc::new(ClientFactory::DirectOpenAi {
            base_url: "http://127.0.0.1:9".to_string(),
            api_key: None,
            http_client: reqwest::Client::new(),
            context_tokens: 8192,
        }),
        request_mutex: Arc::new(TokioMutex::new(())),
        scorer: None,
        final_response_scorer: None,
        tool_output_state: Arc::new(forge_guardrails::ToolOutputCompressionState::new()),
    }
}

struct BinaryChannelStreamClient {
    receiver:
        std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Result<StreamChunk, StreamError>>>>,
}

struct StaticTextClient;

impl BinaryChannelStreamClient {
    fn new(receiver: tokio::sync::mpsc::Receiver<Result<StreamChunk, StreamError>>) -> Self {
        Self {
            receiver: std::sync::Mutex::new(Some(receiver)),
        }
    }
}

impl forge_guardrails::LLMClient for BinaryChannelStreamClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<serde_json::Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        Err(BackendError::new(500, "send should not be used"))
    }

    async fn send_stream(
        &self,
        _messages: Vec<serde_json::Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        Err(StreamError::new("use send_stream_with_options"))
    }

    async fn send_stream_with_options(
        &self,
        _messages: Vec<serde_json::Value>,
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

impl forge_guardrails::LLMClient for StaticTextClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<serde_json::Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        Ok(LLMResponse::Text(TextResponse::new("ok")))
    }

    async fn send_stream(
        &self,
        _messages: Vec<serde_json::Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        Err(StreamError::new("stream should not be used"))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[test]
fn extracts_openai_request_model() {
    let body = br#"{"model":"forge-virtual","messages":[]}"#;
    assert_eq!(extract_openai_model(body, "default"), "forge-virtual");
}

#[test]
fn extracts_anthropic_request_model() {
    let body = br#"{"model":"claude-sonnet","messages":[],"max_tokens":64}"#;
    assert_eq!(extract_anthropic_model(body, "default"), "claude-sonnet");
}

#[test]
fn model_extraction_falls_back_for_invalid_json() {
    assert_eq!(extract_openai_model(b"not json", "default"), "default");
}

#[test]
fn model_extraction_falls_back_for_empty_model() {
    let body = br#"{"model":"   ","messages":[]}"#;
    assert_eq!(extract_anthropic_model(body, "default"), "default");
}

#[tokio::test]
async fn binary_openai_invalid_json_returns_400() {
    let response = chat_completions(State(test_state()), Bytes::from_static(b"not json")).await;

    assert_eq!(response.status().as_u16(), 400);
}

#[tokio::test]
async fn binary_openai_oversized_body_returns_413() {
    let response = chat_completions(
        State(test_state()),
        Bytes::from(vec![b'x'; 17 * 1024 * 1024]),
    )
    .await;

    assert_eq!(response.status().as_u16(), 413);
}

#[tokio::test]
async fn binary_models_endpoint_is_empty_without_explicit_default_model() {
    let response = models(State(test_state_without_default_model())).await;

    assert_eq!(response.status().as_u16(), 200);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(body["data"].as_array().expect("data").len(), 0);
}

#[tokio::test]
async fn binary_openai_missing_model_returns_400_without_explicit_default() {
    let body = Bytes::from(
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        })
        .to_string(),
    );

    let response = chat_completions(State(test_state_without_default_model()), body).await;

    assert_eq!(response.status().as_u16(), 400);
}

#[tokio::test]
async fn binary_anthropic_invalid_json_returns_400() {
    let response = anthropic_messages(
        State(test_state()),
        HeaderMap::new(),
        Bytes::from_static(b"not json"),
    )
    .await;

    assert_eq!(response.status().as_u16(), 400);
}

#[tokio::test]
async fn binary_anthropic_typed_parse_failure_returns_400() {
    let response = anthropic_messages(
        State(test_state()),
        HeaderMap::new(),
        Bytes::from_static(br#"{"model":"claude-test","messages":[]}"#),
    )
    .await;

    assert_eq!(response.status().as_u16(), 400);
}

#[tokio::test]
async fn binary_anthropic_adaptive_thinking_request_is_accepted() {
    let body = Bytes::from(
        json!({
            "model": "claude-opus-4-8",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "xhigh"}
        })
        .to_string(),
    );
    let response = anthropic_messages_with_client(
        test_config(),
        Arc::new(TokioMutex::new(())),
        Arc::new(StaticTextClient),
        body,
    )
    .await;

    assert_eq!(response.status().as_u16(), 200);
}

#[tokio::test]
async fn binary_openai_malformed_tool_request_returns_400() {
    let body = Bytes::from(
        json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "array"}
                }
            }]
        })
        .to_string(),
    );

    let response = chat_completions(State(test_state()), body).await;

    assert_eq!(response.status().as_u16(), 400);
}

#[tokio::test]
async fn external_backend_uses_request_model() {
    let mut upstream = mockito::Server::new_async().await;
    let _mock = upstream
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Json(json!({
            "model": "request-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-routed",
                "object": "chat.completion",
                "created": 1,
                "model": "request-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let state = AppState {
        config: test_config(),
        client_factory: Arc::new(ClientFactory::DirectOpenAi {
            base_url: upstream.url(),
            api_key: None,
            http_client: reqwest::Client::new(),
            context_tokens: 8192,
        }),
        request_mutex: Arc::new(TokioMutex::new(())),
        scorer: None,
        final_response_scorer: None,
        tool_output_state: Arc::new(forge_guardrails::ToolOutputCompressionState::new()),
    };
    let body = Bytes::from(
        json!({
            "model": "request-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        })
        .to_string(),
    );

    let response = chat_completions(State(state), body).await;

    assert_eq!(response.status().as_u16(), 200);
}

#[tokio::test]
async fn binary_anthropic_response_yields_body_chunk_before_backend_final() {
    use tokio::time::{timeout, Duration};

    let config = test_config();
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let client = Arc::new(BinaryChannelStreamClient::new(rx));
    let body = Bytes::from(
        json!({
            "model": "claude-test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        })
        .to_string(),
    );

    let response =
        anthropic_messages_with_client(config, Arc::new(TokioMutex::new(())), client, body).await;
    assert_eq!(response.status().as_u16(), 200);
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

    tx.send(Ok(StreamChunk::new(ChunkType::Final)
        .with_response(LLMResponse::Text(TextResponse::new("first")))))
        .await
        .unwrap();
}
