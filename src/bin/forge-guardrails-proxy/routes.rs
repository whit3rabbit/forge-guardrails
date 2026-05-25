use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::Response;
use axum::routing::{get, options, post};
use axum::Router;
use forge_guardrails::{
    handle_anthropic_messages_with_scorer, handle_chat_completions_with_scorer,
    AnthropicHandlerError, AnthropicHandlerResult, ContextManager, HandlerError, HandlerResult,
    LLMClient, NoCompact, ServerManager, ToolCallScorer,
};
use serde_json::{json, Value};
use tokio::sync::Mutex as TokioMutex;

use crate::client::ClientFactory;
use crate::config::ProxyConfig;
use crate::response::{build_anthropic_sse_response, build_openai_sse_response, build_response};

const MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

#[derive(Clone)]
struct AppState {
    config: Arc<ProxyConfig>,
    client_factory: Arc<ClientFactory>,
    request_mutex: Arc<TokioMutex<()>>,
    scorer: Option<Arc<dyn ToolCallScorer>>,
}

pub(crate) async fn serve(
    config: ProxyConfig,
    client_factory: ClientFactory,
    managed_server: Option<ServerManager>,
    scorer: Option<Arc<dyn ToolCallScorer>>,
) -> Result<(), String> {
    let result = serve_inner(config, client_factory, scorer).await;
    if let Some(server) = managed_server {
        if let Err(err) = server.stop() {
            let stop_err = format!("failed to stop managed backend: {err}");
            if result.is_ok() {
                return Err(stop_err);
            }
            eprintln!("warning: {stop_err}");
        }
    }
    result
}

async fn serve_inner(
    config: ProxyConfig,
    client_factory: ClientFactory,
    scorer: Option<Arc<dyn ToolCallScorer>>,
) -> Result<(), String> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|err| format!("invalid bind address: {err}"))?;
    let state = AppState {
        config: Arc::new(config.clone()),
        client_factory: Arc::new(client_factory),
        request_mutex: Arc::new(TokioMutex::new(())),
        scorer,
    };

    eprintln!(
        "forge-guardrails-proxy listening on http://{}:{}",
        config.host, config.port
    );
    eprintln!(
        "warning: inbound auth is not enforced; do not expose this proxy publicly without an auth layer"
    );
    if config.verbose {
        eprintln!(
            "proxy config: model={}, context_tokens={}, max_retries={}, rescue_enabled={}, serialize_requests={}",
            config.default_model,
            config.context_tokens,
            config.max_retries,
            config.rescue_enabled,
            config.serialize_requests
        );
    }

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/chat/completions", options(cors_preflight))
        .route("/v1/messages", post(anthropic_messages))
        .route("/v1/messages", options(cors_preflight))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|err| format!("failed to bind {addr}: {err}"))?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|err| format!("server failed: {err}"))
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let ctrl_c = async {
            let _ = tokio::signal::ctrl_c().await;
        };
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(err) => {
                    eprintln!("warning: failed to install SIGTERM handler: {err}");
                    std::future::pending::<()>().await;
                }
            }
        };
        tokio::select! {
            _ = ctrl_c => {}
            _ = terminate => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

async fn health() -> Response {
    build_response(200, "application/json", json!({"status": "ok"}).to_string())
}

async fn models(State(state): State<AppState>) -> Response {
    build_response(
        200,
        "application/json",
        json!({
            "object": "list",
            "data": [{
                "id": state.config.default_model,
                "object": "model",
                "created": 0,
                "owned_by": "forge-guardrails"
            }]
        })
        .to_string(),
    )
}

async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    if body.len() > MAX_BODY_SIZE {
        return build_response(
            413,
            "application/json",
            json!({"error": "request too large"}).to_string(),
        );
    }
    let parsed: Value = match serde_json::from_slice(body.as_ref()) {
        Ok(value) => value,
        Err(err) => {
            return build_response(
                400,
                "application/json",
                json!({"error": err.to_string()}).to_string(),
            );
        }
    };
    let model = extract_model_from_value(&parsed, &state.config.default_model);
    let client = Arc::new(state.client_factory.client_for_model(model));
    let context_manager = Arc::new(TokioMutex::new(ContextManager::new(
        Box::new(NoCompact),
        state.config.context_tokens,
        None,
        None,
        None,
    )));

    let guard = if state.config.serialize_requests {
        Some(state.request_mutex.clone().lock_owned().await)
    } else {
        None
    };

    match handle_chat_completions_with_scorer(
        &parsed,
        &client,
        &context_manager,
        state.config.max_retries,
        state.config.rescue_enabled,
        state.scorer.clone(),
    )
    .await
    {
        Ok(HandlerResult::Response(value)) => {
            build_response(200, "application/json", value.to_string())
        }
        Ok(HandlerResult::StreamBody(events)) => build_openai_sse_response(events, guard),
        Err(HandlerError::BadRequest(err)) => {
            build_response(400, "application/json", json!({"error": err}).to_string())
        }
        Err(HandlerError::Upstream(err)) => {
            build_response(502, "application/json", json!({"error": err}).to_string())
        }
    }
}

async fn anthropic_messages(State(state): State<AppState>, body: Bytes) -> Response {
    if body.len() > MAX_BODY_SIZE {
        return build_response(
            413,
            "application/json",
            json!({"error": "request too large"}).to_string(),
        );
    }

    let raw: Value = match serde_json::from_slice(body.as_ref()) {
        Ok(value) => value,
        Err(err) => {
            return build_response(
                400,
                "application/json",
                json!({"error": err.to_string()}).to_string(),
            );
        }
    };
    let model = extract_model_from_value(&raw, &state.config.default_model);
    let client = Arc::new(state.client_factory.client_for_model(model));
    anthropic_messages_with_raw_client(state.config, state.request_mutex, state.scorer, client, raw)
        .await
}

#[cfg(test)]
async fn anthropic_messages_with_client<C: LLMClient + 'static>(
    config: Arc<ProxyConfig>,
    request_mutex: Arc<TokioMutex<()>>,
    client: Arc<C>,
    body: Bytes,
) -> Response {
    if body.len() > MAX_BODY_SIZE {
        return build_response(
            413,
            "application/json",
            json!({"error": "request too large"}).to_string(),
        );
    }

    let raw: Value = match serde_json::from_slice(body.as_ref()) {
        Ok(value) => value,
        Err(err) => {
            return build_response(
                400,
                "application/json",
                json!({"error": err.to_string()}).to_string(),
            );
        }
    };
    anthropic_messages_with_raw_client(config, request_mutex, None, client, raw).await
}

async fn anthropic_messages_with_raw_client<C: LLMClient + 'static>(
    config: Arc<ProxyConfig>,
    request_mutex: Arc<TokioMutex<()>>,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    client: Arc<C>,
    raw: Value,
) -> Response {
    let parsed: anyllm_translate::anthropic::MessageCreateRequest =
        match serde_json::from_value(raw.clone()) {
            Ok(value) => value,
            Err(err) => {
                return build_response(
                    400,
                    "application/json",
                    json!({"error": err.to_string()}).to_string(),
                );
            }
        };
    let context_manager = Arc::new(TokioMutex::new(ContextManager::new(
        Box::new(NoCompact),
        config.context_tokens,
        None,
        None,
        None,
    )));

    let guard = if config.serialize_requests {
        Some(request_mutex.clone().lock_owned().await)
    } else {
        None
    };

    match handle_anthropic_messages_with_scorer(
        &parsed,
        &raw,
        &client,
        &context_manager,
        config.max_retries,
        config.rescue_enabled,
        scorer,
    )
    .await
    {
        Ok(AnthropicHandlerResult::Response(value)) => {
            build_response(200, "application/json", value.to_string())
        }
        Ok(AnthropicHandlerResult::StreamBody(events)) => {
            build_anthropic_sse_response(events, guard)
        }
        Err(AnthropicHandlerError::BadRequest(err)) => {
            build_response(400, "application/json", json!({"error": err}).to_string())
        }
        Err(AnthropicHandlerError::Upstream(err)) => {
            build_response(502, "application/json", json!({"error": err}).to_string())
        }
        Err(AnthropicHandlerError::Internal(err)) => {
            build_response(500, "application/json", json!({"error": err}).to_string())
        }
    }
}

async fn cors_preflight() -> Response {
    build_response(204, "", String::new())
}

#[cfg(test)]
fn extract_openai_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

#[cfg(test)]
fn extract_anthropic_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

#[cfg(test)]
fn extract_json_model(body: &[u8], default_model: &str) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .map(|value| extract_model_from_value(&value, default_model))
        .unwrap_or_else(|| default_model.to_string())
}

fn extract_model_from_value(value: &Value, default_model: &str) -> String {
    value
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| default_model.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_guardrails::{
        ApiFormat, BackendError, ChunkStream, ChunkType, ClassifierModelKind,
        ContextDiscoveryError, LLMRequestOptions, LLMResponse, SamplingParams, ScorerMode,
        StreamChunk, StreamError, TextResponse, ToolSpec,
    };
    use futures_util::StreamExt;

    struct BinaryChannelStreamClient {
        receiver:
            std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Result<StreamChunk, StreamError>>>>,
    }

    impl BinaryChannelStreamClient {
        fn new(receiver: tokio::sync::mpsc::Receiver<Result<StreamChunk, StreamError>>) -> Self {
            Self {
                receiver: std::sync::Mutex::new(Some(receiver)),
            }
        }
    }

    impl LLMClient for BinaryChannelStreamClient {
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
            config: Arc::new(ProxyConfig {
                host: "127.0.0.1".to_string(),
                port: 8081,
                default_model: "default".to_string(),
                context_tokens: 8192,
                max_retries: 0,
                rescue_enabled: true,
                serialize_requests: false,
                verbose: false,
                classifier_dir: None,
                classifier_mode: ScorerMode::Shadow,
                classifier_model: ClassifierModelKind::Quantized,
            }),
            client_factory: Arc::new(ClientFactory::DirectOpenAi {
                base_url: upstream.url(),
                api_key: None,
                context_tokens: 8192,
            }),
            request_mutex: Arc::new(TokioMutex::new(())),
            scorer: None,
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

        let config = Arc::new(ProxyConfig {
            host: "127.0.0.1".to_string(),
            port: 8081,
            default_model: "default".to_string(),
            context_tokens: 8192,
            max_retries: 0,
            rescue_enabled: true,
            serialize_requests: false,
            verbose: false,
            classifier_dir: None,
            classifier_mode: ScorerMode::Shadow,
            classifier_model: ClassifierModelKind::Quantized,
        });
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
            anthropic_messages_with_client(config, Arc::new(TokioMutex::new(())), client, body)
                .await;
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
}
