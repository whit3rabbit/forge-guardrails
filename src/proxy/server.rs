//! Lightweight HTTP server with OpenAI-compatible endpoints.
//!
//! Provides /health, /v1/models, /v1/chat/completions, and CORS preflight.
//! Supports optional request serialization via a single-worker queue for
//! single-GPU environments.

use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::Mutex;

use super::handler::{self, AnthropicHandlerError, AnthropicHandlerResult, HandlerResult};
use crate::clients::base::LLMClient;
use crate::context::manager::ContextManager;

/// Maximum request body size (16 MB).
const MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

/// HTTP server configuration for the OpenAI-compatible proxy.
pub struct HTTPServer {
    /// Host to bind.
    pub host: String,
    /// Port to bind.
    pub port: u16,
    /// Whether to serialize requests (single-worker queue).
    pub serialize_requests: bool,
    /// Maximum retries for tool-call validation.
    pub max_retries: i32,
    /// Whether rescue parsing is enabled.
    pub rescue_enabled: bool,
    /// The model name reported in responses.
    pub model_name: String,
    /// Mutex for request serialization.
    request_mutex: Mutex<()>,
}

impl HTTPServer {
    pub fn new(
        host: &str,
        port: u16,
        serialize_requests: bool,
        max_retries: i32,
        rescue_enabled: bool,
        model_name: &str,
    ) -> Self {
        Self {
            host: host.to_string(),
            port,
            serialize_requests,
            max_retries,
            rescue_enabled,
            model_name: model_name.to_string(),
            request_mutex: Mutex::new(()),
        }
    }

    /// Handle an incoming HTTP request.
    ///
    /// Returns (status_code, content_type, headers, body_string).
    pub async fn handle_request<C: LLMClient>(
        &self,
        method: &str,
        path: &str,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> (u16, &'static str, Vec<(&'static str, &'static str)>, String) {
        let headers = Self::cors_headers();
        match (method, path) {
            ("GET", "/health") => {
                let resp = json!({"status": "ok"});
                (200, "application/json", headers, resp.to_string())
            }

            ("GET", "/v1/models") => {
                let resp = json!({
                    "object": "list",
                    "data": [{
                        "id": self.model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "local"
                    }]
                });
                (200, "application/json", headers, resp.to_string())
            }

            ("OPTIONS", _) => (204, "", headers, String::new()),

            ("POST", "/v1/chat/completions") => {
                let (status, ct, body_str) = self
                    .handle_chat_completions(body, client, context_manager)
                    .await;
                (status, ct, headers, body_str)
            }

            ("POST", "/v1/messages") => {
                let (status, ct, body_str) = self
                    .handle_anthropic_messages(body, client, context_manager)
                    .await;
                (status, ct, headers, body_str)
            }

            _ => (
                404,
                "application/json",
                headers,
                json!({"error": "not found"}).to_string(),
            ),
        }
    }

    /// Handle /v1/messages.
    async fn handle_anthropic_messages<C: LLMClient>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> (u16, &'static str, String) {
        if body.len() > MAX_BODY_SIZE {
            return (
                413,
                "application/json",
                json!({"error": "request too large"}).to_string(),
            );
        }

        let parsed: anyllm_translate::anthropic::MessageCreateRequest =
            match serde_json::from_slice(body) {
                Ok(v) => v,
                Err(e) => {
                    return (
                        400,
                        "application/json",
                        json!({"error": e.to_string()}).to_string(),
                    );
                }
            };

        let _guard = if self.serialize_requests {
            Some(self.request_mutex.lock().await)
        } else {
            None
        };

        match handler::handle_anthropic_messages(
            &parsed,
            client,
            context_manager,
            self.max_retries,
            self.rescue_enabled,
        )
        .await
        {
            Ok(AnthropicHandlerResult::Response(v)) => (200, "application/json", v.to_string()),
            Ok(AnthropicHandlerResult::Events(events)) => {
                (200, "text/event-stream", format_anthropic_sse_body(&events))
            }
            Err(AnthropicHandlerError::BadRequest(e)) => {
                (400, "application/json", json!({"error": e}).to_string())
            }
            Err(AnthropicHandlerError::Upstream(e)) => {
                (502, "application/json", json!({"error": e}).to_string())
            }
            Err(AnthropicHandlerError::Internal(e)) => {
                (500, "application/json", json!({"error": e}).to_string())
            }
        }
    }

    /// Handle /v1/chat/completions.
    async fn handle_chat_completions<C: LLMClient>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> (u16, &'static str, String) {
        if body.len() > MAX_BODY_SIZE {
            return (
                413,
                "application/json",
                json!({"error": "request too large"}).to_string(),
            );
        }

        let parsed: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    400,
                    "application/json",
                    json!({"error": e.to_string()}).to_string(),
                );
            }
        };

        let _guard = if self.serialize_requests {
            Some(self.request_mutex.lock().await)
        } else {
            None
        };

        match handler::handle_chat_completions(
            &parsed,
            client,
            context_manager,
            self.max_retries,
            self.rescue_enabled,
        )
        .await
        {
            Ok(result) => match result {
                HandlerResult::Response(v) => (200, "application/json", v.to_string()),
                HandlerResult::Events(events) => {
                    let body = format_sse_body(&events);
                    (200, "text/event-stream", body)
                }
            },
            Err(e) => (502, "application/json", json!({"error": e}).to_string()),
        }
    }

    /// Serve the HTTP API using axum on a single-threaded local executor.
    ///
    /// Binds to `self.host:self.port`. Because `LLMClient` async methods
    /// produce non-`Send` futures (native AFIT), this server runs on a
    /// `tokio::task::LocalSet` which removes the cross-thread `Send` requirement,
    /// matching Python's single-threaded asyncio model.
    ///
    /// Requests are serialized through `self.request_mutex` when
    /// `serialize_requests=true`, matching Python's `asyncio.Semaphore(1)`.
    pub fn serve_blocking<C>(
        self: Arc<Self>,
        client: Arc<C>,
        ctx: Arc<Mutex<ContextManager>>,
    ) -> anyhow::Result<()>
    where
        C: LLMClient + 'static,
    {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
            let addr: std::net::SocketAddr = format!("{}:{}", self.host, self.port)
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid bind address: {}", e))?;
            let app = Self::build_router(self.clone(), client, ctx);
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
            Ok::<(), anyhow::Error>(())
        })
    }

    /// Serve with graceful shutdown triggered by `shutdown_rx` resolving.
    ///
    /// Same single-threaded LocalSet model as `serve_blocking`.
    pub fn serve_blocking_with_shutdown<C>(
        self: Arc<Self>,
        client: Arc<C>,
        ctx: Arc<Mutex<ContextManager>>,
        shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    ) -> anyhow::Result<()>
    where
        C: LLMClient + 'static,
    {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async move {
            let addr: std::net::SocketAddr = format!("{}:{}", self.host, self.port)
                .parse()
                .map_err(|e| anyhow::anyhow!("Invalid bind address: {}", e))?;
            let app = Self::build_router(self.clone(), client, ctx);
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await?;
            Ok::<(), anyhow::Error>(())
        })
    }

    fn build_router<C>(
        server: Arc<Self>,
        client: Arc<C>,
        ctx: Arc<Mutex<ContextManager>>,
    ) -> axum::Router
    where
        C: LLMClient + 'static,
    {
        use axum::{
            body::Bytes,
            http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
            response::{IntoResponse, Response},
            routing::{get, options, post},
        };

        // Helper: build a Response with status, content-type, body, and CORS headers.
        fn build_response(status: u16, ct: &str, body: String) -> Response {
            let status_code =
                StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            let mut resp = (status_code, body).into_response();
            if !ct.is_empty() {
                if let Ok(v) = HeaderValue::from_str(ct) {
                    resp.headers_mut()
                        .insert(axum::http::header::CONTENT_TYPE, v);
                }
            }
            for (k, v) in HTTPServer::cors_headers() {
                let key_lower = match k {
                    "Access-Control-Allow-Origin" => "access-control-allow-origin",
                    "Access-Control-Allow-Methods" => "access-control-allow-methods",
                    "Access-Control-Allow-Headers" => "access-control-allow-headers",
                    _ => continue,
                };
                if let Ok(hk) = HeaderName::from_bytes(key_lower.as_bytes()) {
                    resp.headers_mut().insert(hk, HeaderValue::from_static(v));
                }
            }
            resp
        }

        // Capture shared state in Arcs for each route closure.
        let srv_health = server.clone();
        let cli_health = client.clone();
        let ctx_health = ctx.clone();

        let srv_models = server.clone();
        let cli_models = client.clone();
        let ctx_models = ctx.clone();

        let srv_chat = server.clone();
        let cli_chat = client.clone();
        let ctx_chat = ctx.clone();

        let srv_messages = server.clone();
        let cli_messages = client.clone();
        let ctx_messages = ctx.clone();

        let health = move || {
            let srv = srv_health.clone();
            let cli = cli_health.clone();
            let ctx = ctx_health.clone();
            async move {
                let (status, ct, _, body) =
                    srv.handle_request("GET", "/health", &[], &cli, &ctx).await;
                build_response(status, ct, body)
            }
        };

        let models = move || {
            let srv = srv_models.clone();
            let cli = cli_models.clone();
            let ctx = ctx_models.clone();
            async move {
                let (status, ct, _, body) = srv
                    .handle_request("GET", "/v1/models", &[], &cli, &ctx)
                    .await;
                build_response(status, ct, body)
            }
        };

        let chat = move |body: Bytes| {
            let srv = srv_chat.clone();
            let cli = cli_chat.clone();
            let ctx = ctx_chat.clone();
            async move {
                let (status, ct, _, resp_body) = srv
                    .handle_request("POST", "/v1/chat/completions", &body, &cli, &ctx)
                    .await;
                build_response(status, ct, resp_body)
            }
        };

        let messages = move |body: Bytes| {
            let srv = srv_messages.clone();
            let cli = cli_messages.clone();
            let ctx = ctx_messages.clone();
            async move {
                let (status, ct, _, resp_body) = srv
                    .handle_request("POST", "/v1/messages", &body, &cli, &ctx)
                    .await;
                build_response(status, ct, resp_body)
            }
        };

        let opts_chat = || async move {
            let mut headers = HeaderMap::new();
            for (k, v) in HTTPServer::cors_headers() {
                let key_lower = match k {
                    "Access-Control-Allow-Origin" => "access-control-allow-origin",
                    "Access-Control-Allow-Methods" => "access-control-allow-methods",
                    "Access-Control-Allow-Headers" => "access-control-allow-headers",
                    _ => continue,
                };
                if let Ok(hk) = HeaderName::from_bytes(key_lower.as_bytes()) {
                    headers.insert(hk, HeaderValue::from_static(v));
                }
            }
            (StatusCode::NO_CONTENT, headers).into_response()
        };

        let opts_messages = || async move {
            let mut headers = HeaderMap::new();
            for (k, v) in HTTPServer::cors_headers() {
                let key_lower = match k {
                    "Access-Control-Allow-Origin" => "access-control-allow-origin",
                    "Access-Control-Allow-Methods" => "access-control-allow-methods",
                    "Access-Control-Allow-Headers" => "access-control-allow-headers",
                    _ => continue,
                };
                if let Ok(hk) = HeaderName::from_bytes(key_lower.as_bytes()) {
                    headers.insert(hk, HeaderValue::from_static(v));
                }
            }
            (StatusCode::NO_CONTENT, headers).into_response()
        };

        axum::Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat))
            .route("/v1/chat/completions", options(opts_chat))
            .route("/v1/messages", post(messages))
            .route("/v1/messages", options(opts_messages))
    }

    /// Build CORS headers for OPTIONS responses.
    pub fn cors_headers() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            (
                "Access-Control-Allow-Headers",
                "Content-Type, Authorization",
            ),
        ]
    }
}

/// Format SSE events into a chunked transfer encoding body.
///
/// Each event is "data: <json>\n\n". Terminator is "data: [DONE]\n\n".
pub fn format_sse_body(events: &[Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {}\n\n", event));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

/// Format Anthropic SSE events.
///
/// Anthropic streams use named SSE events and do not use OpenAI's [DONE]
/// sentinel.
pub fn format_anthropic_sse_body(
    events: &[anyllm_translate::anthropic::streaming::StreamEvent],
) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("event: ");
        body.push_str(anthropic_event_name(event));
        body.push('\n');
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()));
        body.push_str("\n\n");
    }
    body
}

fn anthropic_event_name(
    event: &anyllm_translate::anthropic::streaming::StreamEvent,
) -> &'static str {
    match event {
        anyllm_translate::anthropic::streaming::StreamEvent::MessageStart { .. } => "message_start",
        anyllm_translate::anthropic::streaming::StreamEvent::ContentBlockStart { .. } => {
            "content_block_start"
        }
        anyllm_translate::anthropic::streaming::StreamEvent::ContentBlockDelta { .. } => {
            "content_block_delta"
        }
        anyllm_translate::anthropic::streaming::StreamEvent::ContentBlockStop { .. } => {
            "content_block_stop"
        }
        anyllm_translate::anthropic::streaming::StreamEvent::MessageDelta { .. } => "message_delta",
        anyllm_translate::anthropic::streaming::StreamEvent::MessageStop { .. } => "message_stop",
        anyllm_translate::anthropic::streaming::StreamEvent::Ping { .. } => "ping",
        anyllm_translate::anthropic::streaming::StreamEvent::Error { .. } => "error",
    }
}

/// Parse an HTTP request from raw bytes.
/// Returns (method, path, headers, body) for routing.
///
/// WARNING: This parser is a simple helper designed strictly for testing and local/toy
/// server use. It splits on `\r\n\r\n` and does not handle `Content-Length`, chunked encoding,
/// pipelined requests, or partial reads. For production environments, use a robust HTTP
/// framework such as `axum`, `hyper`, or `tiny_http`.
#[allow(clippy::type_complexity)]
pub fn parse_http_request(raw: &[u8]) -> Option<(String, String, Vec<(String, String)>, Vec<u8>)> {
    let text = std::str::from_utf8(raw).ok()?;
    let mut parts = text.splitn(2, "\r\n\r\n");
    let header_section = parts.next()?;
    let body = parts.next().unwrap_or("").as_bytes();

    let mut lines = header_section.split("\r\n");
    let request_line = lines.next()?;
    let mut rl_parts = request_line.split(' ');
    let method = rl_parts.next()?.to_string();
    let path = rl_parts.next()?.to_string();

    let mut headers = Vec::new();
    for line in lines {
        if let Some((key, value)) = line.split_once(':') {
            headers.push((key.trim().to_string(), value.trim().to_string()));
        }
    }

    Some((method, path, headers, body.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::base::{ApiFormat, ChunkStream, SamplingParams};
    use crate::core::tool_spec::ToolSpec;
    use crate::error::{BackendError, ContextDiscoveryError, StreamError};
    use indexmap::IndexMap;

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
            Err(StreamError::new("not implemented"))
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
}
