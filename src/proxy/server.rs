//! Lightweight HTTP server with OpenAI-compatible endpoints.
//!
//! # Endpoints
//! - `GET /health`
//! - `GET /v1/models`
//! - `POST /v1/chat/completions`
//! - `OPTIONS /v1/chat/completions`
//! - `POST /v1/messages`
//! - `OPTIONS /v1/messages`
//!
//! Supports optional request serialization via a single-worker queue for
//! single-GPU environments.

use std::sync::Arc;

use serde_json::json;
use tokio::sync::Mutex;

use super::response;
use crate::clients::base::LLMClient;
use crate::context::manager::ContextManager;

mod request_handlers;
#[cfg(test)]
const MAX_BODY_SIZE: usize = request_handlers::MAX_BODY_SIZE;
#[cfg(test)]
mod test_helpers;
#[cfg(test)]
pub use test_helpers::{format_anthropic_sse_body, format_sse_body, parse_http_request};

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
    request_mutex: Arc<Mutex<()>>,
}

impl HTTPServer {
    /// Creates a new `HTTPServer` instance with the specified binding and validation options.
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
            request_mutex: Arc::new(Mutex::new(())),
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
}

/// Shared state passed to axum route handlers.
#[derive(Clone)]
pub struct RouterState<C> {
    /// Server configuration and handler methods.
    pub server: Arc<HTTPServer>,
    /// Backend LLM client.
    pub client: Arc<C>,
    /// Per-server context manager.
    pub ctx: Arc<Mutex<ContextManager>>,
}

/// Return the proxy health response.
pub async fn health() -> axum::response::Response {
    response::build_response(200, "application/json", json!({"status": "ok"}).to_string())
}

/// Return the OpenAI-compatible model list.
pub async fn models<C>(
    axum::extract::State(state): axum::extract::State<Arc<RouterState<C>>>,
) -> axum::response::Response
where
    C: LLMClient + 'static,
{
    response::build_response(
        200,
        "application/json",
        json!({
            "object": "list",
            "data": [{
                "id": state.server.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "local"
            }]
        })
        .to_string(),
    )
}

/// Handle an OpenAI-compatible chat completion request.
pub async fn chat<C>(
    axum::extract::State(state): axum::extract::State<Arc<RouterState<C>>>,
    body: axum::body::Bytes,
) -> axum::response::Response
where
    C: LLMClient + 'static,
{
    state
        .server
        .handle_chat_completions_response(&body, &state.client, &state.ctx)
        .await
}

/// Handle an Anthropic-compatible messages request.
pub async fn messages<C>(
    axum::extract::State(state): axum::extract::State<Arc<RouterState<C>>>,
    body: axum::body::Bytes,
) -> axum::response::Response
where
    C: LLMClient + 'static,
{
    state
        .server
        .handle_anthropic_messages_response(&body, &state.client, &state.ctx)
        .await
}

/// Return the chat-completions CORS preflight response.
pub async fn opts_chat() -> axum::response::Response {
    response::cors_preflight_response()
}

/// Return the messages CORS preflight response.
pub async fn opts_messages() -> axum::response::Response {
    response::cors_preflight_response()
}

impl HTTPServer {
    fn build_router<C>(
        server: Arc<Self>,
        client: Arc<C>,
        ctx: Arc<Mutex<ContextManager>>,
    ) -> axum::Router
    where
        C: LLMClient + 'static,
    {
        use axum::{
            routing::{get, options, post},
            Router,
        };

        let state = Arc::new(RouterState {
            server,
            client,
            ctx,
        });

        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models::<C>))
            .route("/v1/chat/completions", post(chat::<C>))
            .route("/v1/chat/completions", options(opts_chat))
            .route("/v1/messages", post(messages::<C>))
            .route("/v1/messages", options(opts_messages))
            .with_state(state)
    }

    /// Build CORS headers for OPTIONS responses.
    pub fn cors_headers() -> Vec<(&'static str, &'static str)> {
        response::cors_headers()
    }
}

#[cfg(test)]
mod tests;
