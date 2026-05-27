//! Lightweight HTTP server with OpenAI-compatible endpoints.
//!
//! Provides /health, /v1/models, /v1/chat/completions, and CORS preflight.
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
            routing::{get, options, post},
        };

        // Capture shared state in Arcs for each route closure.
        let srv_models = server.clone();

        let srv_chat = server.clone();
        let cli_chat = client.clone();
        let ctx_chat = ctx.clone();

        let srv_messages = server.clone();
        let cli_messages = client.clone();
        let ctx_messages = ctx.clone();

        let health = move || async move {
            response::build_response(200, "application/json", json!({"status": "ok"}).to_string())
        };

        let models = move || {
            let srv = srv_models.clone();
            async move {
                response::build_response(
                    200,
                    "application/json",
                    json!({
                        "object": "list",
                        "data": [{
                            "id": srv.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local"
                        }]
                    })
                    .to_string(),
                )
            }
        };

        let chat = move |body: Bytes| {
            let srv = srv_chat.clone();
            let cli = cli_chat.clone();
            let ctx = ctx_chat.clone();
            async move {
                srv.handle_chat_completions_response(&body, &cli, &ctx)
                    .await
            }
        };

        let messages = move |body: Bytes| {
            let srv = srv_messages.clone();
            let cli = cli_messages.clone();
            let ctx = ctx_messages.clone();
            async move {
                srv.handle_anthropic_messages_response(&body, &cli, &ctx)
                    .await
            }
        };

        let opts_chat = || async move { response::cors_preflight_response() };

        let opts_messages = || async move { response::cors_preflight_response() };

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
        response::cors_headers()
    }
}

#[cfg(test)]
mod tests;
