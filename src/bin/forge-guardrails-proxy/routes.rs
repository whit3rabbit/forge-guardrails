//! Server routing and startup logic for the proxy daemon.

pub mod handlers;
#[cfg(test)]
mod tests;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::routing::{get, options, post};
use axum::Router;
use forge_guardrails::{
    init_proxy_classifier_log_sink_from_env, FinalResponseScorer, ServerManager, ToolCallScorer,
    ToolOutputCompressionState,
};
use tokio::sync::Mutex as TokioMutex;

use crate::client::ClientFactory;
use crate::config::ProxyConfig;

/// Shared application state for axum handlers.
#[derive(Clone)]
pub struct AppState {
    pub(crate) config: Arc<ProxyConfig>,
    pub(crate) client_factory: Arc<ClientFactory>,
    pub(crate) request_mutex: Arc<TokioMutex<()>>,
    pub(crate) scorer: Option<Arc<dyn ToolCallScorer>>,
    pub(crate) final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    pub(crate) tool_output_state: Arc<ToolOutputCompressionState>,
}

/// Run the HTTP server daemon using the provided configuration and client factory.
pub(crate) async fn serve(
    config: ProxyConfig,
    client_factory: ClientFactory,
    managed_server: Option<ServerManager>,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
) -> Result<(), String> {
    let result = serve_inner(config, client_factory, scorer, final_response_scorer).await;
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

/// Internal helper to bind TcpListener and start serving the axum Router.
async fn serve_inner(
    config: ProxyConfig,
    client_factory: ClientFactory,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
) -> Result<(), String> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|err| format!("invalid bind address: {err}"))?;
    let state = AppState {
        config: Arc::new(config.clone()),
        client_factory: Arc::new(client_factory),
        request_mutex: Arc::new(TokioMutex::new(())),
        scorer,
        final_response_scorer,
        tool_output_state: Arc::new(ToolOutputCompressionState::new()),
    };
    init_proxy_classifier_log_sink_from_env();

    eprintln!(
        "forge-guardrails-proxy listening on http://{}:{}",
        config.host, config.port
    );
    eprintln!(
        "warning: inbound auth is not enforced; do not expose this proxy publicly without an auth layer"
    );
    if config.verbose {
        eprintln!(
            "proxy config: model={}, context_tokens={}, max_retries={}, rescue_enabled={}, serialize_requests={}, tool_output_compression={}, tool_call_policy={}",
            config.default_model,
            config.context_tokens,
            config.max_retries,
            config.rescue_enabled,
            config.serialize_requests,
            config.tool_output_compression.mode,
            config.tool_call_policy.mode
        );
    }

    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/chat/completions", options(handlers::cors_preflight))
        .route("/v1/messages", post(handlers::anthropic_messages))
        .route("/v1/messages", options(handlers::cors_preflight))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|err| format!("failed to bind {addr}: {err}"))?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|err| format!("server failed: {err}"))
}

/// Helper that waits for SIGINT or SIGTERM and triggers graceful shutdown.
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
