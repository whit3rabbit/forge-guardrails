use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::Response;
use axum::routing::{get, options, post};
use axum::Router;
use forge_guardrails::{ContextManager, HTTPServer, NoCompact, ServerManager};
use serde_json::{json, Value};
use tokio::sync::Mutex as TokioMutex;

use crate::client::ClientFactory;
use crate::config::ProxyConfig;
use crate::response::build_response;

#[derive(Clone)]
struct AppState {
    config: Arc<ProxyConfig>,
    client_factory: Arc<ClientFactory>,
    request_mutex: Arc<TokioMutex<()>>,
}

pub(crate) async fn serve(
    config: ProxyConfig,
    client_factory: ClientFactory,
    managed_server: Option<ServerManager>,
) -> Result<(), String> {
    let result = serve_inner(config, client_factory).await;
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

async fn serve_inner(config: ProxyConfig, client_factory: ClientFactory) -> Result<(), String> {
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|err| format!("invalid bind address: {err}"))?;
    let state = AppState {
        config: Arc::new(config.clone()),
        client_factory: Arc::new(client_factory),
        request_mutex: Arc::new(TokioMutex::new(())),
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
    proxy_post(state, "/v1/chat/completions", body, extract_openai_model).await
}

async fn anthropic_messages(State(state): State<AppState>, body: Bytes) -> Response {
    proxy_post(state, "/v1/messages", body, extract_anthropic_model).await
}

async fn cors_preflight() -> Response {
    build_response(204, "", String::new())
}

async fn proxy_post(
    state: AppState,
    path: &'static str,
    body: Bytes,
    model_from_body: fn(&[u8], &str) -> String,
) -> Response {
    let model = model_from_body(body.as_ref(), &state.config.default_model);
    let client = Arc::new(state.client_factory.client_for_model(model.clone()));
    let context_manager = Arc::new(TokioMutex::new(ContextManager::new(
        Box::new(NoCompact),
        state.config.context_tokens,
        None,
        None,
        None,
    )));
    let server = HTTPServer::new(
        &state.config.host,
        state.config.port,
        false,
        state.config.max_retries,
        state.config.rescue_enabled,
        &model,
    );

    let _guard = if state.config.serialize_requests {
        Some(state.request_mutex.lock().await)
    } else {
        None
    };

    let (status, content_type, _headers, response_body) = server
        .handle_request("POST", path, body.as_ref(), &client, &context_manager)
        .await;
    build_response(status, content_type, response_body)
}

fn extract_openai_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

fn extract_anthropic_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

fn extract_json_model(body: &[u8], default_model: &str) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("model")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|model| !model.is_empty())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| default_model.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
