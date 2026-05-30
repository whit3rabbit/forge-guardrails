//! HTTP route handlers for the proxy.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::Response;
use serde_json::{json, Value};
use tokio::sync::Mutex as TokioMutex;

use forge_guardrails::{
    handle_anthropic_messages_with_scorers_and_tool_controls,
    handle_chat_completions_with_scorers_and_tool_controls, ContextManager, FinalResponseScorer,
    LLMClient, NoCompact, ToolCallScorer, ToolOutputCompressionState,
};

use super::AppState;
use crate::config::ProxyConfig;
use crate::response::{build_anthropic_sse_response, build_openai_sse_response, build_response};

/// Request mapping helper module.
pub mod request_http {
    use anyllm_translate::anthropic::MessageCreateRequest;
    use forge_guardrails::{
        AnthropicEventStream, AnthropicHandlerError, AnthropicHandlerResult, HandlerError,
        HandlerResult, OpenAiEventStream,
    };

    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/proxy/request_http.rs"
    ));
}

/// Handler for the `/health` endpoint.
pub async fn health() -> Response {
    build_response(200, "application/json", json!({"status": "ok"}).to_string())
}

/// Handler for the `/v1/models` endpoint.
pub async fn models(State(state): State<AppState>) -> Response {
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

/// Handler for the `/v1/chat/completions` endpoint.
pub async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Response {
    let parsed = match request_http::parse_openai_body(body.as_ref()) {
        Ok(value) => value,
        Err(response) => return build_http_response(response),
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

    match request_http::openai_handler_http_result(
        handle_chat_completions_with_scorers_and_tool_controls(
            &parsed,
            &client,
            &context_manager,
            state.config.max_retries,
            state.config.rescue_enabled,
            state.scorer.clone(),
            state.final_response_scorer.clone(),
            state.config.tool_output_compression.clone(),
            Some(state.tool_output_state.clone()),
            state.config.tool_call_policy.clone(),
        )
        .await,
    ) {
        request_http::OpenAiHttpResult::Json(response) => build_http_response(response),
        request_http::OpenAiHttpResult::Stream(events) => build_openai_sse_response(events, guard),
    }
}

/// Handler for the `/v1/messages` endpoint.
pub async fn anthropic_messages(State(state): State<AppState>, body: Bytes) -> Response {
    let request = match request_http::parse_anthropic_body(body.as_ref()) {
        Ok(request) => request,
        Err(response) => return build_http_response(response),
    };
    let model = extract_model_from_value(&request.raw, &state.config.default_model);
    let client = Arc::new(state.client_factory.client_for_model(model));
    anthropic_messages_with_request_client(
        state.config,
        state.request_mutex,
        state.scorer,
        state.final_response_scorer,
        state.tool_output_state,
        client,
        request,
    )
    .await
}

#[cfg(test)]
pub(crate) async fn anthropic_messages_with_client<C: LLMClient + 'static>(
    config: Arc<ProxyConfig>,
    request_mutex: Arc<TokioMutex<()>>,
    client: Arc<C>,
    body: Bytes,
) -> Response {
    let request = match request_http::parse_anthropic_body(body.as_ref()) {
        Ok(request) => request,
        Err(response) => return build_http_response(response),
    };
    anthropic_messages_with_request_client(
        config,
        request_mutex,
        None,
        None,
        Arc::new(ToolOutputCompressionState::new()),
        client,
        request,
    )
    .await
}

async fn anthropic_messages_with_request_client<C: LLMClient + 'static>(
    config: Arc<ProxyConfig>,
    request_mutex: Arc<TokioMutex<()>>,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    tool_output_state: Arc<ToolOutputCompressionState>,
    client: Arc<C>,
    request: request_http::ParsedAnthropicRequest,
) -> Response {
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

    match request_http::anthropic_handler_http_result(
        handle_anthropic_messages_with_scorers_and_tool_controls(
            &request.parsed,
            &request.raw,
            &client,
            &context_manager,
            config.max_retries,
            config.rescue_enabled,
            scorer,
            final_response_scorer,
            config.tool_output_compression.clone(),
            Some(tool_output_state),
            config.tool_call_policy.clone(),
        )
        .await,
    ) {
        request_http::AnthropicHttpResult::Json(response) => build_http_response(response),
        request_http::AnthropicHttpResult::Stream(events) => {
            build_anthropic_sse_response(events, guard)
        }
    }
}

/// Handler for OPTIONS CORS preflight requests.
pub async fn cors_preflight() -> Response {
    build_response(204, "", String::new())
}

/// Build an HTTP Response from a JsonHttpResponse.
fn build_http_response(response: request_http::JsonHttpResponse) -> Response {
    let (status, content_type, body) = response.into_parts();
    build_response(status, content_type, body)
}

#[cfg(test)]
pub(crate) fn extract_openai_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

#[cfg(test)]
pub(crate) fn extract_anthropic_model(body: &[u8], default_model: &str) -> String {
    extract_json_model(body, default_model)
}

#[cfg(test)]
pub(crate) fn extract_json_model(body: &[u8], default_model: &str) -> String {
    serde_json::from_slice::<Value>(body)
        .ok()
        .map(|value| extract_model_from_value(&value, default_model))
        .unwrap_or_else(|| default_model.to_string())
}

/// Helper function to extract model string from JSON value.
pub(crate) fn extract_model_from_value(value: &Value, default_model: &str) -> String {
    value
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| default_model.to_string())
}
