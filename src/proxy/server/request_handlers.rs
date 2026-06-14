use std::sync::Arc;

#[cfg(test)]
use serde_json::json;
use tokio::sync::Mutex;

use super::HTTPServer;
use crate::clients::base::LLMClient;
use crate::context::manager::ContextManager;
use crate::proxy::handler;
use crate::proxy::response;

mod request_http {
    use anyllm_translate::anthropic::MessageCreateRequest;

    use crate::proxy::handler::{
        AnthropicEventStream, AnthropicHandlerError, AnthropicHandlerResult, HandlerError,
        HandlerResult, OpenAiEventStream,
    };

    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/proxy/request_http.rs"
    ));
}

#[cfg(test)]
pub(super) use request_http::MAX_BODY_SIZE;

impl HTTPServer {
    /// Handle an incoming HTTP request.
    ///
    /// Returns (status_code, content_type, headers, body_string).
    #[cfg(test)]
    pub async fn handle_request<C: LLMClient + 'static>(
        &self,
        method: &str,
        path: &str,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> (u16, &'static str, Vec<(&'static str, &'static str)>, String) {
        let headers = Self::cors_headers();
        let route_path = path.split('?').next().unwrap_or(path);
        match (method, route_path) {
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
    #[cfg(test)]
    async fn handle_anthropic_messages<C: LLMClient + 'static>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> (u16, &'static str, String) {
        let request = match request_http::parse_anthropic_body(body) {
            Ok(request) => request,
            Err(response) => return response.into_parts(),
        };

        let _guard = if self.serialize_requests {
            Some(self.request_mutex.lock().await)
        } else {
            None
        };

        match request_http::anthropic_handler_http_result(
            handler::handle_anthropic_messages(
                &request.parsed,
                &request.raw,
                client,
                context_manager,
                self.max_retries,
                self.rescue_enabled,
            )
            .await,
        ) {
            request_http::AnthropicHttpResult::Json(response) => response.into_parts(),
            request_http::AnthropicHttpResult::Stream(events) => {
                match response::collect_anthropic_sse_body(events).await {
                    Ok(body) => (200, "text/event-stream", body),
                    Err(e) => {
                        let message = e.to_string();
                        let status = crate::error::BackendError::status_from_display(&message);
                        request_http::upstream_error_response(message, status).into_parts()
                    }
                }
            }
        }
    }

    pub(super) async fn handle_anthropic_messages_response<C: LLMClient + 'static>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> axum::response::Response {
        let request = match request_http::parse_anthropic_body(body) {
            Ok(request) => request,
            Err(response) => return build_http_response(response),
        };

        let guard = if self.serialize_requests {
            Some(self.request_mutex.clone().lock_owned().await)
        } else {
            None
        };

        match request_http::anthropic_handler_http_result(
            handler::handle_anthropic_messages(
                &request.parsed,
                &request.raw,
                client,
                context_manager,
                self.max_retries,
                self.rescue_enabled,
            )
            .await,
        ) {
            request_http::AnthropicHttpResult::Json(response) => build_http_response(response),
            request_http::AnthropicHttpResult::Stream(events) => {
                response::build_anthropic_sse_response(events, guard)
            }
        }
    }

    /// Handle /v1/chat/completions.
    #[cfg(test)]
    async fn handle_chat_completions<C: LLMClient + 'static>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> (u16, &'static str, String) {
        let parsed = match request_http::parse_openai_body(body) {
            Ok(parsed) => parsed,
            Err(response) => return response.into_parts(),
        };

        let _guard = if self.serialize_requests {
            Some(self.request_mutex.lock().await)
        } else {
            None
        };

        match request_http::openai_handler_http_result(
            handler::handle_chat_completions(
                &parsed,
                client,
                context_manager,
                self.max_retries,
                self.rescue_enabled,
            )
            .await,
        ) {
            request_http::OpenAiHttpResult::Json(response) => response.into_parts(),
            request_http::OpenAiHttpResult::Stream(events) => {
                match response::collect_openai_sse_body(events).await {
                    Ok(body) => (200, "text/event-stream", body),
                    Err(e) => {
                        let message = e.to_string();
                        let status = crate::error::BackendError::status_from_display(&message);
                        request_http::upstream_error_response(message, status).into_parts()
                    }
                }
            }
        }
    }

    pub(super) async fn handle_chat_completions_response<C: LLMClient + 'static>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> axum::response::Response {
        let parsed = match request_http::parse_openai_body(body) {
            Ok(parsed) => parsed,
            Err(response) => return build_http_response(response),
        };

        let guard = if self.serialize_requests {
            Some(self.request_mutex.clone().lock_owned().await)
        } else {
            None
        };

        match request_http::openai_handler_http_result(
            handler::handle_chat_completions(
                &parsed,
                client,
                context_manager,
                self.max_retries,
                self.rescue_enabled,
            )
            .await,
        ) {
            request_http::OpenAiHttpResult::Json(response) => build_http_response(response),
            request_http::OpenAiHttpResult::Stream(events) => {
                response::build_openai_sse_response(events, guard)
            }
        }
    }
}

fn build_http_response(response: request_http::JsonHttpResponse) -> axum::response::Response {
    let (status, content_type, body) = response.into_parts();
    response::build_response(status, content_type, body)
}
