use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::Mutex;

use super::{HTTPServer, MAX_BODY_SIZE};
use crate::clients::base::LLMClient;
use crate::context::manager::ContextManager;
use crate::proxy::handler::{
    self, AnthropicHandlerError, AnthropicHandlerResult, HandlerError, HandlerResult,
};
use crate::proxy::response;

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
        if body.len() > MAX_BODY_SIZE {
            return (
                413,
                "application/json",
                json!({"error": "request too large"}).to_string(),
            );
        }

        let raw: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    400,
                    "application/json",
                    json!({"error": e.to_string()}).to_string(),
                );
            }
        };

        let parsed: anyllm_translate::anthropic::MessageCreateRequest =
            match serde_json::from_value(raw.clone()) {
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
            &raw,
            client,
            context_manager,
            self.max_retries,
            self.rescue_enabled,
        )
        .await
        {
            Ok(AnthropicHandlerResult::Response(v)) => (200, "application/json", v.to_string()),
            Ok(AnthropicHandlerResult::StreamBody(events)) => {
                match response::collect_anthropic_sse_body(events).await {
                    Ok(body) => (200, "text/event-stream", body),
                    Err(e) => (
                        502,
                        "application/json",
                        json!({"error": e.to_string()}).to_string(),
                    ),
                }
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

    pub(super) async fn handle_anthropic_messages_response<C: LLMClient + 'static>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> axum::response::Response {
        if body.len() > MAX_BODY_SIZE {
            return response::build_response(
                413,
                "application/json",
                json!({"error": "request too large"}).to_string(),
            );
        }

        let raw: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return response::build_response(
                    400,
                    "application/json",
                    json!({"error": e.to_string()}).to_string(),
                );
            }
        };

        let parsed: anyllm_translate::anthropic::MessageCreateRequest =
            match serde_json::from_value(raw.clone()) {
                Ok(v) => v,
                Err(e) => {
                    return response::build_response(
                        400,
                        "application/json",
                        json!({"error": e.to_string()}).to_string(),
                    );
                }
            };

        let guard = if self.serialize_requests {
            Some(self.request_mutex.clone().lock_owned().await)
        } else {
            None
        };

        match handler::handle_anthropic_messages(
            &parsed,
            &raw,
            client,
            context_manager,
            self.max_retries,
            self.rescue_enabled,
        )
        .await
        {
            Ok(AnthropicHandlerResult::Response(v)) => {
                response::build_response(200, "application/json", v.to_string())
            }
            Ok(AnthropicHandlerResult::StreamBody(events)) => {
                response::build_anthropic_sse_response(events, guard)
            }
            Err(AnthropicHandlerError::BadRequest(e)) => {
                response::build_response(400, "application/json", json!({"error": e}).to_string())
            }
            Err(AnthropicHandlerError::Upstream(e)) => {
                response::build_response(502, "application/json", json!({"error": e}).to_string())
            }
            Err(AnthropicHandlerError::Internal(e)) => {
                response::build_response(500, "application/json", json!({"error": e}).to_string())
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
                HandlerResult::StreamBody(events) => {
                    match response::collect_openai_sse_body(events).await {
                        Ok(body) => (200, "text/event-stream", body),
                        Err(e) => (
                            502,
                            "application/json",
                            json!({"error": e.to_string()}).to_string(),
                        ),
                    }
                }
            },
            Err(HandlerError::BadRequest(e)) => {
                (400, "application/json", json!({"error": e}).to_string())
            }
            Err(HandlerError::Upstream(e)) => {
                (502, "application/json", json!({"error": e}).to_string())
            }
        }
    }

    pub(super) async fn handle_chat_completions_response<C: LLMClient + 'static>(
        &self,
        body: &[u8],
        client: &Arc<C>,
        context_manager: &Arc<Mutex<ContextManager>>,
    ) -> axum::response::Response {
        if body.len() > MAX_BODY_SIZE {
            return response::build_response(
                413,
                "application/json",
                json!({"error": "request too large"}).to_string(),
            );
        }

        let parsed: Value = match serde_json::from_slice(body) {
            Ok(v) => v,
            Err(e) => {
                return response::build_response(
                    400,
                    "application/json",
                    json!({"error": e.to_string()}).to_string(),
                );
            }
        };

        let guard = if self.serialize_requests {
            Some(self.request_mutex.clone().lock_owned().await)
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
            Ok(HandlerResult::Response(v)) => {
                response::build_response(200, "application/json", v.to_string())
            }
            Ok(HandlerResult::StreamBody(events)) => {
                response::build_openai_sse_response(events, guard)
            }
            Err(HandlerError::BadRequest(e)) => {
                response::build_response(400, "application/json", json!({"error": e}).to_string())
            }
            Err(HandlerError::Upstream(e)) => {
                response::build_response(502, "application/json", json!({"error": e}).to_string())
            }
        }
    }
}
