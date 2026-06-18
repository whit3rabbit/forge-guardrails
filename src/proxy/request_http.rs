use serde_json::{json, Value};

pub(crate) const MAX_BODY_SIZE: usize = 16 * 1024 * 1024;

pub(super) struct JsonHttpResponse {
    status: u16,
    content_type: &'static str,
    body: String,
}

impl JsonHttpResponse {
    fn json(status: u16, body: String) -> Self {
        Self {
            status,
            content_type: "application/json",
            body,
        }
    }

    pub(super) fn into_parts(self) -> (u16, &'static str, String) {
        (self.status, self.content_type, self.body)
    }
}

pub(super) struct ParsedAnthropicRequest {
    pub(super) raw: Value,
    pub(super) parsed: MessageCreateRequest,
}

pub(super) enum OpenAiHttpResult {
    Json(JsonHttpResponse),
    Stream(OpenAiEventStream),
}

pub(super) enum AnthropicHttpResult {
    Json(JsonHttpResponse),
    Stream(AnthropicEventStream),
}

pub(super) fn parse_openai_body(body: &[u8]) -> Result<Value, JsonHttpResponse> {
    ensure_body_size(body)?;
    serde_json::from_slice(body).map_err(|err| bad_request_response(err.to_string()))
}

pub(super) fn parse_anthropic_body(
    body: &[u8],
) -> Result<ParsedAnthropicRequest, JsonHttpResponse> {
    ensure_body_size(body)?;
    let raw: Value =
        serde_json::from_slice(body).map_err(|err| bad_request_response(err.to_string()))?;
    let parsed = parse_anthropic_request_value(&raw)?;
    Ok(ParsedAnthropicRequest { raw, parsed })
}

pub(super) fn parse_anthropic_request_value(
    raw: &Value,
) -> Result<MessageCreateRequest, JsonHttpResponse> {
    let parsed_value = anthropic_parse_compat_body(raw);
    serde_json::from_value(parsed_value).map_err(|err| bad_request_response(err.to_string()))
}

fn anthropic_parse_compat_body(raw: &Value) -> Value {
    let mut value = raw.clone();
    let Some(obj) = value.as_object_mut() else {
        return value;
    };
    let Some(thinking) = obj.get("thinking") else {
        return value;
    };
    let Some(kind) = thinking.get("type").and_then(Value::as_str) else {
        return value;
    };

    if kind != "enabled" && kind != "disabled" {
        obj.remove("thinking");
    }
    value
}

pub(super) fn openai_handler_http_result(
    result: Result<HandlerResult, HandlerError>,
) -> OpenAiHttpResult {
    match result {
        Ok(HandlerResult::Response(value)) => {
            OpenAiHttpResult::Json(JsonHttpResponse::json(200, value.to_string()))
        }
        Ok(HandlerResult::StreamBody(events)) => OpenAiHttpResult::Stream(events),
        Ok(HandlerResult::AnthropicResponse(_))
        | Ok(HandlerResult::AnthropicStreamBody(_)) => OpenAiHttpResult::Json(
            internal_error_response("internal response shape mismatch".to_string()),
        ),
        Err(HandlerError::BadRequest(err)) => OpenAiHttpResult::Json(bad_request_response(err)),
        Err(HandlerError::Upstream(err)) => {
            OpenAiHttpResult::Json(upstream_error_response(err, None))
        }
        Err(HandlerError::UpstreamStatus { message, status }) => {
            OpenAiHttpResult::Json(upstream_error_response(message, Some(status)))
        }
    }
}

pub(super) fn anthropic_handler_http_result(
    result: Result<AnthropicHandlerResult, AnthropicHandlerError>,
) -> AnthropicHttpResult {
    match result {
        Ok(AnthropicHandlerResult::Response(value)) => {
            AnthropicHttpResult::Json(JsonHttpResponse::json(200, value.to_string()))
        }
        Ok(AnthropicHandlerResult::StreamBody(events)) => AnthropicHttpResult::Stream(events),
        Err(AnthropicHandlerError::BadRequest(err)) => {
            AnthropicHttpResult::Json(bad_request_response(err))
        }
        Err(AnthropicHandlerError::Upstream(err)) => {
            AnthropicHttpResult::Json(upstream_error_response(err, None))
        }
        Err(AnthropicHandlerError::UpstreamStatus { message, status }) => {
            AnthropicHttpResult::Json(upstream_error_response(message, Some(status)))
        }
        Err(AnthropicHandlerError::Internal(err)) => {
            AnthropicHttpResult::Json(internal_error_response(err))
        }
    }
}

/// Builds the client-facing response for an upstream failure.
///
/// `upstream_status` is the real status from the backend when it is known as a
/// typed value (threaded through `HandlerError::UpstreamStatus`); `None` means
/// the failure was not a typed upstream HTTP error (e.g. a guarded-validation
/// failure) and collapses to a generic 502.
pub(super) fn upstream_error_response(
    message: String,
    upstream_status: Option<i64>,
) -> JsonHttpResponse {
    let client_status = client_status_for_upstream(upstream_status);
    // Surface upstream failures in the proxy log; otherwise they are only
    // visible to the calling client (e.g. an eval harness sees an opaque 502).
    // The proxy has no tracing subscriber, so stderr is the operator-visible
    // channel here, matching the startup banner.
    let upstream = upstream_status
        .map(|status| status.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    eprintln!(
        "warning: upstream request failed (upstream status {upstream}, returning {client_status}): {}",
        truncate_for_log(&message)
    );
    JsonHttpResponse::json(client_status, json!({"error": message}).to_string())
}

/// Maps an upstream status to the status the proxy returns to its client.
///
/// Actionable upstream signals (429 rate-limit, 503 unavailable, 504 gateway
/// timeout) are passed through so clients can distinguish them from a real
/// gateway failure; everything else keeps the historical 502.
fn client_status_for_upstream(upstream_status: Option<i64>) -> u16 {
    match upstream_status {
        Some(429) => 429,
        Some(503) => 503,
        Some(504) => 504,
        _ => 502,
    }
}

/// Truncates a message for a single log line.
fn truncate_for_log(message: &str) -> String {
    const MAX: usize = 500;
    if message.len() <= MAX {
        return message.to_string();
    }
    let mut end = MAX;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &message[..end])
}

fn ensure_body_size(body: &[u8]) -> Result<(), JsonHttpResponse> {
    if body.len() > MAX_BODY_SIZE {
        return Err(JsonHttpResponse::json(
            413,
            json!({"error": "request too large"}).to_string(),
        ));
    }
    Ok(())
}

fn bad_request_response(err: String) -> JsonHttpResponse {
    JsonHttpResponse::json(400, json!({"error": err}).to_string())
}

fn internal_error_response(err: String) -> JsonHttpResponse {
    JsonHttpResponse::json(500, json!({"error": err}).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actionable_upstream_statuses_pass_through_others_stay_502() {
        for upstream in [429, 503, 504] {
            let (status, _, _) =
                upstream_error_response(format!("upstream {upstream}"), Some(upstream))
                    .into_parts();
            assert_eq!(i64::from(status), upstream);
        }

        // A real 500 and a transport failure (status 0) collapse to 502.
        let (status, _, _) =
            upstream_error_response("server error".to_string(), Some(500)).into_parts();
        assert_eq!(status, 502);
        let (status, _, _) =
            upstream_error_response("connection reset".to_string(), Some(0)).into_parts();
        assert_eq!(status, 502);

        // A statusless (non-typed) failure, e.g. guarded-validation, is 502 and
        // is never parsed for an embedded marker.
        let (status, _, _) = upstream_error_response(
            "model failed guarded tool-call validation".to_string(),
            None,
        )
        .into_parts();
        assert_eq!(status, 502);
    }

    #[test]
    fn error_body_is_preserved() {
        let (_, content_type, body) =
            upstream_error_response("rate limited".to_string(), Some(429)).into_parts();
        assert_eq!(content_type, "application/json");
        assert!(body.contains("rate limited"));
    }
}
