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
    let parsed_value = anthropic_parse_compat_body(&raw);
    let parsed =
        serde_json::from_value(parsed_value).map_err(|err| bad_request_response(err.to_string()))?;
    Ok(ParsedAnthropicRequest { raw, parsed })
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
        Err(HandlerError::Upstream(err)) => OpenAiHttpResult::Json(upstream_error_response(err)),
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
            AnthropicHttpResult::Json(upstream_error_response(err))
        }
        Err(AnthropicHandlerError::Internal(err)) => {
            AnthropicHttpResult::Json(internal_error_response(err))
        }
    }
}

pub(super) fn upstream_error_response(err: String) -> JsonHttpResponse {
    JsonHttpResponse::json(502, json!({"error": err}).to_string())
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
