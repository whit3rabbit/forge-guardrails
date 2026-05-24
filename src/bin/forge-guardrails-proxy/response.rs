use anyllm_translate::anthropic::streaming::StreamEvent;
use axum::body::{Body, Bytes};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use forge_guardrails::{AnthropicEventStream, HTTPServer, OpenAiEventStream};
use futures_util::StreamExt;
use std::io;
use tokio::sync::OwnedMutexGuard;

pub(crate) fn build_response(status: u16, content_type: &str, body: String) -> Response {
    let status_code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = (status_code, body).into_response();
    if !content_type.is_empty() {
        if let Ok(value) = HeaderValue::from_str(content_type) {
            response.headers_mut().insert(header::CONTENT_TYPE, value);
        }
    }
    for (name, value) in HTTPServer::cors_headers() {
        if let Some(header_name) = cors_header_name(name) {
            response
                .headers_mut()
                .insert(header_name, HeaderValue::from_static(value));
        }
    }
    response
}

pub(crate) fn build_openai_sse_response(
    events: OpenAiEventStream,
    guard: Option<OwnedMutexGuard<()>>,
) -> Response {
    let mut response = (
        StatusCode::OK,
        Body::from_stream(openai_sse_bytes_stream(events, guard)),
    )
        .into_response();
    insert_sse_headers(response.headers_mut());
    insert_cors_headers(response.headers_mut());
    response
}

pub(crate) fn build_anthropic_sse_response(
    events: AnthropicEventStream,
    guard: Option<OwnedMutexGuard<()>>,
) -> Response {
    let mut response = (
        StatusCode::OK,
        Body::from_stream(anthropic_sse_bytes_stream(events, guard)),
    )
        .into_response();
    insert_sse_headers(response.headers_mut());
    insert_cors_headers(response.headers_mut());
    response
}

fn insert_cors_headers(headers: &mut HeaderMap) {
    for (name, value) in HTTPServer::cors_headers() {
        if let Some(header_name) = cors_header_name(name) {
            headers.insert(header_name, HeaderValue::from_static(value));
        }
    }
}

fn insert_sse_headers(headers: &mut HeaderMap) {
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        HeaderName::from_static("x-accel-buffering"),
        HeaderValue::from_static("no"),
    );
}

fn openai_sse_bytes_stream(
    mut events: OpenAiEventStream,
    guard: Option<OwnedMutexGuard<()>>,
) -> impl futures_core::Stream<Item = Result<Bytes, io::Error>> + Send + 'static {
    async_stream::stream! {
        let _guard = guard;
        while let Some(event) = events.next().await {
            match event {
                Ok(value) => yield Ok(Bytes::from(format!("data: {}\n\n", value))),
                Err(err) => {
                    yield Err(io::Error::other(err.to_string()));
                    return;
                }
            }
        }
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    }
}

fn anthropic_sse_bytes_stream(
    mut events: AnthropicEventStream,
    guard: Option<OwnedMutexGuard<()>>,
) -> impl futures_core::Stream<Item = Result<Bytes, io::Error>> + Send + 'static {
    async_stream::stream! {
        let _guard = guard;
        while let Some(event) = events.next().await {
            match event {
                Ok(event) => {
                    let mut body = String::new();
                    push_anthropic_sse_event(&mut body, &event);
                    yield Ok(Bytes::from(body));
                }
                Err(err) => {
                    yield Err(io::Error::other(err.to_string()));
                    return;
                }
            }
        }
    }
}

fn push_anthropic_sse_event(body: &mut String, event: &StreamEvent) {
    body.push_str("event: ");
    body.push_str(anthropic_event_name(event));
    body.push('\n');
    body.push_str("data: ");
    body.push_str(&serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()));
    body.push_str("\n\n");
}

fn anthropic_event_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::MessageStart { .. } => "message_start",
        StreamEvent::ContentBlockStart { .. } => "content_block_start",
        StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
        StreamEvent::ContentBlockStop { .. } => "content_block_stop",
        StreamEvent::MessageDelta { .. } => "message_delta",
        StreamEvent::MessageStop { .. } => "message_stop",
        StreamEvent::Ping { .. } => "ping",
        StreamEvent::Error { .. } => "error",
    }
}

fn cors_header_name(name: &str) -> Option<HeaderName> {
    match name {
        "Access-Control-Allow-Origin" => {
            Some(HeaderName::from_static("access-control-allow-origin"))
        }
        "Access-Control-Allow-Methods" => {
            Some(HeaderName::from_static("access-control-allow-methods"))
        }
        "Access-Control-Allow-Headers" => {
            Some(HeaderName::from_static("access-control-allow-headers"))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header;

    #[test]
    fn build_response_sets_content_type() {
        let response = build_response(200, "application/json", "{}".to_string());
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
    }

    #[test]
    fn build_response_omits_empty_content_type() {
        let response = build_response(204, "", String::new());
        assert_eq!(response.status(), axum::http::StatusCode::NO_CONTENT);
    }

    #[test]
    fn build_response_sets_cors_headers() {
        let response = build_response(200, "application/json", "{}".to_string());
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .unwrap(),
            "*"
        );
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-methods")
                .unwrap(),
            "GET, POST, OPTIONS"
        );
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-headers")
                .unwrap(),
            "Content-Type, Authorization"
        );
    }
}
