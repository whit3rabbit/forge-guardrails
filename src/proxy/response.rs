//! HTTP response and SSE formatting helpers for the proxy surface.

use anyllm_translate::anthropic::streaming::StreamEvent;
use axum::body::{Body, Bytes};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value;
use std::io;
use tokio::sync::OwnedMutexGuard;

use crate::error::StreamError;

const CORS_HEADERS: [(&str, &str); 3] = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    (
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization",
    ),
];

/// Keep CORS values in one place so route handlers and test helpers cannot drift.
pub(crate) fn cors_headers() -> Vec<(&'static str, &'static str)> {
    CORS_HEADERS.to_vec()
}

/// Build a complete axum response with proxy-wide CORS headers.
pub(crate) fn build_response(status: u16, ct: &str, body: String) -> Response {
    let status_code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut resp = (status_code, body).into_response();
    if !ct.is_empty() {
        if let Ok(value) = HeaderValue::from_str(ct) {
            resp.headers_mut().insert(header::CONTENT_TYPE, value);
        }
    }
    insert_cors_headers(resp.headers_mut());
    resp
}

/// Build a live OpenAI SSE response from JSON chunk events.
pub(crate) fn build_openai_sse_response<S>(
    events: S,
    guard: Option<OwnedMutexGuard<()>>,
) -> Response
where
    S: Stream<Item = Result<Value, StreamError>> + Send + 'static,
{
    let mut resp = (
        StatusCode::OK,
        Body::from_stream(openai_sse_bytes_stream(events, guard)),
    )
        .into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    insert_cors_headers(resp.headers_mut());
    resp
}

/// OPTIONS responses intentionally carry no body but must still advertise CORS.
pub(crate) fn cors_preflight_response() -> Response {
    build_response(204, "", String::new())
}

fn insert_cors_headers(headers: &mut HeaderMap) {
    for (name, value) in CORS_HEADERS {
        if let Some(header_name) = cors_header_name(name) {
            headers.insert(header_name, HeaderValue::from_static(value));
        }
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

/// Format OpenAI-style SSE events.
///
/// OpenAI streams terminate with a synthetic [DONE] sentinel; Anthropic streams do not.
pub fn format_sse_body(events: &[Value]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str(&format!("data: {}\n\n", event));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

pub(crate) async fn collect_openai_sse_body<S>(events: S) -> Result<String, StreamError>
where
    S: Stream<Item = Result<Value, StreamError>>,
{
    let mut events = Box::pin(events);
    let mut body = String::new();
    while let Some(event) = events.next().await {
        body.push_str(&format!("data: {}\n\n", event?));
    }
    body.push_str("data: [DONE]\n\n");
    Ok(body)
}

pub(crate) fn openai_sse_bytes_stream<S>(
    events: S,
    guard: Option<OwnedMutexGuard<()>>,
) -> impl Stream<Item = Result<Bytes, io::Error>> + Send + 'static
where
    S: Stream<Item = Result<Value, StreamError>> + Send + 'static,
{
    async_stream::stream! {
        let _guard = guard;
        let mut events = Box::pin(events);
        while let Some(event) = events.next().await {
            match event {
                Ok(value) => {
                    yield Ok(Bytes::from(format!("data: {}\n\n", value)));
                }
                Err(err) => {
                    yield Err(io::Error::other(err.to_string()));
                    return;
                }
            }
        }
        yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
    }
}

/// Format Anthropic SSE events with named event fields.
pub fn format_anthropic_sse_body(events: &[StreamEvent]) -> String {
    let mut body = String::new();
    for event in events {
        body.push_str("event: ");
        body.push_str(anthropic_event_name(event));
        body.push('\n');
        body.push_str("data: ");
        body.push_str(&serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()));
        body.push_str("\n\n");
    }
    body
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
