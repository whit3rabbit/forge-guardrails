use serde_json::Value;

use crate::proxy::response;

/// Format SSE events into a chunked transfer encoding body.
///
/// Each event is "data: <json>\n\n". Terminator is "data: [DONE]\n\n".
pub fn format_sse_body(events: &[Value]) -> String {
    response::format_sse_body(events)
}

/// Format Anthropic SSE events.
///
/// Anthropic streams use named SSE events and do not use OpenAI's [DONE]
/// sentinel.
pub fn format_anthropic_sse_body(
    events: &[anyllm_translate::anthropic::streaming::StreamEvent],
) -> String {
    response::format_anthropic_sse_body(events)
}

/// Parse an HTTP request from raw bytes.
/// Returns (method, path, headers, body) for routing.
///
/// WARNING: This parser is a simple helper designed strictly for testing and local/toy
/// server use. It splits on `\r\n\r\n` and does not handle `Content-Length`, chunked encoding,
/// pipelined requests, or partial reads. For production environments, use a robust HTTP
/// framework such as `axum`, `hyper`, or `tiny_http`.
#[allow(clippy::type_complexity)]
pub fn parse_http_request(raw: &[u8]) -> Option<(String, String, Vec<(String, String)>, Vec<u8>)> {
    let text = std::str::from_utf8(raw).ok()?;
    let mut parts = text.splitn(2, "\r\n\r\n");
    let header_section = parts.next()?;
    let body = parts.next().unwrap_or("").as_bytes();

    let mut lines = header_section.split("\r\n");
    let request_line = lines.next()?;
    let mut rl_parts = request_line.split(' ');
    let method = rl_parts.next()?.to_string();
    let path = rl_parts.next()?.to_string();

    let mut headers = Vec::new();
    for line in lines {
        if let Some((key, value)) = line.split_once(':') {
            headers.push((key.trim().to_string(), value.trim().to_string()));
        }
    }

    Some((method, path, headers, body.to_vec()))
}
