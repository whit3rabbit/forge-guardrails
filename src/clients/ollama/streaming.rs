use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use super::{response, OllamaClient};
use crate::clients::base::{
    ChunkType, LLMResponse, StreamChunk, TextResponse, TokenUsage, ToolCall,
};
use crate::error::StreamError;

/// Maximum number of bytes allowed for a single in-flight NDJSON line.
const MAX_OLLAMA_NDJSON_LINE_BYTES: usize = 1024 * 1024;

pub(super) fn parse_ollama_ndjson(
    resp: reqwest::Response,
    think: bool,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    let byte_stream = resp.bytes_stream();
    let stream = async_stream::stream! {
        use futures_util::StreamExt;
        let mut inner = Box::pin(byte_stream);
        let mut line_buf = String::new();
        let mut pending_tc: Option<Vec<Value>> = None;
        let mut acc_content = String::new();
        let mut acc_thinking = String::new();
        loop {
            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();
                if line.trim().is_empty() { continue; }
                let obj: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let done = obj.get("done").and_then(|d| d.as_bool()).unwrap_or(false);
                if let Some(msg) = obj.get("message") {
                    if let Some(c) = msg.get("content").and_then(|c| c.as_str()) {
                        if !c.is_empty() {
                            acc_content.push_str(c);
                            yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(c));
                        }
                    }
                    if let Some(thinking) = msg.get("thinking").and_then(|t| t.as_str()) {
                        acc_thinking.push_str(thinking);
                    }
                    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()).cloned() {
                        if !tcs.is_empty() { pending_tc = Some(tcs); }
                    }
                }
                if done {
                    let prompt = obj.get("prompt_eval_count").and_then(|v| v.as_i64()).unwrap_or(0);
                    let completion = obj.get("eval_count").and_then(|v| v.as_i64()).unwrap_or(0);
                    if let Ok(mut guard) = last_usage.lock() {
                        *guard = Some(TokenUsage::new(prompt, completion, prompt + completion));
                    }
                    let response_val = json!({"message": {
                        "content": acc_content.clone(),
                        "thinking": if acc_thinking.is_empty() { Value::Null } else { json!(acc_thinking) },
                    }});
                    let reasoning = OllamaClient::resolve_reasoning(think, &response_val);
                    let final_resp = if let Some(tcs) = pending_tc.take() {
                        let mut calls = Vec::new();
                        let mut bad_args = None;
                        for (i, tc) in tcs.iter().enumerate() {
                            let name = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("");
                            let args_val = tc.get("function").and_then(|f| f.get("arguments"));
                            let args = match response::parse_tool_args_value(args_val) {
                                Ok(args) => args,
                                Err(raw_args) => {
                                    bad_args = Some(raw_args);
                                    break;
                                }
                            };
                            let mut call = ToolCall::new(name, args);
                            if i == 0 { if let Some(r) = &reasoning { call = call.with_reasoning(r); } }
                            calls.push(call);
                        }
                        if let Some(raw_args) = bad_args {
                            let content = acc_content.trim().to_string();
                            LLMResponse::Text(TextResponse::new(if content.is_empty() {
                                raw_args
                            } else {
                                content
                            }))
                        } else {
                            LLMResponse::ToolCalls(calls)
                        }
                    } else {
                        let content = acc_content.trim().to_string();
                        LLMResponse::Text(TextResponse::new(content))
                    };
                    yield Ok(StreamChunk::new(ChunkType::Final).with_response(final_resp));
                    return;
                }
            }
            match inner.next().await {
                Some(Ok(b)) => {
                    if let Err(e) = append_ollama_line_buf_with_limit(&mut line_buf, &b) {
                        yield Err(e);
                        return;
                    }
                }
                Some(Err(e)) => { yield Err(StreamError::new(e.to_string())); return; }
                None => {
                    if line_buf.trim().is_empty() {
                        return;
                    }
                    line_buf.push('\n');
                }
            }
        }
    };
    stream
}

fn append_ollama_line_buf_with_limit(
    line_buf: &mut String,
    chunk: &[u8],
) -> Result<(), StreamError> {
    if line_buf.len().saturating_add(chunk.len()) > MAX_OLLAMA_NDJSON_LINE_BYTES {
        return Err(StreamError::new(format!(
            "Ollama stream line exceeded {} bytes without newline",
            MAX_OLLAMA_NDJSON_LINE_BYTES
        )));
    }

    let decoded = String::from_utf8_lossy(chunk);
    if line_buf.len().saturating_add(decoded.len()) > MAX_OLLAMA_NDJSON_LINE_BYTES {
        return Err(StreamError::new(format!(
            "Ollama stream line exceeded {} bytes without newline",
            MAX_OLLAMA_NDJSON_LINE_BYTES
        )));
    }

    line_buf.push_str(&decoded);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndjson_line_buf_limit_allows_small_chunks() {
        let mut line_buf = String::new();
        let chunk = vec![b'a'; 32];
        let result = append_ollama_line_buf_with_limit(&mut line_buf, &chunk);
        assert!(result.is_ok());
        assert_eq!(line_buf.len(), 32);
    }

    #[test]
    fn ndjson_line_buf_limit_rejects_oversized_line() {
        let mut line_buf = "a".repeat(MAX_OLLAMA_NDJSON_LINE_BYTES);
        let result = append_ollama_line_buf_with_limit(&mut line_buf, b"b");
        assert!(result.is_err());
        assert_eq!(line_buf.len(), MAX_OLLAMA_NDJSON_LINE_BYTES);
        let err = result.err().unwrap();
        assert!(err.to_string().contains("exceeded"));
    }
}
