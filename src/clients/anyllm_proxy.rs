//! Client adapter for a running anyllm_proxy sidecar.
//!
//! This keeps forge-guardrails responsible for interception, validation, and
//! nudging while delegating provider routing and upstream compatibility to
//! anyllm_proxy over its OpenAI-compatible chat completions endpoint.

use std::sync::Mutex;

use futures_util::StreamExt;
use indexmap::IndexMap;
use serde_json::{json, Value};

use crate::clients::base::{
    format_tool, ApiFormat, ChunkStream, ChunkType, LLMClient, LLMResponse, SamplingParams,
    StreamChunk, TextResponse, TokenUsage, ToolCall,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

/// Default anyllm_proxy OpenAI-compatible chat completions endpoint.
pub const DEFAULT_ANYLLM_PROXY_URL: &str = "http://127.0.0.1:3000/v1/chat/completions";

/// LLM client that forwards guarded OpenAI-format calls to anyllm_proxy.
pub struct AnyLlmProxyClient {
    chat_completions_url: String,
    model: String,
    api_key: Option<String>,
    timeout_secs: f64,
    context_length: Option<i64>,
    last_usage: Mutex<Option<TokenUsage>>,
}

impl AnyLlmProxyClient {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            chat_completions_url: DEFAULT_ANYLLM_PROXY_URL.to_string(),
            model: model.into(),
            api_key: None,
            timeout_secs: 300.0,
            context_length: None,
            last_usage: Mutex::new(None),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.chat_completions_url = normalize_chat_completions_url(&url.into());
        self
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_context_length(mut self, tokens: i64) -> Self {
        self.context_length = Some(tokens);
        self
    }

    pub fn with_timeout(mut self, timeout_secs: f64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    fn build_request_body(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
        stream: bool,
    ) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
        });

        if let Some(tool_specs) = tools {
            if !tool_specs.is_empty() {
                let formatted: Vec<Value> = tool_specs.iter().map(format_tool).collect();
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("tools".to_string(), Value::Array(formatted));
                }
            }
        }

        if let Some(params) = sampling {
            if let Some(obj) = body.as_object_mut() {
                for (key, value) in params {
                    obj.insert(key, value);
                }
            }
        }

        body
    }

    async fn send_request(&self, body: Value) -> Result<reqwest::Response, BackendError> {
        let client = reqwest::Client::new();
        let mut req = client
            .post(&self.chat_completions_url)
            .header("content-type", "application/json")
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| BackendError::new(0, e.to_string()))?;
        let status = resp.status().as_u16() as i64;
        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(BackendError::new(status, body_text));
        }
        Ok(resp)
    }

    fn record_usage(&self, usage: Option<&anyllm_translate::openai::ChatUsage>) {
        let token_usage = usage
            .map(|u| {
                TokenUsage::new(
                    u.prompt_tokens as i64,
                    u.completion_tokens as i64,
                    u.total_tokens as i64,
                )
            })
            .unwrap_or_else(TokenUsage::empty);
        if let Ok(mut guard) = self.last_usage.lock() {
            *guard = Some(token_usage);
        }
    }

    fn parse_response(
        &self,
        response: anyllm_translate::openai::ChatCompletionResponse,
    ) -> LLMResponse {
        self.record_usage(response.usage.as_ref());
        parse_openai_response(response)
    }
}

impl LLMClient for AnyLlmProxyClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<TokenUsage> {
        self.last_usage.lock().ok().and_then(|guard| guard.clone())
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        let body = self.build_request_body(messages, tools, sampling, false);
        let resp = self.send_request(body).await?;
        let status = resp.status().as_u16() as i64;
        let response_json = resp
            .json::<anyllm_translate::openai::ChatCompletionResponse>()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;
        Ok(self.parse_response(response_json))
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        let body = self.build_request_body(messages, tools, sampling, true);
        let resp = self
            .send_request(body)
            .await
            .map_err(|e| StreamError::new(e.to_string()))?;
        Ok(Box::pin(parse_openai_sse(resp)))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(self.context_length)
    }
}

fn normalize_chat_completions_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if trimmed.ends_with("/v1/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}

fn parse_openai_response(
    response: anyllm_translate::openai::ChatCompletionResponse,
) -> LLMResponse {
    let Some(choice) = response.choices.into_iter().next() else {
        return LLMResponse::Text(TextResponse::new(""));
    };

    let message = choice.message;
    if let Some(tool_calls) = message.tool_calls {
        if !tool_calls.is_empty() {
            let reasoning = message.reasoning_content.clone();
            let calls = tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, tc)| {
                    let mut call =
                        ToolCall::new(tc.function.name, parse_args_string(&tc.function.arguments))
                            .with_id(tc.id);
                    if index == 0 {
                        if let Some(ref text) = reasoning {
                            call = call.with_reasoning(text);
                        }
                    }
                    call
                })
                .collect();
            return LLMResponse::ToolCalls(calls);
        }
    }

    LLMResponse::Text(TextResponse::new(content_to_string(message.content)))
}

fn content_to_string(content: Option<anyllm_translate::openai::ChatContent>) -> String {
    match content {
        Some(anyllm_translate::openai::ChatContent::Text(text)) => text,
        Some(anyllm_translate::openai::ChatContent::Parts(parts)) => parts
            .into_iter()
            .filter_map(|part| match part {
                anyllm_translate::openai::ChatContentPart::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    }
}

fn parse_args_string(args: &str) -> IndexMap<String, Value> {
    match serde_json::from_str::<Value>(args) {
        Ok(Value::Object(obj)) => obj.into_iter().collect(),
        _ => IndexMap::new(),
    }
}

fn parse_openai_sse(
    resp: reqwest::Response,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    let byte_stream = resp.bytes_stream();
    async_stream::stream! {
        let mut inner = Box::pin(byte_stream);
        let mut line_buf = String::new();
        let mut accumulated_text = String::new();
        let mut accumulated_reasoning = String::new();
        let mut accumulated_tools: Vec<(String, String, String)> = Vec::new();

        loop {
            let chunk = match inner.next().await {
                Some(Ok(bytes)) => bytes,
                Some(Err(e)) => {
                    yield Err(StreamError::new(e.to_string()));
                    return;
                }
                None => break,
            };

            line_buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(newline_pos) = line_buf.find('\n') {
                let raw = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();
                let Some(data) = raw.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    let final_response = final_stream_response(
                        &accumulated_text,
                        &accumulated_reasoning,
                        &accumulated_tools,
                    );
                    yield Ok(StreamChunk::new(ChunkType::Final).with_response(final_response));
                    return;
                }

                let evt: anyllm_translate::openai::ChatCompletionChunk = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                for choice in evt.choices {
                    if let Some(reasoning) = choice.delta.reasoning_content {
                        accumulated_reasoning.push_str(&reasoning);
                    }
                    if let Some(content) = choice.delta.content {
                        accumulated_text.push_str(&content);
                        yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(content));
                    }
                    if let Some(tool_calls) = choice.delta.tool_calls {
                        for tc in tool_calls {
                            let index = tc.index as usize;
                            while accumulated_tools.len() <= index {
                                accumulated_tools.push((String::new(), String::new(), String::new()));
                            }
                            if let Some(id) = tc.id {
                                if !id.is_empty() {
                                    accumulated_tools[index].0 = id;
                                }
                            }
                            if let Some(function) = tc.function {
                                if let Some(name) = function.name {
                                    if !name.is_empty() {
                                        accumulated_tools[index].1 = name;
                                    }
                                }
                                if let Some(args) = function.arguments {
                                    accumulated_tools[index].2.push_str(&args);
                                }
                            }
                        }
                    }
                    if choice.finish_reason.is_some() {
                        let final_response = final_stream_response(
                            &accumulated_text,
                            &accumulated_reasoning,
                            &accumulated_tools,
                        );
                        yield Ok(StreamChunk::new(ChunkType::Final).with_response(final_response));
                        return;
                    }
                }
            }
        }

        let final_response = final_stream_response(
            &accumulated_text,
            &accumulated_reasoning,
            &accumulated_tools,
        );
        yield Ok(StreamChunk::new(ChunkType::Final).with_response(final_response));
    }
}

fn final_stream_response(
    accumulated_text: &str,
    accumulated_reasoning: &str,
    accumulated_tools: &[(String, String, String)],
) -> LLMResponse {
    let calls: Vec<ToolCall> = accumulated_tools
        .iter()
        .filter(|(_, name, _)| !name.is_empty())
        .enumerate()
        .map(|(index, (id, name, args))| {
            let mut call = ToolCall::new(name.clone(), parse_args_string(args)).with_id(id.clone());
            if index == 0 && !accumulated_reasoning.is_empty() {
                call = call.with_reasoning(accumulated_reasoning.to_string());
            }
            call
        })
        .collect();

    if calls.is_empty() {
        LLMResponse::Text(TextResponse::new(accumulated_text.to_string()))
    } else {
        LLMResponse::ToolCalls(calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_full_endpoint() {
        assert_eq!(
            normalize_chat_completions_url("http://localhost:3000/v1/chat/completions"),
            "http://localhost:3000/v1/chat/completions"
        );
    }

    #[test]
    fn normalize_v1_base() {
        assert_eq!(
            normalize_chat_completions_url("http://localhost:3000/v1"),
            "http://localhost:3000/v1/chat/completions"
        );
    }

    #[test]
    fn normalize_server_base() {
        assert_eq!(
            normalize_chat_completions_url("http://localhost:3000"),
            "http://localhost:3000/v1/chat/completions"
        );
    }

    #[test]
    fn parse_args_rejects_non_object() {
        assert!(parse_args_string("[]").is_empty());
    }

    #[test]
    fn parse_args_accepts_object() {
        let args = parse_args_string(r#"{"q":"rust"}"#);
        assert_eq!(args.get("q"), Some(&json!("rust")));
    }
}
