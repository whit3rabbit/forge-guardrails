//! Client adapters for anyllm provider routing.
//!
//! This keeps forge-guardrails responsible for interception, validation, and
//! nudging while delegating provider routing and upstream compatibility to
//! anyllm through either its in-process runtime or a running sidecar.

use std::sync::{Arc, Mutex, RwLock};

use anyllm_proxy::backend::RateLimitHeaders as AnyLlmRateLimitHeaders;
use anyllm_proxy::runtime::{ChatCompletionRuntime, ChatCompletionService};
use futures_util::StreamExt;
use indexmap::IndexMap;
use reqwest::header::HeaderMap;
use serde_json::{json, Value};

use crate::clients::base::{
    format_tool, ApiFormat, ChunkStream, ChunkType, LLMCallInfo, LLMClient, LLMRateLimitInfo,
    LLMResponse, SamplingParams, StreamChunk, TextResponse, TokenUsage, ToolCall,
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
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
}

impl AnyLlmProxyClient {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            chat_completions_url: DEFAULT_ANYLLM_PROXY_URL.to_string(),
            model: model.into(),
            api_key: None,
            timeout_secs: 300.0,
            context_length: None,
            last_usage: Arc::new(Mutex::new(None)),
            last_call_info: Arc::new(Mutex::new(None)),
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
        build_openai_request_body(&self.model, messages, tools, sampling, stream)
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
        record_usage_cell(&self.last_usage, usage);
    }

    fn record_call_info(&self, info: LLMCallInfo) {
        record_call_info_cell(&self.last_call_info, info);
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

    fn last_call_info(&self) -> Option<LLMCallInfo> {
        self.last_call_info
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
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
        let headers = resp.headers().clone();
        let response_json = resp
            .json::<anyllm_translate::openai::ChatCompletionResponse>()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;
        self.record_call_info(sidecar_call_info(
            &self.model,
            &headers,
            Some(response_json.model.clone()),
            response_json.usage.as_ref(),
        ));
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
        self.record_call_info(sidecar_call_info(&self.model, resp.headers(), None, None));
        Ok(Box::pin(parse_openai_sse(
            resp,
            self.last_usage.clone(),
            self.last_call_info.clone(),
            Some(self.model.clone()),
        )))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(self.context_length)
    }
}

/// LLM client that dispatches guarded OpenAI-format calls through
/// `anyllm_proxy::runtime` without embedding anyllm's HTTP router.
pub struct AnyLlmRuntimeClient {
    model: String,
    service: Arc<dyn ChatCompletionService>,
    context_length: Option<i64>,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
}

impl AnyLlmRuntimeClient {
    pub fn new(model: impl Into<String>, service: Arc<dyn ChatCompletionService>) -> Self {
        Self {
            model: model.into(),
            service,
            context_length: None,
            last_usage: Arc::new(Mutex::new(None)),
            last_call_info: Arc::new(Mutex::new(None)),
        }
    }

    pub fn from_runtime(model: impl Into<String>, runtime: ChatCompletionRuntime) -> Self {
        Self::new(model, Arc::new(runtime))
    }

    pub fn from_config(model: impl Into<String>, config: anyllm_proxy::config::Config) -> Self {
        Self::from_runtime(model, ChatCompletionRuntime::from_config(config))
    }

    pub fn from_multi_config(
        model: impl Into<String>,
        config: anyllm_proxy::config::MultiConfig,
    ) -> Self {
        Self::from_runtime(model, ChatCompletionRuntime::from_multi_config(config))
    }

    pub fn from_multi_config_with_model_router(
        model: impl Into<String>,
        config: anyllm_proxy::config::MultiConfig,
        model_router: Option<Arc<RwLock<anyllm_proxy::config::model_router::ModelRouter>>>,
    ) -> Self {
        Self::from_runtime(
            model,
            ChatCompletionRuntime::from_multi_config_with_model_router(config, model_router),
        )
    }

    pub fn with_context_length(mut self, tokens: i64) -> Self {
        self.context_length = Some(tokens);
        self
    }

    fn build_request(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
        stream: bool,
    ) -> Result<anyllm_translate::openai::ChatCompletionRequest, BackendError> {
        let body = build_openai_request_body(&self.model, messages, tools, sampling, stream);
        serde_json::from_value(body).map_err(|e| BackendError::new(400, e.to_string()))
    }
}

impl LLMClient for AnyLlmRuntimeClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<TokenUsage> {
        self.last_usage.lock().ok().and_then(|guard| guard.clone())
    }

    fn last_call_info(&self) -> Option<LLMCallInfo> {
        self.last_call_info
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        let req = self.build_request(messages, tools, sampling, false)?;
        let result = self
            .service
            .complete(req)
            .await
            .map_err(runtime_error_to_backend_error)?;
        let usage = result.usage.as_ref().or(result.response.usage.as_ref());
        record_usage_cell(&self.last_usage, usage);
        record_call_info_cell(
            &self.last_call_info,
            runtime_call_info(
                &result.metadata,
                &result.rate_limits,
                &result.warnings,
                Some(result.response.model.clone()),
                usage,
            ),
        );
        Ok(parse_openai_response(result.response))
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        let req = self
            .build_request(messages, tools, sampling, true)
            .map_err(|e| StreamError::new(e.to_string()))?;
        let result = self
            .service
            .complete_stream(req)
            .await
            .map_err(|e| StreamError::new(runtime_error_to_backend_error(e).to_string()))?;
        let cost_model = result.metadata.mapped_model.clone();
        record_call_info_cell(
            &self.last_call_info,
            runtime_call_info(
                &result.metadata,
                &result.rate_limits,
                &result.warnings,
                None,
                None,
            ),
        );
        Ok(Box::pin(parse_openai_chunks(
            result.chunks,
            self.last_usage.clone(),
            self.last_call_info.clone(),
            Some(cost_model),
        )))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(self.context_length)
    }
}

fn build_openai_request_body(
    model: &str,
    messages: Vec<Value>,
    tools: Option<Vec<ToolSpec>>,
    sampling: Option<SamplingParams>,
    stream: bool,
) -> Value {
    let mut body = json!({
        "model": model,
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

fn record_usage_cell(
    cell: &Arc<Mutex<Option<TokenUsage>>>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) {
    let token_usage = usage
        .map(|u| {
            TokenUsage::new(
                u.prompt_tokens as i64,
                u.completion_tokens as i64,
                u.total_tokens as i64,
            )
        })
        .unwrap_or_else(TokenUsage::empty);
    if let Ok(mut guard) = cell.lock() {
        *guard = Some(token_usage);
    }
}

fn record_call_info_cell(cell: &Arc<Mutex<Option<LLMCallInfo>>>, info: LLMCallInfo) {
    if let Ok(mut guard) = cell.lock() {
        *guard = Some(info);
    }
}

fn runtime_call_info(
    metadata: &anyllm_proxy::runtime::ChatCompletionMetadata,
    rate_limits: &AnyLlmRateLimitHeaders,
    warnings: &anyllm_translate::TranslationWarnings,
    response_model: Option<String>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> LLMCallInfo {
    LLMCallInfo {
        requested_model: Some(metadata.requested_model.clone()),
        response_model,
        selected_backend: Some(metadata.selected_backend.clone()),
        mapped_model: Some(metadata.mapped_model.clone()),
        backend_kind: Some(format!("{:?}", metadata.backend_kind)),
        provider_id: metadata.provider_id.clone(),
        used_responses_api: metadata.used_responses_api,
        degradation_warnings: warnings.as_header_value(),
        cache_status: None,
        rate_limits: rate_limit_info_from_anyllm(rate_limits),
        estimated_cost_usd: estimate_cost_usd(Some(&metadata.mapped_model), usage),
    }
}

fn sidecar_call_info(
    requested_model: &str,
    headers: &HeaderMap,
    response_model: Option<String>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> LLMCallInfo {
    let header_cost =
        header_value(headers, "x-anyllm-cost-usd").and_then(|v| v.parse::<f64>().ok());
    let cost_model = response_model.as_deref().or(Some(requested_model));
    let estimated_cost_usd = header_cost.or_else(|| estimate_cost_usd(cost_model, usage));
    LLMCallInfo {
        requested_model: Some(requested_model.to_string()),
        response_model,
        selected_backend: None,
        mapped_model: None,
        backend_kind: None,
        provider_id: None,
        used_responses_api: false,
        degradation_warnings: header_value(headers, "x-anyllm-degradation"),
        cache_status: header_value(headers, "x-anyllm-cache"),
        rate_limits: rate_limit_info_from_sidecar(headers),
        estimated_cost_usd,
    }
}

fn rate_limit_info_from_anyllm(rate_limits: &AnyLlmRateLimitHeaders) -> LLMRateLimitInfo {
    LLMRateLimitInfo {
        requests_limit: rate_limits.requests_limit.clone(),
        requests_remaining: rate_limits.requests_remaining.clone(),
        requests_reset: rate_limits.requests_reset.clone(),
        tokens_limit: rate_limits.tokens_limit.clone(),
        tokens_remaining: rate_limits.tokens_remaining.clone(),
        tokens_reset: rate_limits.tokens_reset.clone(),
        retry_after: rate_limits.retry_after.clone(),
        organization_id: rate_limits.organization_id.clone(),
    }
}

fn rate_limit_info_from_sidecar(headers: &HeaderMap) -> LLMRateLimitInfo {
    LLMRateLimitInfo {
        requests_limit: header_value(headers, "anthropic-ratelimit-requests-limit"),
        requests_remaining: header_value(headers, "anthropic-ratelimit-requests-remaining"),
        requests_reset: header_value(headers, "anthropic-ratelimit-requests-reset"),
        tokens_limit: header_value(headers, "anthropic-ratelimit-tokens-limit"),
        tokens_remaining: header_value(headers, "anthropic-ratelimit-tokens-remaining"),
        tokens_reset: header_value(headers, "anthropic-ratelimit-tokens-reset"),
        retry_after: header_value(headers, "retry-after"),
        organization_id: header_value(headers, "anthropic-organization-id"),
    }
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn estimate_cost_usd(
    model: Option<&str>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> Option<f64> {
    let model = model?;
    let usage = usage?;
    let (input_cost, output_cost) = anyllm_proxy::cost::pricing().price_for_model(model)?;
    Some(
        input_cost * f64::from(usage.prompt_tokens)
            + output_cost * f64::from(usage.completion_tokens),
    )
}

fn observe_stream_call_info(
    cell: &Arc<Mutex<Option<LLMCallInfo>>>,
    response_model: &str,
    cost_model: &str,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) {
    if let Ok(mut guard) = cell.lock() {
        let info = guard.get_or_insert_with(LLMCallInfo::default);
        if info.response_model.is_none() {
            info.response_model = Some(response_model.to_string());
        }
        if info.estimated_cost_usd.is_none() {
            info.estimated_cost_usd = estimate_cost_usd(Some(cost_model), usage);
        }
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
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
    default_cost_model: Option<String>,
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

                record_usage_cell(&last_usage, evt.usage.as_ref());
                let cost_model = default_cost_model.as_deref().unwrap_or(&evt.model);
                observe_stream_call_info(
                    &last_call_info,
                    &evt.model,
                    cost_model,
                    evt.usage.as_ref(),
                );

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

fn parse_openai_chunks(
    chunks: anyllm_proxy::runtime::ChatCompletionChunkStream,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
    cost_model: Option<String>,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    async_stream::stream! {
        let mut inner = chunks;
        let mut accumulated_text = String::new();
        let mut accumulated_reasoning = String::new();
        let mut accumulated_tools: Vec<(String, String, String)> = Vec::new();

        while let Some(chunk) = inner.next().await {
            let evt = match chunk {
                Ok(evt) => evt,
                Err(e) => {
                    yield Err(StreamError::new(e.to_string()));
                    return;
                }
            };

            record_usage_cell(&last_usage, evt.usage.as_ref());
            let pricing_model = cost_model.as_deref().unwrap_or(&evt.model);
            observe_stream_call_info(
                &last_call_info,
                &evt.model,
                pricing_model,
                evt.usage.as_ref(),
            );

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

fn runtime_error_to_backend_error(
    error: anyllm_proxy::runtime::ChatCompletionError,
) -> BackendError {
    BackendError::new(error.status_code() as i64, error.to_string())
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
