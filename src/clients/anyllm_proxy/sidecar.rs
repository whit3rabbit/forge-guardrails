use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::call_info::{record_call_info_cell, sidecar_call_info};
use super::request::{build_openai_request_body, normalize_chat_completions_url};
use super::response::parse_openai_response;
use super::streaming::parse_openai_sse;
use super::usage::{
    record_usage_cell, record_usage_details_cell, token_usage_from_openai_usage,
    usage_details_from_openai_usage_value,
};
use super::DEFAULT_ANYLLM_PROXY_URL;
use crate::clients::base::{
    ApiFormat, ChunkStream, LLMCallInfo, LLMClient, LLMRequestOptions, LLMResponse,
    LLMResponseEnvelope, LLMUsageDetails, SamplingParams, TokenUsage,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

/// LLM client that forwards guarded OpenAI-format calls to anyllm_proxy.
pub struct AnyLlmProxyClient {
    chat_completions_url: String,
    model: String,
    api_key: Option<String>,
    http_client: reqwest::Client,
    timeout_secs: f64,
    context_length: Option<i64>,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_usage_details: Arc<Mutex<Option<LLMUsageDetails>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
}

impl AnyLlmProxyClient {
    /// Creates a new `AnyLlmProxyClient` for the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            chat_completions_url: DEFAULT_ANYLLM_PROXY_URL.to_string(),
            model: model.into(),
            api_key: None,
            http_client: reqwest::Client::new(),
            timeout_secs: 300.0,
            context_length: None,
            last_usage: Arc::new(Mutex::new(None)),
            last_usage_details: Arc::new(Mutex::new(None)),
            last_call_info: Arc::new(Mutex::new(None)),
        }
    }

    /// Sets the base URL for the sidecar proxy.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.chat_completions_url = normalize_chat_completions_url(&url.into());
        self
    }

    /// Sets the API key used for authenticating with the proxy.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the shared HTTP client used for upstream requests.
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.http_client = client;
        self
    }

    /// Sets the context token length.
    pub fn with_context_length(mut self, tokens: i64) -> Self {
        self.context_length = Some(tokens);
        self
    }

    /// Sets the request timeout in seconds.
    pub fn with_timeout(mut self, timeout_secs: f64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    fn build_request_body(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
        stream: bool,
    ) -> Value {
        build_openai_request_body(&self.model, messages, tools, options, stream)
    }

    async fn send_request(&self, body: Value) -> Result<reqwest::Response, BackendError> {
        let mut req = self
            .http_client
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

    fn last_usage_details(&self) -> Option<LLMUsageDetails> {
        self.last_usage_details
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
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
        self.send_with_options(messages, tools, LLMRequestOptions::from_sampling(sampling))
            .await
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, BackendError> {
        Ok(self
            .send_envelope_with_options(messages, tools, options)
            .await?
            .response)
    }

    async fn send_envelope_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponseEnvelope, BackendError> {
        let body = self.build_request_body(messages, tools, options, false);
        let resp = self.send_request(body).await?;
        let status = resp.status().as_u16() as i64;
        let headers = resp.headers().clone();
        let response_value = resp
            .json::<Value>()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;
        let usage_details = usage_details_from_openai_usage_value(response_value.get("usage"));
        record_usage_details_cell(&self.last_usage_details, usage_details.clone());
        let response_json = serde_json::from_value::<
            anyllm_translate::openai::ChatCompletionResponse,
        >(response_value)
        .map_err(|e| BackendError::new(status, e.to_string()))?;
        let usage = token_usage_from_openai_usage(response_json.usage.as_ref());
        let call_info = sidecar_call_info(
            &self.model,
            &headers,
            Some(response_json.model.clone()),
            response_json.usage.as_ref(),
        );
        self.record_call_info(call_info.clone());
        Ok(LLMResponseEnvelope {
            response: self.parse_response(response_json),
            usage: Some(usage),
            usage_details,
            call_info: Some(call_info),
        })
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        self.send_stream_with_options(messages, tools, LLMRequestOptions::from_sampling(sampling))
            .await
    }

    async fn send_stream_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<ChunkStream, StreamError> {
        let body = self.build_request_body(messages, tools, options, true);
        let resp = self
            .send_request(body)
            .await
            .map_err(|e| StreamError::new(e.to_string()))?;
        let call_info = sidecar_call_info(&self.model, resp.headers(), None, None);
        self.record_call_info(call_info.clone());
        Ok(Box::pin(parse_openai_sse(
            resp,
            self.last_usage.clone(),
            self.last_usage_details.clone(),
            self.last_call_info.clone(),
            Some(call_info),
            Some(self.model.clone()),
        )))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(self.context_length)
    }
}
