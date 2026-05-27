use std::sync::{Arc, Mutex, RwLock};

use ::anyllm_proxy::runtime::{ChatCompletionRuntime, ChatCompletionService};
use serde_json::Value;

use super::call_info::{record_call_info_cell, runtime_call_info};
use super::request::build_openai_request_body;
use super::response::parse_openai_response;
use super::streaming::parse_openai_chunks;
use super::usage::{record_usage_cell, record_usage_details_cell, usage_details_from_openai_usage};
use crate::clients::base::{
    ApiFormat, ChunkStream, LLMCallInfo, LLMClient, LLMRequestOptions, LLMResponse,
    LLMUsageDetails, SamplingParams, TokenUsage,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

/// LLM client that dispatches guarded OpenAI-format calls through
/// `::anyllm_proxy::runtime` without embedding anyllm's HTTP router.
pub struct AnyLlmRuntimeClient {
    model: String,
    service: Arc<dyn ChatCompletionService>,
    context_length: Option<i64>,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_usage_details: Arc<Mutex<Option<LLMUsageDetails>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
}

impl AnyLlmRuntimeClient {
    /// Creates a new `AnyLlmRuntimeClient` with the given model and underlying chat completion service.
    pub fn new(model: impl Into<String>, service: Arc<dyn ChatCompletionService>) -> Self {
        Self {
            model: model.into(),
            service,
            context_length: None,
            last_usage: Arc::new(Mutex::new(None)),
            last_usage_details: Arc::new(Mutex::new(None)),
            last_call_info: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates an `AnyLlmRuntimeClient` directly from an existing runtime.
    pub fn from_runtime(model: impl Into<String>, runtime: ChatCompletionRuntime) -> Self {
        Self::new(model, Arc::new(runtime))
    }

    /// Creates an `AnyLlmRuntimeClient` from a configuration object.
    pub fn from_config(model: impl Into<String>, config: ::anyllm_proxy::config::Config) -> Self {
        Self::from_runtime(model, ChatCompletionRuntime::from_config(config))
    }

    /// Creates an `AnyLlmRuntimeClient` from a multi-provider config.
    pub fn from_multi_config(
        model: impl Into<String>,
        config: ::anyllm_proxy::config::MultiConfig,
    ) -> Self {
        Self::from_runtime(model, ChatCompletionRuntime::from_multi_config(config))
    }

    /// Creates an `AnyLlmRuntimeClient` from a multi-provider config and custom model router.
    pub fn from_multi_config_with_model_router(
        model: impl Into<String>,
        config: ::anyllm_proxy::config::MultiConfig,
        model_router: Option<Arc<RwLock<::anyllm_proxy::config::model_router::ModelRouter>>>,
    ) -> Self {
        Self::from_runtime(
            model,
            ChatCompletionRuntime::from_multi_config_with_model_router(config, model_router),
        )
    }

    /// Sets the context token length.
    pub fn with_context_length(mut self, tokens: i64) -> Self {
        self.context_length = Some(tokens);
        self
    }

    /// Reuse the same anyllm runtime service with a different requested model.
    ///
    /// Usage and call metadata are intentionally fresh per clone so request
    /// observers do not read state from a previous model route.
    pub fn for_model(&self, model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            service: self.service.clone(),
            context_length: self.context_length,
            last_usage: Arc::new(Mutex::new(None)),
            last_usage_details: Arc::new(Mutex::new(None)),
            last_call_info: Arc::new(Mutex::new(None)),
        }
    }

    fn build_request(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
        stream: bool,
    ) -> Result<anyllm_translate::openai::ChatCompletionRequest, BackendError> {
        let body = build_openai_request_body(&self.model, messages, tools, options, stream);
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
        let req = self.build_request(messages, tools, options, false)?;
        let result = self
            .service
            .complete(req)
            .await
            .map_err(runtime_error_to_backend_error)?;
        let usage = result.usage.as_ref().or(result.response.usage.as_ref());
        record_usage_cell(&self.last_usage, usage);
        record_usage_details_cell(
            &self.last_usage_details,
            usage_details_from_openai_usage(usage),
        );
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
        self.send_stream_with_options(messages, tools, LLMRequestOptions::from_sampling(sampling))
            .await
    }

    async fn send_stream_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<ChunkStream, StreamError> {
        let req = self
            .build_request(messages, tools, options, true)
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
            self.last_usage_details.clone(),
            self.last_call_info.clone(),
            Some(cost_model),
        )))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(self.context_length)
    }
}

fn runtime_error_to_backend_error(
    error: ::anyllm_proxy::runtime::ChatCompletionError,
) -> BackendError {
    BackendError::new(error.status_code() as i64, error.to_string())
}
