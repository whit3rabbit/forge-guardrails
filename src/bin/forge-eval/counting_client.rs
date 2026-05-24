use std::sync::atomic::{AtomicI32, Ordering};

use forge_guardrails::clients::base::LLMCallInfo;
use forge_guardrails::{ApiFormat, ChunkStream, LLMClient, LLMResponse, SamplingParams, ToolSpec};
use serde_json::Value;

pub(crate) struct CountingClient<C> {
    inner: C,
    calls: AtomicI32,
}

impl<C> CountingClient<C> {
    pub(crate) fn new(inner: C) -> Self {
        Self {
            inner,
            calls: AtomicI32::new(0),
        }
    }

    pub(crate) fn calls(&self) -> i32 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl<C: LLMClient> LLMClient for CountingClient<C> {
    fn api_format(&self) -> ApiFormat {
        self.inner.api_format()
    }

    fn last_usage(&self) -> Option<forge_guardrails::TokenUsage> {
        self.inner.last_usage()
    }

    fn last_call_info(&self) -> Option<LLMCallInfo> {
        self.inner.last_call_info()
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.send(messages, tools, sampling).await
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.send_stream(messages, tools, sampling).await
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        self.inner.get_context_length().await
    }
}
