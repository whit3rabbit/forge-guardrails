use std::sync::atomic::{AtomicI32, Ordering};

use forge_guardrails::clients::base::LLMCallInfo;
use forge_guardrails::{
    ApiFormat, ChunkStream, LLMClient, LLMRequestOptions, LLMResponse, LLMResponseEnvelope,
    SamplingParams, ToolSpec,
};
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

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.send_with_options(messages, tools, options).await
    }

    async fn send_envelope_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponseEnvelope, forge_guardrails::BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner
            .send_envelope_with_options(messages, tools, options)
            .await
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

    async fn send_stream_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner
            .send_stream_with_options(messages, tools, options)
            .await
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        self.inner.get_context_length().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use forge_guardrails::{
        BackendError, ChunkType, ContextDiscoveryError, StreamChunk, StreamError, TextResponse,
    };
    use futures_util::{stream, StreamExt};
    use serde_json::json;

    use super::*;

    #[derive(Default)]
    struct FakeClient {
        send_options: Mutex<Option<LLMRequestOptions>>,
        stream_options: Mutex<Option<LLMRequestOptions>>,
    }

    impl LLMClient for FakeClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }

        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, BackendError> {
            Ok(LLMResponse::Text(TextResponse::new("ok")))
        }

        async fn send_with_options(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            options: LLMRequestOptions,
        ) -> Result<LLMResponse, BackendError> {
            *self.send_options.lock().unwrap() = Some(options);
            Ok(LLMResponse::Text(TextResponse::new("ok")))
        }

        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, StreamError> {
            let final_chunk = StreamChunk::new(ChunkType::Final)
                .with_response(LLMResponse::Text(TextResponse::new("ok")));
            Ok(Box::pin(stream::iter(vec![Ok(final_chunk)])))
        }

        async fn send_stream_with_options(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            options: LLMRequestOptions,
        ) -> Result<ChunkStream, StreamError> {
            *self.stream_options.lock().unwrap() = Some(options);
            self.send_stream(Vec::new(), None, None).await
        }

        async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
            Ok(None)
        }
    }

    #[tokio::test]
    async fn send_with_options_counts_once_and_forwards_options() {
        let client = CountingClient::new(FakeClient::default());
        let options = LLMRequestOptions {
            passthrough: Some(serde_json::Map::from_iter([(
                "user".to_string(),
                json!("eval"),
            )])),
            ..LLMRequestOptions::default()
        };

        client
            .send_with_options(Vec::new(), None, options.clone())
            .await
            .expect("send should succeed");

        assert_eq!(client.calls(), 1);
        assert_eq!(*client.inner.send_options.lock().unwrap(), Some(options));
    }

    #[tokio::test]
    async fn send_stream_with_options_counts_once_and_forwards_options() {
        let client = CountingClient::new(FakeClient::default());
        let options = LLMRequestOptions {
            passthrough: Some(serde_json::Map::from_iter([(
                "user".to_string(),
                json!("eval"),
            )])),
            ..LLMRequestOptions::default()
        };

        let mut stream = client
            .send_stream_with_options(Vec::new(), None, options.clone())
            .await
            .expect("stream should start");
        assert!(stream.next().await.expect("final chunk").is_ok());

        assert_eq!(client.calls(), 1);
        assert_eq!(*client.inner.stream_options.lock().unwrap(), Some(options));
    }
}
