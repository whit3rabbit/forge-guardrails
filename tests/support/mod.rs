use forge_guardrails::{
    ApiFormat, ChunkStream, ChunkType, LLMClient, LLMResponse, SamplingParams, StreamChunk,
    ToolSpec,
};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    FinalOnly,
    Unsupported,
}

pub struct ScriptedLlmClient {
    responses: Vec<LLMResponse>,
    calls: AtomicUsize,
    api_format: ApiFormat,
    context_length: Option<i64>,
    stream_mode: StreamMode,
}

impl ScriptedLlmClient {
    pub fn new(responses: Vec<LLMResponse>) -> Self {
        Self {
            responses,
            calls: AtomicUsize::new(0),
            api_format: ApiFormat::OpenAI,
            context_length: Some(4096),
            stream_mode: StreamMode::FinalOnly,
        }
    }

    #[allow(dead_code)]
    pub fn from_texts(responses: Vec<&str>) -> Self {
        Self::new(
            responses
                .into_iter()
                .map(|content| LLMResponse::Text(forge_guardrails::TextResponse::new(content)))
                .collect(),
        )
    }

    pub fn with_stream_mode(mut self, stream_mode: StreamMode) -> Self {
        self.stream_mode = stream_mode;
        self
    }

    #[allow(dead_code)]
    pub fn with_api_format(mut self, api_format: ApiFormat) -> Self {
        self.api_format = api_format;
        self
    }

    #[allow(dead_code)]
    pub fn with_context_length(mut self, context_length: Option<i64>) -> Self {
        self.context_length = context_length;
        self
    }

    pub fn calls(&self) -> i32 {
        self.calls.load(Ordering::SeqCst) as i32
    }

    fn next_response(&self) -> LLMResponse {
        let idx = self.calls.fetch_add(1, Ordering::SeqCst);
        self.responses
            .get(idx)
            .cloned()
            .unwrap_or_else(|| panic!("ScriptedLlmClient exhausted at call {idx}"))
    }
}

impl LLMClient for ScriptedLlmClient {
    fn api_format(&self) -> ApiFormat {
        self.api_format
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        Ok(self.next_response())
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        if self.stream_mode == StreamMode::Unsupported {
            return Err(forge_guardrails::StreamError::new(
                "streaming is not supported by ScriptedLlmClient",
            ));
        }
        let response = self.next_response();
        Ok(Box::pin(futures_util::stream::iter(vec![Ok(
            StreamChunk::new(ChunkType::Final).with_response(response),
        )])))
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        Ok(self.context_length)
    }
}
