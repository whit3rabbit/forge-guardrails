use forge_guardrails::{
    AnthropicClient, AnyLlmProxyClient, AnyLlmRuntimeClient, ApiFormat, BackendError, ChunkStream,
    ContextDiscoveryError, LLMCallInfo, LLMClient, LLMRequestOptions, LLMResponse, LlamafileClient,
    OllamaClient, SamplingParams, StreamError, TokenUsage, ToolSpec,
};
use serde_json::Value;

pub(crate) enum ClientFactory {
    Runtime(AnyLlmRuntimeClient),
    DirectOpenAi {
        base_url: String,
        api_key: Option<String>,
        context_tokens: i64,
    },
    DirectAnthropic {
        base_url: String,
        api_key: Option<String>,
        context_tokens: i64,
    },
    DirectLlamafile {
        base_url: String,
        mode: String,
        context_tokens: i64,
    },
    ManagedLlamafile {
        gguf_path: String,
        base_url: String,
        mode: String,
    },
    ManagedOllama {
        model: String,
        context_tokens: i64,
    },
}

pub(crate) enum RoutedClient {
    Runtime(AnyLlmRuntimeClient),
    DirectOpenAi(AnyLlmProxyClient),
    DirectAnthropic(AnthropicClient, i64),
    DirectLlamafile(LlamafileClient, i64),
    ManagedLlamafile(LlamafileClient),
    ManagedOllama(OllamaClient),
}

impl ClientFactory {
    pub(crate) fn client_for_model(&self, model: String) -> RoutedClient {
        match self {
            Self::Runtime(client) => RoutedClient::Runtime(client.for_model(model)),
            Self::DirectOpenAi {
                base_url,
                api_key,
                context_tokens,
            } => {
                let mut client = AnyLlmProxyClient::new(model)
                    .with_base_url(base_url)
                    .with_context_length(*context_tokens);
                if let Some(api_key) = api_key {
                    client = client.with_api_key(api_key.clone());
                }
                RoutedClient::DirectOpenAi(client)
            }
            Self::DirectAnthropic {
                base_url,
                api_key,
                context_tokens,
            } => RoutedClient::DirectAnthropic(
                AnthropicClient::new(model, api_key.clone()).with_base_url(base_url),
                *context_tokens,
            ),
            Self::DirectLlamafile {
                base_url,
                mode,
                context_tokens,
            } => RoutedClient::DirectLlamafile(
                LlamafileClient::new(model)
                    .with_base_url(base_url)
                    .with_mode(mode),
                *context_tokens,
            ),
            Self::ManagedLlamafile {
                gguf_path,
                base_url,
                mode,
            } => RoutedClient::ManagedLlamafile(
                LlamafileClient::new(gguf_path)
                    .with_base_url(base_url)
                    .with_mode(mode),
            ),
            Self::ManagedOllama {
                model,
                context_tokens,
            } => {
                let client = OllamaClient::new(model.clone());
                client.set_num_ctx(Some(*context_tokens));
                RoutedClient::ManagedOllama(client)
            }
        }
    }
}

impl LLMClient for RoutedClient {
    fn api_format(&self) -> ApiFormat {
        match self {
            Self::Runtime(client) => client.api_format(),
            Self::DirectOpenAi(client) => client.api_format(),
            Self::DirectAnthropic(client, _) => client.api_format(),
            Self::DirectLlamafile(client, _) => client.api_format(),
            Self::ManagedLlamafile(client) => client.api_format(),
            Self::ManagedOllama(client) => client.api_format(),
        }
    }

    fn last_usage(&self) -> Option<TokenUsage> {
        match self {
            Self::Runtime(client) => client.last_usage(),
            Self::DirectOpenAi(client) => client.last_usage(),
            Self::DirectAnthropic(client, _) => client.last_usage(),
            Self::DirectLlamafile(client, _) => client.last_usage(),
            Self::ManagedLlamafile(client) => client.last_usage(),
            Self::ManagedOllama(client) => client.last_usage(),
        }
    }

    fn last_call_info(&self) -> Option<LLMCallInfo> {
        match self {
            Self::Runtime(client) => client.last_call_info(),
            Self::DirectOpenAi(client) => client.last_call_info(),
            Self::DirectAnthropic(client, _) => client.last_call_info(),
            Self::DirectLlamafile(client, _) => client.last_call_info(),
            Self::ManagedLlamafile(client) => client.last_call_info(),
            Self::ManagedOllama(client) => client.last_call_info(),
        }
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        match self {
            Self::Runtime(client) => client.send(messages, tools, sampling).await,
            Self::DirectOpenAi(client) => client.send(messages, tools, sampling).await,
            Self::DirectAnthropic(client, _) => client.send(messages, tools, sampling).await,
            Self::DirectLlamafile(client, _) => client.send(messages, tools, sampling).await,
            Self::ManagedLlamafile(client) => client.send(messages, tools, sampling).await,
            Self::ManagedOllama(client) => client.send(messages, tools, sampling).await,
        }
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, BackendError> {
        match self {
            Self::Runtime(client) => client.send_with_options(messages, tools, options).await,
            Self::DirectOpenAi(client) => client.send_with_options(messages, tools, options).await,
            Self::DirectAnthropic(client, _) => {
                client.send_with_options(messages, tools, options).await
            }
            Self::DirectLlamafile(client, _) => {
                client.send_with_options(messages, tools, options).await
            }
            Self::ManagedLlamafile(client) => {
                client.send_with_options(messages, tools, options).await
            }
            Self::ManagedOllama(client) => client.send_with_options(messages, tools, options).await,
        }
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        match self {
            Self::Runtime(client) => client.send_stream(messages, tools, sampling).await,
            Self::DirectOpenAi(client) => client.send_stream(messages, tools, sampling).await,
            Self::DirectAnthropic(client, _) => client.send_stream(messages, tools, sampling).await,
            Self::DirectLlamafile(client, _) => client.send_stream(messages, tools, sampling).await,
            Self::ManagedLlamafile(client) => client.send_stream(messages, tools, sampling).await,
            Self::ManagedOllama(client) => client.send_stream(messages, tools, sampling).await,
        }
    }

    async fn send_stream_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<ChunkStream, StreamError> {
        match self {
            Self::Runtime(client) => {
                client
                    .send_stream_with_options(messages, tools, options)
                    .await
            }
            Self::DirectOpenAi(client) => {
                client
                    .send_stream_with_options(messages, tools, options)
                    .await
            }
            Self::DirectAnthropic(client, _) => {
                client
                    .send_stream_with_options(messages, tools, options)
                    .await
            }
            Self::DirectLlamafile(client, _) => {
                client
                    .send_stream_with_options(messages, tools, options)
                    .await
            }
            Self::ManagedLlamafile(client) => {
                client
                    .send_stream_with_options(messages, tools, options)
                    .await
            }
            Self::ManagedOllama(client) => {
                client
                    .send_stream_with_options(messages, tools, options)
                    .await
            }
        }
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        match self {
            Self::Runtime(client) => client.get_context_length().await,
            Self::DirectOpenAi(client) => client.get_context_length().await,
            Self::DirectAnthropic(_, context_tokens) => Ok(Some(*context_tokens)),
            Self::DirectLlamafile(_, context_tokens) => Ok(Some(*context_tokens)),
            Self::ManagedLlamafile(client) => client.get_context_length().await,
            Self::ManagedOllama(client) => client.get_context_length().await,
        }
    }
}
