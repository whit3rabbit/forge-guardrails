use forge_guardrails::{
    AnthropicClient, AnyLlmProxyClient, AnyLlmRuntimeClient, ApiFormat, BackendError, ChunkStream,
    ContextDiscoveryError, LLMCallInfo, LLMClient, LLMRequestOptions, LLMResponse,
    LLMResponseEnvelope, LlamafileClient, OllamaClient, SamplingParams, StreamError, TokenUsage,
    ToolSpec,
};
use serde_json::Value;

pub(crate) enum ClientFactory {
    Runtime(AnyLlmRuntimeClient),
    DirectOpenAi {
        base_url: String,
        api_key: Option<String>,
        http_client: reqwest::Client,
        context_tokens: i64,
    },
    DirectAnthropic {
        base_url: String,
        api_key: Option<String>,
        http_client: reqwest::Client,
        context_tokens: i64,
    },
    DirectLlamafile {
        base_url: String,
        mode: String,
        http_client: reqwest::Client,
        context_tokens: i64,
    },
    ManagedLlamafile {
        gguf_path: String,
        base_url: String,
        mode: String,
        http_client: reqwest::Client,
    },
    ManagedOllama {
        model: String,
        http_client: reqwest::Client,
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
                http_client,
                context_tokens,
            } => {
                let mut client = AnyLlmProxyClient::new(model)
                    .with_base_url(base_url)
                    .with_http_client(http_client.clone())
                    .with_context_length(*context_tokens);
                if let Some(api_key) = api_key {
                    client = client.with_api_key(api_key.clone());
                }
                RoutedClient::DirectOpenAi(client)
            }
            Self::DirectAnthropic {
                base_url,
                api_key,
                http_client,
                context_tokens,
            } => RoutedClient::DirectAnthropic(
                AnthropicClient::new(model, api_key.clone())
                    .with_base_url(base_url)
                    .with_http_client(http_client.clone()),
                *context_tokens,
            ),
            Self::DirectLlamafile {
                base_url,
                mode,
                http_client,
                context_tokens,
            } => RoutedClient::DirectLlamafile(
                LlamafileClient::new(model)
                    .with_base_url(base_url)
                    .with_mode(mode)
                    .with_http_client(http_client.clone()),
                *context_tokens,
            ),
            Self::ManagedLlamafile {
                gguf_path,
                base_url,
                mode,
                http_client,
            } => RoutedClient::ManagedLlamafile(
                LlamafileClient::new(gguf_path)
                    .with_base_url(base_url)
                    .with_mode(mode)
                    .with_http_client(http_client.clone()),
            ),
            Self::ManagedOllama {
                model,
                http_client,
                context_tokens,
            } => {
                let client = OllamaClient::new(model.clone()).with_http_client(http_client.clone());
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

    async fn send_envelope_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponseEnvelope, BackendError> {
        match self {
            Self::Runtime(client) => {
                client
                    .send_envelope_with_options(messages, tools, options)
                    .await
            }
            Self::DirectOpenAi(client) => {
                client
                    .send_envelope_with_options(messages, tools, options)
                    .await
            }
            Self::DirectAnthropic(client, _) => {
                client
                    .send_envelope_with_options(messages, tools, options)
                    .await
            }
            Self::DirectLlamafile(client, _) => {
                client
                    .send_envelope_with_options(messages, tools, options)
                    .await
            }
            Self::ManagedLlamafile(client) => {
                client
                    .send_envelope_with_options(messages, tools, options)
                    .await
            }
            Self::ManagedOllama(client) => {
                client
                    .send_envelope_with_options(messages, tools, options)
                    .await
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn direct_openai_clients_share_transport_but_keep_usage_isolated() {
        let mut upstream = mockito::Server::new_async().await;
        let _mock = upstream
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "request-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let factory = ClientFactory::DirectOpenAi {
            base_url: upstream.url(),
            api_key: None,
            http_client: reqwest::Client::new(),
            context_tokens: 8192,
        };

        let first = factory.client_for_model("first-model".to_string());
        let second = factory.client_for_model("second-model".to_string());
        first
            .send_with_options(
                vec![json!({"role": "user", "content": "hello"})],
                None,
                LLMRequestOptions::default(),
            )
            .await
            .expect("request");

        assert_eq!(
            first.last_usage(),
            Some(TokenUsage::new(2, 3, 5)),
            "requesting client records usage"
        );
        assert_eq!(
            second.last_usage(),
            None,
            "routed clients keep usage isolated"
        );
    }
}
