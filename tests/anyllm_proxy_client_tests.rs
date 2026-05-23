use anyllm_proxy::config::{
    BackendAuth, BackendConfig, BackendKind, ModelMapping, MultiConfig, OpenAIApiFormat, TlsConfig,
};
use forge_guardrails::{
    AnyLlmProxyClient, AnyLlmRuntimeClient, ChunkType, LLMClient, LLMResponse, SamplingParams,
    ToolSpec,
};
use futures_util::StreamExt;
use indexmap::IndexMap;
use serde_json::json;

fn search_spec() -> ToolSpec {
    ToolSpec::from_json_schema(
        "search",
        "Search",
        &json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }),
    )
    .expect("valid tool spec")
}

fn anyllm_backend_config(kind: BackendKind, base_url: String) -> BackendConfig {
    BackendConfig {
        kind,
        api_key: "test-key".to_string(),
        base_url,
        api_format: OpenAIApiFormat::Chat,
        model_mapping: ModelMapping {
            big_model: "gpt-4o".to_string(),
            small_model: "gpt-4o-mini".to_string(),
        },
        tls: TlsConfig::default(),
        backend_auth: BackendAuth::BearerToken("test-key".to_string()),
        log_bodies: false,
        omit_stream_options: false,
        stream_timeout_secs: 900,
        bedrock_credentials: None,
    }
}

fn anyllm_multi_config(backend: BackendConfig) -> MultiConfig {
    let mut backends = IndexMap::new();
    backends.insert("openai".to_string(), backend);
    MultiConfig {
        listen_port: 0,
        log_bodies: false,
        default_backend: "openai".to_string(),
        backends,
        expose_degradation_warnings: false,
    }
}

#[tokio::test]
async fn anyllm_proxy_client_sends_request_and_parses_text() {
    let mut server = mockito::Server::new_async().await;
    let mut sampling = SamplingParams::new();
    sampling.insert("temperature".into(), json!(0.2));
    sampling.insert("max_tokens".into(), json!(64));

    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::Json(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "temperature": 0.2,
            "max_tokens": 64,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }]
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = AnyLlmProxyClient::new("gpt-4o-mini")
        .with_base_url(server.url())
        .with_api_key("test-key");
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            Some(vec![search_spec()]),
            Some(sampling),
        )
        .await
        .expect("request succeeds");

    match response {
        LLMResponse::Text(text) => assert_eq!(text.content, "hi"),
        other => panic!("expected text response, got {other:?}"),
    }
    assert_eq!(client.last_usage().unwrap().total_tokens, 5);
}

#[tokio::test]
async fn anyllm_proxy_client_parses_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "{\"query\":\"rust\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = AnyLlmProxyClient::new("gpt-4o-mini").with_base_url(server.url());
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("request succeeds");

    match response {
        LLMResponse::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].id.as_deref(), Some("call_123"));
            assert_eq!(calls[0].tool, "search");
            assert_eq!(calls[0].args.get("query"), Some(&json!("rust")));
        }
        other => panic!("expected tool call response, got {other:?}"),
    }
}

#[tokio::test]
async fn anyllm_proxy_client_reports_upstream_error() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(429)
        .with_body("rate limited")
        .create_async()
        .await;

    let client = AnyLlmProxyClient::new("gpt-4o-mini").with_base_url(server.url());
    let err = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect_err("upstream error should fail");

    assert!(err.to_string().contains("status 429"));
    assert!(err.to_string().contains("rate limited"));
}

#[tokio::test]
async fn anyllm_runtime_client_preserves_openai_fields_and_parses_usage() {
    let mut server = mockito::Server::new_async().await;
    let mut sampling = SamplingParams::new();
    sampling.insert("temperature".into(), json!(0.2));
    sampling.insert("seed".into(), json!(42));
    sampling.insert("top_k".into(), json!(50));
    sampling.insert("min_p".into(), json!(0.1));
    sampling.insert("repeat_penalty".into(), json!(1.05));
    sampling.insert(
        "chat_template_kwargs".into(),
        json!({"enable_thinking": false}),
    );

    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::Json(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "temperature": 0.2,
            "seed": 42,
            "top_k": 50,
            "min_p": 0.1,
            "repeat_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": false},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }]
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-runtime-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let runtime_config =
        anyllm_multi_config(anyllm_backend_config(BackendKind::OpenAI, server.url()));
    let client = AnyLlmRuntimeClient::from_multi_config("gpt-4o-mini", runtime_config);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            Some(vec![search_spec()]),
            Some(sampling),
        )
        .await
        .expect("runtime request succeeds");

    match response {
        LLMResponse::Text(text) => assert_eq!(text.content, "ok"),
        other => panic!("expected text response, got {other:?}"),
    }
    assert_eq!(client.last_usage().unwrap().total_tokens, 11);
}

#[tokio::test]
async fn anyllm_runtime_client_parses_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-runtime-2",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_runtime",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "{\"query\":\"anyllm\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let runtime_config =
        anyllm_multi_config(anyllm_backend_config(BackendKind::OpenAI, server.url()));
    let client = AnyLlmRuntimeClient::from_multi_config("gpt-4o-mini", runtime_config);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("runtime request succeeds");

    match response {
        LLMResponse::ToolCalls(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].id.as_deref(), Some("call_runtime"));
            assert_eq!(calls[0].tool, "search");
            assert_eq!(calls[0].args.get("query"), Some(&json!("anyllm")));
        }
        other => panic!("expected tool call response, got {other:?}"),
    }
    assert_eq!(client.last_usage().unwrap().total_tokens, 10);
}

#[tokio::test]
async fn anyllm_runtime_client_records_stream_usage() {
    let mut server = mockito::Server::new_async().await;
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-runtime-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-runtime-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
        "data: [DONE]\n\n"
    );
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;

    let runtime_config =
        anyllm_multi_config(anyllm_backend_config(BackendKind::OpenAI, server.url()));
    let client = AnyLlmRuntimeClient::from_multi_config("gpt-4o-mini", runtime_config);
    let mut stream = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("runtime stream starts");

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.expect("stream chunk"));
    }

    assert_eq!(chunks[0].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[0].content, "hi");
    assert_eq!(chunks.last().unwrap().chunk_type, ChunkType::Final);
    assert_eq!(client.last_usage().unwrap().total_tokens, 5);
}

#[tokio::test]
async fn anyllm_runtime_client_maps_runtime_errors() {
    let backend = anyllm_backend_config(
        BackendKind::Anthropic,
        "https://api.anthropic.com".to_string(),
    );
    let runtime_config = anyllm_multi_config(backend);
    let client = AnyLlmRuntimeClient::from_multi_config("claude-3-5-sonnet", runtime_config);
    let err = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect_err("unsupported runtime backend should fail");

    assert!(err.to_string().contains("status 400"));
    assert!(err.to_string().contains("does not support"));
}
