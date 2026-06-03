//! Integration tests for anyllm proxy client tests.

use anyllm_proxy::config::model_router::{Deployment, ModelRouter};
use anyllm_proxy::config::{
    BackendAuth, BackendConfig, BackendKind, ModelMapping, MultiConfig, OpenAIApiFormat, TlsConfig,
};
use forge_guardrails::{
    handle_chat_completions, AnyLlmProxyClient, AnyLlmRuntimeClient, ChunkType, ContextManager,
    HandlerResult, LLMClient, LLMRequestOptions, LLMResponse, NoCompact, SamplingParams, ToolSpec,
};
use futures_util::StreamExt;
use indexmap::IndexMap;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex as TokioMutex;

fn assert_positive_cost(cost: Option<f64>) {
    let cost = cost.expect("estimated cost is available");
    assert!(cost > 0.0, "estimated cost should be positive, got {cost}");
}

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
                        "properties": {"query": {"title": "Query", "type": "string"}},
                        "required": ["query"],
                        "title": "SearchParams",
                        "type": "object",
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

    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.requested_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.response_model.as_deref(), Some("gpt-4o-mini"));
    assert!(info.selected_backend.is_none());
    assert!(info.mapped_model.is_none());
    assert!(info.provider_id.is_none());
    assert!(!info.used_responses_api);
    assert_positive_cost(info.estimated_cost_usd);
}

#[tokio::test]
async fn anyllm_proxy_client_preserves_cache_passthrough_and_records_cached_tokens() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Json(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "prompt_cache_key": "tenant-a-tools-v1",
            "prompt_cache_retention": "24h",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "properties": {"query": {"title": "Query", "type": "string"}},
                        "required": ["query"],
                        "title": "SearchParams",
                        "type": "object",
                    }
                }
            }]
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-cache",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached ok"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 4,
                    "total_tokens": 104,
                    "prompt_tokens_details": {"cached_tokens": 64}
                }
            })
            .to_string(),
        )
        .create_async()
        .await;

    let mut passthrough = serde_json::Map::new();
    passthrough.insert("prompt_cache_key".to_string(), json!("tenant-a-tools-v1"));
    passthrough.insert("prompt_cache_retention".to_string(), json!("24h"));

    let client = AnyLlmProxyClient::new("gpt-4o-mini").with_base_url(server.url());
    let response = client
        .send_with_options(
            vec![json!({"role": "user", "content": "hello"})],
            Some(vec![search_spec()]),
            LLMRequestOptions {
                passthrough: Some(passthrough),
                ..Default::default()
            },
        )
        .await
        .expect("request succeeds");

    match response {
        LLMResponse::Text(text) => assert_eq!(text.content, "cached ok"),
        other => panic!("expected text response, got {other:?}"),
    }
    let details = client.last_usage_details().expect("usage details");
    assert_eq!(details.cached_prompt_tokens, Some(64));
}

#[tokio::test]
async fn anyllm_proxy_client_parses_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_header("anthropic-ratelimit-requests-limit", "100")
        .with_header("anthropic-ratelimit-tokens-remaining", "900")
        .with_header("retry-after", "2")
        .with_header("anthropic-organization-id", "org_test")
        .with_header("x-anyllm-degradation", "top_k")
        .with_header("x-anyllm-cache", "miss")
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
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "prompt_cache_hit_tokens": 20,
                    "prompt_cache_miss_tokens": 3
                }
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

    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.degradation_warnings.as_deref(), Some("top_k"));
    assert_eq!(info.cache_status.as_deref(), Some("miss"));
    assert_eq!(info.rate_limits.requests_limit.as_deref(), Some("100"));
    assert_eq!(info.rate_limits.tokens_remaining.as_deref(), Some("900"));
    assert_eq!(info.rate_limits.retry_after.as_deref(), Some("2"));
    assert_eq!(
        info.rate_limits.organization_id.as_deref(),
        Some("org_test")
    );
    assert_positive_cost(info.estimated_cost_usd);
    let details = client.last_usage_details().expect("usage details");
    assert_eq!(details.cached_prompt_tokens, Some(20));
    assert_eq!(details.cache_miss_prompt_tokens, Some(3));
    assert_eq!(details.prompt_cache_hit_tokens, Some(20));
    assert_eq!(details.prompt_cache_miss_tokens, Some(3));
}

#[tokio::test]
async fn anyllm_proxy_client_records_stream_headers_and_cost() {
    let mut server = mockito::Server::new_async().await;
    let sse = concat!(
        "data:{\"id\":\"chatcmpl-sidecar-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data:{\"id\":\"chatcmpl-sidecar-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n",
        "data:[DONE]\n\n"
    );
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Json(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_header("anthropic-ratelimit-requests-remaining", "12")
        .with_header("x-anyllm-degradation", "cache_control")
        .with_header("x-anyllm-cache", "hit")
        .with_header("x-anyllm-cost-usd", "0.0042")
        .with_body(sse)
        .create_async()
        .await;

    let client = AnyLlmProxyClient::new("gpt-4o-mini").with_base_url(server.url());
    let mut stream = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("sidecar stream starts");

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.expect("stream chunk"));
    }

    assert_eq!(chunks[0].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[0].content, "hi");
    let final_chunk = chunks.last().unwrap();
    assert_eq!(final_chunk.chunk_type, ChunkType::Final);
    assert_eq!(final_chunk.usage.as_ref().unwrap().total_tokens, 5);
    let final_info = final_chunk.call_info.as_ref().expect("final call info");
    assert_eq!(final_info.requested_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(final_info.response_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(
        final_info.rate_limits.requests_remaining.as_deref(),
        Some("12")
    );
    assert_positive_cost(final_info.estimated_cost_usd);
    assert_eq!(client.last_usage().unwrap().total_tokens, 5);

    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.requested_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.response_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.degradation_warnings.as_deref(), Some("cache_control"));
    assert_eq!(info.cache_status.as_deref(), Some("hit"));
    assert_eq!(info.rate_limits.requests_remaining.as_deref(), Some("12"));
    assert_eq!(info.estimated_cost_usd, Some(0.0042));
}

#[tokio::test]
async fn anyllm_proxy_client_processes_final_unterminated_sse_line() {
    let mut server = mockito::Server::new_async().await;
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-sidecar-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data:{\"id\":\"chatcmpl-sidecar-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}"
    );
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;

    let client = AnyLlmProxyClient::new("gpt-4o-mini").with_base_url(server.url());
    let mut stream = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("sidecar stream starts");

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
                        "properties": {"query": {"title": "Query", "type": "string"}},
                        "required": ["query"],
                        "title": "SearchParams",
                        "type": "object",
                    }
                }
            }]
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_header("x-ratelimit-limit-requests", "1000")
        .with_header("x-ratelimit-remaining-requests", "999")
        .with_header("x-ratelimit-limit-tokens", "200000")
        .with_header("x-ratelimit-remaining-tokens", "199000")
        .with_header("retry-after", "1")
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

    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.requested_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.response_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.selected_backend.as_deref(), Some("openai"));
    assert_eq!(info.mapped_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.backend_kind.as_deref(), Some("OpenAI"));
    assert!(info.provider_id.is_none());
    assert!(!info.used_responses_api);
    assert!(info.degradation_warnings.is_none());
    assert_eq!(info.rate_limits.requests_limit.as_deref(), Some("1000"));
    assert_eq!(info.rate_limits.tokens_remaining.as_deref(), Some("199000"));
    assert_eq!(info.rate_limits.retry_after.as_deref(), Some("1"));
    assert_positive_cost(info.estimated_cost_usd);
}

#[tokio::test]
async fn anyllm_runtime_client_for_model_uses_requested_model_with_fresh_state() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Json(json!({
            "model": "request-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-runtime-for-model",
                "object": "chat.completion",
                "created": 1,
                "model": "request-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "routed"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let config = anyllm_multi_config(anyllm_backend_config(BackendKind::OpenAI, server.url()));
    let base_client =
        AnyLlmRuntimeClient::from_multi_config("base-model", config).with_context_length(8192);
    let request_client = base_client.for_model("request-model");

    let response = request_client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("runtime response");

    match response {
        LLMResponse::Text(text) => assert_eq!(text.content, "routed"),
        other => panic!("expected text response, got {other:?}"),
    }
    assert_eq!(request_client.last_usage().unwrap().total_tokens, 5);
    assert!(base_client.last_usage().is_none());
    assert!(base_client.last_call_info().is_none());
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

    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.selected_backend.as_deref(), Some("openai"));
    assert_eq!(info.mapped_model.as_deref(), Some("gpt-4o-mini"));
    assert_positive_cost(info.estimated_cost_usd);
}

#[tokio::test]
async fn anyllm_runtime_client_routes_model_and_handler_emits_tool_call() {
    let fallback = mockito::Server::new_async().await;
    let mut routed = mockito::Server::new_async().await;
    let _mock = routed
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_header("x-ratelimit-remaining-requests", "77")
        .with_body(
            json!({
                "id": "chatcmpl-routed",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_routed",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "{\"query\":\"router\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let mut backends = IndexMap::new();
    backends.insert(
        "fallback".to_string(),
        anyllm_backend_config(BackendKind::OpenAI, fallback.url()),
    );
    backends.insert(
        "routed".to_string(),
        anyllm_backend_config(BackendKind::OpenAI, routed.url()),
    );
    let runtime_config = MultiConfig {
        listen_port: 0,
        log_bodies: false,
        default_backend: "fallback".to_string(),
        backends,
        expose_degradation_warnings: false,
    };

    let deployment = Arc::new(Deployment::new(
        "routed".to_string(),
        "gpt-4o-mini".to_string(),
        None,
        None,
    ));
    let mut routes = HashMap::new();
    routes.insert("forge-virtual".to_string(), vec![deployment]);
    let router = Arc::new(RwLock::new(ModelRouter::new(routes)));

    let client = Arc::new(AnyLlmRuntimeClient::from_multi_config_with_model_router(
        "forge-virtual",
        runtime_config,
        Some(router),
    ));
    let context_manager = Arc::new(TokioMutex::new(ContextManager::new(
        Box::new(NoCompact),
        4096,
        None,
        None,
        None,
    )));
    let body = json!({
        "model": "forge-virtual",
        "messages": [{"role": "user", "content": "find router docs"}],
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
    });

    let result = handle_chat_completions(&body, &client, &context_manager, 1, true)
        .await
        .expect("handler succeeds");

    match result {
        HandlerResult::Response(value) => {
            let calls = value["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], "search");
            assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
        }
        _ => panic!("expected non-streaming response"),
    }

    assert_eq!(client.last_usage().unwrap().total_tokens, 9);
    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.requested_model.as_deref(), Some("forge-virtual"));
    assert_eq!(info.response_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.selected_backend.as_deref(), Some("routed"));
    assert_eq!(info.mapped_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.backend_kind.as_deref(), Some("OpenAI"));
    assert_eq!(info.rate_limits.requests_remaining.as_deref(), Some("77"));
    assert_positive_cost(info.estimated_cost_usd);
}

#[tokio::test]
async fn anyllm_runtime_client_supports_mlx_openai_compatible_eval_backend() {
    let mut server = mockito::Server::new_async().await;
    let model = "mlx-community/Llama-3.2-3B-Instruct-4bit";
    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Json(json!({
            "model": model,
            "messages": [{"role": "user", "content": "hello from mac"}],
            "stream": false,
            "max_tokens": 16
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "chatcmpl-mlx",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "mlx ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let mut sampling = SamplingParams::new();
    sampling.insert("max_tokens".into(), json!(16));

    let mut backends = IndexMap::new();
    backends.insert(
        "mlx".to_string(),
        anyllm_backend_config(BackendKind::OpenAI, server.url()),
    );
    let runtime_config = MultiConfig {
        listen_port: 0,
        log_bodies: false,
        default_backend: "mlx".to_string(),
        backends,
        expose_degradation_warnings: false,
    };

    let client = AnyLlmRuntimeClient::from_multi_config(model, runtime_config);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello from mac"})],
            None,
            Some(sampling),
        )
        .await
        .expect("mlx-compatible runtime request succeeds");

    match response {
        LLMResponse::Text(text) => assert_eq!(text.content, "mlx ok"),
        other => panic!("expected text response, got {other:?}"),
    }
    assert_eq!(client.last_usage().unwrap().total_tokens, 6);

    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.requested_model.as_deref(), Some(model));
    assert_eq!(info.response_model.as_deref(), Some(model));
    assert_eq!(info.selected_backend.as_deref(), Some("mlx"));
    assert_eq!(info.mapped_model.as_deref(), Some(model));
    assert_eq!(info.backend_kind.as_deref(), Some("OpenAI"));
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
        .with_header("x-ratelimit-remaining-requests", "88")
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

    let info = client.last_call_info().expect("call info recorded");
    assert_eq!(info.requested_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.response_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.selected_backend.as_deref(), Some("openai"));
    assert_eq!(info.mapped_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(info.backend_kind.as_deref(), Some("OpenAI"));
    assert_eq!(info.rate_limits.requests_remaining.as_deref(), Some("88"));
    assert!(info.degradation_warnings.is_none());
    assert_positive_cost(info.estimated_cost_usd);
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
