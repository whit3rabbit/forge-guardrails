use forge_guardrails::{AnyLlmProxyClient, LLMClient, LLMResponse, SamplingParams, ToolSpec};
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
