//! Integration tests for ollama client tests.

use forge_guardrails::{BackendError, ChunkType, LLMClient, LLMResponse, OllamaClient};
use futures_util::StreamExt;
use serde_json::json;

#[tokio::test]
async fn ollama_auto_think_unsupported_retries_without_think_and_persists() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _unsupported = server
        .mock("POST", "/api/chat")
        .match_body(mockito::Matcher::Json(json!({
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "think": true
        })))
        .with_status(400)
        .with_header("content-type", "application/json")
        .with_body(json!({"error": "thinking is not supported"}).to_string())
        .create_async()
        .await;

    let _retry = server
        .mock("POST", "/api/chat")
        .match_body(mockito::Matcher::Json(json!({
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false
        })))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "message": {"content": "ok"},
                "prompt_eval_count": 2,
                "eval_count": 3
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = OllamaClient::new("deepseek-r1:8b")
        .with_base_url(url)
        .with_timeout(5.0);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("retry succeeds");

    match response {
        LLMResponse::Text(text) => assert_eq!(text.content, "ok"),
        other => panic!("expected text response, got {other:?}"),
    }
    assert!(!client.is_think_enabled());
    assert!(client.is_think_resolved());
    let usage = client.get_last_usage().expect("usage recorded");
    assert_eq!(usage.prompt_tokens, 2);
    assert_eq!(usage.completion_tokens, 3);
}

#[tokio::test]
async fn ollama_explicit_think_unsupported_returns_error_without_retry() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _unsupported = server
        .mock("POST", "/api/chat")
        .match_body(mockito::Matcher::Json(json!({
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "think": true
        })))
        .with_status(400)
        .with_header("content-type", "application/json")
        .with_body(json!({"error": "thinking is not supported"}).to_string())
        .create_async()
        .await;

    let client = OllamaClient::new("deepseek-r1:8b")
        .with_base_url(url)
        .with_think(Some(true))
        .with_timeout(5.0);
    let err = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect_err("explicit think should fail");

    match err {
        BackendError::ThinkingNotSupported {
            model,
            status_code,
            body,
        } => {
            assert_eq!(model, "deepseek-r1:8b");
            assert_eq!(status_code, 400);
            assert!(body.contains("thinking is not supported"));
        }
        other => panic!("expected thinking-not-supported error, got {other:?}"),
    }
    assert!(client.is_think_enabled());
    assert!(client.is_think_resolved());
}

#[tokio::test]
async fn ollama_streaming_auto_think_unsupported_retries_without_think() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _unsupported = server
        .mock("POST", "/api/chat")
        .match_body(mockito::Matcher::Json(json!({
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "think": true
        })))
        .with_status(400)
        .with_header("content-type", "application/json")
        .with_body(json!({"error": "thinking is not supported"}).to_string())
        .create_async()
        .await;

    let _retry = server
        .mock("POST", "/api/chat")
        .match_body(mockito::Matcher::Json(json!({
            "model": "deepseek-r1:8b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        })))
        .with_status(200)
        .with_header("content-type", "application/x-ndjson")
        .with_body(concat!(
            "{\"message\": {\"content\": \"ok\"}, \"done\": false}\n",
            "{\"message\": {\"content\": \"\"}, \"done\": true, \"prompt_eval_count\": 4, \"eval_count\": 5}\n",
        ))
        .create_async()
        .await;

    let client = OllamaClient::new("deepseek-r1:8b")
        .with_base_url(url)
        .with_timeout(5.0);
    let mut stream = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("retry stream opens");

    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.expect("stream chunk"));
    }

    assert_eq!(chunks[0].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[0].content, "ok");
    assert_eq!(chunks[1].chunk_type, ChunkType::Final);
    match chunks[1].response.clone().expect("final response") {
        LLMResponse::Text(text) => assert_eq!(text.content, "ok"),
        other => panic!("expected text response, got {other:?}"),
    }
    assert!(!client.is_think_enabled());
    assert!(client.is_think_resolved());
}
