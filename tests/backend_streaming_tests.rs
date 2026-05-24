//! Integration tests for backend streaming tests.

use forge_guardrails::{AnthropicClient, ChunkType, LLMClient, LlamafileClient, OllamaClient};
use futures_util::StreamExt;
use serde_json::json;
use std::path::Path;

#[tokio::test]
async fn test_anthropic_streaming_request() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    // Anthropic SSE protocol: content_block_delta events carry text,
    // message_delta carries usage, message_stop triggers the Final chunk.
    let sse_body = concat!(
        "data: {\"type\": \"message_start\", \"message\": {\"usage\": {\"input_tokens\": 10, \"output_tokens\": 0}}}\n\n",
        "data: {\"type\": \"content_block_delta\", \"delta\": {\"type\": \"text_delta\", \"text\": \"hello \"}}\n\n",
        "data: {\"type\": \"content_block_delta\", \"delta\": {\"type\": \"text_delta\", \"text\": \"world\"}}\n\n",
        "data: {\"type\": \"message_delta\", \"usage\": {\"output_tokens\": 5}}\n\n",
        "data: {\"type\": \"message_stop\"}\n\n",
    );

    let _mock = server
        .mock("POST", "/messages")
        .match_header("content-type", "application/json")
        .match_body(mockito::Matcher::Json(json!({
            "model": "claude-3",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4096,
            "stream": true
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let client = AnthropicClient::new("claude-3", Some("test-key".to_string()))
        .with_base_url(url)
        .with_timeout(5.0);

    let stream_res = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await;
    assert!(stream_res.is_ok());

    let mut chunks = Vec::new();
    let mut stream = stream_res.unwrap();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[0].content, "hello ");
    assert_eq!(chunks[1].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[1].content, "world");
    assert_eq!(chunks[2].chunk_type, ChunkType::Final);
    let usage = client.last_usage().expect("stream usage recorded");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[tokio::test]
async fn test_ollama_streaming_request() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _mock = server.mock("POST", "/api/chat")
        .match_body(mockito::Matcher::Json(json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        })))
        .with_status(200)
        .with_header("content-type", "application/x-ndjson")
        .with_body("{\"message\": {\"content\": \"hello \"}, \"done\": false}\n{\"message\": {\"content\": \"world\"}, \"done\": false}\n{\"message\": {\"content\": \"\"}, \"done\": true, \"prompt_eval_count\": 5, \"eval_count\": 5}\n")
        .create_async()
        .await;

    let client = OllamaClient::new("llama3")
        .with_base_url(url)
        .with_timeout(5.0);

    let stream_res = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await;
    assert!(stream_res.is_ok());

    let mut chunks = Vec::new();
    let mut stream = stream_res.unwrap();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[0].content, "hello ");
    assert_eq!(chunks[1].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[1].content, "world");
    assert_eq!(chunks[2].chunk_type, ChunkType::Final);
    let usage = client.last_usage().expect("stream usage recorded");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 5);
    assert_eq!(usage.total_tokens, 10);
}

#[tokio::test]
async fn test_ollama_native_malformed_args_returns_text() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _mock = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "run", "arguments": "{broken"}}
                    ]
                },
                "prompt_eval_count": 1,
                "eval_count": 1
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = OllamaClient::new("llama3")
        .with_base_url(url)
        .with_timeout(5.0);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("ollama response parsed");

    match response {
        forge_guardrails::LLMResponse::Text(text) => assert_eq!(text.content, "{broken"),
        other => panic!("expected text response, got {other:?}"),
    }
}

#[tokio::test]
async fn test_ollama_streaming_malformed_args_returns_text() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _mock = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/x-ndjson")
        .with_body(concat!(
            "{\"message\": {\"tool_calls\": [{\"function\": {\"name\": \"run\", \"arguments\": \"{broken\"}}]}, \"done\": false}\n",
            "{\"message\": {\"content\": \"\"}, \"done\": true, \"prompt_eval_count\": 1, \"eval_count\": 1}\n",
        ))
        .create_async()
        .await;

    let client = OllamaClient::new("llama3")
        .with_base_url(url)
        .with_timeout(5.0);
    let stream_res = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("ollama stream opened");

    let mut final_response = None;
    let mut stream = stream_res;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("stream chunk");
        if chunk.chunk_type == ChunkType::Final {
            final_response = chunk.response;
        }
    }

    match final_response.expect("final response") {
        forge_guardrails::LLMResponse::Text(text) => assert_eq!(text.content, "{broken"),
        other => panic!("expected text response, got {other:?}"),
    }
}

#[tokio::test]
async fn test_llamafile_streaming_records_usage_after_finish() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let sse_body = concat!(
        "data: {\"choices\": [{\"delta\": {\"content\": \"hello\"}, \"finish_reason\": null}]}\n\n",
        "data: {\"choices\": [{\"delta\": {}, \"finish_reason\": \"stop\"}]}\n\n",
        "data: {\"choices\": [], \"usage\": {\"prompt_tokens\": 7, \"completion_tokens\": 3, \"total_tokens\": 10}}\n\n",
        "data: [DONE]\n\n",
    );

    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .match_body(mockito::Matcher::Json(json!({
            "model": "t",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "cache_prompt": true
        })))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let client = LlamafileClient::new(Path::new("t.gguf"))
        .with_base_url(format!("{}/v1", url))
        .with_timeout(5.0);

    let stream_res = client
        .send_stream(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await;
    assert!(stream_res.is_ok());

    let mut chunks = Vec::new();
    let mut stream = stream_res.unwrap();
    while let Some(chunk) = stream.next().await {
        chunks.push(chunk.unwrap());
    }

    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].chunk_type, ChunkType::TextDelta);
    assert_eq!(chunks[0].content, "hello");
    assert_eq!(chunks[1].chunk_type, ChunkType::Final);
    let usage = client.last_usage().expect("stream usage recorded");
    assert_eq!(usage.prompt_tokens, 7);
    assert_eq!(usage.completion_tokens, 3);
    assert_eq!(usage.total_tokens, 10);
}

#[tokio::test]
async fn test_llamafile_native_malformed_args_returns_text() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "choices": [{
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "run", "arguments": "{broken"}}
                        ]
                    }
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = LlamafileClient::new(Path::new("t.gguf"))
        .with_base_url(format!("{}/v1", url))
        .with_timeout(5.0);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "hello"})],
            None,
            None,
        )
        .await
        .expect("llamafile response parsed");

    match response {
        forge_guardrails::LLMResponse::Text(text) => assert_eq!(text.content, ""),
        other => panic!("expected text response, got {other:?}"),
    }
}
