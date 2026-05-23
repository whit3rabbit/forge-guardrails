use forge_guardrails::guardrails::ResponseValidator;
use forge_guardrails::{
    extract_sampling, format_tool_call_id, handle_chat_completions, respond_spec, AnthropicClient,
    ApiFormat, ChunkStream, ContextManager, ErrorTracker, ForgeError, HandlerResult, LLMClient,
    LLMResponse, LlamafileClient, Message, MessageMeta, MessageRole, MessageType, NoCompact,
    OllamaClient, SamplingParams, StreamChunk, ToolSpec,
};
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

fn golden() -> Value {
    serde_json::from_str(include_str!("parity/fixtures/python_golden.json"))
        .expect("valid Python golden fixture")
}

fn golden_case<'a>(fixtures: &'a Value, name: &str) -> &'a Value {
    fixtures
        .get("cases")
        .and_then(|cases| cases.get(name))
        .unwrap_or_else(|| panic!("missing golden case {name}"))
}

fn context_manager() -> ContextManager {
    ContextManager::new(Box::new(NoCompact), 4096, None, None, None)
}

fn proxy_context() -> Arc<Mutex<ContextManager>> {
    Arc::new(Mutex::new(context_manager()))
}

fn run_spec() -> ToolSpec {
    ToolSpec::from_json_schema(
        "run",
        "Run command",
        &json!({
            "type": "object",
            "properties": {
                "x": {"type": "integer"}
            }
        }),
    )
    .expect("valid run spec")
}

fn tool_calls_payload(calls: &[forge_guardrails::ToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|call| {
                json!({
                    "tool": call.tool,
                    "args": serde_json::to_value(&call.args).expect("args serialize"),
                    "reasoning": call.reasoning,
                })
            })
            .collect(),
    )
}

fn messages_payload(messages: &[Message]) -> Value {
    Value::Array(
        messages
            .iter()
            .map(|message| {
                let mut item = serde_json::Map::new();
                item.insert("role".to_string(), json!(message.role.as_str()));
                item.insert("content".to_string(), json!(message.content));
                item.insert(
                    "type".to_string(),
                    json!(message.metadata.msg_type.as_str()),
                );
                if let Some(tool_name) = &message.tool_name {
                    item.insert("tool_name".to_string(), json!(tool_name));
                }
                if let Some(tool_call_id) = &message.tool_call_id {
                    item.insert("tool_call_id".to_string(), json!(tool_call_id));
                }
                if let Some(tool_calls) = &message.tool_calls {
                    item.insert(
                        "tool_calls".to_string(),
                        Value::Array(
                            tool_calls
                                .iter()
                                .map(|call| {
                                    json!({
                                        "name": call.name,
                                        "args": call.args,
                                        "call_id": call.call_id,
                                    })
                                })
                                .collect(),
                        ),
                    );
                }
                Value::Object(item)
            })
            .collect(),
    )
}

fn map_from_object(value: Value) -> IndexMap<String, Value> {
    value
        .as_object()
        .expect("object args")
        .iter()
        .map(|(key, val)| (key.clone(), val.clone()))
        .collect()
}

struct TextSequenceClient {
    responses: Vec<String>,
    calls: AtomicI32,
}

impl TextSequenceClient {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: responses.into_iter().map(str::to_string).collect(),
            calls: AtomicI32::new(0),
        }
    }

    fn calls(&self) -> i32 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl LLMClient for TextSequenceClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        let idx = self.calls.fetch_add(1, Ordering::SeqCst) as usize;
        let content = self
            .responses
            .get(idx)
            .or_else(|| self.responses.last())
            .cloned()
            .unwrap_or_default();
        Ok(LLMResponse::Text(forge_guardrails::TextResponse::new(
            content,
        )))
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        Err(forge_guardrails::StreamError::new("not used"))
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct BackendErrorClient {
    calls: AtomicI32,
}

impl BackendErrorClient {
    fn new() -> Self {
        Self {
            calls: AtomicI32::new(0),
        }
    }

    fn calls(&self) -> i32 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl LLMClient for BackendErrorClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(forge_guardrails::BackendError::new(503, "backend down"))
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        Err(forge_guardrails::StreamError::new("not used"))
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct NoFinalStreamClient {
    calls: AtomicI32,
}

impl NoFinalStreamClient {
    fn new() -> Self {
        Self {
            calls: AtomicI32::new(0),
        }
    }

    fn calls(&self) -> i32 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl LLMClient for NoFinalStreamClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        Ok(LLMResponse::Text(forge_guardrails::TextResponse::new(
            "not used",
        )))
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let chunks = futures_util::stream::iter(vec![Ok(StreamChunk::new(
            forge_guardrails::ChunkType::TextDelta,
        )
        .with_content("partial"))]);
        Ok(Box::pin(chunks))
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct RespondOnlyClient;

impl LLMClient for RespondOnlyClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, forge_guardrails::BackendError> {
        Ok(LLMResponse::ToolCalls(vec![
            forge_guardrails::ToolCall::new("respond", map_from_object(json!({"message": "done"}))),
        ]))
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, forge_guardrails::StreamError> {
        Err(forge_guardrails::StreamError::new("not used"))
    }

    async fn get_context_length(
        &self,
    ) -> Result<Option<i64>, forge_guardrails::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn python_golden_anthropic_conversion_matches_request_body() {
    let fixtures = golden();
    for case_name in [
        "anthropic_conversion_unpaired",
        "anthropic_conversion_fallback_id",
    ] {
        let case = golden_case(&fixtures, case_name);
        let mut server = mockito::Server::new_async().await;
        let url = server.url();

        let mut expected_body = serde_json::Map::new();
        expected_body.insert("model".to_string(), json!("claude-3"));
        expected_body.insert("max_tokens".to_string(), json!(4096));
        expected_body.insert("messages".to_string(), case["expected"]["messages"].clone());
        if let Some(system) = case["expected"].get("system") {
            if !system.is_null() {
                expected_body.insert("system".to_string(), system.clone());
            }
        }

        let _mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::Json(Value::Object(expected_body)))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let client = AnthropicClient::new("claude-3", None)
            .with_base_url(url)
            .with_timeout(5.0);
        client
            .send(
                case["input"].as_array().expect("input array").to_vec(),
                None,
                None,
            )
            .await
            .expect("anthropic request accepted");
    }
}

#[tokio::test]
async fn python_golden_inference_retry_budget_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "inference_retry_budget");
    let expected = &case["expected"];

    let client = TextSequenceClient::new(vec!["bad 1", "bad 2", "bad 3", "bad 4"]);
    let mut messages = vec![Message::new(
        MessageRole::User,
        "start",
        MessageMeta::new(MessageType::UserInput),
    )];
    let mut context = context_manager();
    let validator = ResponseValidator::new(vec!["respond".to_string()], false, None);
    let mut tracker = ErrorTracker::new(
        case["input"]["max_retries"].as_i64().expect("max_retries") as i32,
        2,
    );
    let mut counter = 0;
    let result = forge_guardrails::run_inference(
        &mut messages,
        &client,
        &mut context,
        &validator,
        &mut tracker,
        &[respond_spec()],
        &mut counter,
        0,
        "",
        Some(
            case["input"]["max_attempts"]
                .as_i64()
                .expect("max_attempts") as i32,
        ),
        false,
        None,
        None,
    )
    .await;

    match result {
        Err(ForgeError::ToolCall(err)) => {
            assert_eq!(expected["error_type"], "ToolCallError");
            assert_eq!(
                client.calls(),
                expected["attempts"].as_i64().unwrap() as i32
            );
            assert_eq!(
                err.raw_response.as_deref(),
                expected["raw_response"].as_str()
            );
            assert_eq!(messages_payload(&messages), expected["messages"]);
        }
        other => panic!("expected ToolCallError, got {other:?}"),
    }
}

#[tokio::test]
async fn python_golden_backend_errors_do_not_retry() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "backend_error_propagation");
    let expected = &case["expected"];

    let client = BackendErrorClient::new();
    let mut messages = vec![Message::new(
        MessageRole::User,
        "start",
        MessageMeta::new(MessageType::UserInput),
    )];
    let mut context = context_manager();
    let validator = ResponseValidator::new(vec!["respond".to_string()], false, None);
    let mut tracker = ErrorTracker::new(2, 2);
    let mut counter = 0;
    let result = forge_guardrails::run_inference(
        &mut messages,
        &client,
        &mut context,
        &validator,
        &mut tracker,
        &[respond_spec()],
        &mut counter,
        0,
        "",
        Some(10),
        false,
        None,
        None,
    )
    .await;

    match result {
        Err(ForgeError::Backend(err)) => {
            assert_eq!(expected["error_type"], "BackendError");
            assert_eq!(
                client.calls(),
                expected["attempts"].as_i64().unwrap() as i32
            );
            match err {
                forge_guardrails::BackendError::Generic { status_code, body } => {
                    assert_eq!(status_code, case["input"]["status_code"].as_i64().unwrap());
                    assert_eq!(body, case["input"]["body"].as_str().unwrap());
                }
                other => panic!("expected generic backend error, got {other:?}"),
            }
        }
        other => panic!("expected BackendError, got {other:?}"),
    }
}

#[tokio::test]
async fn python_golden_stream_without_final_is_error() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "streaming_without_final");
    let expected = &case["expected"];

    let client = NoFinalStreamClient::new();
    let mut messages = vec![Message::new(
        MessageRole::User,
        "start",
        MessageMeta::new(MessageType::UserInput),
    )];
    let mut context = context_manager();
    let validator = ResponseValidator::new(vec!["respond".to_string()], false, None);
    let mut tracker = ErrorTracker::new(2, 2);
    let mut counter = 0;
    let result = forge_guardrails::run_inference(
        &mut messages,
        &client,
        &mut context,
        &validator,
        &mut tracker,
        &[respond_spec()],
        &mut counter,
        0,
        "",
        Some(10),
        true,
        None,
        None,
    )
    .await;

    match result {
        Err(ForgeError::Stream(err)) => {
            assert_eq!(expected["error_type"], "StreamError");
            assert_eq!(
                client.calls(),
                expected["attempts"].as_i64().unwrap() as i32
            );
            assert!(err.to_string().contains("Stream ended without FINAL chunk"));
        }
        other => panic!("expected StreamError, got {other:?}"),
    }
}

#[test]
fn python_golden_tool_call_id_generation_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "tool_call_id_generation");
    let expected = &case["expected"];
    let mut counter = case["input"]["starting_counter"]
        .as_i64()
        .expect("starting counter");

    let call_ids: Vec<Value> = case["input"]["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|_| {
            let id = format_tool_call_id(counter);
            counter += 1;
            json!(id)
        })
        .collect();

    assert_eq!(Value::Array(call_ids), expected["call_ids"]);
    assert_eq!(counter, expected["next_counter"].as_i64().unwrap());
}

#[tokio::test]
async fn python_golden_llamafile_reasoning_extraction_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "llamafile_reasoning_extraction");
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "choices": [{
                    "message": {"content": case["input"]["content"].clone()}
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = LlamafileClient::new(Path::new("t.gguf"))
        .with_base_url(format!("{}/v1", url))
        .with_mode("prompt")
        .with_timeout(5.0);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "run"})],
            Some(vec![run_spec()]),
            None,
        )
        .await
        .expect("llamafile response parsed");

    match response {
        LLMResponse::ToolCalls(calls) => {
            assert_eq!(tool_calls_payload(&calls), case["expected"]["tool_calls"]);
        }
        other => panic!("expected tool calls, got {other:?}"),
    }
}

#[tokio::test]
async fn python_golden_ollama_thinking_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "ollama_thinking");
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _mock = server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "message": case["input"]["message"].clone(),
                "prompt_eval_count": 1,
                "eval_count": 1
            })
            .to_string(),
        )
        .create_async()
        .await;

    let client = OllamaClient::new("reason-model")
        .with_base_url(url)
        .with_think(Some(true))
        .with_timeout(5.0);
    let response = client
        .send(
            vec![json!({"role": "user", "content": "run"})],
            Some(vec![run_spec()]),
            None,
        )
        .await
        .expect("ollama response parsed");

    match response {
        LLMResponse::ToolCalls(calls) => {
            assert_eq!(tool_calls_payload(&calls), case["expected"]["tool_calls"]);
        }
        other => panic!("expected tool calls, got {other:?}"),
    }
}

#[test]
fn python_golden_proxy_sampling_fields_match() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "proxy_sampling_fields");
    let sampling = extract_sampling(&case["input"]).expect("sampling fields");
    assert_eq!(Value::Object(sampling), case["expected"]);
}

#[tokio::test]
async fn python_golden_proxy_respond_stripping_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "proxy_respond_stripping");
    let client = Arc::new(RespondOnlyClient);
    let context = proxy_context();

    let response = handle_chat_completions(&case["input"], &client, &context, 2, true)
        .await
        .expect("proxy response");

    match response {
        HandlerResult::Response(value) => {
            let choice = &value["choices"][0];
            assert_eq!(choice["message"]["content"], case["expected"]["content"]);
            assert_eq!(choice["finish_reason"], case["expected"]["finish_reason"]);
        }
        other => panic!("expected response, got {other:?}"),
    }
}
