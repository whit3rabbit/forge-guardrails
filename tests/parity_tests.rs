#![allow(dead_code, unused_imports)]

use forge_guardrails::guardrails::ResponseValidator;
use forge_guardrails::{
    build_tool_prompt, extract_sampling, fold_and_serialize, format_tool, format_tool_call_id,
    handle_chat_completions, respond_spec, respond_tool, unknown_tool_nudge, AnthropicClient,
    ApiFormat, ChunkStream, CompactStrategy, ContextManager, ErrorTracker, ForgeError,
    HandlerResult, LLMClient, LLMResponse, LlamafileClient, Message, MessageMeta, MessageRole,
    MessageType, NoCompact, OllamaClient, SamplingParams, StepEnforcer, StreamChunk, TieredCompact,
    ToolCallInfo, ToolDef, ToolResolutionError, ToolSpec, Workflow, WorkflowRunner,
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

fn empty_spec(name: &str) -> ToolSpec {
    ToolSpec::from_json_schema(
        name,
        format!("{name} tool"),
        &json!({
            "type": "object",
            "properties": {}
        }),
    )
    .expect("valid empty spec")
}

fn seed_messages() -> Vec<Message> {
    vec![
        Message::new(
            MessageRole::System,
            "sys",
            MessageMeta::new(MessageType::SystemPrompt),
        ),
        Message::new(
            MessageRole::User,
            "start",
            MessageMeta::new(MessageType::UserInput),
        ),
    ]
}

fn lookup_tool(_args: Vec<String>) -> Result<String, ToolResolutionError> {
    Ok("lookup ok".to_string())
}

fn analyze_tool(_args: Vec<String>) -> Result<String, ToolResolutionError> {
    Ok("analyze ok".to_string())
}

fn soft_fail_tool(_args: Vec<String>) -> Result<String, ToolResolutionError> {
    Err(ToolResolutionError::new("try again"))
}

fn hard_fail_callable() -> forge_guardrails::workflow::ToolCallable {
    Arc::new(|_args| {
        Box::pin(async {
            Err(forge_guardrails::error::ToolError::Execution(
                "boom".to_string(),
            ))
        })
            as futures_util::future::BoxFuture<
                'static,
                Result<Value, forge_guardrails::error::ToolError>,
            >
    })
}

fn respond_tool_def() -> ToolDef {
    respond_tool()
}

fn parity_workflow(tools: IndexMap<String, ToolDef>, required_steps: Vec<&str>) -> Workflow {
    Workflow::new(
        "wf",
        "Parity workflow",
        tools,
        required_steps.into_iter().map(str::to_string).collect(),
        "respond".to_string().into(),
        "sys",
    )
    .expect("valid workflow")
}

fn scripted_call(name: &str, args: Value) -> forge_guardrails::ToolCall {
    forge_guardrails::ToolCall::new(name, map_from_object(args))
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
                if let Some(step_index) = message.metadata.step_index {
                    item.insert("step_index".to_string(), json!(step_index));
                }
                if let Some(original_type) = message.metadata.original_type {
                    item.insert("original_type".to_string(), json!(original_type.as_str()));
                }
                if let Some(token_estimate) = message.metadata.token_estimate {
                    item.insert("token_estimate".to_string(), json!(token_estimate));
                }
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

fn json_dumps_python(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(val) => val.to_string(),
        Value::Number(val) => val.to_string(),
        Value::String(val) => serde_json::to_string(val).expect("string serialize"),
        Value::Array(values) => {
            let inner = values
                .iter()
                .map(json_dumps_python)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{}]", inner)
        }
        Value::Object(values) => {
            let inner = values
                .iter()
                .map(|(key, val)| {
                    let key = serde_json::to_string(key).expect("key serialize");
                    format!("{}: {}", key, json_dumps_python(val))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{}}}", inner)
        }
    }
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

struct ScriptedClient {
    responses: Vec<LLMResponse>,
    calls: AtomicI32,
}

impl ScriptedClient {
    fn new(responses: Vec<LLMResponse>) -> Self {
        Self {
            responses,
            calls: AtomicI32::new(0),
        }
    }

    fn calls(&self) -> i32 {
        self.calls.load(Ordering::SeqCst)
    }
}

impl LLMClient for ScriptedClient {
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
        Ok(self
            .responses
            .get(idx)
            .or_else(|| self.responses.last())
            .cloned()
            .unwrap_or_else(|| LLMResponse::Text(forge_guardrails::TextResponse::new(""))))
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

async fn run_workflow_capture(
    workflow: Workflow,
    responses: Vec<LLMResponse>,
    max_iterations: i32,
    max_tool_errors: i32,
) -> (Result<Value, ForgeError>, Vec<Message>, i32) {
    let client = Arc::new(ScriptedClient::new(responses));
    let emitted: Arc<std::sync::Mutex<Vec<Message>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let emitted_cb = emitted.clone();
    let runner = WorkflowRunner::new(
        client.clone(),
        Arc::new(Mutex::new(context_manager())),
        max_iterations,
        2,
        max_tool_errors,
        false,
        None,
        Some(Box::new(move |msg: &Message| {
            emitted_cb.lock().expect("message lock").push(msg.clone());
        })),
        false,
        None,
    );
    let result = runner
        .run(&workflow, "ignored", None, Some(seed_messages()), None)
        .await;
    let messages = emitted.lock().expect("message lock").clone();
    (result, messages, client.calls())
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

#[tokio::test]
async fn python_golden_text_retry_history_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "text_retry_history");
    let expected = &case["expected"];

    let client = ScriptedClient::new(vec![
        LLMResponse::Text(forge_guardrails::TextResponse::new("plain answer")),
        LLMResponse::ToolCalls(vec![scripted_call("run", json!({"x": 1}))]),
    ]);
    let mut messages = vec![Message::new(
        MessageRole::User,
        "start",
        MessageMeta::new(MessageType::UserInput),
    )];
    let mut context = context_manager();
    let validator = ResponseValidator::new(vec!["run".to_string()], false, None);
    let mut tracker = ErrorTracker::new(2, 2);
    let mut counter = 0;
    let result = forge_guardrails::run_inference(
        &mut messages,
        &client,
        &mut context,
        &validator,
        &mut tracker,
        &[run_spec()],
        &mut counter,
        0,
        "",
        Some(3),
        false,
        None,
        None,
    )
    .await
    .expect("inference should succeed")
    .expect("inference result");

    assert_eq!(
        result.attempts,
        expected["attempts"].as_i64().unwrap() as i32
    );
    assert_eq!(
        result.tool_call_counter,
        expected["next_counter"].as_i64().unwrap()
    );
    assert_eq!(messages_payload(&messages), expected["messages"]);
    match result.response {
        LLMResponse::ToolCalls(calls) => {
            assert_eq!(tool_calls_payload(&calls), expected["response"]);
        }
        other => panic!("expected tool calls, got {other:?}"),
    }
}

#[tokio::test]
async fn python_golden_unknown_tool_history_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "unknown_tool_history");
    let expected = &case["expected"];

    let client = ScriptedClient::new(vec![
        LLMResponse::ToolCalls(vec![scripted_call("bogus", json!({}))]),
        LLMResponse::ToolCalls(vec![scripted_call("run", json!({"x": 1}))]),
    ]);
    let mut messages = vec![Message::new(
        MessageRole::User,
        "start",
        MessageMeta::new(MessageType::UserInput),
    )];
    let mut context = context_manager();
    let validator = ResponseValidator::new(vec!["run".to_string()], false, None);
    let mut tracker = ErrorTracker::new(2, 2);
    let mut counter = 0;
    let result = forge_guardrails::run_inference(
        &mut messages,
        &client,
        &mut context,
        &validator,
        &mut tracker,
        &[run_spec()],
        &mut counter,
        0,
        "",
        Some(3),
        false,
        None,
        None,
    )
    .await
    .expect("inference should succeed")
    .expect("inference result");

    assert_eq!(
        result.attempts,
        expected["attempts"].as_i64().unwrap() as i32
    );
    assert_eq!(
        result.tool_call_counter,
        expected["next_counter"].as_i64().unwrap()
    );
    assert_eq!(messages_payload(&messages), expected["messages"]);
    match result.response {
        LLMResponse::ToolCalls(calls) => {
            assert_eq!(tool_calls_payload(&calls), expected["response"]);
        }
        other => panic!("expected tool calls, got {other:?}"),
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

#[test]
fn python_golden_toolspec_schema_output_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "toolspec_schema_output");
    let spec =
        ToolSpec::from_json_schema("search_tool", "Search", &case["input"]).expect("valid spec");
    let schema = spec.get_json_schema();
    assert_eq!(schema, case["expected"]["schema"]);
    assert_eq!(
        json_dumps_python(&schema),
        case["expected"]["schema_json"]
            .as_str()
            .expect("schema json")
    );
}

#[test]
fn python_golden_format_tool_output_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "format_tool_output");
    let spec =
        ToolSpec::from_json_schema("search_tool", "Search", &case["input"]).expect("valid spec");
    let tool = format_tool(&spec);
    assert_eq!(tool, case["expected"]["tool"]);
    assert_eq!(
        json_dumps_python(&tool),
        case["expected"]["tool_json"].as_str().expect("tool json")
    );
}

#[test]
fn python_golden_tool_prompt_text_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "tool_prompt_text");
    let spec =
        ToolSpec::from_json_schema("search", "Search docs", &case["input"]).expect("valid spec");
    assert_eq!(
        build_tool_prompt(&[spec]),
        case["expected"].as_str().expect("prompt text")
    );
}

#[test]
fn python_golden_unknown_tool_order_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "unknown_tool_order");
    let tools = case["input"]["available_tools"]
        .as_array()
        .expect("available tools")
        .iter()
        .map(|v| v.as_str().expect("tool"))
        .collect::<Vec<_>>();
    let called = case["input"]["called_tool"].as_str().expect("called tool");
    assert_eq!(
        unknown_tool_nudge(called, &tools),
        case["expected"].as_str().expect("nudge")
    );
}

#[tokio::test]
async fn python_golden_step_nudge_history_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "step_nudge_history");
    let expected = &case["expected"];

    let mut tools = IndexMap::new();
    tools.insert(
        "lookup".to_string(),
        ToolDef::new(
            empty_spec("lookup"),
            lookup_tool as fn(Vec<String>) -> Result<String, ToolResolutionError>,
        ),
    );
    tools.insert("respond".to_string(), respond_tool_def());
    let workflow = parity_workflow(tools, vec!["lookup"]);
    let (result, messages, calls) = run_workflow_capture(
        workflow,
        vec![
            LLMResponse::ToolCalls(vec![scripted_call(
                "respond",
                json!({"message": "too soon"}),
            )]),
            LLMResponse::ToolCalls(vec![scripted_call("lookup", json!({}))]),
            LLMResponse::ToolCalls(vec![scripted_call(
                "respond",
                json!({"message": "lookup complete"}),
            )]),
        ],
        10,
        2,
    )
    .await;

    assert_eq!(result.expect("workflow succeeds"), expected["result"]);
    assert_eq!(json!(calls), expected["attempts"]);
    assert_eq!(messages_payload(&messages), expected["messages"]);
}

#[tokio::test]
async fn python_golden_prerequisite_nudge_history_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "prerequisite_nudge_history");
    let expected = &case["expected"];

    let mut tools = IndexMap::new();
    tools.insert(
        "lookup".to_string(),
        ToolDef::new(
            empty_spec("lookup"),
            lookup_tool as fn(Vec<String>) -> Result<String, ToolResolutionError>,
        ),
    );
    tools.insert(
        "analyze".to_string(),
        ToolDef::new(
            empty_spec("analyze"),
            analyze_tool as fn(Vec<String>) -> Result<String, ToolResolutionError>,
        )
        .with_prerequisites(vec![
            forge_guardrails::workflow::PrerequisiteSpec::NameOnly("lookup".to_string()),
        ]),
    );
    tools.insert("respond".to_string(), respond_tool_def());
    let workflow = parity_workflow(tools, vec!["lookup", "analyze"]);
    let (result, messages, calls) = run_workflow_capture(
        workflow,
        vec![
            LLMResponse::ToolCalls(vec![scripted_call("analyze", json!({}))]),
            LLMResponse::ToolCalls(vec![scripted_call("lookup", json!({}))]),
            LLMResponse::ToolCalls(vec![scripted_call("analyze", json!({}))]),
            LLMResponse::ToolCalls(vec![scripted_call(
                "respond",
                json!({"message": "analysis complete"}),
            )]),
        ],
        10,
        2,
    )
    .await;

    assert_eq!(result.expect("workflow succeeds"), expected["result"]);
    assert_eq!(json!(calls), expected["attempts"]);
    assert_eq!(messages_payload(&messages), expected["messages"]);
}

#[tokio::test]
async fn python_golden_tool_resolution_soft_error_budget_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "tool_resolution_soft_error_budget");
    let expected = &case["expected"];

    let mut tools = IndexMap::new();
    tools.insert(
        "lookup".to_string(),
        ToolDef::new(
            empty_spec("lookup"),
            soft_fail_tool as fn(Vec<String>) -> Result<String, ToolResolutionError>,
        ),
    );
    tools.insert("respond".to_string(), respond_tool_def());
    let workflow = parity_workflow(tools, vec![]);
    let (result, messages, calls) = run_workflow_capture(
        workflow,
        vec![
            LLMResponse::ToolCalls(vec![scripted_call("lookup", json!({}))]),
            LLMResponse::ToolCalls(vec![scripted_call(
                "respond",
                json!({"message": "soft resolution recovered"}),
            )]),
        ],
        10,
        0,
    )
    .await;

    assert_eq!(result.expect("workflow succeeds"), expected["result"]);
    assert_eq!(json!(calls), expected["attempts"]);
    assert_eq!(messages_payload(&messages), expected["messages"]);
}

#[tokio::test]
async fn python_golden_hard_tool_execution_error_budget_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "hard_tool_execution_error_budget");
    let expected = &case["expected"];

    let mut tools = IndexMap::new();
    tools.insert(
        "explode".to_string(),
        ToolDef::new(empty_spec("explode"), hard_fail_callable()),
    );
    tools.insert("respond".to_string(), respond_tool_def());
    let workflow = parity_workflow(tools, vec![]);
    let (result, messages, calls) = run_workflow_capture(
        workflow,
        vec![LLMResponse::ToolCalls(vec![scripted_call(
            "explode",
            json!({}),
        )])],
        3,
        0,
    )
    .await;

    match result {
        Err(ForgeError::ToolExecution(err)) => {
            assert_eq!(expected["error_type"], "ToolExecutionError");
            assert_eq!(json!(err.tool_name), expected["tool_name"]);
            assert_eq!(json!(err.cause), expected["cause"]);
        }
        other => panic!("expected ToolExecutionError, got {other:?}"),
    }
    assert_eq!(json!(calls), expected["attempts"]);
    assert_eq!(messages_payload(&messages), expected["messages"]);
}

#[test]
fn python_golden_compaction_phases_match() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "compaction_phases");
    let mut messages = vec![
        Message::new(
            MessageRole::System,
            "sys",
            MessageMeta::new(MessageType::SystemPrompt),
        ),
        Message::new(
            MessageRole::User,
            "usr",
            MessageMeta::new(MessageType::UserInput),
        ),
    ];
    for step in 0..3 {
        messages.push(Message::new(
            MessageRole::Assistant,
            "thinking",
            MessageMeta::new(MessageType::Reasoning).with_step_index(step),
        ));
        messages.push(
            Message::new(
                MessageRole::Assistant,
                "",
                MessageMeta::new(MessageType::ToolCall).with_step_index(step),
            )
            .with_tool_calls(vec![ToolCallInfo::new(
                "run",
                Some(map_from_object(json!({ "x": step }))),
                format!("call_{step}"),
            )]),
        );
        messages.push(
            Message::new(
                MessageRole::Tool,
                "x".repeat(250),
                MessageMeta::new(MessageType::ToolResult).with_step_index(step),
            )
            .with_tool_name("run")
            .with_tool_call_id(format!("call_{step}")),
        );
        messages.push(Message::new(
            MessageRole::Assistant,
            "text",
            MessageMeta::new(MessageType::TextResponse).with_step_index(step),
        ));
        messages.push(Message::new(
            MessageRole::User,
            "retry",
            MessageMeta::new(MessageType::RetryNudge).with_step_index(step),
        ));
    }

    for (name, thresholds) in [
        ("phase1", [0.0, 100.0, 100.0]),
        ("phase2", [0.0, 0.0, 100.0]),
        ("phase3", [0.0, 0.0, 0.0]),
    ] {
        let strategy = TieredCompact::new(1).with_phase_thresholds(thresholds);
        let (result, phase) = strategy.compact(&messages, 100, None);
        assert_eq!(
            phase,
            case["expected"][name]["phase"].as_i64().expect("phase")
        );
        assert_eq!(
            messages_payload(&result),
            case["expected"][name]["messages"]
        );
    }
}

#[test]
fn python_golden_fold_and_serialize_reasoning_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "fold_and_serialize_reasoning");

    fn tool_call(call_id: &str, x: i64) -> Message {
        Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "run",
            Some(map_from_object(json!({ "x": x }))),
            call_id,
        )])
    }

    let cases = [
        (
            "before_tool_call",
            vec![
                Message::new(
                    MessageRole::Assistant,
                    "think first",
                    MessageMeta::new(MessageType::Reasoning),
                ),
                tool_call("call_000000000", 1),
            ],
        ),
        (
            "orphan_before_user",
            vec![
                Message::new(
                    MessageRole::Assistant,
                    "orphan",
                    MessageMeta::new(MessageType::Reasoning),
                ),
                Message::new(
                    MessageRole::User,
                    "next",
                    MessageMeta::new(MessageType::UserInput),
                ),
            ],
        ),
        (
            "consecutive_before_tool_call",
            vec![
                Message::new(
                    MessageRole::Assistant,
                    "first",
                    MessageMeta::new(MessageType::Reasoning),
                ),
                Message::new(
                    MessageRole::Assistant,
                    "second",
                    MessageMeta::new(MessageType::Reasoning),
                ),
                tool_call("call_000000001", 2),
            ],
        ),
    ];

    for (name, messages) in cases {
        assert_eq!(
            Value::Array(fold_and_serialize(&messages, "openai")),
            case["expected"][name]
        );
    }
}

#[test]
fn python_golden_pending_steps_only_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "pending_steps_only");
    let mut enforcer = StepEnforcer::new(
        vec!["lookup".to_string(), "analyze".to_string()],
        indexmap::IndexSet::from(["respond".to_string()]),
        None,
        3,
        2,
    );
    enforcer.record("lookup", None);
    assert_eq!(json!(enforcer.pending()), case["expected"]["pending"]);
}

#[tokio::test]
async fn python_golden_max_iterations_pending_error_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "max_iterations_pending_error");
    let expected = &case["expected"];

    let mut tools = IndexMap::new();
    tools.insert(
        "lookup".to_string(),
        ToolDef::new(
            empty_spec("lookup"),
            lookup_tool as fn(Vec<String>) -> Result<String, ToolResolutionError>,
        ),
    );
    tools.insert(
        "analyze".to_string(),
        ToolDef::new(
            empty_spec("analyze"),
            analyze_tool as fn(Vec<String>) -> Result<String, ToolResolutionError>,
        ),
    );
    tools.insert("respond".to_string(), respond_tool_def());
    let workflow = parity_workflow(tools, vec!["lookup", "analyze"]);
    let (result, messages, _calls) = run_workflow_capture(
        workflow,
        vec![LLMResponse::ToolCalls(vec![scripted_call(
            "lookup",
            json!({}),
        )])],
        1,
        2,
    )
    .await;

    match result {
        Err(ForgeError::MaxIterations(err)) => {
            assert_eq!(expected["error_type"], "MaxIterationsError");
            let completed: Vec<String> = err.completed_steps.keys().cloned().collect();
            assert_eq!(json!(completed), expected["completed"]);
            assert_eq!(json!(err.pending_steps), expected["pending"]);
        }
        other => panic!("expected MaxIterationsError, got {other:?}"),
    }
    assert_eq!(messages_payload(&messages), expected["messages"]);
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
async fn python_golden_llamafile_malformed_args_matches() {
    let fixtures = golden();
    let case = golden_case(&fixtures, "llamafile_malformed_args");
    let mut server = mockito::Server::new_async().await;
    let url = server.url();

    let _mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "choices": case["input"]["choices"].clone(),
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
        .send(vec![json!({"role": "user", "content": "run"})], None, None)
        .await
        .expect("llamafile response parsed");

    match response {
        LLMResponse::Text(text) => assert_eq!(
            text.content,
            case["expected"]["content"].as_str().expect("content")
        ),
        other => panic!("expected text response, got {other:?}"),
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
