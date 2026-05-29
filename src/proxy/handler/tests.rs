use super::*;
use crate::clients::base::{
    ApiFormat, ChunkStream, ChunkType, LLMRequestOptions, LLMResponseEnvelope, LLMUsageDetails,
    SamplingParams, StreamChunk, TokenUsage, ToolCall,
};
use crate::clients::AnthropicClient;
use crate::core::tool_spec::ToolSpec;
use crate::{
    ClassifierAction, FinalResponseClass, FinalResponseContext, FinalResponseScore, ToolCallClass,
    ToolCallPolicyConfig, ToolCallScore, ToolCallScorer, ToolOutputCompressionConfig,
    ToolOutputCompressionMode, ToolOutputCompressionState,
};
use anyllm_translate::anthropic::MessageCreateRequest;
use indexmap::IndexMap;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn filter_respond_removes_respond() {
    let calls = vec![
        ToolCall::new("respond", {
            let mut m = IndexMap::new();
            m.insert("message".into(), json!("hi"));
            m
        }),
        ToolCall::new("search", IndexMap::new()),
    ];
    let filtered = filter_respond(&calls);
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].tool, "search");
}

#[test]
fn filter_respond_keeps_all_real() {
    let calls = vec![
        ToolCall::new("search", IndexMap::new()),
        ToolCall::new("read", IndexMap::new()),
    ];
    let filtered = filter_respond(&calls);
    assert_eq!(filtered.len(), 2);
}

#[test]
fn process_response_text_non_streaming() {
    let resp = LLMResponse::Text(TextResponse::new("hello"));
    let result = process_response(&resp, "model", false);
    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "hello");
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
        }
        _ => panic!("expected Response"),
    }
}

#[tokio::test]
async fn classifier_log_sink_writes_one_event_per_line() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "forge-proxy-classifier-{}-{}.jsonl",
        std::process::id(),
        super::classifier_log::unix_ms()
    ));
    std::fs::remove_file(&path).ok();
    let first = json!({"kind": "tool_call", "tool": "search"});
    let second = json!({"kind": "tool_call", "tool": "read"});

    let sink = super::classifier_log::ClassifierLogSink::spawn(
        super::classifier_log::ClassifierLogConfig::new(path.clone()).with_queue_capacity(4),
    )
    .expect("sink");
    sink.emit(first.clone());
    sink.emit(second.clone());
    sink.flush().await.expect("flush");

    let text = std::fs::read_to_string(&path).expect("jsonl");
    let rows = text
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("json row"))
        .collect::<Vec<_>>();
    assert_eq!(rows, vec![first, second]);
    std::fs::remove_file(path).ok();
}

#[tokio::test]
async fn classifier_log_sink_redacts_sensitive_payload_fields() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "forge-proxy-classifier-redacted-{}-{}.jsonl",
        std::process::id(),
        super::classifier_log::unix_ms()
    ));
    std::fs::remove_file(&path).ok();
    let event = json!({
        "kind": "final_response",
        "user_request": "secret request",
        "initial_user_request": "initial secret",
        "workflow_state": {"recent_errors": ["secret error"]},
        "required_facts": ["secret fact"],
        "candidate_call": {"name": "search", "arguments": {"q": "secret query"}},
        "tool_results": [{"tool_name": "search", "content": "secret result"}],
        "candidate_final_response": "secret answer",
        "label": "valid_final_response"
    });

    let sink = super::classifier_log::ClassifierLogSink::spawn(
        super::classifier_log::ClassifierLogConfig::new(path.clone()).with_redaction(true),
    )
    .expect("sink");
    sink.emit(event);
    sink.flush().await.expect("flush");

    let text = std::fs::read_to_string(&path).expect("jsonl");
    assert!(!text.contains("secret"));
    let row = serde_json::from_str::<Value>(text.trim()).expect("json row");
    assert_eq!(row["user_request"], "[redacted]");
    assert_eq!(row["candidate_call"]["arguments"], "[redacted]");
    assert_eq!(row["tool_results"][0]["content"], "[redacted]");
    std::fs::remove_file(path).ok();
}

#[tokio::test]
async fn classifier_log_sink_drops_oversized_events() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "forge-proxy-classifier-oversized-{}-{}.jsonl",
        std::process::id(),
        super::classifier_log::unix_ms()
    ));
    std::fs::remove_file(&path).ok();
    let sink = super::classifier_log::ClassifierLogSink::spawn(
        super::classifier_log::ClassifierLogConfig::new(path.clone()).with_max_event_bytes(32),
    )
    .expect("sink");
    sink.emit(json!({"payload": "x".repeat(128)}));
    sink.flush().await.expect("flush");

    let text = std::fs::read_to_string(&path).expect("jsonl");
    assert!(text.is_empty());
    std::fs::remove_file(path).ok();
}

async fn collect_stream_events(result: HandlerResult) -> Vec<Value> {
    match result {
        HandlerResult::StreamBody(stream) => collect_openai_events(stream).await.unwrap(),
        other => panic!("expected StreamBody, got {other:?}"),
    }
}

fn stream_from_response(response: LLMResponse) -> ChunkStream {
    Box::pin(futures_util::stream::iter(vec![Ok(StreamChunk::new(
        ChunkType::Final,
    )
    .with_response(response))]))
}

#[tokio::test]
async fn process_response_text_streaming() {
    let resp = LLMResponse::Text(TextResponse::new("hello"));
    let result = process_response(&resp, "model", true);
    let events = collect_stream_events(result).await;
    assert!(!events.is_empty());
    let last = events.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

#[test]
fn process_response_tool_calls_non_streaming() {
    let calls = vec![ToolCall::new("search", IndexMap::new())];
    let resp = LLMResponse::ToolCalls(calls);
    let result = process_response(&resp, "model", false);
    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        }
        _ => panic!("expected Response"),
    }
}

#[test]
fn process_response_empty_tool_calls() {
    let resp = LLMResponse::ToolCalls(vec![]);
    let result = process_response(&resp, "model", false);
    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "");
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
        }
        _ => panic!("expected Response"),
    }
}

#[test]
fn parse_tool_specs_basic() {
    let schema = json!({
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    });
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search things",
            "parameters": schema.clone()
        }
    })];
    let specs = parse_tool_specs(&tools).expect("valid tools");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "search");
    assert_eq!(specs[0].get_json_schema(), schema);
}

#[test]
fn parse_tool_specs_empty() {
    let specs = parse_tool_specs(&[]).expect("empty tools");
    assert!(specs.is_empty());
}

#[test]
fn parse_tool_specs_accepts_missing_parameters_as_no_args() {
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "ping",
            "description": "Ping"
        }
    })];
    let specs = parse_tool_specs(&tools).expect("missing parameters is no-arg tool");
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "ping");
    assert_eq!(
        specs[0].get_json_schema(),
        json!({"type": "object", "properties": {}})
    );
}

#[test]
fn parse_tool_specs_rejects_malformed_tools() {
    let cases = [
        (
            json!({"type": "custom", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}),
            "must be a function tool",
        ),
        (
            json!({"type": "function", "function": {"name": "", "parameters": {"type": "object", "properties": {}}}}),
            "must not be empty",
        ),
        (
            json!({"type": "function", "function": {"name": "search", "parameters": {"type": "array"}}}),
            "must have type 'object'",
        ),
        (
            json!({"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": []}}}),
            "invalid schema",
        ),
    ];

    for (tool, expected) in cases {
        let err = parse_tool_specs(&[tool]).expect_err("invalid tool");
        assert!(
            err.message().contains(expected),
            "expected '{expected}' in '{}'",
            err.message()
        );
    }
}

#[test]
fn parse_tool_specs_rejects_duplicate_names() {
    let tools = vec![
        json!({"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}),
        json!({"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}),
    ];
    let err = parse_tool_specs(&tools).expect_err("duplicate rejected");
    assert!(err.message().contains("duplicates tool 'search'"));
}

#[test]
fn parse_tool_specs_rejects_reserved_respond_name() {
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "respond",
            "description": "Client-owned respond",
            "parameters": {"type": "object", "properties": {}}
        }
    })];
    let err = parse_tool_specs(&tools).expect_err("reserved name rejected");
    assert!(err.message().contains("tool name 'respond' is reserved"));
}

#[test]
fn extract_sampling_from_body() {
    let body = json!({
        "messages": [],
        "temperature": 0.7,
        "top_p": 0.9,
        "seed": 42
    });
    let s = extract_sampling(&body).unwrap();
    assert_eq!(s["temperature"], 0.7);
    assert_eq!(s["seed"], 42);
}

#[test]
fn extract_sampling_no_sampling_fields() {
    let body = json!({"messages": []});
    assert!(extract_sampling(&body).is_none());
}

// Integration-style tests for handle_chat_completions with a mock client.
struct MockTextClient;

impl LLMClient for MockTextClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Ok(LLMResponse::Text(TextResponse::new("mock response")))
    }
    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
            "mock response",
        ))))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct MockOptionsClient {
    last_options: std::sync::Mutex<Option<LLMRequestOptions>>,
    usage: Option<TokenUsage>,
    usage_details: Option<LLMUsageDetails>,
}

struct MockStreamingOptionsClient {
    send_calls: AtomicUsize,
    stream_calls: AtomicUsize,
}

struct IsolatedUsageClient;

struct StreamMetadataClient;

impl LLMClient for IsolatedUsageClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<TokenUsage> {
        Some(TokenUsage::new(999, 999, 1998))
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Ok(LLMResponse::Text(TextResponse::new("unused")))
    }

    async fn send_envelope_with_options(
        &self,
        messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _options: LLMRequestOptions,
    ) -> Result<LLMResponseEnvelope, crate::error::BackendError> {
        let content = messages
            .first()
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        tokio::time::sleep(Duration::from_millis(25)).await;
        let prompt_tokens = if content == "first" { 1 } else { 2 };
        Ok(LLMResponseEnvelope {
            response: LLMResponse::Text(TextResponse::new(content)),
            usage: Some(TokenUsage::new(prompt_tokens, 10, prompt_tokens + 10)),
            usage_details: None,
            call_info: None,
        })
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Err(crate::error::StreamError::new("not implemented"))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

impl LLMClient for StreamMetadataClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<TokenUsage> {
        Some(TokenUsage::new(999, 999, 1998))
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Ok(LLMResponse::Text(TextResponse::new("unused")))
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Err(crate::error::StreamError::new("use options"))
    }

    async fn send_stream_with_options(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _options: LLMRequestOptions,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Ok(Box::pin(futures_util::stream::iter(vec![
            Ok(StreamChunk::new(ChunkType::TextDelta).with_content("streamed")),
            Ok(StreamChunk::new(ChunkType::Final)
                .with_response(LLMResponse::Text(TextResponse::new("streamed")))
                .with_metadata(Some(TokenUsage::new(4, 2, 6)), None, None)),
        ])))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

impl MockStreamingOptionsClient {
    fn new() -> Self {
        Self {
            send_calls: AtomicUsize::new(0),
            stream_calls: AtomicUsize::new(0),
        }
    }
}

impl LLMClient for MockStreamingOptionsClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        self.send_calls.fetch_add(1, Ordering::SeqCst);
        Ok(LLMResponse::Text(TextResponse::new("non-stream")))
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        self.send_calls.fetch_add(1, Ordering::SeqCst);
        self.send(messages, tools, options.sampling).await
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Err(crate::error::StreamError::new("use stream_with_options"))
    }

    async fn send_stream_with_options(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _options: LLMRequestOptions,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Box::pin(futures_util::stream::iter(vec![
            Ok(StreamChunk::new(ChunkType::TextDelta).with_content("first")),
            Ok(StreamChunk::new(ChunkType::Final)
                .with_response(LLMResponse::Text(TextResponse::new("first")))),
        ])))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

impl MockOptionsClient {
    fn new(usage: Option<TokenUsage>) -> Self {
        Self {
            last_options: std::sync::Mutex::new(None),
            usage,
            usage_details: None,
        }
    }

    fn new_with_details(usage: Option<TokenUsage>, usage_details: Option<LLMUsageDetails>) -> Self {
        Self {
            last_options: std::sync::Mutex::new(None),
            usage,
            usage_details,
        }
    }
}

impl LLMClient for MockOptionsClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<TokenUsage> {
        self.usage.clone()
    }

    fn last_usage_details(&self) -> Option<LLMUsageDetails> {
        self.usage_details.clone()
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Ok(LLMResponse::Text(TextResponse::new("options response")))
    }

    async fn send_with_options(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        *self.last_options.lock().unwrap() = Some(options);
        Ok(LLMResponse::Text(TextResponse::new("options response")))
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
            "options response",
        ))))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct MockToolCallClient;

struct SleepingToolScorer;

impl ToolCallScorer for SleepingToolScorer {
    fn score(
        &self,
        _ctx: &crate::ScoringContext,
        _candidate: &ToolCall,
    ) -> anyhow::Result<ToolCallScore> {
        std::thread::sleep(Duration::from_millis(150));
        Ok(ToolCallScore {
            label: ToolCallClass::Valid,
            confidence: 1.0,
            logits: vec![9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            action: ClassifierAction::Allow,
            model_version: "sleeping-test".to_string(),
            latency_ms: 150.0,
        })
    }
}

struct SequenceFinalScorer {
    calls: AtomicUsize,
}

impl SequenceFinalScorer {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
}

impl FinalResponseScorer for SequenceFinalScorer {
    fn score(&self, _ctx: &FinalResponseContext) -> anyhow::Result<FinalResponseScore> {
        let idx = self.calls.fetch_add(1, Ordering::SeqCst);
        if idx == 0 {
            Ok(FinalResponseScore {
                label: FinalResponseClass::MissingToolFact,
                confidence: 0.99,
                logits: vec![0.0, 9.0, 0.0, 0.0, 0.0],
                action: ClassifierAction::AdvisoryNudge,
                model_version: "fake-final".to_string(),
                latency_ms: 1.0,
            })
        } else {
            Ok(FinalResponseScore {
                label: FinalResponseClass::ValidFinalResponse,
                confidence: 1.0,
                logits: vec![9.0, 0.0, 0.0, 0.0, 0.0],
                action: ClassifierAction::Allow,
                model_version: "fake-final".to_string(),
                latency_ms: 1.0,
            })
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn proxy_tool_scoring_does_not_block_current_thread_runtime() {
    let messages = vec![Message::new(
        MessageRole::User,
        "search for the current status",
        MessageMeta::new(MessageType::UserInput),
    )];
    let tool_specs = vec![ToolSpec::from_json_schema(
        "search",
        "Search for information.",
        &json!({"type": "object", "properties": {}}),
    )
    .expect("tool spec")];
    let tool_calls = vec![ToolCall::new("search", IndexMap::new())];
    let scorer = Arc::new(SleepingToolScorer);

    let scoring = score_proxy_tool_calls(Some(scorer), &messages, &tool_calls, None, &tool_specs);
    tokio::pin!(scoring);
    let early = tokio::time::timeout(Duration::from_millis(25), &mut scoring).await;

    assert!(
        early.is_err(),
        "scoring should be running on a blocking worker"
    );
    assert!(scoring.await.is_none());
}

impl LLMClient for MockToolCallClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        let mut args = IndexMap::new();
        args.insert("message".into(), json!("responded text"));
        Ok(LLMResponse::ToolCalls(vec![ToolCall::new("respond", args)]))
    }
    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        let mut args = IndexMap::new();
        args.insert("message".into(), json!("responded text"));
        Ok(stream_from_response(LLMResponse::ToolCalls(vec![
            ToolCall::new("respond", args),
        ])))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct MockPassthroughToolCallClient;

impl LLMClient for MockPassthroughToolCallClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Ok(LLMResponse::ToolCalls(vec![ToolCall::new(
            "search",
            IndexMap::new(),
        )]))
    }
    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Ok(stream_from_response(LLMResponse::ToolCalls(vec![
            ToolCall::new("search", IndexMap::new()),
        ])))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

fn dummy_ctx() -> ContextManager {
    ContextManager::new(
        Box::new(crate::context::strategies::NoCompact),
        4096,
        None,
        None,
        None,
    )
}

fn search_tool_json() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        }
    })
}

fn bash_tool_json() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    })
}

fn grep_tool_json() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files with grep",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"]
            }
        }
    })
}

fn find_definition_tool_json() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": "Find a symbol definition",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"]
            }
        }
    })
}

#[tokio::test]
async fn handle_no_tools_passthrough() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false
    });
    let client = Arc::new(MockTextClient);
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
    match result.unwrap() {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "mock response");
        }
        _ => panic!("expected Response"),
    }
}

#[tokio::test]
async fn handle_no_tools_forwards_passthrough_options_and_usage() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "stream": false,
        "max_tokens": 128,
        "stop": ["done"],
        "tool_choice": {"type": "function", "function": {"name": "search"}},
        "response_format": {"type": "json_object"},
        "temperature": 0.7
    });
    let client = Arc::new(MockOptionsClient::new(Some(TokenUsage::new(11, 5, 16))));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;

    match result.unwrap() {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "options response");
            assert_eq!(v["usage"]["prompt_tokens"], 11);
            assert_eq!(v["usage"]["completion_tokens"], 5);
            assert_eq!(v["usage"]["total_tokens"], 16);
        }
        _ => panic!("expected Response"),
    }

    let options = client
        .last_options
        .lock()
        .unwrap()
        .clone()
        .expect("options recorded");
    let passthrough = options.passthrough.expect("passthrough");
    assert_eq!(passthrough["model"], "request-model");
    assert_eq!(passthrough["max_tokens"], 128);
    assert_eq!(passthrough["stop"], json!(["done"]));
    assert_eq!(
        passthrough["tool_choice"],
        json!({"type": "function", "function": {"name": "search"}})
    );
    assert_eq!(
        passthrough["response_format"],
        json!({"type": "json_object"})
    );
    assert!(!passthrough.contains_key("messages"));
    assert!(!passthrough.contains_key("stream"));
    assert!(!passthrough.contains_key("temperature"));
    assert!(!passthrough.contains_key("_forge"));
    assert!(options.inbound_anthropic_body.is_none());
}

#[tokio::test]
async fn handle_no_tools_uses_envelope_usage_per_concurrent_response() {
    let first_body = json!({
        "messages": [{"role": "user", "content": "first"}],
        "model": "request-model"
    });
    let second_body = json!({
        "messages": [{"role": "user", "content": "second"}],
        "model": "request-model"
    });
    let client = Arc::new(IsolatedUsageClient);
    let first_ctx = Arc::new(Mutex::new(dummy_ctx()));
    let second_ctx = Arc::new(Mutex::new(dummy_ctx()));

    let (first, second) = tokio::join!(
        handle_chat_completions(&first_body, &client, &first_ctx, 3, true),
        handle_chat_completions(&second_body, &client, &second_ctx, 3, true),
    );

    let first = match first.expect("first response") {
        HandlerResult::Response(value) => value,
        _ => panic!("expected first response"),
    };
    let second = match second.expect("second response") {
        HandlerResult::Response(value) => value,
        _ => panic!("expected second response"),
    };
    assert_eq!(first["usage"]["prompt_tokens"], 1);
    assert_eq!(second["usage"]["prompt_tokens"], 2);
}

#[tokio::test]
async fn handle_no_tools_stream_usage_requires_include_usage_and_is_final_only() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let client = Arc::new(MockOptionsClient::new(Some(TokenUsage::new(11, 5, 16))));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let events = collect_stream_events(
        handle_chat_completions(&body, &client, &ctx, 3, true)
            .await
            .expect("handler result"),
    )
    .await;

    let usage_events: Vec<&Value> = events
        .iter()
        .filter(|event| event.get("usage").is_some())
        .collect();
    assert_eq!(usage_events.len(), 1);
    assert_eq!(usage_events[0]["choices"][0]["finish_reason"], "stop");
    assert_eq!(usage_events[0]["usage"]["total_tokens"], 16);

    let body_without_usage = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "stream": true
    });
    let events = collect_stream_events(
        handle_chat_completions(&body_without_usage, &client, &ctx, 3, true)
            .await
            .expect("handler result"),
    )
    .await;
    assert!(events.iter().all(|event| event.get("usage").is_none()));
}

#[tokio::test]
async fn handle_no_tools_stream_uses_final_chunk_usage_before_last_usage() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "stream": true,
        "stream_options": {"include_usage": true}
    });
    let client = Arc::new(StreamMetadataClient);
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let events = collect_stream_events(
        handle_chat_completions(&body, &client, &ctx, 3, true)
            .await
            .expect("handler result"),
    )
    .await;

    let usage_event = events
        .iter()
        .find(|event| event.get("usage").is_some())
        .expect("usage event");
    assert_eq!(usage_event["usage"]["prompt_tokens"], 4);
    assert_eq!(usage_event["usage"]["completion_tokens"], 2);
    assert_eq!(usage_event["usage"]["total_tokens"], 6);
}

#[tokio::test]
async fn handle_no_tools_rejects_required_steps_contract() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "_forge": {"required_steps": ["search"]}
    });
    let client = Arc::new(MockOptionsClient::new(None));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let err = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect_err("required steps without tools");
    assert!(matches!(err, HandlerError::BadRequest(_)));
    assert!(err.message().contains("requires tools"));
}

#[tokio::test]
async fn proxy_step_contract_rejects_invalid_names() {
    let cases = [
        (
            json!({"required_steps": ["missing"]}),
            "required_steps contains unknown tool 'missing'",
        ),
        (
            json!({"required_steps": [""]}),
            "required_steps contains an empty tool name",
        ),
        (
            json!({"required_steps": ["search", "search"]}),
            "required_steps contains duplicate tool 'search'",
        ),
        (
            json!({"required_steps": ["search"], "terminal_tools": ["finish"]}),
            "terminal_tools contains unknown tool 'finish'",
        ),
        (
            json!({"required_steps": ["search"], "terminal_tools": [""]}),
            "terminal_tools contains an empty tool name",
        ),
        (
            json!({"required_steps": ["search"], "terminal_tools": ["respond", "respond"]}),
            "terminal_tools contains duplicate tool 'respond'",
        ),
        (
            json!({"required_steps": ["search"], "terminal_tools": ["search"]}),
            "terminal_tools contains required step 'search'",
        ),
    ];

    let client = Arc::new(MockOptionsClient::new(None));
    for (forge, expected) in cases {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "request-model",
            "tools": [search_tool_json()],
            "_forge": forge
        });
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let err = handle_chat_completions(&body, &client, &ctx, 3, true)
            .await
            .expect_err("invalid _forge contract");
        assert!(matches!(err, HandlerError::BadRequest(_)));
        assert!(
            err.message().contains(expected),
            "expected '{expected}' in '{}'",
            err.message()
        );
    }
}

#[tokio::test]
async fn handle_no_tools_emits_cache_usage_details() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "stream": false
    });
    let details = LLMUsageDetails {
        cached_prompt_tokens: Some(8),
        prompt_cache_hit_tokens: Some(8),
        prompt_cache_miss_tokens: Some(3),
        cache_miss_prompt_tokens: Some(3),
        ..Default::default()
    };
    let client = Arc::new(MockOptionsClient::new_with_details(
        Some(TokenUsage::new(11, 5, 16)),
        Some(details),
    ));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;

    match result.unwrap() {
        HandlerResult::Response(v) => {
            assert_eq!(v["usage"]["prompt_tokens"], 11);
            assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 8);
            assert_eq!(v["usage"]["prompt_cache_hit_tokens"], 8);
            assert_eq!(v["usage"]["prompt_cache_miss_tokens"], 3);
        }
        _ => panic!("expected Response"),
    }
}

struct MockRespondOptionsClient {
    last_options: std::sync::Mutex<Option<LLMRequestOptions>>,
}

impl MockRespondOptionsClient {
    fn new() -> Self {
        Self {
            last_options: std::sync::Mutex::new(None),
        }
    }
}

impl LLMClient for MockRespondOptionsClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        let mut args = IndexMap::new();
        args.insert("message".into(), json!("done"));
        Ok(LLMResponse::ToolCalls(vec![ToolCall::new("respond", args)]))
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        *self.last_options.lock().unwrap() = Some(options.clone());
        self.send(messages, tools, options.sampling).await
    }

    async fn send_stream(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
            "done",
        ))))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn handle_tools_forwards_prompt_cache_passthrough_fields() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "stream": false,
        "prompt_cache_key": "tenant-a-tools-v1",
        "prompt_cache_retention": "24h",
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
    let client = Arc::new(MockRespondOptionsClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "done");
        }
        _ => panic!("expected Response"),
    }

    let options = client
        .last_options
        .lock()
        .unwrap()
        .clone()
        .expect("options recorded");
    let passthrough = options.passthrough.expect("passthrough");
    assert_eq!(passthrough["prompt_cache_key"], "tenant-a-tools-v1");
    assert_eq!(passthrough["prompt_cache_retention"], "24h");
}

#[tokio::test]
async fn handle_tools_rejects_tool_choice_none() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "tool_choice": "none",
        "tools": [search_tool_json()]
    });
    let client = Arc::new(MockRespondOptionsClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let err = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect_err("tool_choice none rejected");
    assert!(matches!(err, HandlerError::BadRequest(_)));
    assert!(err.message().contains("tool_choice=none"));
}

#[tokio::test]
async fn handle_required_steps_rejects_forced_tool_choice() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "tool_choice": {"type": "function", "function": {"name": "search"}},
        "tools": [search_tool_json()],
        "_forge": {
            "required_steps": ["search"],
            "terminal_tools": ["respond"]
        }
    });
    let client = Arc::new(MockRespondOptionsClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let err = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect_err("forced tool choice rejected");
    assert!(matches!(err, HandlerError::BadRequest(_)));
    assert!(err.message().contains("forced tool_choice"));
}

#[tokio::test]
async fn handle_tools_strips_response_format_from_passthrough() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "response_format": {"type": "json_object"},
        "prompt_cache_key": "tenant-a-tools-v1",
        "tools": [search_tool_json()]
    });
    let client = Arc::new(MockRespondOptionsClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler");

    let options = client
        .last_options
        .lock()
        .unwrap()
        .clone()
        .expect("options recorded");
    let passthrough = options.passthrough.expect("passthrough");
    assert_eq!(passthrough["prompt_cache_key"], "tenant-a-tools-v1");
    assert!(!passthrough.contains_key("response_format"));
}

#[test]
fn guarded_anthropic_body_rejects_incompatible_tool_choice() {
    let err = sanitize_guarded_anthropic_body(
        Some(Arc::new(json!({"tool_choice": {"type": "none"}}))),
        false,
    )
    .expect_err("tool_choice none rejected");
    assert!(err.message().contains("tool_choice=none"));

    let err = sanitize_guarded_anthropic_body(
        Some(Arc::new(
            json!({"tool_choice": {"type": "tool", "name": "search"}}),
        )),
        true,
    )
    .expect_err("forced tool choice rejected");
    assert!(err.message().contains("forced tool_choice"));
}

#[tokio::test]
async fn handle_no_tools_streaming_uses_stream_client() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "request-model",
        "stream": true,
        "temperature": 0.7
    });
    let client = Arc::new(MockStreamingOptionsClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    assert_eq!(client.send_calls.load(Ordering::SeqCst), 0);
    assert_eq!(client.stream_calls.load(Ordering::SeqCst), 1);

    let events = collect_stream_events(result).await;
    assert_eq!(events[0]["choices"][0]["delta"]["content"], "first");
    assert_eq!(
        events.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[tokio::test]
async fn anthropic_no_tools_streaming_uses_stream_client() {
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
    });
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let client = Arc::new(MockStreamingOptionsClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    assert_eq!(client.send_calls.load(Ordering::SeqCst), 0);
    assert_eq!(client.stream_calls.load(Ordering::SeqCst), 1);

    let events = match result {
        AnthropicHandlerResult::StreamBody(stream) => {
            collect_anthropic_events(stream).await.expect("events")
        }
        other => panic!("expected StreamBody, got {other:?}"),
    };
    let body = crate::proxy::server::format_anthropic_sse_body(events.as_slice());
    assert!(body.contains("event: message_start"));
    assert!(body.contains("event: content_block_delta"));
    assert!(body.contains("first"));
    assert!(!body.contains("[DONE]"));
}

#[tokio::test]
async fn anthropic_messages_translates_nonzero_usage() {
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let client = Arc::new(MockOptionsClient::new(Some(TokenUsage::new(13, 7, 20))));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true).await;

    match result.unwrap() {
        AnthropicHandlerResult::Response(v) => {
            assert_eq!(v["content"][0]["text"], "options response");
            assert_eq!(v["usage"]["input_tokens"], 13);
            assert_eq!(v["usage"]["output_tokens"], 7);
        }
        _ => panic!("expected Response"),
    }

    let options = client
        .last_options
        .lock()
        .unwrap()
        .clone()
        .expect("options recorded");
    assert_eq!(options.inbound_anthropic_body.as_deref(), Some(&raw));
}

#[tokio::test]
async fn anthropic_messages_strips_forge_extension_before_upstream_forwarding() {
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "_forge": {"tool_output_compression": "disabled"}
    });
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let client = Arc::new(MockOptionsClient::new(Some(TokenUsage::new(13, 7, 20))));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    let options = client
        .last_options
        .lock()
        .unwrap()
        .clone()
        .expect("options recorded");
    let forwarded = options
        .inbound_anthropic_body
        .as_deref()
        .expect("raw anthropic body");
    assert!(forwarded.get("_forge").is_none());
    assert_eq!(forwarded["model"], "claude-3");
}

#[tokio::test]
async fn anthropic_messages_includes_cache_usage_details() {
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let details = LLMUsageDetails {
        cached_prompt_tokens: Some(13),
        cache_creation_prompt_tokens: Some(5),
        cache_read_input_tokens: Some(13),
        cache_creation_input_tokens: Some(5),
        ..Default::default()
    };
    let client = Arc::new(MockOptionsClient::new_with_details(
        Some(TokenUsage::new(20, 7, 27)),
        Some(details),
    ));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true).await;

    match result.unwrap() {
        AnthropicHandlerResult::Response(v) => {
            assert_eq!(v["usage"]["input_tokens"], 20);
            assert_eq!(v["usage"]["output_tokens"], 7);
            assert_eq!(v["usage"]["cache_read_input_tokens"], 13);
            assert_eq!(v["usage"]["cache_creation_input_tokens"], 5);
        }
        _ => panic!("expected Response"),
    }
}

#[tokio::test]
async fn anthropic_messages_clean_path_preserves_cache_control_to_backend() {
    let mut server = mockito::Server::new_async().await;
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": {"type": "ephemeral"}
            }]
        }]
    });
    let mock = server
        .mock("POST", "/messages")
        .match_body(mockito::Matcher::Json(raw.clone()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 3, "output_tokens": 1}
            })
            .to_string(),
        )
        .create_async()
        .await;
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let client = Arc::new(
        AnthropicClient::new("fallback-model", Some("test-key".to_string()))
            .with_base_url(server.url())
            .with_timeout(5.0),
    );
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true).await;

    match result.unwrap() {
        AnthropicHandlerResult::Response(v) => {
            assert_eq!(v["content"][0]["text"], "ok");
            assert_eq!(v["usage"]["input_tokens"], 3);
            assert_eq!(v["usage"]["output_tokens"], 1);
        }
        _ => panic!("expected Response"),
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn anthropic_messages_with_tools_injects_respond_to_raw_backend_body() {
    let mut server = mockito::Server::new_async().await;
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "name": "search",
            "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }
        }]
    });
    let mut expected = raw.clone();
    expected["tools"].as_array_mut().expect("tools").push(
        crate::clients::anthropic::convert::convert_tools(&[crate::tools::respond::respond_spec()])
            [0]
        .clone(),
    );
    let mock = server
        .mock("POST", "/messages")
        .match_body(mockito::Matcher::Json(expected))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-3",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_respond",
                    "name": "respond",
                    "input": {"message": "ok"}
                }],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 3, "output_tokens": 1}
            })
            .to_string(),
        )
        .create_async()
        .await;
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let client = Arc::new(
        AnthropicClient::new("fallback-model", Some("test-key".to_string()))
            .with_base_url(server.url())
            .with_timeout(5.0),
    );
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true).await;

    match result.unwrap() {
        AnthropicHandlerResult::Response(v) => {
            assert_eq!(v["content"][0]["text"], "ok");
        }
        _ => panic!("expected Response"),
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn anthropic_messages_streaming_preserves_cache_control_to_backend() {
    let mut server = mockito::Server::new_async().await;
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "stream": true,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": {"type": "ephemeral"}
            }]
        }]
    });
    let sse = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
    let mock = server
        .mock("POST", "/messages")
        .match_body(mockito::Matcher::Json(raw.clone()))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse)
        .create_async()
        .await;
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let client = Arc::new(
        AnthropicClient::new("fallback-model", Some("test-key".to_string()))
            .with_base_url(server.url())
            .with_timeout(5.0),
    );
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    let events = match result {
        AnthropicHandlerResult::StreamBody(stream) => {
            collect_anthropic_events(stream).await.expect("events")
        }
        other => panic!("expected StreamBody, got {other:?}"),
    };
    let body = crate::proxy::server::format_anthropic_sse_body(events.as_slice());
    assert!(body.contains("ok"));
    assert!(!body.contains("[DONE]"));
    mock.assert_async().await;
}

#[tokio::test]
async fn handle_no_tools_tool_calls_return_upstream_error() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false
    });
    let client = Arc::new(MockPassthroughToolCallClient);
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
    let err = result.expect_err("unexpected tool calls should fail");
    assert!(matches!(err, HandlerError::Upstream(_)));
    assert!(err.message().contains("without tools"));
}

#[tokio::test]
async fn handle_no_tools_tool_calls_return_stream_error() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": true
    });
    let client = Arc::new(MockPassthroughToolCallClient);
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("stream body");
    let HandlerResult::StreamBody(stream) = result else {
        panic!("expected stream body");
    };
    let err = collect_openai_events(stream)
        .await
        .expect_err("unexpected tool calls should fail stream");
    assert!(err.to_string().contains("without tools"));
}

#[tokio::test]
async fn handle_tools_respond_stripped() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false,
        "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    });
    let client = Arc::new(MockToolCallClient);
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
    match result.unwrap() {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "responded text");
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
        }
        _ => panic!("expected Response"),
    }
}

struct MockWorkflowContractClient {
    responses: Vec<LLMResponse>,
    calls: std::sync::Mutex<usize>,
    sent_messages: std::sync::Mutex<Vec<Vec<Value>>>,
    sent_tools: std::sync::Mutex<Vec<Vec<String>>>,
}

impl MockWorkflowContractClient {
    fn new(responses: Vec<LLMResponse>) -> Self {
        Self {
            responses,
            calls: std::sync::Mutex::new(0),
            sent_messages: std::sync::Mutex::new(Vec::new()),
            sent_tools: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn sent_messages(&self) -> Vec<Vec<Value>> {
        self.sent_messages.lock().unwrap().clone()
    }

    fn sent_tools(&self) -> Vec<Vec<String>> {
        self.sent_tools.lock().unwrap().clone()
    }
}

impl LLMClient for MockWorkflowContractClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        self.sent_messages.lock().unwrap().push(messages);
        self.sent_tools.lock().unwrap().push(
            tools
                .unwrap_or_default()
                .into_iter()
                .map(|tool| tool.name)
                .collect(),
        );
        let mut calls = self.calls.lock().unwrap();
        let response =
            self.responses.get(*calls).cloned().unwrap_or_else(|| {
                panic!("MockWorkflowContractClient exhausted at call {}", *calls)
            });
        *calls += 1;
        Ok(response)
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        self.send(messages, tools, options.sampling).await
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        let response = self
            .send(messages, tools, sampling)
            .await
            .map_err(|err| crate::error::StreamError::new(err.to_string()))?;
        Ok(stream_from_response(response))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn tool_output_compression_default_disabled_leaves_prior_tool_result_unchanged() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let raw_tool_output = "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz";
    let body = json!({
        "messages": [
            {"role": "user", "content": "summarize previous search"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{\"query\":\"x\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_search",
                "name": "search",
                "content": raw_tool_output
            }
        ],
        "model": "test-model"
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    let sent = client.sent_messages();
    assert_eq!(sent[0][2]["content"], raw_tool_output);
}

#[tokio::test]
async fn tool_output_compression_request_disabled_overrides_process_default() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let raw_tool_output = "\u{1b}[31mOPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\u{1b}[0m";
    let body = json!({
        "messages": [
            {"role": "user", "content": "summarize previous search"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{\"query\":\"x\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_search",
                "name": "search",
                "content": raw_tool_output
            }
        ],
        "model": "test-model",
        "_forge": {
            "tool_output_compression": "disabled"
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Standard),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    let sent = client.sent_messages();
    assert_eq!(sent[0][2]["content"], raw_tool_output);
}

#[tokio::test]
async fn tool_output_compression_request_can_disable_secret_redaction_only() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let body = json!({
        "messages": [
            {"role": "user", "content": "summarize previous search"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{\"query\":\"x\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_search",
                "name": "search",
                "content": "\u{1b}[31mOPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\u{1b}[0m"
            }
        ],
        "model": "test-model",
        "_forge": {
            "tool_output_compression": {
                "mode": "safe",
                "redact_secrets": false
            }
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    let sent = client.sent_messages();
    let content = sent[0][2]["content"].as_str().expect("tool content");
    assert!(content.contains("sk-abcdefghijklmnopqrstuvwxyz"));
    assert!(!content.contains("[REDACTED_SECRET]"));
    assert!(!content.contains("\u{1b}[31m"));
}

#[tokio::test]
async fn tool_output_compression_opt_in_compresses_prior_tool_result_and_preserves_ids() {
    let mut response_args = IndexMap::new();
    response_args.insert("query".into(), json!("next"));
    let client = Arc::new(MockWorkflowContractClient::new(vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("search", response_args)]),
    ]));
    let body = json!({
        "messages": [
            {"role": "user", "content": "continue"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{\"query\":\"x\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_search",
                "name": "search",
                "content": "\u{1b}[31mOPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\u{1b}[0m"
            }
        ],
        "model": "test-model",
        "tools": [search_tool_json()],
        "_forge": {
            "tool_output_compression": {"mode": "safe"}
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    let sent = client.sent_messages();
    assert_eq!(sent[0][2]["tool_call_id"], "call_search");
    assert_eq!(sent[0][2]["name"], "search");
    let content = sent[0][2]["content"].as_str().expect("tool content");
    assert!(content.contains("[REDACTED_SECRET]"));
    assert!(!content.contains("\u{1b}[31m"));

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(
                v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "search"
            );
        }
        _ => panic!("expected Response"),
    }
}

#[tokio::test]
async fn tool_output_compression_opt_in_runs_for_no_tools_passthrough() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let body = json!({
        "messages": [
            {"role": "user", "content": "summarize previous output"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_shell",
                    "type": "function",
                    "function": {"name": "run_command", "arguments": "{\"command\":\"cargo test\"}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_shell",
                "content": "\u{1b}[31merror: failed\u{1b}[0m\nnoise\n"
            }
        ],
        "model": "test-model",
        "_forge": {
            "tool_output_compression": "standard"
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    let sent = client.sent_messages();
    let content = sent[0][2]["content"].as_str().expect("tool content");
    assert_eq!(sent[0][2]["tool_call_id"], "call_shell");
    assert!(content.contains("error: failed"));
    assert!(!content.contains("\u{1b}[31m"));
}

#[tokio::test]
async fn tool_output_compression_request_method_overrides_process_default() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let repeated = (0..30)
        .map(|idx| {
            format!(
                "error: repeated dependency resolution failure in workspace crate alpha at module_{idx}\n"
            )
        })
        .collect::<String>();
    let body = json!({
        "messages": [
            {"role": "user", "content": "summarize previous output"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_custom",
                    "type": "function",
                    "function": {"name": "custom_tool", "arguments": "{}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_custom",
                "name": "custom_tool",
                "content": repeated
            }
        ],
        "model": "test-model",
        "_forge": {
            "tool_output_compression": {"method": "repair"}
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Aggressive),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    let sent = client.sent_messages();
    let content = sent[0][2]["content"].as_str().expect("tool content");
    assert!(content.starts_with("[Forge RePair Dictionary]"));
    assert!(!content.starts_with("[Forge LZW Dictionary]"));
}

#[tokio::test]
async fn tool_output_compression_request_rejects_invalid_method() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let body = json!({
        "messages": [{"role": "user", "content": "summarize previous output"}],
        "model": "test-model",
        "_forge": {
            "tool_output_compression": {"method": "gzip"}
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let err = handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect_err("invalid method");

    assert!(matches!(err, HandlerError::BadRequest(_)));
    assert!(err
        .message()
        .contains("method must be lzw, repair, or auto"));
}

#[tokio::test]
async fn tool_output_compression_request_rejects_unbounded_max_output_bytes() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let body = json!({
        "messages": [{"role": "user", "content": "summarize previous output"}],
        "model": "test-model",
        "_forge": {
            "tool_output_compression": {
                "mode": "safe",
                "max_output_bytes": 1048577
            }
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let err = handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect_err("oversized max_output_bytes");

    assert!(matches!(err, HandlerError::BadRequest(_)));
    assert!(err.message().contains("max_output_bytes must be <="));
}

#[tokio::test]
async fn tool_output_compression_dedups_repeated_tool_results_by_session() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let repeated = (0..200)
        .map(|idx| format!("large unique output line {idx}"))
        .collect::<Vec<_>>()
        .join("\n");
    let body = json!({
        "messages": [
            {"role": "user", "content": "summarize prior output"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_one",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"custom-tool\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_one", "name": "bash", "content": repeated},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_two",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"custom-tool\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_two", "name": "bash", "content": repeated}
        ],
        "model": "test-model",
        "_forge": {
            "tool_output_compression": {
                "mode": "standard",
                "session_id": "session-a",
                "dedup": true
            }
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_chat_completions_with_scorers_and_tool_output_compression(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::from_mode(ToolOutputCompressionMode::Disabled),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    let sent = client.sent_messages();
    assert!(!sent[0][2]["content"]
        .as_str()
        .expect("first content")
        .contains("Duplicate of call"));
    assert!(sent[0][4]["content"]
        .as_str()
        .expect("second content")
        .contains("Duplicate of call #"));
}

#[tokio::test]
async fn anthropic_tool_output_compression_rebuilds_messages_when_content_changes() {
    let client = Arc::new(MockWorkflowContractClient::new(vec![LLMResponse::Text(
        TextResponse::new("ok"),
    )]));
    let raw_for_parse = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "summarize"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": "search",
                    "input": {"query": "x"}
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_search",
                    "content": "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
                }]
            }
        ]
    });
    let mut raw_with_forge = raw_for_parse.clone();
    raw_with_forge["_forge"] = json!({
        "tool_output_compression": {"mode": "safe"}
    });
    let body: MessageCreateRequest = serde_json::from_value(raw_for_parse).expect("request");
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    handle_anthropic_messages_with_scorers_and_tool_output_compression(
        &body,
        &raw_with_forge,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    let sent = client.sent_messages();
    let wire = serde_json::to_string(&sent[0]).expect("wire json");
    assert!(wire.contains("[REDACTED_SECRET]"));
    assert!(!wire.contains("sk-abcdefghijklmnopqrstuvwxyz"));
}

#[tokio::test]
async fn anthropic_tool_output_compression_patches_raw_body_and_preserves_cache_control() {
    let mut server = mockito::Server::new_async().await;
    let raw_for_parse = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "summarize",
                    "cache_control": {"type": "ephemeral"}
                }]
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_search",
                    "name": "search",
                    "input": {"query": "x"}
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_search",
                    "content": "sk-\u{1b}[31mabcdefghijklmnopqrstuvwxyz\u{1b}[0m",
                    "is_error": true
                }]
            }
        ],
        "metadata": {"user_id": "user-123"},
        "thinking": {"type": "disabled"}
    });
    let mut raw_with_forge = raw_for_parse.clone();
    raw_with_forge["_forge"] = json!({
        "tool_output_compression": {"mode": "safe"}
    });
    let mut expected = raw_for_parse.clone();
    expected["messages"][2]["content"][0]["content"] = json!("[REDACTED_SECRET]");

    let mock = server
        .mock("POST", "/messages")
        .match_body(mockito::Matcher::Json(expected))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            json!({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 3, "output_tokens": 1}
            })
            .to_string(),
        )
        .create_async()
        .await;
    let body: MessageCreateRequest = serde_json::from_value(raw_for_parse).expect("request");
    let client = Arc::new(
        AnthropicClient::new("fallback-model", Some("test-key".to_string()))
            .with_base_url(server.url())
            .with_timeout(5.0),
    );
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_anthropic_messages_with_scorers_and_tool_output_compression(
        &body,
        &raw_with_forge,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        Some(Arc::new(ToolOutputCompressionState::new())),
    )
    .await
    .expect("handler result");

    match result {
        AnthropicHandlerResult::Response(value) => {
            assert_eq!(value["content"][0]["text"], "ok");
        }
        _ => panic!("expected Response"),
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn tool_call_policy_lsp_nudge_fires_only_when_replacement_tool_exists() {
    let mut grep_args = IndexMap::new();
    grep_args.insert("pattern".into(), json!("UserService"));
    let mut definition_args = IndexMap::new();
    definition_args.insert("symbol".into(), json!("UserService"));
    let client = Arc::new(MockWorkflowContractClient::new(vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("grep", grep_args.clone())]),
        LLMResponse::ToolCalls(vec![ToolCall::new("find_definition", definition_args)]),
    ]));
    let body = json!({
        "messages": [{"role": "user", "content": "find UserService"}],
        "model": "test-model",
        "tools": [grep_tool_json(), find_definition_tool_json()],
        "_forge": {
            "tool_call_policy": {"lsp_first": true}
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions_with_scorers_and_tool_controls(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        None,
        ToolCallPolicyConfig::disabled(),
    )
    .await
    .expect("handler result");

    match result {
        HandlerResult::Response(value) => {
            assert_eq!(
                value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "find_definition"
            );
        }
        _ => panic!("expected Response"),
    }
    assert_eq!(*client.calls.lock().unwrap(), 2);
    let sent = serde_json::to_string(&client.sent_messages()).expect("sent messages");
    assert!(sent.contains("[ToolCallPolicyNudge]"));
    assert!(sent.contains("find_definition"));

    let client = Arc::new(MockWorkflowContractClient::new(vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("grep", grep_args)]),
    ]));
    let body = json!({
        "messages": [{"role": "user", "content": "find UserService"}],
        "model": "test-model",
        "tools": [grep_tool_json()],
        "_forge": {
            "tool_call_policy": {"lsp_first": true}
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions_with_scorers_and_tool_controls(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        None,
        ToolCallPolicyConfig::disabled(),
    )
    .await
    .expect("handler result");

    match result {
        HandlerResult::Response(value) => {
            assert_eq!(
                value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "grep"
            );
        }
        _ => panic!("expected Response"),
    }
    assert_eq!(*client.calls.lock().unwrap(), 1);
}

#[tokio::test]
async fn tool_call_policy_quiet_command_never_mutates_and_nudges_once() {
    let mut bash_args = IndexMap::new();
    bash_args.insert("command".into(), json!("cargo test"));
    let client = Arc::new(MockWorkflowContractClient::new(vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("bash", bash_args.clone())]),
        LLMResponse::ToolCalls(vec![ToolCall::new("bash", bash_args)]),
    ]));
    let body = json!({
        "messages": [{"role": "user", "content": "run tests"}],
        "model": "test-model",
        "tools": [bash_tool_json()],
        "_forge": {
            "tool_call_policy": {"quiet_commands": true}
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions_with_scorers_and_tool_controls(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        None,
        ToolOutputCompressionConfig::disabled(),
        None,
        ToolCallPolicyConfig::disabled(),
    )
    .await
    .expect("handler result");

    let response = match result {
        HandlerResult::Response(value) => value,
        _ => panic!("expected Response"),
    };
    let returned_args = response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .expect("arguments string");
    let returned_args: Value = serde_json::from_str(returned_args).expect("arguments json");
    assert_eq!(returned_args["command"], "cargo test");

    assert_eq!(*client.calls.lock().unwrap(), 2);
    let sent_messages = client.sent_messages();
    let retry_wire = serde_json::to_string(&sent_messages[1]).expect("retry messages");
    assert!(retry_wire.contains("[ToolCallPolicyNudge]"));
    assert!(retry_wire.contains("cargo test --quiet"));
    let assistant_arguments = sent_messages[1]
        .iter()
        .find_map(|message| message.get("tool_calls").and_then(Value::as_array))
        .and_then(|calls| calls.first())
        .and_then(|call| call.get("function"))
        .and_then(|function| function.get("arguments"))
        .and_then(Value::as_str)
        .expect("assistant tool-call arguments");
    let assistant_arguments: Value =
        serde_json::from_str(assistant_arguments).expect("assistant arguments json");
    assert_eq!(assistant_arguments["command"], "cargo test");
}

#[tokio::test]
async fn final_response_scorer_retries_proxy_respond_before_returning() {
    let mut bad_args = IndexMap::new();
    bad_args.insert("message".into(), json!("bad"));
    let mut good_args = IndexMap::new();
    good_args.insert("message".into(), json!("good"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", bad_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", good_args)]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let final_scorer = Arc::new(SequenceFinalScorer::new());
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false,
        "tools": [search_tool_json()]
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions_with_scorers(
        &body,
        &client,
        &ctx,
        3,
        true,
        None,
        Some(final_scorer),
    )
    .await
    .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "good");
        }
        _ => panic!("expected Response"),
    }
    assert_eq!(*client.calls.lock().unwrap(), 2);
    let sent_messages = serde_json::to_string(&client.sent_messages()).expect("messages");
    assert!(sent_messages.contains("[FinalResponseNudge]"));
}

fn legacy_list_accounts_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "legacy_list_accounts",
            "description": "List available accounts",
            "parameters": {"type": "object", "properties": {}}
        }
    })
}

fn legacy_submit_audit_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "legacy_submit_audit",
            "description": "Submit the final audit",
            "parameters": {
                "type": "object",
                "properties": {"report": {"type": "string"}}
            }
        }
    })
}

fn legacy_fetch_account_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "legacy_fetch_account",
            "description": "Fetch one account",
            "parameters": {
                "type": "object",
                "additionalProperties": false,
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"]
            }
        }
    })
}

#[tokio::test]
async fn proxy_real_terminal_tool_omits_synthetic_respond_tool() {
    let mut terminal_args = IndexMap::new();
    terminal_args.insert("report".into(), json!("done"));
    let responses = vec![LLMResponse::ToolCalls(vec![ToolCall::new(
        "legacy_submit_audit",
        terminal_args,
    )])];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "audit account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool(), legacy_submit_audit_tool()],
        "_forge": {
            "terminal_tools": ["legacy_submit_audit"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(
                v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "legacy_submit_audit"
            );
        }
        _ => panic!("expected Response"),
    }
    assert_eq!(
        client.sent_tools()[0],
        vec![
            "legacy_list_accounts".to_string(),
            "legacy_submit_audit".to_string()
        ]
    );
}

#[tokio::test]
async fn proxy_respond_only_terminal_still_injects_respond_tool() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("done"));
    let responses = vec![LLMResponse::ToolCalls(vec![ToolCall::new(
        "respond",
        respond_args,
    )])];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "audit account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
            assert_eq!(v["choices"][0]["message"]["content"], "done");
        }
        _ => panic!("expected Response"),
    }
    assert_eq!(
        client.sent_tools()[0],
        vec!["legacy_list_accounts".to_string(), "respond".to_string()]
    );
}

#[tokio::test]
async fn proxy_mixed_terminal_tools_filters_respond_when_real_terminal_exists() {
    let mut terminal_args = IndexMap::new();
    terminal_args.insert("report".into(), json!("done"));
    let responses = vec![LLMResponse::ToolCalls(vec![ToolCall::new(
        "legacy_submit_audit",
        terminal_args,
    )])];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "audit account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool(), legacy_submit_audit_tool()],
        "_forge": {
            "terminal_tools": ["respond", "legacy_submit_audit"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(
                v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "legacy_submit_audit"
            );
        }
        _ => panic!("expected Response"),
    }
    assert_eq!(
        client.sent_tools()[0],
        vec![
            "legacy_list_accounts".to_string(),
            "legacy_submit_audit".to_string()
        ]
    );
}

#[tokio::test]
async fn proxy_required_steps_block_premature_respond() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("too soon"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "audit account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            let calls = v["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], json!("legacy_list_accounts"));
        }
        _ => panic!("expected Response"),
    }

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("[StepEnforcementError]"));
    assert!(second_wire.contains("legacy_list_accounts"));
}

#[tokio::test]
async fn proxy_required_steps_retry_empty_tool_batch() {
    let responses = vec![
        LLMResponse::ToolCalls(Vec::new()),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "audit account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            let calls = v["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], json!("legacy_list_accounts"));
        }
        _ => panic!("expected Response"),
    }

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("Your previous response was not a valid tool call"));
    assert!(!second_wire.contains("\"tool_calls\":[]"));
}

#[tokio::test]
async fn proxy_retries_invalid_tool_arguments() {
    let mut bad_args = IndexMap::new();
    bad_args.insert("account_id".into(), json!(42));
    let mut good_args = IndexMap::new();
    good_args.insert("account_id".into(), json!("ACC-123"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_fetch_account", bad_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_fetch_account", good_args)]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "fetch account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_fetch_account_tool()]
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            let calls = v["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], json!("legacy_fetch_account"));
            assert_eq!(
                calls[0]["function"]["arguments"],
                json!("{\"account_id\":\"ACC-123\"}")
            );
        }
        _ => panic!("expected Response"),
    }

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("[InvalidArguments]"));
    assert!(second_wire.contains("account_id must be string, got number"));
}

#[tokio::test]
async fn proxy_retries_invalid_tool_arguments_streaming() {
    let mut bad_args = IndexMap::new();
    bad_args.insert("account_id".into(), json!(42));
    let mut good_args = IndexMap::new();
    good_args.insert("account_id".into(), json!("ACC-123"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_fetch_account", bad_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_fetch_account", good_args)]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "fetch account"}],
        "model": "test-model",
        "stream": true,
        "tools": [legacy_fetch_account_tool()]
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    let events = collect_stream_events(result).await;
    let event_text = serde_json::to_string(&events).expect("events json");
    assert!(event_text.contains("legacy_fetch_account"));
    assert!(event_text.contains("ACC-123"));

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("[InvalidArguments]"));
    assert!(second_wire.contains("account_id must be string, got number"));
}

#[tokio::test]
async fn proxy_required_steps_use_prior_tool_result_history() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("done"));
    let client = Arc::new(MockWorkflowContractClient::new(vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
    ]));
    let body = json!({
        "messages": [
            {"role": "user", "content": "audit account"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_list",
                    "type": "function",
                    "function": {"name": "legacy_list_accounts", "arguments": "{}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_list",
                "name": "legacy_list_accounts",
                "content": "ACC-12345"
            }
        ],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
            assert_eq!(v["choices"][0]["message"]["content"], "done");
        }
        _ => panic!("expected Response"),
    }
    let wire = serde_json::to_string(&client.sent_messages()).expect("wire json");
    assert!(!wire.contains("[StepEnforcementError]"));
}

#[tokio::test]
async fn proxy_required_steps_ignore_unresolved_assistant_tool_call() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("too soon"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [
            {"role": "user", "content": "audit account"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_list",
                    "type": "function",
                    "function": {"name": "legacy_list_accounts", "arguments": "{}"}
                }]
            }
        ],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            let calls = v["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], json!("legacy_list_accounts"));
        }
        _ => panic!("expected Response"),
    }

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("[StepEnforcementError]"));
}

#[tokio::test]
async fn proxy_required_steps_ignore_failed_prior_tool_result_history() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("too soon"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [
            {"role": "user", "content": "audit account"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_list",
                    "type": "function",
                    "function": {"name": "legacy_list_accounts", "arguments": "{}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_list",
                "name": "legacy_list_accounts",
                "content": "[ToolError] timeout"
            }
        ],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(
                v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "legacy_list_accounts"
            );
        }
        _ => panic!("expected Response"),
    }

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("[StepEnforcementError]"));
}

#[tokio::test]
async fn proxy_required_steps_treat_absent_status_as_success_without_broad_text_heuristic() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("done"));
    let client = Arc::new(MockWorkflowContractClient::new(vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
    ]));
    let body = json!({
        "messages": [
            {"role": "user", "content": "audit account"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_list",
                    "type": "function",
                    "function": {"name": "legacy_list_accounts", "arguments": "{}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_list",
                "name": "legacy_list_accounts",
                "content": "0 failed checks"
            }
        ],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
            assert_eq!(v["choices"][0]["message"]["content"], "done");
        }
        _ => panic!("expected Response"),
    }
    let wire = serde_json::to_string(&client.sent_messages()).expect("wire json");
    assert!(!wire.contains("[StepEnforcementError]"));
}

#[test]
fn record_completed_proxy_tool_results_keys_status_by_tool_call_id() {
    let raw_messages = vec![
        json!({
            "role": "tool",
            "tool_call_id": "call_list",
            "content": "[ToolError] stale text",
            "_forge": {"tool_status": "ok"}
        }),
        json!({"role": "user", "content": "not the tool result index"}),
    ];
    let messages = vec![
        Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "legacy_list_accounts",
            Some(IndexMap::new()),
            "call_list",
        )]),
        Message::new(
            MessageRole::User,
            "index padding",
            MessageMeta::new(MessageType::UserInput),
        ),
        Message::new(
            MessageRole::Tool,
            "[ToolError] text would fail without keyed status",
            MessageMeta::new(MessageType::ToolResult),
        )
        .with_tool_name("legacy_list_accounts")
        .with_tool_call_id("call_list"),
    ];
    let mut enforcer = StepEnforcer::new(
        vec!["legacy_list_accounts".to_string()],
        IndexSet::from_iter(["respond".to_string()]),
        None,
        3,
        2,
    );

    record_completed_proxy_tool_results(&raw_messages, &messages, &mut enforcer);

    assert!(enforcer.is_satisfied());
}

#[tokio::test]
async fn proxy_required_steps_ignore_non_ok_prior_tool_status() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("too soon"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [
            {"role": "user", "content": "audit account"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_list",
                    "type": "function",
                    "function": {"name": "legacy_list_accounts", "arguments": "{}"}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_list",
                "name": "legacy_list_accounts",
                "content": "ACC-12345",
                "_forge": {"tool_status": "error"}
            }
        ],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            assert_eq!(
                v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                "legacy_list_accounts"
            );
        }
        _ => panic!("expected Response"),
    }
}

#[tokio::test]
async fn proxy_required_steps_block_premature_real_terminal_tool() {
    let mut terminal_args = IndexMap::new();
    terminal_args.insert("report".into(), json!("too soon"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_submit_audit", terminal_args)]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "audit account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool(), legacy_submit_audit_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond", "legacy_submit_audit"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            let calls = v["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], json!("legacy_list_accounts"));
        }
        _ => panic!("expected Response"),
    }

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("[StepEnforcementError]"));
    assert!(second_wire.contains("legacy_submit_audit"));
}

#[tokio::test]
async fn proxy_required_steps_retry_mixed_terminal_batch() {
    let mut respond_args = IndexMap::new();
    respond_args.insert("message".into(), json!("done"));
    let responses = vec![
        LLMResponse::ToolCalls(vec![
            ToolCall::new("legacy_list_accounts", IndexMap::new()),
            ToolCall::new("respond", respond_args),
        ]),
        LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
    ];
    let client = Arc::new(MockWorkflowContractClient::new(responses));
    let body = json!({
        "messages": [{"role": "user", "content": "audit account"}],
        "model": "test-model",
        "stream": false,
        "tools": [legacy_list_accounts_tool()],
        "_forge": {
            "required_steps": ["legacy_list_accounts"],
            "terminal_tools": ["respond"]
        }
    });
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    let result = handle_chat_completions(&body, &client, &ctx, 3, true)
        .await
        .expect("handler result");

    match result {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            let calls = v["choices"][0]["message"]["tool_calls"]
                .as_array()
                .expect("tool calls");
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0]["function"]["name"], json!("legacy_list_accounts"));
        }
        _ => panic!("expected Response"),
    }

    let sent = client.sent_messages();
    assert_eq!(sent.len(), 2);
    let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
    assert!(second_wire.contains("[StepEnforcementError]"));
    assert!(second_wire.contains("Do not combine terminal and non-terminal tools"));
}

struct MockAlwaysTextClient;
impl LLMClient for MockAlwaysTextClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Ok(LLMResponse::Text(TextResponse::new("always text")))
    }
    async fn send_stream(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
            "always text",
        ))))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn handle_retries_exhausted_errors_by_default() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false,
        "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    });
    let client = Arc::new(MockAlwaysTextClient);
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 2, true).await;
    let err = result.expect_err("guardrail failure should surface as upstream error");
    assert!(matches!(err, HandlerError::Upstream(_)));
    assert!(err
        .message()
        .contains("model failed guarded tool-call validation after retries"));
}

struct MockTextSequenceClient {
    responses: Vec<String>,
    calls: std::sync::Mutex<usize>,
}

impl MockTextSequenceClient {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: responses.into_iter().map(str::to_string).collect(),
            calls: std::sync::Mutex::new(0),
        }
    }
}

impl LLMClient for MockTextSequenceClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        let mut calls = self.calls.lock().unwrap();
        let content = self
            .responses
            .get(*calls)
            .cloned()
            .unwrap_or_else(|| panic!("MockTextSequenceClient exhausted at call {}", *calls));
        *calls += 1;
        Ok(LLMResponse::Text(TextResponse::new(content)))
    }
    async fn send_stream(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        let mut calls = self.calls.lock().unwrap();
        let content = self
            .responses
            .get(*calls)
            .cloned()
            .unwrap_or_else(|| panic!("MockTextSequenceClient exhausted at call {}", *calls));
        *calls += 1;
        Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
            content,
        ))))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn handle_retries_exhausted_raw_response_requires_opt_in() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false,
        "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    });
    let client = Arc::new(MockTextSequenceClient::new(vec!["first bad", "raw final"]));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 1, true).await;
    let err = result.expect_err("default rejects raw fallback");
    assert!(matches!(err, HandlerError::Upstream(_)));

    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false,
        "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
        "_forge": {"return_raw_on_guardrail_failure": true}
    });
    let client = Arc::new(MockTextSequenceClient::new(vec!["first bad", "raw final"]));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 1, true).await;
    match result.unwrap() {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["message"]["content"], "raw final");
            assert_eq!(v["choices"][0]["finish_reason"], "stop");
        }
        _ => panic!("expected Response"),
    }
}

#[tokio::test]
async fn handle_retries_exhausted_returns_raw_response_streaming() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": true,
        "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}],
        "_forge": {"return_raw_on_guardrail_failure": true}
    });
    let client = Arc::new(MockTextSequenceClient::new(vec!["first bad", "raw final"]));
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 1, true).await;
    let events = collect_stream_events(result.unwrap()).await;
    assert_eq!(events[0]["choices"][0]["delta"]["content"], "raw final");
    assert_eq!(
        events.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

struct MockMixedToolClient;
impl LLMClient for MockMixedToolClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        let mut respond_args = IndexMap::new();
        respond_args.insert("message".into(), json!("ignored text"));
        let mut search_args = IndexMap::new();
        search_args.insert("query".into(), json!("test"));
        Ok(LLMResponse::ToolCalls(vec![
            ToolCall::new("respond", respond_args),
            ToolCall::new("search", search_args),
        ]))
    }
    async fn send_stream(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        let mut respond_args = IndexMap::new();
        respond_args.insert("message".into(), json!("ignored text"));
        let mut search_args = IndexMap::new();
        search_args.insert("query".into(), json!("test"));
        Ok(stream_from_response(LLMResponse::ToolCalls(vec![
            ToolCall::new("respond", respond_args),
            ToolCall::new("search", search_args),
        ])))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

struct MockGuardedStreamingClient {
    stream_calls: AtomicUsize,
}

impl MockGuardedStreamingClient {
    fn new() -> Self {
        Self {
            stream_calls: AtomicUsize::new(0),
        }
    }
}

impl LLMClient for MockGuardedStreamingClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Err(crate::error::BackendError::new(
            500,
            "send should not be used",
        ))
    }

    async fn send_stream(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        let call = self.stream_calls.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::new(ChunkType::ToolCallDelta).with_content("leaky-bogus")),
                Ok(
                    StreamChunk::new(ChunkType::Final).with_response(LLMResponse::ToolCalls(vec![
                        ToolCall::new("bogus", IndexMap::new()),
                    ])),
                ),
            ])))
        } else {
            let mut args = IndexMap::new();
            args.insert("q".into(), json!("safe"));
            Ok(stream_from_response(LLMResponse::ToolCalls(vec![
                ToolCall::new("search", args),
            ])))
        }
    }

    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn guarded_streaming_holds_invalid_tool_chunks_until_validated() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": true,
        "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    });
    let client = Arc::new(MockGuardedStreamingClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 2, true)
        .await
        .expect("handler result");

    assert_eq!(client.stream_calls.load(Ordering::SeqCst), 2);
    let events = collect_stream_events(result).await;
    let body = serde_json::to_string(&events).unwrap();
    assert!(!body.contains("leaky-bogus"));
    assert!(!body.contains("bogus"));
    assert!(body.contains("search"));
    assert_eq!(
        events.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
}

#[tokio::test]
async fn anthropic_guarded_streaming_holds_invalid_tool_chunks_until_validated() {
    let raw = json!({
        "model": "claude-3",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "tools": [{
            "name": "search",
            "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }
        }]
    });
    let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
    let client = Arc::new(MockGuardedStreamingClient::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 2, true)
        .await
        .expect("handler result");

    assert_eq!(client.stream_calls.load(Ordering::SeqCst), 2);
    let events = match result {
        AnthropicHandlerResult::StreamBody(stream) => {
            collect_anthropic_events(stream).await.expect("events")
        }
        other => panic!("expected StreamBody, got {other:?}"),
    };
    let body = crate::proxy::server::format_anthropic_sse_body(events.as_slice());
    assert!(!body.contains("leaky-bogus"));
    assert!(!body.contains("bogus"));
    assert!(body.contains("search"));
}

#[tokio::test]
async fn handle_mixed_tools_drops_respond() {
    let body = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test-model",
        "stream": false,
        "tools": [
            {"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}
        ]
    });
    let client = Arc::new(MockMixedToolClient);
    let ctx = Arc::new(Mutex::new(dummy_ctx()));
    let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
    match result.unwrap() {
        HandlerResult::Response(v) => {
            assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            let tcs = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
            assert_eq!(tcs.len(), 1);
            assert_eq!(tcs[0]["function"]["name"], "search");
        }
        _ => panic!("expected Response"),
    }
}

struct MockSamplingTracker {
    last_sampling: std::sync::Mutex<Option<SamplingParams>>,
}
impl MockSamplingTracker {
    fn new() -> Self {
        Self {
            last_sampling: std::sync::Mutex::new(None),
        }
    }
}
impl LLMClient for MockSamplingTracker {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }
    async fn send(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        *self.last_sampling.lock().unwrap() = sampling;
        Ok(LLMResponse::Text(TextResponse::new("ok")))
    }
    async fn send_stream(
        &self,
        _m: Vec<Value>,
        _t: Option<Vec<ToolSpec>>,
        _s: Option<SamplingParams>,
    ) -> Result<ChunkStream, crate::error::StreamError> {
        Err(crate::error::StreamError::new("not implemented"))
    }
    async fn get_context_length(&self) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
        Ok(Some(4096))
    }
}

#[tokio::test]
async fn sampling_per_call_no_mutation() {
    let client = Arc::new(MockSamplingTracker::new());
    let ctx = Arc::new(Mutex::new(dummy_ctx()));

    // First call with sampling.
    let body1 = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "model": "test", "temperature": 0.7
    });
    handle_chat_completions(&body1, &client, &ctx, 0, true)
        .await
        .unwrap();
    let s1 = client.last_sampling.lock().unwrap().clone();
    assert_eq!(
        s1.as_ref().and_then(|m| m.get("temperature")),
        Some(&json!(0.7))
    );

    // Second call without sampling: should be None, not persisted from call 1.
    let body2 = json!({"messages": [{"role": "user", "content": "hi"}], "model": "test"});
    handle_chat_completions(&body2, &client, &ctx, 0, true)
        .await
        .unwrap();
    let s2 = client.last_sampling.lock().unwrap().clone();
    assert!(s2.is_none());
}
