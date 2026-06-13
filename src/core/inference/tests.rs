use super::*;
use crate::clients::base::{
    ApiFormat, ChunkStream, LLMClient, LLMRequestOptions, LLMResponse, SamplingParams,
    TextResponse, ToolCall,
};
use crate::context::manager::ContextManager;
use crate::context::strategies::NoCompact;
use crate::core::message::{Message, MessageMeta, MessageRole, MessageType};
use crate::core::tool_spec::ToolSpec;
use crate::guardrails::{ErrorTracker, ResponseValidator};
use indexmap::IndexMap;
use serde_json::json;
use std::sync::Arc;

#[test]
fn tool_call_id_format() {
    assert_eq!(format_tool_call_id(0), "call_000000000");
    assert_eq!(format_tool_call_id(1), "call_000000001");
    assert_eq!(format_tool_call_id(42), "call_000000042");
    assert_eq!(format_tool_call_id(999999999), "call_999999999");
}

#[test]
fn inference_result_fields() {
    let result = InferenceResult {
        response: LLMResponse::Text(crate::clients::base::TextResponse::new("hi")),
        usage: None,
        usage_details: None,
        call_info: None,
        provider_response: None,
        provider_events: None,
        new_messages: vec![],
        tool_call_counter: 5,
        attempts: 1,
    };
    assert_eq!(result.tool_call_counter, 5);
    assert_eq!(result.attempts, 1);
}

#[test]
fn next_unique_tool_call_id_skips_existing_history_ids() {
    let messages = vec![
        Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![crate::core::message::ToolCallInfo::new(
            "prior",
            Some(IndexMap::new()),
            "call_000000000",
        )]),
        Message::new(
            MessageRole::Tool,
            "prior result",
            MessageMeta::new(MessageType::ToolResult),
        )
        .with_tool_name("prior")
        .with_tool_call_id("call_000000000"),
    ];
    let mut seen = existing_tool_call_ids(&messages);
    let mut counter = 0;

    assert_eq!(
        next_unique_tool_call_id(&mut counter, &mut seen),
        "call_000000001"
    );
    assert_eq!(counter, 2);
}

struct RetryRecordingClient {
    raw_bodies: std::sync::Mutex<Vec<Option<Arc<Value>>>>,
    initial_messages: std::sync::Mutex<Vec<Option<Arc<[Value]>>>>,
}

impl RetryRecordingClient {
    fn new() -> Self {
        Self {
            raw_bodies: std::sync::Mutex::new(Vec::new()),
            initial_messages: std::sync::Mutex::new(Vec::new()),
        }
    }
}

impl LLMClient for RetryRecordingClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        _sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        Ok(LLMResponse::Text(TextResponse::new("unused")))
    }

    async fn send_with_options(
        &self,
        _messages: Vec<Value>,
        _tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, crate::error::BackendError> {
        let mut raw_bodies = self.raw_bodies.lock().unwrap();
        let attempt = raw_bodies.len();
        raw_bodies.push(options.inbound_anthropic_body);
        drop(raw_bodies);
        self.initial_messages
            .lock()
            .unwrap()
            .push(options.initial_openai_messages);

        if attempt == 0 {
            Ok(LLMResponse::Text(TextResponse::new("not a tool call")))
        } else {
            let mut args = IndexMap::new();
            args.insert("message".to_string(), json!("ok"));
            Ok(LLMResponse::ToolCalls(vec![ToolCall::new("respond", args)]))
        }
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

#[tokio::test]
async fn raw_anthropic_body_is_cleared_after_clean_attempt() {
    let client = RetryRecordingClient::new();
    let raw = Arc::new(json!({
        "model": "claude-3",
        "max_tokens": 64,
        "system": [{
            "type": "text",
            "text": "system",
            "cache_control": {"type": "ephemeral"}
        }],
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "hi",
                "cache_control": {"type": "ephemeral"}
            }]
        }]
    }));
    let mut messages = vec![Message::new(
        MessageRole::User,
        "hi",
        MessageMeta::new(MessageType::UserInput),
    )];
    let initial_messages: Arc<[Value]> =
        Arc::from(fold_and_serialize(&messages, "openai").into_boxed_slice());
    let mut context = ContextManager::new(Box::new(NoCompact), 4096, None, None, None);
    let validator = ResponseValidator::new(vec!["respond".to_string()], false, None);
    let mut tracker = ErrorTracker::new(3, 2);
    let mut counter = 0;
    let tools = vec![crate::tools::respond::respond_spec()];

    let result = run_inference_with_options(
        &mut messages,
        &client,
        &mut context,
        &validator,
        &mut tracker,
        &tools,
        &mut counter,
        0,
        "",
        Some(3),
        false,
        None,
        LLMRequestOptions {
            inbound_anthropic_body: Some(raw.clone()),
            initial_openai_messages: Some(initial_messages.clone()),
            ..Default::default()
        },
    )
    .await
    .expect("inference")
    .expect("result");

    assert_eq!(result.attempts, 2);
    let raw_bodies = client.raw_bodies.lock().unwrap().clone();
    assert_eq!(raw_bodies.len(), 2);
    assert!(raw_bodies[0]
        .as_ref()
        .is_some_and(|body| Arc::ptr_eq(body, &raw)));
    assert!(raw_bodies[1].is_none());
    let recorded_initial_messages = client.initial_messages.lock().unwrap().clone();
    assert_eq!(recorded_initial_messages.len(), 2);
    assert!(recorded_initial_messages[0]
        .as_ref()
        .is_some_and(|messages| Arc::ptr_eq(messages, &initial_messages)));
    assert!(recorded_initial_messages[1].is_none());
}
