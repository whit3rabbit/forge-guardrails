//! Request handler: bridges HTTP layer and inference with guardrails.
//!
//! handle_chat_completions is the main entry point for /v1/chat/completions.
//! It converts inbound OpenAI messages, runs inference with validation/retry,
//! then strips respond() calls from output.

use crate::client::LLMClient;
use crate::context::ContextManager;
use crate::proxy::{
    extract_sampling, has_respond_tool, openai_to_messages, respond_tool_openai,
    strip_respond_calls, text_response_to_openai, text_to_sse_events, tool_calls_to_openai,
    tool_calls_to_sse_events,
};
use crate::respond::RESPOND_TOOL_NAME;
use crate::streaming::{LLMResponse, TextResponse, ToolCall};
use crate::tool_spec::ToolSpec;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Result of handling a chat completion request.
#[derive(Debug)]
pub enum HandlerResult {
    /// Non-streaming: single OpenAI response object.
    Response(Value),
    /// Streaming: list of SSE event objects.
    Events(Vec<Value>),
}

/// Main handler for /v1/chat/completions.
///
/// When no tools are present, passes through to backend directly (no guardrails).
/// When tools are present, injects a respond tool if not already provided,
/// runs inference with validation/retry, then strips respond() calls from output.
///
/// Sampling fields are extracted per-request and passed as a dict (or None);
/// they never persist on the client between calls.
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat_completions<C: LLMClient>(
    body: &Value,
    client: &Arc<C>,
    _context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    _rescue_enabled: bool,
) -> Result<HandlerResult, String> {
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("missing or invalid messages field")?;

    let model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    let stream = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let tools_raw = body
        .get("tools")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default();

    let sampling = extract_sampling(body);

    // Convert inbound OpenAI messages to internal format.
    let internal_msgs = openai_to_messages(messages);

    // Serialize for the client's API format.
    let api_format = client.api_format().as_str();
    let serialized: Vec<Value> = internal_msgs
        .iter()
        .map(|m| m.serialize(api_format))
        .collect();

    // If no tools, pass through directly.
    if tools_raw.is_empty() {
        return run_passthrough(client, &serialized, None, &sampling, model_name, stream).await;
    }

    // Tools present: inject respond tool if needed.
    let mut tools_with_respond = tools_raw.clone();
    if !has_respond_tool(&tools_with_respond) {
        tools_with_respond.push(respond_tool_openai());
    }

    // Parse tools into ToolSpec for the client.
    let tool_specs = parse_tool_specs(&tools_with_respond);

    // Run inference with retry.
    let response = run_with_retry(
        client,
        &serialized,
        Some(&tool_specs),
        &sampling,
        max_retries,
    )
    .await;

    match response {
        Ok(llm_response) => {
            let result = process_response(&llm_response, model_name, stream);
            Ok(result)
        }
        Err(_) => Err("Inference failed".to_string()),
    }
}

/// Pass-through mode: no tools, no guardrails, direct backend call.
async fn run_passthrough<C: LLMClient>(
    client: &Arc<C>,
    messages: &[Value],
    tools: Option<&[ToolSpec]>,
    sampling: &Option<serde_json::Map<String, Value>>,
    model_name: &str,
    stream: bool,
) -> Result<HandlerResult, String> {
    let sampling_clone = sampling.clone();
    let llm_response = client
        .send(messages.to_vec(), tools.map(|t| t.to_vec()), sampling_clone)
        .await
        .map_err(|e| e.to_string())?;

    Ok(process_response(&llm_response, model_name, stream))
}

/// Run inference with retry for tool-guardrail mode.
async fn run_with_retry<C: LLMClient>(
    client: &Arc<C>,
    messages: &[Value],
    tools: Option<&[ToolSpec]>,
    sampling: &Option<serde_json::Map<String, Value>>,
    max_retries: i32,
) -> Result<LLMResponse, String> {
    let mut last_response: Option<LLMResponse> = None;

    for attempt in 0..=max_retries {
        let sampling_clone = sampling.clone();
        let response = client
            .send(messages.to_vec(), tools.map(|t| t.to_vec()), sampling_clone)
            .await
            .map_err(|e| e.to_string())?;

        match &response {
            LLMResponse::ToolCalls(calls) => {
                let (real_calls, respond_text) = strip_respond_calls(calls);

                if !real_calls.is_empty() {
                    return Ok(LLMResponse::ToolCalls(filter_respond(calls)));
                }

                if let Some(text) = respond_text {
                    return Ok(LLMResponse::Text(TextResponse::new(text)));
                }

                // Empty tool calls list: return empty text.
                return Ok(LLMResponse::Text(TextResponse::new("")));
            }
            LLMResponse::Text(_) => {
                last_response = Some(response);
                if attempt >= max_retries {
                    return Ok(last_response.unwrap_or(LLMResponse::Text(TextResponse::new(""))));
                }
                continue;
            }
        }
    }

    Ok(last_response.unwrap_or(LLMResponse::Text(TextResponse::new(""))))
}

/// Remove respond() calls, keeping only real tool calls.
pub fn filter_respond(calls: &[ToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .filter(|c| c.tool != RESPOND_TOOL_NAME)
        .cloned()
        .collect()
}

/// Convert LLM response to OpenAI format (streaming or non-streaming).
pub fn process_response(response: &LLMResponse, model_name: &str, stream: bool) -> HandlerResult {
    match response {
        LLMResponse::ToolCalls(calls) => {
            if calls.is_empty() {
                let text_resp = TextResponse::new("");
                if stream {
                    HandlerResult::Events(text_to_sse_events("", model_name, 0))
                } else {
                    HandlerResult::Response(text_response_to_openai(&text_resp, model_name))
                }
            } else if stream {
                HandlerResult::Events(tool_calls_to_sse_events(calls, model_name))
            } else {
                HandlerResult::Response(tool_calls_to_openai(calls, model_name))
            }
        }
        LLMResponse::Text(text) => {
            if stream {
                HandlerResult::Events(text_to_sse_events(&text.content, model_name, 0))
            } else {
                HandlerResult::Response(text_response_to_openai(text, model_name))
            }
        }
    }
}

/// Parse OpenAI-format tool definitions into ToolSpec objects.
pub fn parse_tool_specs(tools: &[Value]) -> Vec<ToolSpec> {
    let mut specs = Vec::new();
    for tool in tools {
        if let Some(func) = tool.get("function") {
            let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let description = func
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let schema = func
                .get("parameters")
                .cloned()
                .unwrap_or(json!({"type": "object", "properties": {}}));

            if let Ok(spec) = ToolSpec::from_json_schema(name, description, &schema) {
                specs.push(spec);
            }
        }
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::ToolCall;
    use indexmap::IndexMap;
    use serde_json::json;

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

    #[test]
    fn process_response_text_streaming() {
        let resp = LLMResponse::Text(TextResponse::new("hello"));
        let result = process_response(&resp, "model", true);
        match result {
            HandlerResult::Events(events) => {
                assert!(!events.is_empty());
                let last = events.last().unwrap();
                assert_eq!(last["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Events"),
        }
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
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search things",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        })];
        let specs = parse_tool_specs(&tools);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "search");
    }

    #[test]
    fn parse_tool_specs_empty() {
        let specs = parse_tool_specs(&[]);
        assert!(specs.is_empty());
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
        fn api_format(&self) -> crate::client::ApiFormat {
            crate::client::ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<crate::client::SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            Ok(LLMResponse::Text(TextResponse::new("mock response")))
        }
        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<crate::client::SamplingParams>,
        ) -> Result<crate::client::ChunkStream, crate::error::StreamError> {
            Err(crate::error::StreamError::new("not implemented"))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    struct MockToolCallClient;

    impl LLMClient for MockToolCallClient {
        fn api_format(&self) -> crate::client::ApiFormat {
            crate::client::ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<crate::client::SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            let mut args = IndexMap::new();
            args.insert("message".into(), json!("responded text"));
            Ok(LLMResponse::ToolCalls(vec![ToolCall::new("respond", args)]))
        }
        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<crate::client::SamplingParams>,
        ) -> Result<crate::client::ChunkStream, crate::error::StreamError> {
            Err(crate::error::StreamError::new("not implemented"))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    fn dummy_ctx() -> ContextManager {
        ContextManager::new(Box::new(crate::compact::NoCompact), 4096, None, None, None)
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

    struct MockAlwaysTextClient;
    impl LLMClient for MockAlwaysTextClient {
        fn api_format(&self) -> crate::client::ApiFormat {
            crate::client::ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<crate::client::SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            Ok(LLMResponse::Text(TextResponse::new("always text")))
        }
        async fn send_stream(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<crate::client::SamplingParams>,
        ) -> Result<crate::client::ChunkStream, crate::error::StreamError> {
            Err(crate::error::StreamError::new("not implemented"))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    #[tokio::test]
    async fn handle_retries_exhausted_returns_text() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false,
            "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        });
        let client = Arc::new(MockAlwaysTextClient);
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 2, true).await;
        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "always text");
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Response"),
        }
    }

    struct MockMixedToolClient;
    impl LLMClient for MockMixedToolClient {
        fn api_format(&self) -> crate::client::ApiFormat {
            crate::client::ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<crate::client::SamplingParams>,
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
            _s: Option<crate::client::SamplingParams>,
        ) -> Result<crate::client::ChunkStream, crate::error::StreamError> {
            Err(crate::error::StreamError::new("not implemented"))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    #[tokio::test]
    async fn handle_mixed_tools_drops_respond() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false,
            "tools": [
                {"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "respond", "description": "r", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}}
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
        last_sampling: std::sync::Mutex<Option<crate::client::SamplingParams>>,
    }
    impl MockSamplingTracker {
        fn new() -> Self {
            Self {
                last_sampling: std::sync::Mutex::new(None),
            }
        }
    }
    impl LLMClient for MockSamplingTracker {
        fn api_format(&self) -> crate::client::ApiFormat {
            crate::client::ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            sampling: Option<crate::client::SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            *self.last_sampling.lock().unwrap() = sampling;
            Ok(LLMResponse::Text(TextResponse::new("ok")))
        }
        async fn send_stream(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<crate::client::SamplingParams>,
        ) -> Result<crate::client::ChunkStream, crate::error::StreamError> {
            Err(crate::error::StreamError::new("not implemented"))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
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
}
