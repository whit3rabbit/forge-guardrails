//! Anthropic Messages API client adapter.
//!
//! Converts messages from the common wire format to Anthropic's format before
//! each API call. Uses reqwest for HTTP. The sampling parameter is accepted
//! for protocol symmetry but ignored (Anthropic controls sampling server-side).

pub(crate) mod convert;

use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::clients::base::{
    ApiFormat, ChunkStream, ChunkType, LLMClient, LLMRequestOptions, LLMResponse, SamplingParams,
    StreamChunk, TokenUsage,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

/// Client for Anthropic Messages API (Claude models).
pub struct AnthropicClient {
    base_url: String,
    model: String,
    api_key: Option<String>,
    max_tokens: i64,
    timeout_secs: f64,
    max_retries: i64,
    tool_choice: Option<String>,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
}

impl AnthropicClient {
    /// Creates a new `AnthropicClient` for the given model.
    pub fn new(model: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: "https://api.anthropic.com/v1".to_string(),
            model: model.into(),
            api_key,
            max_tokens: 4096,
            timeout_secs: 300.0,
            max_retries: 3,
            tool_choice: None,
            last_usage: Arc::new(Mutex::new(None)),
        }
    }

    /// Sets the base URL for the Anthropic API.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets the max tokens parameter for completions.
    pub fn with_max_tokens(mut self, max_tokens: i64) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the request timeout in seconds.
    pub fn with_timeout(mut self, timeout_secs: f64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Sets the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: i64) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Sets the tool choice configuration.
    pub fn with_tool_choice(mut self, tool_choice: impl Into<String>) -> Self {
        self.tool_choice = Some(tool_choice.into());
        self
    }

    fn record_usage(&self, response: &Value) {
        let usage = response.get("usage");
        let prompt = usage
            .and_then(|u| u.get("input_tokens"))
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let completion = usage
            .and_then(|u| u.get("output_tokens"))
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let token_usage = TokenUsage::new(prompt, completion, prompt + completion);
        if let Ok(mut guard) = self.last_usage.lock() {
            *guard = Some(token_usage);
        }
    }

    /// Returns the token usage of the last request made by this client, if any.
    pub fn get_last_usage(&self) -> Option<TokenUsage> {
        self.last_usage.lock().ok().and_then(|guard| guard.clone())
    }

    fn build_body_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
        stream: bool,
    ) -> Value {
        if let Some(mut body) = options.inbound_anthropic_body {
            if let Some(obj) = body.as_object_mut() {
                if stream {
                    obj.insert("stream".to_string(), Value::Bool(true));
                } else {
                    obj.remove("stream");
                }
                obj.entry("model".to_string())
                    .or_insert_with(|| Value::String(self.model.clone()));
            }
            return body;
        }

        let (_system, mut body) = convert::build_request_body(
            &self.model,
            &messages,
            self.max_tokens,
            tools.as_deref(),
            self.tool_choice.as_deref(),
        );
        apply_rebuilt_anthropic_passthrough(options.passthrough.as_ref(), &mut body);
        if stream {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("stream".to_string(), Value::Bool(true));
            }
        }
        body
    }
}

fn apply_rebuilt_anthropic_passthrough(
    passthrough: Option<&serde_json::Map<String, Value>>,
    body: &mut Value,
) {
    let Some(passthrough) = passthrough else {
        return;
    };
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    if let Some(model) = passthrough.get("model").and_then(Value::as_str) {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }
    if let Some(max_tokens) = passthrough
        .get("max_completion_tokens")
        .or_else(|| passthrough.get("max_tokens"))
    {
        obj.insert("max_tokens".to_string(), max_tokens.clone());
    }
    if let Some(stop) = passthrough.get("stop") {
        obj.insert("stop_sequences".to_string(), stop.clone());
    }
    if let Some(tool_choice) = passthrough.get("tool_choice") {
        if let Some(mapped) = openai_tool_choice_to_anthropic(tool_choice) {
            obj.insert("tool_choice".to_string(), mapped);
        }
    }
    if let Some(user) = passthrough.get("user").and_then(Value::as_str) {
        obj.insert("metadata".to_string(), serde_json::json!({"user_id": user}));
    }
}

fn openai_tool_choice_to_anthropic(value: &Value) -> Option<Value> {
    match value {
        Value::String(choice) if choice == "required" => Some(serde_json::json!({"type": "any"})),
        Value::String(choice) if choice == "auto" || choice == "none" => {
            Some(serde_json::json!({"type": choice}))
        }
        Value::Object(obj) => obj
            .get("function")
            .and_then(|func| func.get("name"))
            .and_then(Value::as_str)
            .map(|name| serde_json::json!({"type": "tool", "name": name})),
        _ => None,
    }
}

impl LLMClient for AnthropicClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<crate::clients::base::TokenUsage> {
        self.last_usage.lock().ok().and_then(|g| g.clone())
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        self.send_with_options(messages, tools, LLMRequestOptions::from_sampling(sampling))
            .await
    }

    async fn send_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<LLMResponse, BackendError> {
        let sampling = options.sampling.clone();
        if let Some(sp) = &sampling {
            log::debug!(
                "AnthropicClient: ignoring sampling keys: {:?}",
                sp.keys().collect::<Vec<_>>()
            );
        }

        let body = self.build_body_with_options(messages, tools, options, false);

        let client = reqwest::Client::new();
        let mut req = client
            .post(format!("{}/messages", self.base_url))
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.header("x-api-key", key.as_str());
        }

        let resp = req
            .send()
            .await
            .map_err(|e| BackendError::new(0, e.to_string()))?;
        let status = resp.status().as_u16() as i64;

        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(BackendError::new(status, body_text));
        }

        let response_json: Value = resp
            .json()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;

        self.record_usage(&response_json);
        Ok(convert::parse_response(&response_json))
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        self.send_stream_with_options(messages, tools, LLMRequestOptions::from_sampling(sampling))
            .await
    }

    async fn send_stream_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
    ) -> Result<ChunkStream, StreamError> {
        let sampling = options.sampling.clone();
        if let Some(sp) = &sampling {
            log::debug!(
                "AnthropicClient: ignoring sampling keys: {:?}",
                sp.keys().collect::<Vec<_>>()
            );
        }

        let body = self.build_body_with_options(messages, tools, options, true);

        let client = reqwest::Client::new();
        let mut req = client
            .post(format!("{}/messages", self.base_url))
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body);

        if let Some(ref key) = self.api_key {
            req = req.header("x-api-key", key.as_str());
        }

        let resp = req
            .send()
            .await
            .map_err(|e| StreamError::new(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16() as i64;
            let body_text = resp.text().await.unwrap_or_default();
            return Err(StreamError::new(format!(
                "Backend error (status {}): {}",
                status, body_text
            )));
        }

        // Incremental SSE streaming aligned with Python AnthropicClient.send_stream().
        // Tracks:
        //   content_block_start  → detect tool_use blocks by index
        //   content_block_delta  → text_delta (TEXT_DELTA) or input_json_delta (TOOL_CALL_DELTA)
        //   content_block_stop   → reset current tool index
        //   message_stop         → emit FINAL chunk built from accumulated state
        //   message_delta        → capture usage (input/output_tokens)
        let byte_stream = resp.bytes_stream();
        let last_usage = self.last_usage.clone();
        let stream = async_stream::stream! {
            use futures_util::StreamExt;
            // SSE line buffer for partial-line data across byte chunks.
            let mut line_buf = String::new();
            let mut inner = Box::pin(byte_stream);

            // Accumulated stream state.
            let mut accumulated_text = String::new();
            // (name, args_json) per tool_use block.
            let mut tool_blocks: Vec<(String, String)> = Vec::new();
            let mut current_tool_idx: i64 = -1;
            let mut usage_input: i64 = 0;
            let mut usage_output: i64 = 0;

            loop {
                match inner.next().await {
                    Some(Ok(bytes)) => {
                        line_buf.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Some(Err(e)) => {
                        yield Err(StreamError::new(e.to_string()));
                        return;
                    }
                    None => break,
                }
                // Process complete lines from the buffer.
                while let Some(newline_pos) = line_buf.find('\n') {
                    let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                    line_buf = line_buf[newline_pos + 1..].to_string();

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" { continue; }
                        let evt: Value = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        match evt.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                            "content_block_start" => {
                                if let Some("tool_use") = evt
                                    .get("content_block")
                                    .and_then(|b| b.get("type"))
                                    .and_then(|t| t.as_str())
                                {
                                    let name = evt
                                        .get("content_block")
                                        .and_then(|b| b.get("name"))
                                        .and_then(|n| n.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    tool_blocks.push((name, String::new()));
                                    current_tool_idx = (tool_blocks.len() as i64) - 1;
                                }
                            }
                            "content_block_delta" => {
                                if let Some(delta) = evt.get("delta") {
                                    match delta.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                                        "text_delta" => {
                                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                accumulated_text.push_str(text);
                                                yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(text));
                                            }
                                        }
                                        "input_json_delta" => {
                                            let idx = current_tool_idx;
                                            if idx >= 0 {
                                                if let Some(partial) = delta.get("partial_json").and_then(|p| p.as_str()) {
                                                    if let Some(block) = tool_blocks.get_mut(idx as usize) {
                                                        block.1.push_str(partial);
                                                    }
                                                    yield Ok(StreamChunk::new(ChunkType::ToolCallDelta).with_content(partial));
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_stop" => {
                                current_tool_idx = -1;
                            }
                            "message_delta" => {
                                // Capture usage tokens from message_delta.
                                if let Some(usage) = evt.get("usage") {
                                    usage_input = usage.get("input_tokens").and_then(|t| t.as_i64()).unwrap_or(usage_input);
                                    usage_output = usage.get("output_tokens").and_then(|t| t.as_i64()).unwrap_or(usage_output);
                                }
                            }
                            "message_start" => {
                                // Initial usage (prompt tokens) from message_start.
                                if let Some(msg) = evt.get("message") {
                                    if let Some(usage) = msg.get("usage") {
                                        usage_input = usage.get("input_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                                        usage_output = usage.get("output_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                                    }
                                }
                            }
                            "message_stop" => {
                                if let Ok(mut guard) = last_usage.lock() {
                                    *guard = Some(TokenUsage::new(usage_input, usage_output, usage_input + usage_output));
                                }
                                // Build the final LLMResponse matching Python's message_stop handler.
                                let final_resp = if !tool_blocks.is_empty() {
                                    let reasoning = if accumulated_text.is_empty() { None } else { Some(accumulated_text.clone()) };
                                    let calls: Vec<crate::clients::base::ToolCall> = tool_blocks
                                        .iter()
                                        .enumerate()
                                        .map(|(i, (name, args_json))| {
                                            let args = serde_json::from_str::<Value>(args_json)
                                                .ok()
                                                .and_then(|v| v.as_object().cloned())
                                                .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                                                .unwrap_or_default();
                                            let mut call = crate::clients::base::ToolCall::new(name, args);
                                            if i == 0 {
                                                if let Some(ref r) = reasoning {
                                                    call = call.with_reasoning(r);
                                                }
                                            }
                                            call
                                        })
                                        .collect();
                                    crate::clients::base::LLMResponse::ToolCalls(calls)
                                } else {
                                    crate::clients::base::LLMResponse::Text(
                                        crate::clients::base::TextResponse::new(&accumulated_text)
                                    )
                                };
                                yield Ok(StreamChunk::new(ChunkType::Final).with_response(final_resp));
                                return;
                            }
                            _ => {}
                        }
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        Ok(Some(200_000))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::base::LLMResponse;
    use serde_json::json;

    fn make_tool_spec(name: &str, desc: &str, props: &[(&str, &str)]) -> ToolSpec {
        let mut properties = json!({});
        let prop_map = properties.as_object_mut().expect("object");
        for (pname, ptype) in props {
            prop_map.insert(pname.to_string(), json!({"type": ptype}));
        }
        let schema = json!({"type": "object", "properties": properties});
        ToolSpec::from_json_schema(name, desc, &schema).expect("valid spec")
    }

    #[test]
    fn convert_tools_basic() {
        let spec = make_tool_spec("read_file", "Read a file", &[("path", "string")]);
        let tools = convert::convert_tools(&[spec]);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "read_file");
    }

    #[test]
    fn convert_tools_enum() {
        let mut properties = json!({});
        if let Some(m) = properties.as_object_mut() {
            m.insert(
                "mode".to_string(),
                json!({"type": "string", "enum": ["fast", "slow"]}),
            );
        }
        let schema = json!({"type": "object", "properties": properties});
        let spec = ToolSpec::from_json_schema("run", "Run", &schema).expect("ok");
        let tools = convert::convert_tools(&[spec]);
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn convert_tools_optional_params() {
        let schema = json!({
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 10}},
            "required": ["query"],
        });
        let spec = ToolSpec::from_json_schema("search", "Search", &schema).expect("ok");
        assert_eq!(convert::convert_tools(&[spec]).len(), 1);
    }

    #[test]
    fn convert_tools_multiple() {
        let specs = vec![
            make_tool_spec("a", "A", &[("x", "string")]),
            make_tool_spec("b", "B", &[("y", "number")]),
        ];
        assert_eq!(convert::convert_tools(&specs).len(), 2);
    }

    #[test]
    fn convert_messages_system_extraction() {
        let msgs = vec![
            json!({"role": "system", "content": "You are helpful."}),
            json!({"role": "user", "content": "Hello"}),
        ];
        let (sys, conv) = convert::convert_messages(&msgs);
        assert_eq!(sys, Some(json!("You are helpful.")));
        assert_eq!(conv.len(), 1);
    }

    #[test]
    fn convert_messages_user_assistant() {
        let msgs = vec![
            json!({"role": "user", "content": "Hi"}),
            json!({"role": "assistant", "content": "Hello!"}),
        ];
        let (sys, conv) = convert::convert_messages(&msgs);
        assert!(sys.is_none());
        assert_eq!(conv.len(), 2);
    }

    #[test]
    fn convert_messages_tool_call() {
        let msgs = vec![
            json!({"role": "user", "content": "Read"}),
            json!({"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_123",
                "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/a\"}"},
            }]}),
            json!({"role": "tool", "tool_call_id": "call_123", "content": "data"}),
        ];
        let (_, conv) = convert::convert_messages(&msgs);
        assert_eq!(conv.len(), 3);
        let content = conv[1]["content"].as_array().expect("array");
        let tu = content
            .iter()
            .find(|b| b["type"] == "tool_use")
            .expect("found");
        assert_eq!(tu["name"], "read_file");
        assert_eq!(tu["input"]["path"], "/tmp/a");
    }

    #[test]
    fn convert_messages_tool_result() {
        let msgs = vec![json!({"role": "tool", "tool_call_id": "c1", "content": "result"})];
        let (_, conv) = convert::convert_messages(&msgs);
        assert_eq!(conv[0]["role"], "user");
        let blocks = conv[0]["content"].as_array().expect("array");
        assert_eq!(blocks[0]["type"], "tool_result");
    }

    #[test]
    fn convert_messages_unpaired_tool_use() {
        let msgs = vec![
            json!({
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "abc", "function": {"name": "run", "arguments": "{}"}}],
            }),
            json!({"role": "user", "content": "next"}),
        ];
        let (_, conv) = convert::convert_messages(&msgs);
        let blocks = conv[1]["content"].as_array().expect("array");
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "abc");
        assert_eq!(blocks[0]["is_error"], true);
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "next");
    }

    #[test]
    fn convert_messages_consecutive_merging() {
        let msgs = vec![
            json!({"role": "user", "content": "First"}),
            json!({"role": "user", "content": "Second"}),
        ];
        let (_, conv) = convert::convert_messages(&msgs);
        assert_eq!(conv.len(), 1);
        let blocks = conv[0]["content"].as_array().expect("array");
        assert_eq!(blocks[0]["text"], "First");
        assert_eq!(blocks[1]["text"], "Second");
    }

    #[test]
    fn convert_messages_multi_step() {
        let msgs = vec![
            json!({"role": "system", "content": "Sys"}),
            json!({"role": "user", "content": "Do"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "read", "arguments": "{}"}},
            ]}),
            json!({"role": "tool", "tool_call_id": "c1", "content": "data"}),
            json!({"role": "assistant", "content": "Done"}),
        ];
        let (sys, conv) = convert::convert_messages(&msgs);
        assert_eq!(sys, Some(json!("Sys")));
        assert_eq!(conv.len(), 4);
    }

    #[test]
    fn convert_messages_arguments_as_dict() {
        let msgs = vec![json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "run", "arguments": {"path": "/tmp/a"}}}],
        })];
        let (_, conv) = convert::convert_messages(&msgs);
        let content = conv[0]["content"].as_array().expect("array");
        let tu = content
            .iter()
            .find(|b| b["type"] == "tool_use")
            .expect("found");
        assert_eq!(tu["input"]["path"], "/tmp/a");
    }

    #[test]
    fn convert_messages_missing_tool_id_uses_non_empty_fallback() {
        let msgs = vec![
            json!({
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": "run", "arguments": "{}"}}],
            }),
            json!({"role": "user", "content": "next"}),
        ];
        let (_, conv) = convert::convert_messages(&msgs);
        let tool_use = conv[0]["content"].as_array().expect("array")[0].clone();
        assert_eq!(tool_use["id"], "toolu_0");
        let synthetic = conv[1]["content"].as_array().expect("array")[0].clone();
        assert_eq!(synthetic["tool_use_id"], "toolu_0");
    }

    #[test]
    fn convert_messages_merges_array_and_text_same_role() {
        let msgs = vec![
            json!({"role": "tool", "tool_call_id": "c1", "content": "result"}),
            json!({"role": "user", "content": "follow up"}),
        ];
        let (_, conv) = convert::convert_messages(&msgs);
        assert_eq!(conv.len(), 1);
        let blocks = conv[0]["content"].as_array().expect("array");
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], "follow up");
    }

    #[test]
    fn parse_response_text() {
        let r = json!({"content": [{"type": "text", "text": "Hello"}]});
        match convert::parse_response(&r) {
            LLMResponse::Text(tr) => assert_eq!(tr.content, "Hello"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_response_tool_use() {
        let r = json!({"content": [
            {"type": "tool_use", "id": "tu1", "name": "read", "input": {"path": "/x"}},
        ]});
        match convert::parse_response(&r) {
            LLMResponse::ToolCalls(c) => {
                assert_eq!(c[0].tool, "read");
                assert_eq!(c[0].args["path"], "/x");
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_response_tool_use_with_reasoning() {
        let r = json!({"content": [
            {"type": "text", "text": "Thinking..."},
            {"type": "tool_use", "id": "tu1", "name": "run", "input": {}},
        ]});
        match convert::parse_response(&r) {
            LLMResponse::ToolCalls(c) => {
                assert_eq!(c[0].reasoning, Some("Thinking...".to_string()));
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_response_empty_content() {
        let r = json!({"content": []});
        match convert::parse_response(&r) {
            LLMResponse::Text(tr) => assert_eq!(tr.content, ""),
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn get_context_length_returns_200k() {
        let client = AnthropicClient::new("claude-3", None);
        assert_eq!(
            client.get_context_length().await.expect("ok"),
            Some(200_000)
        );
    }

    #[test]
    fn record_usage_extracts_tokens() {
        let client = AnthropicClient::new("claude-3", None);
        client.record_usage(&json!({"usage": {"input_tokens": 42, "output_tokens": 7}}));
        let u = client.get_last_usage().expect("set");
        assert_eq!(u.prompt_tokens, 42);
        assert_eq!(u.total_tokens, 49);
    }

    #[test]
    fn rebuilt_body_maps_passthrough_to_anthropic_fields() {
        let client = AnthropicClient::new("fallback-model", None);
        let mut passthrough = serde_json::Map::new();
        passthrough.insert("model".to_string(), json!("request-model"));
        passthrough.insert("max_tokens".to_string(), json!(128));
        passthrough.insert("stop".to_string(), json!(["done"]));
        passthrough.insert(
            "tool_choice".to_string(),
            json!({"type": "function", "function": {"name": "search"}}),
        );
        passthrough.insert("user".to_string(), json!("user-123"));

        let body = client.build_body_with_options(
            vec![json!({"role": "user", "content": "hi"})],
            Some(vec![make_tool_spec(
                "search",
                "Search",
                &[("query", "string")],
            )]),
            LLMRequestOptions {
                passthrough: Some(passthrough),
                ..Default::default()
            },
            false,
        );

        assert_eq!(body["model"], "request-model");
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["stop_sequences"], json!(["done"]));
        assert_eq!(
            body["tool_choice"],
            json!({"type": "tool", "name": "search"})
        );
        assert_eq!(body["metadata"]["user_id"], "user-123");
    }

    #[tokio::test]
    async fn raw_anthropic_body_is_sent_verbatim_for_clean_path() {
        let mut server = mockito::Server::new_async().await;
        let raw = json!({
            "model": "claude-request",
            "max_tokens": 128,
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
            }],
            "metadata": {"user_id": "user-123"}
        });
        let mock = server
            .mock("POST", "/messages")
            .match_header("content-type", "application/json")
            .match_body(mockito::Matcher::Json(raw.clone()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 10, "output_tokens": 2}
                })
                .to_string(),
            )
            .create_async()
            .await;

        let client = AnthropicClient::new("fallback-model", Some("test-key".to_string()))
            .with_base_url(server.url())
            .with_timeout(5.0);
        let result = client
            .send_with_options(
                vec![json!({"role": "user", "content": "mutated"})],
                None,
                LLMRequestOptions {
                    inbound_anthropic_body: Some(raw),
                    ..Default::default()
                },
            )
            .await
            .expect("accepted");

        match result {
            LLMResponse::Text(text) => assert_eq!(text.content, "ok"),
            _ => panic!("expected text"),
        }
        let usage = client.last_usage().expect("usage");
        assert_eq!(usage.total_tokens, 12);
        mock.assert_async().await;
    }
}
