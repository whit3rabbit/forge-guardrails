//! Anthropic Messages API client adapter.
//!
//! Converts messages from the common wire format to Anthropic's format before
//! each API call. Uses reqwest for HTTP. The sampling parameter is accepted
//! for protocol symmetry but ignored (Anthropic controls sampling server-side).

pub(crate) mod convert;

use std::sync::Mutex;

use serde_json::Value;

use crate::client::{ApiFormat, ChunkStream, LLMClient, SamplingParams, TokenUsage};
use crate::error::{BackendError, ContextDiscoveryError, StreamError};
use crate::tool_spec::ToolSpec;

/// Client for Anthropic Messages API (Claude models).
pub struct AnthropicClient {
    base_url: String,
    model: String,
    api_key: Option<String>,
    max_tokens: i64,
    timeout_secs: f64,
    max_retries: i64,
    tool_choice: Option<String>,
    last_usage: Mutex<Option<TokenUsage>>,
}

impl AnthropicClient {
    pub fn new(model: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            base_url: "https://api.anthropic.com/v1".to_string(),
            model: model.into(),
            api_key,
            max_tokens: 4096,
            timeout_secs: 300.0,
            max_retries: 3,
            tool_choice: None,
            last_usage: Mutex::new(None),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: i64) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_timeout(mut self, timeout_secs: f64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    pub fn with_max_retries(mut self, max_retries: i64) -> Self {
        self.max_retries = max_retries;
        self
    }

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

    pub fn get_last_usage(&self) -> Option<TokenUsage> {
        self.last_usage.lock().ok().and_then(|guard| guard.clone())
    }
}

impl LLMClient for AnthropicClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<crate::streaming::LLMResponse, BackendError> {
        if let Some(sp) = &sampling {
            log::debug!(
                "AnthropicClient: ignoring sampling keys: {:?}",
                sp.keys().collect::<Vec<_>>()
            );
        }

        let (_system, body) = convert::build_request_body(
            &self.model,
            &messages,
            self.max_tokens,
            tools.as_deref(),
            self.tool_choice.as_deref(),
        );

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
        if let Some(sp) = &sampling {
            log::debug!(
                "AnthropicClient: ignoring sampling keys: {:?}",
                sp.keys().collect::<Vec<_>>()
            );
        }

        let (_system, body) = convert::build_request_body(
            &self.model,
            &messages,
            self.max_tokens,
            tools.as_deref(),
            self.tool_choice.as_deref(),
        );

        let mut body = body;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".to_string(), Value::Bool(true));
        }

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

        // SSE streaming: yield accumulated chunks.
        // Full SSE parsing requires a proper event parser. This placeholder
        // returns the accumulated response as a single Final chunk once the
        // stream completes.
        let byte_stream = resp.bytes_stream();
        let stream = async_stream::stream! {
            use futures_util::StreamExt;
            let mut buf = Vec::new();
            let mut inner = Box::pin(byte_stream);
            while let Some(chunk) = inner.next().await {
                match chunk {
                    Ok(bytes) => buf.extend_from_slice(&bytes),
                    Err(e) => {
                        yield Err(StreamError::new(e.to_string()));
                        return;
                    }
                }
            }
            let body_str = String::from_utf8_lossy(&buf);
            // Extract JSON from SSE data lines
            for line in body_str.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" { continue; }
                    if let Ok(evt) = serde_json::from_str::<Value>(data) {
                        match evt.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                            "content_block_delta" => {
                                if let Some(delta) = evt.get("delta") {
                                    if delta.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                                        if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                            yield Ok(crate::streaming::StreamChunk::new(
                                                crate::streaming::ChunkType::TextDelta,
                                            ).with_content(text));
                                        }
                                    }
                                }
                            }
                            "message_delta" => {
                                let response = evt.get("message").cloned().unwrap_or(serde_json::json!({}));
                                let parsed = convert::parse_response(&response);
                                yield Ok(crate::streaming::StreamChunk::new(
                                    crate::streaming::ChunkType::Final,
                                ).with_response(parsed));
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
    use crate::streaming::LLMResponse;
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
        properties.as_object_mut().map(|m| {
            m.insert(
                "mode".to_string(),
                json!({"type": "string", "enum": ["fast", "slow"]}),
            );
        });
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
        let msgs = vec![json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "abc", "function": {"name": "run", "arguments": "{}"}}],
        })];
        let (_, conv) = convert::convert_messages(&msgs);
        let has_synthetic = conv.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_array())
                .map(|b| b.iter().any(|bl| bl["is_error"] == true))
                .unwrap_or(false)
        });
        assert!(has_synthetic);
    }

    #[test]
    fn convert_messages_consecutive_merging() {
        let msgs = vec![
            json!({"role": "user", "content": "First"}),
            json!({"role": "user", "content": "Second"}),
        ];
        let (_, conv) = convert::convert_messages(&msgs);
        assert_eq!(conv.len(), 1);
        assert_eq!(conv[0]["content"], "First\nSecond");
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
}
