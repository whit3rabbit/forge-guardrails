use serde_json::{json, Map, Value};
use std::time::Duration;

use super::{helpers, streaming, LlamafileClient, LlamafileMode};
use crate::clients::base::{
    format_tool, ChunkStream, ChunkType, LLMResponse, SamplingParams, StreamChunk, TextResponse,
};
use crate::clients::openai_compat;
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, StreamError};
use crate::prompts::build_tool_prompt;

const UPSTREAM_SEND_MAX_ATTEMPTS: usize = 3;
const UPSTREAM_SEND_RETRY_DELAY: Duration = Duration::from_millis(50);

impl LlamafileClient {
    pub(super) async fn native_send(
        &self,
        messages: Vec<Value>,
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
        passthrough: Option<&Map<String, Value>>,
    ) -> Result<LLMResponse, BackendError> {
        let body = self.build_native_body(&messages, tools, sampling, passthrough);
        let resp = self
            .send_chat_completions(&body)
            .await
            .map_err(|e| BackendError::new(0, e.to_string()))?;
        let status = resp.status().as_u16() as i64;
        if status == 500 {
            return Ok(LLMResponse::Text(TextResponse::new(
                resp.text().await.unwrap_or_default(),
            )));
        }
        if !resp.status().is_success() {
            return Err(BackendError::new(
                status,
                resp.text().await.unwrap_or_default(),
            ));
        }
        let rj: Value = resp
            .json()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;
        self.record_usage(&rj);
        Ok(self.parse_native_response(&rj))
    }

    pub(super) async fn prompt_send(
        &self,
        messages: Vec<Value>,
        tools: &[ToolSpec],
        sampling: Option<&SamplingParams>,
        passthrough: Option<&Map<String, Value>>,
    ) -> Result<LLMResponse, BackendError> {
        let body = self.build_prompt_body(&messages, tools, sampling, passthrough);
        let resp = self
            .send_chat_completions(&body)
            .await
            .map_err(|e| BackendError::new(0, e.to_string()))?;
        let status = resp.status().as_u16() as i64;
        if status == 500 {
            return Ok(LLMResponse::Text(TextResponse::new(
                resp.text().await.unwrap_or_default(),
            )));
        }
        if !resp.status().is_success() {
            return Err(BackendError::new(
                status,
                resp.text().await.unwrap_or_default(),
            ));
        }
        let rj: Value = resp
            .json()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;
        self.record_usage(&rj);
        Ok(self.parse_prompt_response(&rj, tools))
    }

    pub(super) async fn stream_send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
        passthrough: Option<Map<String, Value>>,
        mode: LlamafileMode,
    ) -> Result<ChunkStream, StreamError> {
        let body = self.build_stream_body(
            &messages,
            tools.as_deref(),
            sampling.as_ref(),
            passthrough.as_ref(),
            mode,
        );
        let resp = self
            .send_chat_completions(&body)
            .await
            .map_err(|e| StreamError::new(e.to_string()))?;

        let status = resp.status().as_u16() as i64;
        if status == 500 {
            let text = resp.text().await.unwrap_or_default();
            let chunk = StreamChunk::new(ChunkType::Final)
                .with_response(LLMResponse::Text(TextResponse::new(text)));
            return Ok(Box::pin(futures_util::stream::once(
                async move { Ok(chunk) },
            )));
        }
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(StreamError::new(format!(
                "Backend error (status {}): {}",
                status, text
            )));
        }

        let tool_names: Vec<String> = tools
            .unwrap_or_default()
            .iter()
            .map(|t| t.name.clone())
            .collect();
        let stream = streaming::parse_openai_sse(
            resp,
            self.think,
            tool_names,
            mode == LlamafileMode::Prompt,
            self.last_usage.clone(),
            self.slot_id.unwrap_or(0),
        );
        Ok(Box::pin(stream))
    }

    async fn send_chat_completions(
        &self,
        body: &Value,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let url = format!("{}/chat/completions", self.base_url);
        let timeout = Duration::from_secs_f64(self.timeout_secs);
        let mut attempt = 0;

        loop {
            attempt += 1;
            let result = self
                .http_client
                .post(&url)
                .timeout(timeout)
                .json(body)
                .send()
                .await;

            match result {
                Ok(resp) => return Ok(resp),
                Err(err) if attempt < UPSTREAM_SEND_MAX_ATTEMPTS && retryable_send_error(&err) => {
                    tokio::time::sleep(UPSTREAM_SEND_RETRY_DELAY * attempt as u32).await;
                }
                Err(err) => return Err(err),
            }
        }
    }

    pub(super) fn build_native_body(
        &self,
        messages: &[Value],
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
        passthrough: Option<&Map<String, Value>>,
    ) -> Value {
        let mut merged = helpers::merge_messages(messages);
        openai_compat::normalize_openai_message_tool_call_ids(&mut merged);
        let mut body = Value::Object(passthrough.cloned().unwrap_or_default());
        if let Some(obj) = body.as_object_mut() {
            obj.entry("model").or_insert_with(|| json!(self.model));
            obj.insert("messages".into(), json!(merged));
            obj.insert("stream".into(), Value::Bool(false));
            obj.insert("cache_prompt".into(), json!(self.cache_prompt));
        }
        if let Some(tl) = tools {
            if !tl.is_empty() {
                let fmt: Vec<Value> = tl.iter().map(format_tool).collect();
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("tools".into(), json!(fmt));
                }
            }
        }
        self.apply_common_body_options(sampling, &mut body);
        body
    }

    pub(super) fn build_prompt_body(
        &self,
        messages: &[Value],
        tools: &[ToolSpec],
        sampling: Option<&SamplingParams>,
        passthrough: Option<&Map<String, Value>>,
    ) -> Value {
        let mut downgraded = helpers::downgrade_messages_for_prompt(messages);
        inject_tool_prompt(&mut downgraded, tools);
        let mut body = Value::Object(passthrough.cloned().unwrap_or_default());
        if let Some(obj) = body.as_object_mut() {
            obj.entry("model").or_insert_with(|| json!(self.model));
            obj.insert("messages".into(), json!(downgraded));
            obj.insert("stream".into(), Value::Bool(false));
            obj.insert("cache_prompt".into(), json!(self.cache_prompt));
        }
        self.apply_common_body_options(sampling, &mut body);
        body
    }

    pub(super) fn build_stream_body(
        &self,
        messages: &[Value],
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
        passthrough: Option<&Map<String, Value>>,
        mode: LlamafileMode,
    ) -> Value {
        let mut prepared = helpers::merge_messages(messages);
        if mode == LlamafileMode::Native {
            openai_compat::normalize_openai_message_tool_call_ids(&mut prepared);
        }
        let mut body = Value::Object(passthrough.cloned().unwrap_or_default());
        if let Some(obj) = body.as_object_mut() {
            obj.entry("model").or_insert_with(|| json!(self.model));
            obj.insert("messages".into(), json!(prepared));
            obj.insert("stream".into(), Value::Bool(true));
            obj.insert("stream_options".into(), json!({"include_usage": true}));
            obj.insert("cache_prompt".into(), json!(self.cache_prompt));
        }
        if mode == LlamafileMode::Native {
            if let Some(tl) = tools {
                if !tl.is_empty() {
                    let fmt: Vec<Value> = tl.iter().map(format_tool).collect();
                    if let Some(obj) = body.as_object_mut() {
                        obj.insert("tools".into(), json!(fmt));
                    }
                }
            }
        } else if let Some(tl) = tools {
            let mut downgraded = helpers::downgrade_messages_for_prompt(messages);
            inject_tool_prompt(&mut downgraded, tl);
            if let Some(obj) = body.as_object_mut() {
                obj.insert("messages".into(), json!(downgraded));
            }
        }

        self.apply_common_body_options(sampling, &mut body);
        body
    }

    fn apply_common_body_options(&self, sampling: Option<&SamplingParams>, body: &mut Value) {
        helpers::apply_sampling(
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.repeat_penalty,
            self.presence_penalty,
            &self.chat_template_kwargs,
            &self.sampling_defaults,
            sampling,
            body,
        );
        if let Some(s) = self.slot_id {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("slot_id".into(), json!(s));
            }
        }
    }
}

fn retryable_send_error(err: &reqwest::Error) -> bool {
    !err.is_status()
        && !err.is_builder()
        && !err.is_body()
        && !err.is_decode()
        && !err.is_timeout()
        && (err.is_connect() || err.is_request())
}

fn inject_tool_prompt(messages: &mut [Value], tools: &[ToolSpec]) {
    let tool_prompt = build_tool_prompt(tools);
    if let Some(first) = messages.first_mut() {
        let c_str = first
            .get("content")
            .and_then(|c| c.as_str())
            .map(str::to_string);
        if let Some(c) = c_str {
            if let Some(obj) = first.as_object_mut() {
                obj.insert("content".into(), json!(format!("{}\n\n{}", tool_prompt, c)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use futures_util::StreamExt;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    fn run_tool() -> ToolSpec {
        ToolSpec::from_json_schema(
            "run",
            "Run",
            &json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        )
        .expect("valid spec")
    }

    #[test]
    fn native_body_includes_non_empty_tools() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let tool = run_tool();
        let body = c.build_native_body(
            &[json!({"role": "user", "content": "go"})],
            Some(&[tool]),
            None,
            None,
        );
        assert_eq!(body["tools"].as_array().expect("tools").len(), 1);
    }

    #[test]
    fn native_body_omits_empty_tools() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_native_body(
            &[json!({"role": "user", "content": "go"})],
            Some(&[]),
            None,
            None,
        );
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn native_body_rewrites_forge_internal_tool_call_ids() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_native_body(
            &[
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_000000000",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"}
                    }]
                }),
                json!({
                    "role": "tool",
                    "tool_call_id": "call_000000000",
                    "name": "run",
                    "content": "ok"
                }),
            ],
            None,
            None,
            None,
        );
        let messages = body["messages"].as_array().expect("messages");
        let id = messages[0]["tool_calls"][0]["id"].as_str().expect("id");

        assert!(is_mistral_safe_id(id), "{id}");
        assert_eq!(messages[1]["tool_call_id"].as_str(), Some(id));
    }

    fn is_mistral_safe_id(id: &str) -> bool {
        id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric())
    }

    #[test]
    fn prompt_body_downgrades_and_injects_tool_prompt() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let tool = run_tool();
        let body = c.build_prompt_body(
            &[
                json!({"role": "user", "content": "go"}),
                json!({"role": "tool", "content": "done"}),
            ],
            &[tool],
            None,
            None,
        );
        let messages = body["messages"].as_array().expect("messages");
        assert!(messages[0]["content"]
            .as_str()
            .expect("content")
            .contains("run"));
        assert!(messages[0]["content"]
            .as_str()
            .expect("content")
            .contains("\n\ngo"));
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn stream_body_includes_usage_options() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_stream_body(
            &[json!({"role": "user", "content": "go"})],
            None,
            None,
            None,
            LlamafileMode::Native,
        );
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn stream_native_body_rewrites_forge_internal_tool_call_ids() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_stream_body(
            &[
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_000000000",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"}
                    }]
                }),
                json!({
                    "role": "tool",
                    "tool_call_id": "call_000000000",
                    "name": "run",
                    "content": "ok"
                }),
            ],
            None,
            None,
            None,
            LlamafileMode::Native,
        );
        let messages = body["messages"].as_array().expect("messages");
        let id = messages[0]["tool_calls"][0]["id"].as_str().expect("id");

        assert!(is_mistral_safe_id(id), "{id}");
        assert_eq!(messages[1]["tool_call_id"].as_str(), Some(id));
    }

    #[test]
    fn stream_prompt_body_downgrades_tool_calls_and_injects_prompt() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let tool = run_tool();
        let body = c.build_stream_body(
            &[json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "run", "arguments": "{\"path\":\"/x\"}"}}],
            })],
            Some(&[tool]),
            None,
            None,
            LlamafileMode::Prompt,
        );
        let content = body["messages"][0]["content"].as_str().expect("content");
        assert!(content.contains("\"tool\": \"run\""));
        assert!(content.contains("\n\n"));
    }

    #[test]
    fn sampling_absent_by_default() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_native_body(&[], None, None, None);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn sampling_populated() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.7);
        let body = c.build_native_body(&[], None, None, None);
        assert_eq!(body["temperature"], 0.7);
    }

    #[test]
    fn sampling_per_call_override() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let body = c.build_native_body(&[], None, Some(&sp), None);
        assert_eq!(body["temperature"], 0.9);
    }

    #[test]
    fn sampling_instance_immutability() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let _ = c.build_native_body(&[], None, Some(&sp), None);
        let body = c.build_native_body(&[], None, None, None);
        assert_eq!(body["temperature"], 0.5);
    }

    #[test]
    fn slot_id_injection() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_slot_id(3);
        let body = c.build_native_body(&[], None, None, None);
        assert_eq!(body["slot_id"], 3);
    }

    #[test]
    fn slot_id_default_noop() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_native_body(&[], None, None, None);
        assert!(body.get("slot_id").is_none());
    }

    #[test]
    fn recommended_sampling_explicit_override() {
        let c = LlamafileClient::new(Path::new("qwen3:8b-q4_K_M.gguf"))
            .with_recommended_sampling(true)
            .with_temperature(0.99);
        let body = c.build_native_body(&[], None, None, None);
        assert_eq!(body["temperature"], 0.99);
    }

    #[tokio::test]
    async fn native_send_retries_pre_response_transport_error() {
        let base_url = retry_server(
            "application/json",
            json!({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })
            .to_string(),
        )
        .await;
        let c = LlamafileClient::new(Path::new("t.gguf")).with_base_url(base_url);

        let response = c
            .native_send(
                vec![json!({"role": "user", "content": "go"})],
                None,
                None,
                None,
            )
            .await
            .expect("request retried");

        match response {
            LLMResponse::Text(text) => assert_eq!(text.content, "ok"),
            _ => panic!("expected text response"),
        }
    }

    #[tokio::test]
    async fn stream_send_retries_pre_response_transport_error() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],",
            "\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
            "data: [DONE]\n\n"
        )
        .to_string();
        let base_url = retry_server("text/event-stream", body).await;
        let c = LlamafileClient::new(Path::new("t.gguf")).with_base_url(base_url);

        let mut stream = c
            .stream_send(
                vec![json!({"role": "user", "content": "go"})],
                None,
                None,
                None,
                LlamafileMode::Native,
            )
            .await
            .expect("stream request retried");
        let mut final_text = None;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("stream chunk");
            if chunk.chunk_type == ChunkType::Final {
                if let Some(LLMResponse::Text(text)) = chunk.response {
                    final_text = Some(text.content);
                }
            }
        }

        assert_eq!(final_text.as_deref(), Some("ok"));
    }

    async fn retry_server(content_type: &'static str, body: String) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind retry server");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let (first, _) = listener.accept().await.expect("first connection");
            drop(first);

            let (mut second, _) = listener.accept().await.expect("second connection");
            let mut buffer = [0_u8; 4096];
            let _ = second.read(&mut buffer).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            second
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        });
        format!("http://{addr}/v1")
    }
}
