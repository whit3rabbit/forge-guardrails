use serde_json::{json, Value};

use super::{helpers, streaming, LlamafileClient, LlamafileMode};
use crate::clients::base::{
    format_tool, ChunkStream, ChunkType, LLMResponse, SamplingParams, StreamChunk, TextResponse,
};
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, StreamError};
use crate::prompts::build_tool_prompt;

impl LlamafileClient {
    pub(super) async fn native_send(
        &self,
        messages: Vec<Value>,
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        let body = self.build_native_body(&messages, tools, sampling);
        let resp = reqwest::Client::new()
            .post(format!("{}/chat/completions", self.base_url))
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body)
            .send()
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
    ) -> Result<LLMResponse, BackendError> {
        let body = self.build_prompt_body(&messages, tools, sampling);
        let resp = reqwest::Client::new()
            .post(format!("{}/chat/completions", self.base_url))
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body)
            .send()
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
        mode: LlamafileMode,
    ) -> Result<ChunkStream, StreamError> {
        let body = self.build_stream_body(&messages, tools.as_deref(), sampling.as_ref(), mode);
        let resp = reqwest::Client::new()
            .post(format!("{}/chat/completions", self.base_url))
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body)
            .send()
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

    pub(super) fn build_native_body(
        &self,
        messages: &[Value],
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
    ) -> Value {
        let merged = helpers::merge_messages(messages);
        let mut body = json!({
            "model": self.model,
            "messages": merged,
            "stream": false,
            "cache_prompt": self.cache_prompt
        });
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
    ) -> Value {
        let mut downgraded = helpers::downgrade_messages_for_prompt(messages);
        inject_tool_prompt(&mut downgraded, tools);
        let mut body = json!({
            "model": self.model,
            "messages": downgraded,
            "stream": false,
            "cache_prompt": self.cache_prompt
        });
        self.apply_common_body_options(sampling, &mut body);
        body
    }

    pub(super) fn build_stream_body(
        &self,
        messages: &[Value],
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
        mode: LlamafileMode,
    ) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": helpers::merge_messages(messages),
            "stream": true,
            "stream_options": {"include_usage": true},
            "cache_prompt": self.cache_prompt
        });
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

    use serde_json::json;

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
        );
        assert_eq!(body["tools"].as_array().expect("tools").len(), 1);
    }

    #[test]
    fn native_body_omits_empty_tools() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body =
            c.build_native_body(&[json!({"role": "user", "content": "go"})], Some(&[]), None);
        assert!(body.get("tools").is_none());
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
            LlamafileMode::Native,
        );
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
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
            LlamafileMode::Prompt,
        );
        let content = body["messages"][0]["content"].as_str().expect("content");
        assert!(content.contains("\"tool\": \"run\""));
        assert!(content.contains("\n\n"));
    }

    #[test]
    fn sampling_absent_by_default() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_native_body(&[], None, None);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn sampling_populated() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.7);
        let body = c.build_native_body(&[], None, None);
        assert_eq!(body["temperature"], 0.7);
    }

    #[test]
    fn sampling_per_call_override() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let body = c.build_native_body(&[], None, Some(&sp));
        assert_eq!(body["temperature"], 0.9);
    }

    #[test]
    fn sampling_instance_immutability() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let _ = c.build_native_body(&[], None, Some(&sp));
        let body = c.build_native_body(&[], None, None);
        assert_eq!(body["temperature"], 0.5);
    }

    #[test]
    fn slot_id_injection() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_slot_id(3);
        let body = c.build_native_body(&[], None, None);
        assert_eq!(body["slot_id"], 3);
    }

    #[test]
    fn slot_id_default_noop() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let body = c.build_native_body(&[], None, None);
        assert!(body.get("slot_id").is_none());
    }

    #[test]
    fn recommended_sampling_explicit_override() {
        let c = LlamafileClient::new(Path::new("qwen3:8b-q4_K_M.gguf"))
            .with_recommended_sampling(true)
            .with_temperature(0.99);
        let body = c.build_native_body(&[], None, None);
        assert_eq!(body["temperature"], 0.99);
    }
}
