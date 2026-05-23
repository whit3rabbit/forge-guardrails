//! Llamafile (llama-server) client adapter using OpenAI-compatible chat API.
//!
//! Supports three modes: native (tools parameter), prompt (inject tool
//! descriptions into prompt), and auto (tries native, falls back on HTTP
//! error). Context length discovered from server properties endpoint.

pub(crate) mod helpers;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use serde_json::{json, Map, Value};

use crate::clients::base::{
    ApiFormat, ChunkStream, ChunkType, LLMClient, LLMResponse, SamplingParams, StreamChunk,
    TextResponse, TokenUsage, ToolCall,
};
use crate::clients::sampling::get_sampling_defaults;
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};
use crate::prompts::{build_tool_prompt, extract_tool_call};

/// Function calling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamafileMode {
    Native,
    Prompt,
    Auto,
}

/// Client for Llamafile using the OpenAI-compatible chat completions API.
pub struct LlamafileClient {
    base_url: String,
    model: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i64>,
    min_p: Option<f64>,
    repeat_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    chat_template_kwargs: Option<Map<String, Value>>,
    mode: LlamafileMode,
    resolved_mode: Mutex<Option<LlamafileMode>>,
    timeout_secs: f64,
    think: bool,
    cache_prompt: bool,
    slot_id: Option<i64>,
    last_usage: Arc<Mutex<HashMap<i64, TokenUsage>>>,
    recommended_sampling: bool,
    sampling_defaults: Option<Map<String, Value>>,
}

impl LlamafileClient {
    pub fn new(gguf_path: impl AsRef<Path>) -> Self {
        let model = helpers::extract_model_identity(gguf_path.as_ref());
        Self {
            base_url: "http://localhost:8080/v1".to_string(),
            model,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            presence_penalty: None,
            chat_template_kwargs: None,
            mode: LlamafileMode::Auto,
            resolved_mode: Mutex::new(None),
            timeout_secs: 300.0,
            think: true,
            cache_prompt: true,
            slot_id: None,
            last_usage: Arc::new(Mutex::new(HashMap::new())),
            recommended_sampling: false,
            sampling_defaults: None,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }
    pub fn with_top_p(mut self, v: f64) -> Self {
        self.top_p = Some(v);
        self
    }
    pub fn with_top_k(mut self, v: i64) -> Self {
        self.top_k = Some(v);
        self
    }
    pub fn with_min_p(mut self, v: f64) -> Self {
        self.min_p = Some(v);
        self
    }
    pub fn with_repeat_penalty(mut self, v: f64) -> Self {
        self.repeat_penalty = Some(v);
        self
    }
    pub fn with_presence_penalty(mut self, v: f64) -> Self {
        self.presence_penalty = Some(v);
        self
    }
    pub fn with_chat_template_kwargs(mut self, kw: Map<String, Value>) -> Self {
        self.chat_template_kwargs = Some(kw);
        self
    }

    pub fn with_mode(mut self, mode: &str) -> Self {
        self.mode = match mode {
            "native" => LlamafileMode::Native,
            "prompt" => LlamafileMode::Prompt,
            _ => LlamafileMode::Auto,
        };
        if self.mode != LlamafileMode::Auto {
            if let Ok(mut g) = self.resolved_mode.lock() {
                *g = Some(self.mode);
            }
        }
        self
    }

    pub fn with_timeout(mut self, s: f64) -> Self {
        self.timeout_secs = s;
        self
    }
    pub fn with_think(mut self, t: Option<bool>) -> Self {
        self.think = t.unwrap_or(true);
        self
    }
    pub fn with_cache_prompt(mut self, c: bool) -> Self {
        self.cache_prompt = c;
        self
    }
    pub fn with_slot_id(mut self, s: i64) -> Self {
        self.slot_id = Some(s);
        self
    }

    pub fn with_recommended_sampling(mut self, enabled: bool) -> Self {
        self.recommended_sampling = enabled;
        if enabled {
            let d = get_sampling_defaults(&self.model);
            if !d.is_empty() {
                self.sampling_defaults = Some(d);
            }
        }
        self
    }

    pub fn model_identity(&self) -> &str {
        &self.model
    }

    fn get_resolved_mode(&self) -> Option<LlamafileMode> {
        self.resolved_mode.lock().ok().and_then(|g| *g)
    }

    fn set_resolved_mode(&self, m: LlamafileMode) {
        if let Ok(mut g) = self.resolved_mode.lock() {
            *g = Some(m);
        }
    }

    fn record_usage(&self, response: &Value) {
        let u = response.get("usage");
        let p = u
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let c = u
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|t| t.as_i64())
            .unwrap_or(0);
        let key = self.slot_id.unwrap_or(0);
        if let Ok(mut g) = self.last_usage.lock() {
            g.insert(key, TokenUsage::new(p, c, p + c));
        }
    }

    pub fn get_usage(&self, slot: i64) -> Option<TokenUsage> {
        self.last_usage.lock().ok()?.get(&slot).cloned()
    }

    fn parse_native_response(&self, response: &Value) -> LLMResponse {
        let choice = response.get("choices").and_then(|c| c.get(0));
        let message = choice.and_then(|c| c.get("message"));
        let reasoning = helpers::resolve_reasoning(self.think, response);

        if let Some(tcs) = message
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())
        {
            if !tcs.is_empty() {
                let mut calls = Vec::new();
                for (i, tc) in tcs.iter().enumerate() {
                    let name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("");
                    let id = tc.get("id").and_then(|i| i.as_str()).map(|s| s.to_string());
                    let args_raw = tc.get("function").and_then(|f| f.get("arguments"));
                    let args = parse_args(args_raw);
                    let mut call = ToolCall::new(name, args);
                    if let Some(id_val) = id {
                        call = call.with_id(id_val);
                    }
                    if i == 0 {
                        if let Some(r) = reasoning.as_ref() {
                            call = call.with_reasoning(r);
                        }
                    }
                    calls.push(call);
                }
                return LLMResponse::ToolCalls(calls);
            }
        }

        let content = message
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let cleaned = if self.think {
            helpers::strip_reasoning_tags(content)
        } else {
            content.to_string()
        };
        LLMResponse::Text(TextResponse::new(cleaned))
    }

    fn parse_prompt_response(&self, response: &Value, tools: &[ToolSpec]) -> LLMResponse {
        let content = response
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        let (reasoning, cleaned) = if self.think {
            helpers::extract_think_tags(content)
        } else {
            (String::new(), content.to_string())
        };
        let mut calls = extract_tool_call(&cleaned, &names);
        if calls.is_empty() {
            LLMResponse::Text(TextResponse::new(cleaned))
        } else {
            if let Some(first) = calls.first_mut() {
                if !reasoning.is_empty() {
                    *first = first.clone().with_reasoning(reasoning);
                }
            }
            LLMResponse::ToolCalls(calls)
        }
    }

    async fn native_send(
        &self,
        messages: Vec<Value>,
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        let merged = helpers::merge_messages(&messages);
        let mut body = json!({"model": self.model, "messages": merged, "stream": false, "cache_prompt": self.cache_prompt});
        if let Some(tl) = tools {
            if !tl.is_empty() {
                let fmt: Vec<Value> = tl.iter().map(crate::clients::base::format_tool).collect();
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("tools".into(), json!(fmt));
                }
            }
        }
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
            &mut body,
        );
        if let Some(s) = self.slot_id {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("slot_id".into(), json!(s));
            }
        }
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

    async fn prompt_send(
        &self,
        messages: Vec<Value>,
        tools: &[ToolSpec],
        sampling: Option<&SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        let tool_prompt = build_tool_prompt(tools);
        let mut downgraded = helpers::downgrade_messages_for_prompt(&messages);
        if let Some(first) = downgraded.first_mut() {
            let c_str = first
                .get("content")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string());
            if let Some(c) = c_str {
                if let Some(obj) = first.as_object_mut() {
                    obj.insert("content".into(), json!(format!("{}\n\n{}", tool_prompt, c)));
                }
            }
        }
        let mut body = json!({"model": self.model, "messages": downgraded, "stream": false, "cache_prompt": self.cache_prompt});
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
            &mut body,
        );
        if let Some(s) = self.slot_id {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("slot_id".into(), json!(s));
            }
        }
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
}

fn parse_args(args_raw: Option<&Value>) -> IndexMap<String, Value> {
    match args_raw {
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(obj)) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            _ => IndexMap::new(),
        },
        Some(Value::Object(obj)) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => IndexMap::new(),
    }
}

impl LLMClient for LlamafileClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::OpenAI
    }

    fn last_usage(&self) -> Option<crate::clients::base::TokenUsage> {
        let slot = self.slot_id.unwrap_or(0);
        self.last_usage.lock().ok()?.get(&slot).cloned()
    }

    async fn send(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<LLMResponse, BackendError> {
        match self.get_resolved_mode() {
            Some(LlamafileMode::Prompt) => {
                self.prompt_send(messages, &tools.unwrap_or_default(), sampling.as_ref())
                    .await
            }
            Some(LlamafileMode::Native) => {
                self.native_send(messages, tools.as_deref(), sampling.as_ref())
                    .await
            }
            _ => {
                if tools.as_ref().is_none_or(|t| t.is_empty()) {
                    self.set_resolved_mode(LlamafileMode::Native);
                    return self
                        .native_send(messages, tools.as_deref(), sampling.as_ref())
                        .await;
                }
                match self
                    .native_send(messages.clone(), tools.as_deref(), sampling.as_ref())
                    .await
                {
                    Ok(resp) => {
                        self.set_resolved_mode(LlamafileMode::Native);
                        Ok(resp)
                    }
                    Err(_) => {
                        self.set_resolved_mode(LlamafileMode::Prompt);
                        self.prompt_send(messages, &tools.unwrap_or_default(), sampling.as_ref())
                            .await
                    }
                }
            }
        }
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        let resolved = self.get_resolved_mode();
        if resolved.is_none() && tools.as_ref().is_some_and(|t| !t.is_empty()) {
            let _ = self
                .send(messages.clone(), tools.clone(), sampling.clone())
                .await;
        }
        let mode = self.get_resolved_mode().unwrap_or(LlamafileMode::Native);

        let mut body = json!({
            "model": self.model,
            "messages": helpers::merge_messages(&messages),
            "stream": true,
            // stream_options: include_usage so the final SSE chunk carries
            // token counts, matching Python's {"stream_options": {"include_usage": True}}.
            "stream_options": {"include_usage": true},
            "cache_prompt": self.cache_prompt
        });
        if mode == LlamafileMode::Native {
            if let Some(tl) = &tools {
                if !tl.is_empty() {
                    let fmt: Vec<Value> =
                        tl.iter().map(crate::clients::base::format_tool).collect();
                    if let Some(obj) = body.as_object_mut() {
                        obj.insert("tools".into(), json!(fmt));
                    }
                }
            }
        } else if let Some(tl) = &tools {
            let tp = build_tool_prompt(tl);
            let dw = helpers::downgrade_messages_for_prompt(&messages);
            if let Some(obj) = body.as_object_mut() {
                obj.insert("messages".into(), json!(dw));
            }
            if let Some(msgs) = body.get_mut("messages").and_then(|m| m.as_array_mut()) {
                if let Some(first) = msgs.first_mut() {
                    let c = first
                        .get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string());
                    if let Some(c) = c {
                        if let Some(obj) = first.as_object_mut() {
                            obj.insert("content".into(), json!(format!("{}\n\n{}", tp, c)));
                        }
                    }
                }
            }
        }

        helpers::apply_sampling(
            self.temperature,
            self.top_p,
            self.top_k,
            self.min_p,
            self.repeat_penalty,
            self.presence_penalty,
            &self.chat_template_kwargs,
            &self.sampling_defaults,
            sampling.as_ref(),
            &mut body,
        );
        if let Some(s) = self.slot_id {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("slot_id".into(), json!(s));
            }
        }

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

        let think = self.think;
        let tool_names: Vec<String> = tools
            .unwrap_or_default()
            .iter()
            .map(|t| t.name.clone())
            .collect();
        let stream = parse_openai_sse(
            resp,
            think,
            tool_names,
            mode == LlamafileMode::Prompt,
            self.last_usage.clone(),
            self.slot_id.unwrap_or(0),
        );
        Ok(Box::pin(stream))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        let server_url = self.base_url.trim_end_matches("/v1").trim_end_matches('/');
        let resp = reqwest::Client::new()
            .get(format!("{}/props", server_url))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| ContextDiscoveryError::new(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(ContextDiscoveryError::new(format!(
                "Status {}",
                resp.status()
            )));
        }
        let json: Value = resp
            .json()
            .await
            .map_err(|e| ContextDiscoveryError::new(e.to_string()))?;
        Ok(json
            .get("default_generation_settings")
            .and_then(|s| s.get("n_ctx"))
            .and_then(|n| n.as_i64()))
    }
}

fn parse_openai_sse(
    resp: reqwest::Response,
    think: bool,
    tool_names: Vec<String>,
    is_prompt: bool,
    last_usage: Arc<Mutex<HashMap<i64, TokenUsage>>>,
    slot_id: i64,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    let byte_stream = resp.bytes_stream();
    let stream = async_stream::stream! {
        use futures_util::StreamExt;
        let mut inner = Box::pin(byte_stream);
        let mut line_buf = String::new();

        let mut acc_content = String::new();
        let mut acc_reasoning = String::new();
        // tool_call_parts: index -> (name, accumulated_args_json, optional_id)
        let mut acc_tools: Vec<(String, String, Option<String>)> = Vec::new();
        let mut final_response: Option<LLMResponse> = None;

        loop {
            // Drain buffered lines first
            while let Some(newline_pos) = line_buf.find('\n') {
                let raw = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();
                let Some(data) = raw.strip_prefix("data: ") else { continue; };
                if data == "[DONE]" {
                    match final_response.take() {
                        Some(response) => {
                            yield Ok(StreamChunk::new(ChunkType::Final).with_response(response));
                        }
                        None => {
                            yield Err(StreamError::default());
                        }
                    }
                    return;
                }
                let evt: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(usage) = evt.get("usage") {
                    let prompt = usage.get("prompt_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                    let completion = usage.get("completion_tokens").and_then(|t| t.as_i64()).unwrap_or(0);
                    if let Ok(mut guard) = last_usage.lock() {
                        guard.insert(slot_id, TokenUsage::new(prompt, completion, prompt + completion));
                    }
                }

                // Usage-only chunk (no choices) — Python: self._record_usage(chunk)
                if !evt.get("choices").is_some_and(|c| c.as_array().map(|a| !a.is_empty()).unwrap_or(false)) {
                    continue;
                }

                let choice = &evt["choices"][0];
                let delta = choice.get("delta");

                if let Some(d) = delta {
                    // reasoning_content (llama-server reasoning field)
                    if let Some(rc) = d.get("reasoning_content").and_then(|r| r.as_str()) {
                        if !rc.is_empty() {
                            acc_reasoning.push_str(rc);
                        }
                    }

                    // Regular content
                    if let Some(text) = d.get("content").and_then(|c| c.as_str()) {
                        if !text.is_empty() {
                            acc_content.push_str(text);
                            yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(text));
                        }
                    }

                    // Tool call deltas
                    if let Some(tcs) = d.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tcs {
                            let idx = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                            while acc_tools.len() <= idx {
                                acc_tools.push((String::new(), String::new(), None));
                            }
                            if let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) {
                                acc_tools[idx].0 = name.to_string();
                            }
                            if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")).and_then(|a| a.as_str()) {
                                acc_tools[idx].1.push_str(args);
                                yield Ok(StreamChunk::new(ChunkType::ToolCallDelta).with_content(args));
                            }
                            if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                                acc_tools[idx].2 = Some(id.to_string());
                            }
                        }
                    }
                }

                // finish_reason present → build Final chunk
                if choice.get("finish_reason").and_then(|r| r.as_str()).is_some() {
                    let response = if !acc_tools.is_empty() {
                        let reasoning = if think {
                            helpers::resolve_full_reasoning(&acc_reasoning, &acc_content)
                        } else {
                            None
                        };
                        let mut calls = Vec::new();
                        let mut bad_args = false;
                        for (i, (name, args_json, id)) in acc_tools.iter().enumerate() {
                            let args = if args_json.is_empty() {
                                IndexMap::new()
                            } else {
                                match serde_json::from_str::<Value>(args_json)
                                    .ok()
                                    .and_then(|v| v.as_object().cloned())
                                    .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect::<IndexMap<String, Value>>())
                                {
                                    Some(a) => a,
                                    None => {
                                        bad_args = true;
                                        break;
                                    }
                                }
                            };
                            let mut call = ToolCall::new(name, args);
                            if let Some(id_val) = id {
                                call = call.with_id(id_val);
                            }
                            if i == 0 {
                                if let Some(r) = &reasoning {
                                    call = call.with_reasoning(r);
                                }
                            }
                            calls.push(call);
                        }
                        // Python: on bad JSON args, fall back to text
                        if bad_args {
                            let content = if acc_content.is_empty() {
                                acc_tools.first().map(|(_, a, _)| a.as_str()).unwrap_or("").to_string()
                            } else {
                                acc_content.clone()
                            };
                            LLMResponse::Text(TextResponse::new(content))
                        } else {
                            LLMResponse::ToolCalls(calls)
                        }
                    } else if is_prompt {
                        // Prompt-injected mode: extract JSON tool calls from text
                        let (think_text, cleaned) = helpers::extract_think_tags(&acc_content);
                        let names: Vec<&str> = tool_names.iter().map(|n| n.as_str()).collect();
                        let extracted = extract_tool_call(&cleaned, &names);
                        if extracted.is_empty() {
                            LLMResponse::Text(TextResponse::new(cleaned))
                        } else {
                            let mut result = extracted;
                            if let Some(first) = result.first_mut() {
                                let r = if think {
                                    helpers::resolve_full_reasoning(&acc_reasoning, &think_text)
                                } else {
                                    None
                                };
                                if let Some(r_val) = r {
                                    *first = first.clone().with_reasoning(&r_val);
                                }
                            }
                            LLMResponse::ToolCalls(result)
                        }
                    } else {
                        let cleaned = if think {
                            helpers::strip_reasoning_tags(&acc_content)
                        } else {
                            acc_content.clone()
                        };
                        LLMResponse::Text(TextResponse::new(cleaned))
                    };
                    final_response = Some(response);
                }
            }

            // Read next bytes from network
            match inner.next().await {
                Some(Ok(bytes)) => line_buf.push_str(&String::from_utf8_lossy(&bytes)),
                Some(Err(e)) => { yield Err(StreamError::new(e.to_string())); return; }
                None => {
                    match final_response.take() {
                        Some(response) => {
                            yield Ok(StreamChunk::new(ChunkType::Final).with_response(response));
                        }
                        None => {
                            yield Err(StreamError::default());
                        }
                    }
                    return;
                }
            }
        }
    };
    stream
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::Path;

    #[test]
    fn native_mode_resolved() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_mode("native");
        assert_eq!(c.get_resolved_mode(), Some(LlamafileMode::Native));
    }

    #[test]
    fn prompt_mode_resolved() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_mode("prompt");
        assert_eq!(c.get_resolved_mode(), Some(LlamafileMode::Prompt));
    }

    #[test]
    fn auto_mode_unresolved() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_mode("auto");
        assert_eq!(c.get_resolved_mode(), None);
    }

    #[test]
    fn think_default_true() {
        assert!(LlamafileClient::new(Path::new("t.gguf")).think);
    }
    #[test]
    fn think_explicit_false() {
        assert!(
            !LlamafileClient::new(Path::new("t.gguf"))
                .with_think(Some(false))
                .think
        );
    }

    #[test]
    fn sampling_absent_by_default() {
        let _c = LlamafileClient::new(Path::new("t.gguf"));
        let mut body = json!({});
        helpers::apply_sampling(
            None, None, None, None, None, None, &None, &None, None, &mut body,
        );
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn sampling_populated() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.7);
        let mut body = json!({});
        helpers::apply_sampling(
            c.temperature,
            c.top_p,
            c.top_k,
            c.min_p,
            c.repeat_penalty,
            c.presence_penalty,
            &c.chat_template_kwargs,
            &c.sampling_defaults,
            None,
            &mut body,
        );
        assert_eq!(body["temperature"], 0.7);
    }

    #[test]
    fn sampling_per_call_override() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let mut body = json!({});
        helpers::apply_sampling(
            c.temperature,
            c.top_p,
            c.top_k,
            c.min_p,
            c.repeat_penalty,
            c.presence_penalty,
            &c.chat_template_kwargs,
            &c.sampling_defaults,
            Some(&sp),
            &mut body,
        );
        assert_eq!(body["temperature"], 0.9);
    }

    #[test]
    fn sampling_instance_immutability() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let mut body = json!({});
        helpers::apply_sampling(
            c.temperature,
            c.top_p,
            c.top_k,
            c.min_p,
            c.repeat_penalty,
            c.presence_penalty,
            &c.chat_template_kwargs,
            &c.sampling_defaults,
            Some(&sp),
            &mut body,
        );
        let mut body2 = json!({});
        helpers::apply_sampling(
            c.temperature,
            c.top_p,
            c.top_k,
            c.min_p,
            c.repeat_penalty,
            c.presence_penalty,
            &c.chat_template_kwargs,
            &c.sampling_defaults,
            None,
            &mut body2,
        );
        assert_eq!(body2["temperature"], 0.5);
    }

    #[test]
    fn slot_id_injection() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_slot_id(3);
        let mut body = json!({});
        if let Some(s) = c.slot_id {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("slot_id".into(), json!(s));
            }
        }
        assert_eq!(body["slot_id"], 3);
    }

    #[test]
    fn slot_id_default_noop() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let mut body = json!({});
        if let Some(s) = c.slot_id {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("slot_id".into(), json!(s));
            }
        }
        assert!(body.get("slot_id").is_none());
    }

    #[test]
    fn context_url_strips_v1() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_base_url("http://localhost:8080/v1");
        assert!(c.base_url.ends_with("/v1"));
    }

    #[test]
    fn parse_native_tool_call() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read", "arguments": "{\"path\": \"/x\"}"}},
        ]}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "read");
                assert_eq!(calls[0].args["path"], "/x");
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_native_text() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"content": "Hello"}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::Text(tr) => assert_eq!(tr.content, "Hello"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_native_args_as_dict() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"tool_calls": [
            {"function": {"name": "run", "arguments": {"x": 1}}},
        ]}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::ToolCalls(calls) => assert_eq!(calls[0].args["x"], 1),
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn parse_native_null_content() {
        let c = LlamafileClient::new(Path::new("t.gguf"));
        let resp = json!({"choices": [{"message": {"content": null}}]});
        match c.parse_native_response(&resp) {
            LLMResponse::Text(tr) => assert_eq!(tr.content, ""),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn parse_prompt_strips_think_tags_and_attaches_reasoning() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_mode("prompt");
        let tool =
            ToolSpec::from_json_schema("run", "Run", &json!({"type": "object", "properties": {}}))
                .expect("ok");
        let resp = json!({"choices": [{"message": {"content": "<think >reason</think >{\"tool\":\"run\",\"args\":{}}"}}]});

        match c.parse_prompt_response(&resp, &[tool]) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "run");
                assert_eq!(calls[0].reasoning, Some("reason".to_string()));
            }
            _ => panic!("expected tool calls"),
        }
    }

    #[test]
    fn recommended_sampling_unknown() {
        let c = LlamafileClient::new(Path::new("unknown.gguf")).with_recommended_sampling(true);
        assert!(c.sampling_defaults.is_none());
    }

    #[test]
    fn recommended_sampling_known() {
        let c =
            LlamafileClient::new(Path::new("qwen3:8b-q4_K_M.gguf")).with_recommended_sampling(true);
        assert!(c.sampling_defaults.is_some());
    }

    #[test]
    fn recommended_sampling_explicit_override() {
        let c = LlamafileClient::new(Path::new("qwen3:8b-q4_K_M.gguf"))
            .with_recommended_sampling(true)
            .with_temperature(0.99);
        let mut body = json!({});
        helpers::apply_sampling(
            c.temperature,
            c.top_p,
            c.top_k,
            c.min_p,
            c.repeat_penalty,
            c.presence_penalty,
            &c.chat_template_kwargs,
            &c.sampling_defaults,
            None,
            &mut body,
        );
        assert_eq!(body["temperature"], 0.99);
    }
}
