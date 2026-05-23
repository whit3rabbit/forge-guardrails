//! Ollama native API client adapter.
//!
//! Uses the /api/chat endpoint with the tools parameter for native function
//! calling. Messages pass through to Ollama unchanged. Supports think mode
//! with auto-detection from model name keywords and fallback on unsupported.

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

/// Keywords that indicate a model supports thinking/reasoning mode.
const THINK_KEYWORDS: &[&str] = &["think", "reasoning", "r1", "deepseek", "qwq"];

/// Client using Ollama's native function calling via /api/chat.
pub struct OllamaClient {
    base_url: String,
    model: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i64>,
    min_p: Option<f64>,
    repeat_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    timeout_secs: f64,
    /// Whether think mode is active. Mutex for interior mutability — the
    /// think-unsupported fallback must persist this across &self calls,
    /// matching Python's `self._think = False` mutation pattern.
    think: Mutex<bool>,
    think_resolved: Mutex<bool>,
    num_ctx: Mutex<Option<i64>>,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    sampling_defaults: Option<Map<String, Value>>,
}

impl OllamaClient {
    pub fn new(model: impl Into<String>) -> Self {
        let model_str = model.into();
        let (think, think_resolved) = Self::detect_think_mode(&model_str);
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: model_str,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            presence_penalty: None,
            timeout_secs: 300.0,
            think: Mutex::new(think),
            think_resolved: Mutex::new(think_resolved),
            num_ctx: Mutex::new(None),
            last_usage: Arc::new(Mutex::new(None)),
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
    pub fn with_timeout(mut self, s: f64) -> Self {
        self.timeout_secs = s;
        self
    }

    pub fn with_think(mut self, think: Option<bool>) -> Self {
        match think {
            Some(t) => {
                self.think = Mutex::new(t);
                self.think_resolved = Mutex::new(true);
            }
            None => {
                let (d, r) = Self::detect_think_mode(&self.model);
                self.think = Mutex::new(d);
                self.think_resolved = Mutex::new(r);
            }
        }
        self
    }

    pub fn with_recommended_sampling(mut self, enabled: bool) -> Self {
        if enabled {
            let d = get_sampling_defaults(&self.model);
            if !d.is_empty() {
                self.sampling_defaults = Some(d);
            }
        }
        self
    }

    fn detect_think_mode(model: &str) -> (bool, bool) {
        let lower = model.to_lowercase();
        let matches = THINK_KEYWORDS.iter().any(|kw| lower.contains(kw));
        (matches, false)
    }

    pub fn is_think_enabled(&self) -> bool {
        self.think.lock().map(|g| *g).unwrap_or(false)
    }
    pub fn is_think_resolved(&self) -> bool {
        self.think_resolved.lock().map(|g| *g).unwrap_or(false)
    }

    pub fn set_num_ctx(&self, ctx: Option<i64>) {
        if let Ok(mut g) = self.num_ctx.lock() {
            *g = ctx;
        }
    }

    pub fn build_options(&self, sampling: Option<&SamplingParams>) -> Map<String, Value> {
        let mut opts = Map::new();
        if let Some(ref defaults) = self.sampling_defaults {
            for (k, v) in defaults {
                opts.insert(k.clone(), v.clone());
            }
        }
        if let Some(t) = self.temperature {
            opts.insert("temperature".into(), json!(t));
        }
        if let Some(t) = self.top_p {
            opts.insert("top_p".into(), json!(t));
        }
        if let Some(k) = self.top_k {
            opts.insert("top_k".into(), json!(k));
        }
        if let Some(m) = self.min_p {
            opts.insert("min_p".into(), json!(m));
        }
        if let Some(r) = self.repeat_penalty {
            opts.insert("repeat_penalty".into(), json!(r));
        }
        if let Some(p) = self.presence_penalty {
            opts.insert("presence_penalty".into(), json!(p));
        }
        if let Some(sp) = sampling {
            for (k, v) in sp {
                if matches!(
                    k.as_str(),
                    "temperature"
                        | "top_p"
                        | "top_k"
                        | "min_p"
                        | "repeat_penalty"
                        | "presence_penalty"
                        | "seed"
                ) {
                    opts.insert(k.clone(), v.clone());
                }
            }
        }
        if let Ok(guard) = self.num_ctx.lock() {
            if let Some(ctx) = *guard {
                opts.insert("num_ctx".into(), json!(ctx));
            }
        }
        opts
    }

    pub fn resolve_reasoning(think: bool, response: &Value) -> Option<String> {
        if !think {
            return None;
        }
        let message = response.get("message");
        if let Some(r) = message
            .and_then(|m| m.get("thinking"))
            .and_then(|r| r.as_str())
        {
            if !r.is_empty() {
                return Some(r.to_string());
            }
        }
        message
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
    }

    fn record_usage(&self, response: &Value) {
        let p = response
            .get("prompt_eval_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let c = response
            .get("eval_count")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if let Ok(mut g) = self.last_usage.lock() {
            *g = Some(TokenUsage::new(p, c, p + c));
        }
    }

    pub fn get_last_usage(&self) -> Option<TokenUsage> {
        self.last_usage.lock().ok().and_then(|g| g.clone())
    }

    fn is_think_unsupported_error(response: &Value) -> bool {
        response
            .get("error")
            .and_then(|e| e.as_str())
            .map(|s| {
                let l = s.to_lowercase();
                l.contains("think") && l.contains("support")
                    || l.contains("thinking") && l.contains("not")
            })
            .unwrap_or(false)
    }

    fn build_request_body(
        &self,
        messages: Vec<Value>,
        tools: Option<&[ToolSpec]>,
        sampling: Option<&SamplingParams>,
        think: bool,
    ) -> Value {
        let mut body = json!({"model": self.model, "messages": messages, "stream": false});
        if think {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("think".into(), json!(true));
            }
        }
        if let Some(tl) = tools {
            if !tl.is_empty() {
                let fmt: Vec<Value> = tl.iter().map(|t| json!({
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.get_json_schema()},
                })).collect();
                if let Some(obj) = body.as_object_mut() {
                    obj.insert("tools".into(), json!(fmt));
                }
            }
        }
        let opts = self.build_options(sampling);
        if !opts.is_empty() {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("options".into(), Value::Object(opts));
            }
        }
        body
    }

    fn parse_send_response(&self, response: &Value, think: bool) -> LLMResponse {
        if let Some(calls) = self.parse_tool_calls(response, think) {
            return LLMResponse::ToolCalls(calls);
        }
        let content = response
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        LLMResponse::Text(TextResponse::new(content))
    }

    fn parse_tool_calls(&self, response: &Value, think: bool) -> Option<Vec<ToolCall>> {
        let tcs = response
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|tc| tc.as_array())?;
        if tcs.is_empty() {
            return None;
        }
        let reasoning = Self::resolve_reasoning(think, response);
        let mut calls = Vec::new();
        for (i, tc) in tcs.iter().enumerate() {
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let args_val = tc.get("function").and_then(|f| f.get("arguments"));
            let args = match args_val {
                Some(Value::Object(obj)) => {
                    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                }
                Some(Value::String(s)) => serde_json::from_str::<Value>(s)
                    .ok()
                    .and_then(|v| v.as_object().cloned())
                    .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default(),
                _ => IndexMap::new(),
            };
            let mut call = ToolCall::new(name, args);
            if i == 0 {
                if let Some(r) = reasoning.as_ref() {
                    call = call.with_reasoning(r);
                }
            }
            calls.push(call);
        }
        Some(calls)
    }
}

impl LLMClient for OllamaClient {
    fn api_format(&self) -> ApiFormat {
        ApiFormat::Ollama
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
        let think = self.think.lock().map(|g| *g).unwrap_or(false);
        let think_resolved = self.think_resolved.lock().map(|g| *g).unwrap_or(false);
        let body =
            self.build_request_body(messages.clone(), tools.as_deref(), sampling.as_ref(), think);
        let resp = match reqwest::Client::new()
            .post(format!("{}/api/chat", self.base_url))
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return Err(if e.is_timeout() {
                    BackendError::new(408, e.to_string())
                } else {
                    BackendError::new(0, e.to_string())
                })
            }
        };
        let status = resp.status().as_u16() as i64;
        if status == 500 {
            return Ok(LLMResponse::Text(TextResponse::new(
                resp.text().await.unwrap_or_default(),
            )));
        }
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            // Detect think-unsupported on non-200 non-500 — may be a 400.
            if status == 400 {
                if let Ok(ej) = serde_json::from_str::<Value>(&text) {
                    if Self::is_think_unsupported_error(&ej) {
                        if !think_resolved {
                            // Persist: disable think for future calls (Python parity).
                            if let Ok(mut g) = self.think.lock() {
                                *g = false;
                            }
                            if let Ok(mut g) = self.think_resolved.lock() {
                                *g = true;
                            }
                            let retry_body = self.build_request_body(
                                messages,
                                tools.as_deref(),
                                sampling.as_ref(),
                                false,
                            );
                            let retry_resp = reqwest::Client::new()
                                .post(format!("{}/api/chat", self.base_url))
                                .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
                                .json(&retry_body)
                                .send()
                                .await
                                .map_err(|e| BackendError::new(0, e.to_string()))?;
                            let rs = retry_resp.status().as_u16() as i64;
                            if rs == 500 {
                                return Ok(LLMResponse::Text(TextResponse::new(
                                    retry_resp.text().await.unwrap_or_default(),
                                )));
                            }
                            if !retry_resp.status().is_success() {
                                return Err(BackendError::new(
                                    rs,
                                    retry_resp.text().await.unwrap_or_default(),
                                ));
                            }
                            let rj2: Value = retry_resp
                                .json()
                                .await
                                .map_err(|e| BackendError::new(rs, e.to_string()))?;
                            self.record_usage(&rj2);
                            return Ok(self.parse_send_response(&rj2, false));
                        } else {
                            return Err(BackendError::ThinkingNotSupported {
                                model: self.model.clone(),
                                status_code: status,
                                body: text,
                            });
                        }
                    }
                }
            }
            return Err(BackendError::new(status, text));
        }
        let rj: Value = resp
            .json()
            .await
            .map_err(|e| BackendError::new(status, e.to_string()))?;
        // Mark resolved after first successful call.
        if !think_resolved {
            if let Ok(mut g) = self.think_resolved.lock() {
                *g = true;
            }
        }
        self.record_usage(&rj);
        Ok(self.parse_send_response(&rj, think))
    }

    async fn send_stream(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        sampling: Option<SamplingParams>,
    ) -> Result<ChunkStream, StreamError> {
        let think = self.think.lock().map(|g| *g).unwrap_or(false);
        let think_resolved = self.think_resolved.lock().map(|g| *g).unwrap_or(false);
        let mut body =
            self.build_request_body(messages.clone(), tools.as_deref(), sampling.as_ref(), think);
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".to_string(), Value::Bool(true));
        }
        let resp = match reqwest::Client::new()
            .post(format!("{}/api/chat", self.base_url))
            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                return Err(if e.is_timeout() {
                    StreamError::new(format!("Backend error (status 408): {}", e))
                } else {
                    StreamError::new(e.to_string())
                })
            }
        };
        let status = resp.status().as_u16() as i64;
        if status == 500 {
            let text = resp.text().await.unwrap_or_default();
            let chunk = StreamChunk::new(ChunkType::Final)
                .with_response(LLMResponse::Text(TextResponse::new(text)));
            return Ok(Box::pin(futures_util::stream::once(
                async move { Ok(chunk) },
            )));
        }
        if status == 400 {
            let bt = resp.text().await.unwrap_or_default();
            if let Ok(ej) = serde_json::from_str::<Value>(&bt) {
                if Self::is_think_unsupported_error(&ej) {
                    if !think_resolved {
                        // Persist: disable think for future calls (Python parity).
                        if let Ok(mut g) = self.think.lock() {
                            *g = false;
                        }
                        if let Ok(mut g) = self.think_resolved.lock() {
                            *g = true;
                        }
                        let rb = self.build_request_body(
                            messages,
                            tools.as_deref(),
                            sampling.as_ref(),
                            false,
                        );
                        let mut rb_obj = rb;
                        if let Some(obj) = rb_obj.as_object_mut() {
                            obj.insert("stream".to_string(), Value::Bool(true));
                        }
                        let rr = reqwest::Client::new()
                            .post(format!("{}/api/chat", self.base_url))
                            .timeout(std::time::Duration::from_secs_f64(self.timeout_secs))
                            .json(&rb_obj)
                            .send()
                            .await
                            .map_err(|e| StreamError::new(e.to_string()))?;
                        return Ok(Box::pin(parse_ollama_ndjson(
                            rr,
                            false,
                            self.last_usage.clone(),
                        )));
                    } else {
                        return Err(StreamError::new(format!(
                            "Thinking mode not supported for model '{}'",
                            self.model
                        )));
                    }
                }
            }
            return Err(StreamError::new(format!(
                "Backend error (status 400): {}",
                bt
            )));
        }
        if !resp.status().is_success() {
            return Err(StreamError::new(format!(
                "Backend error (status {}): {}",
                status,
                resp.text().await.unwrap_or_default()
            )));
        }
        // Mark resolved after first successful call.
        if !think_resolved {
            if let Ok(mut g) = self.think_resolved.lock() {
                *g = true;
            }
        }
        Ok(Box::pin(parse_ollama_ndjson(
            resp,
            think,
            self.last_usage.clone(),
        )))
    }

    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError> {
        let guard = self
            .num_ctx
            .lock()
            .map_err(|e| ContextDiscoveryError::new(e.to_string()))?;
        Ok(*guard)
    }
}

fn parse_ollama_ndjson(
    resp: reqwest::Response,
    think: bool,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    let byte_stream = resp.bytes_stream();
    let stream = async_stream::stream! {
        use futures_util::StreamExt;
        let mut inner = Box::pin(byte_stream);
        let mut line_buf = String::new();
        let mut pending_tc: Option<Vec<Value>> = None;
        let mut acc_content = String::new();
        let mut acc_thinking = String::new();
        loop {
            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();
                if line.trim().is_empty() { continue; }
                let obj: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let done = obj.get("done").and_then(|d| d.as_bool()).unwrap_or(false);
                if let Some(msg) = obj.get("message") {
                    if let Some(c) = msg.get("content").and_then(|c| c.as_str()) {
                        if !c.is_empty() {
                            acc_content.push_str(c);
                            yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(c));
                        }
                    }
                    if let Some(thinking) = msg.get("thinking").and_then(|t| t.as_str()) {
                        acc_thinking.push_str(thinking);
                    }
                    if let Some(tcs) = msg.get("tool_calls").and_then(|t| t.as_array()).cloned() {
                        if !tcs.is_empty() { pending_tc = Some(tcs); }
                    }
                }
                if done {
                    let prompt = obj.get("prompt_eval_count").and_then(|v| v.as_i64()).unwrap_or(0);
                    let completion = obj.get("eval_count").and_then(|v| v.as_i64()).unwrap_or(0);
                    if let Ok(mut guard) = last_usage.lock() {
                        *guard = Some(TokenUsage::new(prompt, completion, prompt + completion));
                    }
                    let response_val = json!({"message": {
                        "content": acc_content.clone(),
                        "thinking": if acc_thinking.is_empty() { Value::Null } else { json!(acc_thinking) },
                    }});
                    let reasoning = OllamaClient::resolve_reasoning(think, &response_val);
                    let final_resp = if let Some(tcs) = pending_tc.take() {
                        let mut calls = Vec::new();
                        for (i, tc) in tcs.iter().enumerate() {
                            let name = tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()).unwrap_or("");
                            let args_val = tc.get("function").and_then(|f| f.get("arguments"));
                            let args = match args_val {
                                Some(Value::Object(obj)) => obj.iter().map(|(k,v)|(k.clone(),v.clone())).collect(),
                                Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok()
                                    .and_then(|v| v.as_object().cloned())
                                    .map(|obj| obj.iter().map(|(k,v)|(k.clone(),v.clone())).collect())
                                    .unwrap_or_default(),
                                _ => IndexMap::new(),
                            };
                            let mut call = ToolCall::new(name, args);
                            if i == 0 { if let Some(r) = &reasoning { call = call.with_reasoning(r); } }
                            calls.push(call);
                        }
                        LLMResponse::ToolCalls(calls)
                    } else {
                        let content = acc_content.trim().to_string();
                        LLMResponse::Text(TextResponse::new(content))
                    };
                    yield Ok(StreamChunk::new(ChunkType::Final).with_response(final_resp));
                    return;
                }
            }
            match inner.next().await {
                Some(Ok(b)) => line_buf.push_str(&String::from_utf8_lossy(&b)),
                Some(Err(e)) => { yield Err(StreamError::new(e.to_string())); return; }
                None => {
                    if line_buf.trim().is_empty() {
                        return;
                    }
                    line_buf.push('\n');
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

    // --- Think mode detection ---
    #[test]
    fn think_heuristic_match() {
        let (t, _) = OllamaClient::detect_think_mode("deepseek-r1:8b");
        assert!(t);
    }
    #[test]
    fn think_heuristic_qwq() {
        let (t, _) = OllamaClient::detect_think_mode("qwq:32b");
        assert!(t);
    }
    #[test]
    fn think_heuristic_no_match() {
        let (t, _) = OllamaClient::detect_think_mode("llama3:8b");
        assert!(!t);
    }
    #[test]
    fn think_explicit_true() {
        let c = OllamaClient::new("llama3").with_think(Some(true));
        assert!(c.is_think_enabled());
        assert!(c.is_think_resolved());
    }
    #[test]
    fn think_explicit_false() {
        let c = OllamaClient::new("deepseek-r1").with_think(Some(false));
        assert!(!c.is_think_enabled());
        assert!(c.is_think_resolved());
    }

    // --- Options ---
    #[test]
    fn options_no_ctx() {
        let c = OllamaClient::new("llama3");
        assert!(c.build_options(None).is_empty());
    }
    #[test]
    fn options_with_ctx() {
        let c = OllamaClient::new("llama3");
        c.set_num_ctx(Some(8192));
        assert_eq!(c.build_options(None).get("num_ctx"), Some(&json!(8192)));
    }
    #[test]
    fn options_all_params() {
        let c = OllamaClient::new("llama3")
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_top_k(40)
            .with_min_p(0.05)
            .with_repeat_penalty(1.1)
            .with_presence_penalty(0.5);
        let o = c.build_options(None);
        assert_eq!(o.get("temperature"), Some(&json!(0.7)));
        assert_eq!(o.get("top_k"), Some(&json!(40)));
    }

    // --- Per-call sampling ---
    #[test]
    fn per_call_override() {
        let c = OllamaClient::new("llama3").with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        assert_eq!(
            c.build_options(Some(&sp)).get("temperature"),
            Some(&json!(0.9))
        );
    }
    #[test]
    fn per_call_none_uses_instance() {
        let c = OllamaClient::new("llama3").with_temperature(0.5);
        assert_eq!(c.build_options(None).get("temperature"), Some(&json!(0.5)));
    }
    #[test]
    fn per_call_instance_immutability() {
        let c = OllamaClient::new("llama3").with_temperature(0.5);
        let mut sp = SamplingParams::new();
        sp.insert("temperature".into(), json!(0.9));
        let _ = c.build_options(Some(&sp));
        assert_eq!(c.build_options(None).get("temperature"), Some(&json!(0.5)));
    }

    // --- Temperature ---
    #[test]
    fn temp_default_absent() {
        assert!(OllamaClient::new("llama3")
            .build_options(None)
            .get("temperature")
            .is_none());
    }
    #[test]
    fn temp_explicit() {
        assert_eq!(
            OllamaClient::new("llama3")
                .with_temperature(0.3)
                .build_options(None)
                .get("temperature"),
            Some(&json!(0.3))
        );
    }

    // --- Context ---
    #[test]
    fn ctx_none_default() {
        let c = OllamaClient::new("llama3");
        assert!(c.num_ctx.lock().expect("l").is_none());
    }
    #[test]
    fn ctx_set_and_clear() {
        let c = OllamaClient::new("llama3");
        c.set_num_ctx(Some(4096));
        assert_eq!(*c.num_ctx.lock().expect("l"), Some(4096));
        c.set_num_ctx(None);
        assert!(c.num_ctx.lock().expect("l").is_none());
    }

    // --- Reasoning ---
    #[test]
    fn reasoning_message_thinking_preferred() {
        assert_eq!(
            OllamaClient::resolve_reasoning(
                true,
                &json!({"message": {"thinking": "s", "content": "c"}})
            ),
            Some("s".into())
        );
    }
    #[test]
    fn reasoning_content_fallback() {
        assert_eq!(
            OllamaClient::resolve_reasoning(true, &json!({"message": {"content": "c"}})),
            Some("c".into())
        );
    }
    #[test]
    fn reasoning_disabled() {
        assert!(
            OllamaClient::resolve_reasoning(false, &json!({"message": {"thinking": "s"}}))
                .is_none()
        );
    }
    #[test]
    fn reasoning_empty() {
        assert!(
            OllamaClient::resolve_reasoning(true, &json!({"message": {"content": ""}})).is_none()
        );
    }

    // --- Response parsing ---
    #[test]
    fn parse_tool_call() {
        let c = OllamaClient::new("llama3");
        let r = json!({"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read", "arguments": {"path": "/tmp/x"}}},
        ]}});
        match c.parse_send_response(&r, true) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].tool, "read");
                assert_eq!(calls[0].args["path"], "/tmp/x");
            }
            _ => panic!("expected tool calls"),
        }
    }
    #[test]
    fn parse_text() {
        let c = OllamaClient::new("llama3");
        match c.parse_send_response(&json!({"message": {"content": "Hi"}}), true) {
            LLMResponse::Text(t) => assert_eq!(t.content, "Hi"),
            _ => panic!("expected text"),
        }
    }
    #[test]
    fn parse_empty_tool_calls() {
        let c = OllamaClient::new("llama3");
        match c.parse_send_response(
            &json!({"message": {"content": "No tools", "tool_calls": []}}),
            true,
        ) {
            LLMResponse::Text(t) => assert_eq!(t.content, "No tools"),
            _ => panic!("expected text"),
        }
    }

    // --- Request body ---
    #[test]
    fn request_body_structure() {
        let c = OllamaClient::new("llama3").with_temperature(0.7);
        let b = c.build_request_body(
            vec![json!({"role": "user", "content": "Hi"})],
            None,
            None,
            false,
        );
        assert_eq!(b["model"], "llama3");
        assert_eq!(b["stream"], false);
        assert_eq!(b["options"]["temperature"], 0.7);
    }
    #[test]
    fn request_body_think() {
        let c = OllamaClient::new("llama3").with_think(Some(true));
        assert_eq!(
            c.build_request_body(vec![], None, None, true)["think"],
            true
        );
    }
    #[test]
    fn request_body_tools() {
        let s = ToolSpec::from_json_schema(
            "run",
            "Run",
            &json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        )
        .expect("ok");
        let c = OllamaClient::new("llama3");
        let b = c.build_request_body(
            vec![json!({"role": "user", "content": "Go"})],
            Some(&[s]),
            None,
            false,
        );
        assert_eq!(b["tools"].as_array().expect("a").len(), 1);
    }

    // --- Tool role pass-through ---
    #[test]
    fn tool_role_passthrough() {
        let c = OllamaClient::new("llama3");
        let msgs = vec![json!({"role": "tool", "content": "data"})];
        let b = c.build_request_body(msgs.clone(), None, None, false);
        assert_eq!(b["messages"][0]["role"], "tool");
    }

    // --- Usage ---
    #[test]
    fn usage_from_response() {
        let c = OllamaClient::new("llama3");
        c.record_usage(&json!({"prompt_eval_count": 100, "eval_count": 50}));
        let u = c.get_last_usage().expect("set");
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.total_tokens, 150);
    }

    // --- Recommended sampling ---
    #[test]
    fn rec_unknown() {
        assert!(OllamaClient::new("unknown")
            .with_recommended_sampling(true)
            .sampling_defaults
            .is_none());
    }
    #[test]
    fn rec_known() {
        assert!(OllamaClient::new("qwen3:8b-q4_K_M")
            .with_recommended_sampling(true)
            .sampling_defaults
            .is_some());
    }
    #[test]
    fn rec_explicit_override() {
        let c = OllamaClient::new("qwen3:8b-q4_K_M")
            .with_recommended_sampling(true)
            .with_temperature(0.99);
        assert_eq!(c.build_options(None).get("temperature"), Some(&json!(0.99)));
    }

    // --- API format ---
    #[test]
    fn api_format_ollama() {
        assert_eq!(OllamaClient::new("llama3").api_format(), ApiFormat::Ollama);
    }

    // --- Tool formatting ---
    #[test]
    fn tool_format_basic() {
        let s = ToolSpec::from_json_schema(
            "t",
            "T",
            &json!({"type": "object", "properties": {"x": {"type": "string"}}}),
        )
        .expect("ok");
        let c = OllamaClient::new("llama3");
        let b = c.build_request_body(vec![], Some(&[s]), None, false);
        assert_eq!(b["tools"][0]["type"], "function");
    }
    #[test]
    fn tool_format_enum() {
        let s = ToolSpec::from_json_schema(
            "e",
            "E",
            &json!({"type": "object", "properties": {"m": {"type": "string", "enum": ["a","b"]}}}),
        )
        .expect("ok");
        let c = OllamaClient::new("llama3");
        let b = c.build_request_body(vec![], Some(&[s]), None, false);
        assert!(b["tools"][0]["function"]["parameters"]
            .get("properties")
            .is_some());
    }

    // --- Reasoning on first tool call ---
    #[test]
    fn reasoning_first_tool_call() {
        let c = OllamaClient::new("llama3").with_think(Some(true));
        let r = json!({"message": {"thinking": "thinking", "content": "", "tool_calls": [
            {"function": {"name": "run", "arguments": {}}}
        ]}});
        match c.parse_send_response(&r, true) {
            LLMResponse::ToolCalls(calls) => {
                assert_eq!(calls[0].reasoning, Some("thinking".into()))
            }
            _ => panic!("expected tool calls"),
        }
    }
}
