//! Llamafile (llama-server) client adapter using OpenAI-compatible chat API.
//!
//! Supports three modes: native (tools parameter), prompt (inject tool
//! descriptions into prompt), and auto (tries native, falls back on HTTP
//! error). Context length discovered from server properties endpoint.

pub(crate) mod helpers;
mod request;
mod response;
mod streaming;

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value};

use crate::clients::base::{
    ApiFormat, ChunkStream, LLMClient, LLMResponse, SamplingParams, TokenUsage,
};
use crate::clients::sampling::get_sampling_defaults;
use crate::core::tool_spec::ToolSpec;
use crate::error::{BackendError, ContextDiscoveryError, StreamError};

/// Function calling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamafileMode {
    /// Native tool call support via JSON parameters.
    Native,
    /// Fallback tool call support via prompt injection.
    Prompt,
    /// Automated detection of native support with prompt fallback.
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
    /// Creates a new `LlamafileClient` representing the model file at the given path.
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

    /// Sets the base URL for the Llamafile endpoint.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
    /// Sets the temperature sampling parameter.
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }
    /// Sets the top_p sampling parameter.
    pub fn with_top_p(mut self, v: f64) -> Self {
        self.top_p = Some(v);
        self
    }
    /// Sets the top_k sampling parameter.
    pub fn with_top_k(mut self, v: i64) -> Self {
        self.top_k = Some(v);
        self
    }
    /// Sets the min_p sampling parameter.
    pub fn with_min_p(mut self, v: f64) -> Self {
        self.min_p = Some(v);
        self
    }
    /// Sets the repeat_penalty sampling parameter.
    pub fn with_repeat_penalty(mut self, v: f64) -> Self {
        self.repeat_penalty = Some(v);
        self
    }
    /// Sets the presence_penalty sampling parameter.
    pub fn with_presence_penalty(mut self, v: f64) -> Self {
        self.presence_penalty = Some(v);
        self
    }
    /// Sets custom chat template keyword arguments.
    pub fn with_chat_template_kwargs(mut self, kw: Map<String, Value>) -> Self {
        self.chat_template_kwargs = Some(kw);
        self
    }

    /// Sets the function calling mode (native, prompt, or auto).
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

    /// Sets the request timeout in seconds.
    pub fn with_timeout(mut self, s: f64) -> Self {
        self.timeout_secs = s;
        self
    }
    /// Sets whether thinking/reasoning parsing is enabled.
    pub fn with_think(mut self, t: Option<bool>) -> Self {
        self.think = t.unwrap_or(true);
        self
    }
    /// Sets whether prompt caching is enabled.
    pub fn with_cache_prompt(mut self, c: bool) -> Self {
        self.cache_prompt = c;
        self
    }
    /// Sets the server slot ID to query usage on.
    pub fn with_slot_id(mut self, s: i64) -> Self {
        self.slot_id = Some(s);
        self
    }

    /// Sets whether recommended sampling defaults are used.
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

    /// Returns the model identity string.
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

    /// Returns the token usage of the last request for the given slot.
    pub fn get_usage(&self, slot: i64) -> Option<TokenUsage> {
        self.last_usage.lock().ok()?.get(&slot).cloned()
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
        self.stream_send(messages, tools, sampling, mode).await
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

#[cfg(test)]
mod tests {
    use super::*;
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
    fn context_url_strips_v1() {
        let c = LlamafileClient::new(Path::new("t.gguf")).with_base_url("http://localhost:8080/v1");
        assert!(c.base_url.ends_with("/v1"));
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
}
