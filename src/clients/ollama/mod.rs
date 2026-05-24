//! Ollama native API client adapter.
//!
//! Uses the /api/chat endpoint with the tools parameter for native function
//! calling. Messages pass through to Ollama unchanged. Supports think mode
//! with auto-detection from model name keywords and fallback on unsupported.

mod helpers;
mod request;
mod response;
mod streaming;

use std::sync::{Arc, Mutex};

use serde_json::{Map, Value};

use crate::clients::base::TokenUsage;
use crate::clients::sampling::get_sampling_defaults;

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
    /// Creates a new `OllamaClient` for the given model.
    pub fn new(model: impl Into<String>) -> Self {
        let model_str = model.into();
        let (think, think_resolved) = helpers::detect_think_mode(&model_str);
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

    /// Sets the base URL for the Ollama endpoint.
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
    /// Sets the request timeout in seconds.
    pub fn with_timeout(mut self, s: f64) -> Self {
        self.timeout_secs = s;
        self
    }

    /// Sets whether thinking mode is active.
    pub fn with_think(mut self, think: Option<bool>) -> Self {
        match think {
            Some(t) => {
                self.think = Mutex::new(t);
                self.think_resolved = Mutex::new(true);
            }
            None => {
                let (d, r) = helpers::detect_think_mode(&self.model);
                self.think = Mutex::new(d);
                self.think_resolved = Mutex::new(r);
            }
        }
        self
    }

    /// Sets whether recommended sampling defaults are used.
    pub fn with_recommended_sampling(mut self, enabled: bool) -> Self {
        if enabled {
            let d = get_sampling_defaults(&self.model);
            if !d.is_empty() {
                self.sampling_defaults = Some(d);
            }
        }
        self
    }

    /// Returns true if thinking mode is active.
    pub fn is_think_enabled(&self) -> bool {
        self.think.lock().map(|g| *g).unwrap_or(false)
    }
    /// Returns true if thinking mode support has been resolved.
    pub fn is_think_resolved(&self) -> bool {
        self.think_resolved.lock().map(|g| *g).unwrap_or(false)
    }

    /// Sets the model context window size parameter.
    pub fn set_num_ctx(&self, ctx: Option<i64>) {
        if let Ok(mut g) = self.num_ctx.lock() {
            *g = ctx;
        }
    }
}
