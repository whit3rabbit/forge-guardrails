//! Anthropic Messages API client adapter.
//!
//! Converts messages from the common wire format to Anthropic's format before
//! each API call. Uses reqwest for HTTP. The sampling parameter is accepted
//! for protocol symmetry but ignored (Anthropic controls sampling server-side).

pub(crate) mod convert;
mod request;
mod streaming;
mod transport;
mod usage;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};

use crate::clients::base::{LLMUsageDetails, TokenUsage};

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
    last_usage_details: Arc<Mutex<Option<LLMUsageDetails>>>,
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
            last_usage_details: Arc::new(Mutex::new(None)),
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

    /// Returns the token usage of the last request made by this client, if any.
    pub fn get_last_usage(&self) -> Option<TokenUsage> {
        self.last_usage.lock().ok().and_then(|guard| guard.clone())
    }

    /// Returns provider-specific cache usage details from the last request, if any.
    pub fn get_last_usage_details(&self) -> Option<LLMUsageDetails> {
        self.last_usage_details
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }
}
