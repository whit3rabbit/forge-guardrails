use serde_json::Value;

use super::AnthropicClient;
use crate::clients::base::{LLMUsageDetails, TokenUsage};

impl AnthropicClient {
    pub(super) fn record_usage(&self, response: &Value) {
        let (token_usage, details) = usage_from_response(response);
        if let Ok(mut guard) = self.last_usage.lock() {
            *guard = Some(token_usage);
        }
        if let Ok(mut guard) = self.last_usage_details.lock() {
            *guard = details;
        }
    }
}

pub(super) fn usage_from_response(response: &Value) -> (TokenUsage, Option<LLMUsageDetails>) {
    let usage = response.get("usage");
    let prompt = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(|t| t.as_i64())
        .unwrap_or(0);
    let cache_creation = usage_i64(usage, "cache_creation_input_tokens");
    let cache_read = usage_i64(usage, "cache_read_input_tokens");
    let thinking_output = usage
        .and_then(|u| u.get("output_tokens_details"))
        .and_then(|details| details.get("thinking_tokens"))
        .and_then(Value::as_i64);
    let prompt_total = prompt + cache_creation.unwrap_or(0) + cache_read.unwrap_or(0);
    let completion = usage
        .and_then(|u| u.get("output_tokens"))
        .and_then(|t| t.as_i64())
        .unwrap_or(0);
    let token_usage = TokenUsage::new(prompt_total, completion, prompt_total + completion);
    (
        token_usage,
        anthropic_usage_details(cache_creation, cache_read, thinking_output),
    )
}

pub(super) fn usage_i64(usage: Option<&Value>, key: &str) -> Option<i64> {
    usage.and_then(|u| u.get(key)).and_then(Value::as_i64)
}

pub(super) fn anthropic_usage_details(
    cache_creation: Option<i64>,
    cache_read: Option<i64>,
    thinking_output: Option<i64>,
) -> Option<LLMUsageDetails> {
    let details = LLMUsageDetails {
        cached_prompt_tokens: cache_read,
        cache_creation_prompt_tokens: cache_creation,
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: cache_creation,
        anthropic_thinking_output_tokens: thinking_output,
        ..Default::default()
    };
    if details.is_empty() {
        None
    } else {
        Some(details)
    }
}
