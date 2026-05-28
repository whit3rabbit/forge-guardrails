use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::clients::base::{LLMUsageDetails, TokenUsage};

pub(super) fn record_usage_cell(
    cell: &Arc<Mutex<Option<TokenUsage>>>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) {
    let token_usage = token_usage_from_openai_usage(usage);
    if let Ok(mut guard) = cell.lock() {
        *guard = Some(token_usage);
    }
}

pub(super) fn token_usage_from_openai_usage(
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> TokenUsage {
    usage
        .map(|u| {
            TokenUsage::new(
                u.prompt_tokens as i64,
                u.completion_tokens as i64,
                u.total_tokens as i64,
            )
        })
        .unwrap_or_else(TokenUsage::empty)
}

pub(super) fn record_usage_details_cell(
    cell: &Arc<Mutex<Option<LLMUsageDetails>>>,
    details: Option<LLMUsageDetails>,
) {
    if let Ok(mut guard) = cell.lock() {
        *guard = details;
    }
}

pub(super) fn usage_details_from_openai_usage(
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> Option<LLMUsageDetails> {
    let cached = usage
        .and_then(|u| u.prompt_tokens_details.as_ref())
        .and_then(cached_tokens_from_details);
    let details = LLMUsageDetails {
        cached_prompt_tokens: cached,
        ..Default::default()
    };
    if details.is_empty() {
        None
    } else {
        Some(details)
    }
}

pub(super) fn usage_details_from_openai_usage_value(
    usage: Option<&Value>,
) -> Option<LLMUsageDetails> {
    let cached = usage
        .and_then(|u| u.get("prompt_tokens_details"))
        .and_then(cached_tokens_from_details)
        .or_else(|| {
            usage
                .and_then(|u| u.get("input_token_details"))
                .and_then(cached_tokens_from_details)
        });
    let deepseek_hit = usage_i64(usage, "prompt_cache_hit_tokens");
    let deepseek_miss = usage_i64(usage, "prompt_cache_miss_tokens");
    let details = LLMUsageDetails {
        cached_prompt_tokens: cached.or(deepseek_hit),
        cache_miss_prompt_tokens: deepseek_miss,
        prompt_cache_hit_tokens: deepseek_hit,
        prompt_cache_miss_tokens: deepseek_miss,
        ..Default::default()
    };
    if details.is_empty() {
        None
    } else {
        Some(details)
    }
}

fn cached_tokens_from_details(details: &Value) -> Option<i64> {
    details.get("cached_tokens").and_then(Value::as_i64)
}

fn usage_i64(usage: Option<&Value>, key: &str) -> Option<i64> {
    usage.and_then(|u| u.get(key)).and_then(Value::as_i64)
}
