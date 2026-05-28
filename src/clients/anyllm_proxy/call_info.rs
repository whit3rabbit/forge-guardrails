use std::sync::{Arc, Mutex};

use ::anyllm_proxy::backend::RateLimitHeaders as AnyLlmRateLimitHeaders;
use reqwest::header::HeaderMap;

use crate::clients::base::{LLMCallInfo, LLMRateLimitInfo};

pub(super) fn record_call_info_cell(cell: &Arc<Mutex<Option<LLMCallInfo>>>, info: LLMCallInfo) {
    if let Ok(mut guard) = cell.lock() {
        *guard = Some(info);
    }
}

pub(super) fn runtime_call_info(
    metadata: &::anyllm_proxy::runtime::ChatCompletionMetadata,
    rate_limits: &AnyLlmRateLimitHeaders,
    warnings: &anyllm_translate::TranslationWarnings,
    response_model: Option<String>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> LLMCallInfo {
    LLMCallInfo {
        requested_model: Some(metadata.requested_model.clone()),
        response_model,
        selected_backend: Some(metadata.selected_backend.clone()),
        mapped_model: Some(metadata.mapped_model.clone()),
        backend_kind: Some(format!("{:?}", metadata.backend_kind)),
        provider_id: metadata.provider_id.clone(),
        used_responses_api: metadata.used_responses_api,
        degradation_warnings: warnings.as_header_value(),
        cache_status: None,
        rate_limits: rate_limit_info_from_anyllm(rate_limits),
        estimated_cost_usd: estimate_cost_usd(Some(&metadata.mapped_model), usage),
    }
}

pub(super) fn sidecar_call_info(
    requested_model: &str,
    headers: &HeaderMap,
    response_model: Option<String>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> LLMCallInfo {
    let header_cost =
        header_value(headers, "x-anyllm-cost-usd").and_then(|v| v.parse::<f64>().ok());
    let cost_model = response_model.as_deref().or(Some(requested_model));
    let estimated_cost_usd = header_cost.or_else(|| estimate_cost_usd(cost_model, usage));
    LLMCallInfo {
        requested_model: Some(requested_model.to_string()),
        response_model,
        selected_backend: None,
        mapped_model: None,
        backend_kind: None,
        provider_id: None,
        used_responses_api: false,
        degradation_warnings: header_value(headers, "x-anyllm-degradation"),
        cache_status: header_value(headers, "x-anyllm-cache"),
        rate_limits: rate_limit_info_from_sidecar(headers),
        estimated_cost_usd,
    }
}

fn rate_limit_info_from_anyllm(rate_limits: &AnyLlmRateLimitHeaders) -> LLMRateLimitInfo {
    LLMRateLimitInfo {
        requests_limit: rate_limits.requests_limit.clone(),
        requests_remaining: rate_limits.requests_remaining.clone(),
        requests_reset: rate_limits.requests_reset.clone(),
        tokens_limit: rate_limits.tokens_limit.clone(),
        tokens_remaining: rate_limits.tokens_remaining.clone(),
        tokens_reset: rate_limits.tokens_reset.clone(),
        retry_after: rate_limits.retry_after.clone(),
        organization_id: rate_limits.organization_id.clone(),
    }
}

fn rate_limit_info_from_sidecar(headers: &HeaderMap) -> LLMRateLimitInfo {
    LLMRateLimitInfo {
        requests_limit: header_value(headers, "anthropic-ratelimit-requests-limit"),
        requests_remaining: header_value(headers, "anthropic-ratelimit-requests-remaining"),
        requests_reset: header_value(headers, "anthropic-ratelimit-requests-reset"),
        tokens_limit: header_value(headers, "anthropic-ratelimit-tokens-limit"),
        tokens_remaining: header_value(headers, "anthropic-ratelimit-tokens-remaining"),
        tokens_reset: header_value(headers, "anthropic-ratelimit-tokens-reset"),
        retry_after: header_value(headers, "retry-after"),
        organization_id: header_value(headers, "anthropic-organization-id"),
    }
}

fn header_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(super) fn estimate_cost_usd(
    model: Option<&str>,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) -> Option<f64> {
    let model = model?;
    let usage = usage?;
    let pricing = ::anyllm_proxy::cost::pricing();
    pricing.price_for_model(model)?;
    Some(pricing.cost_for_usage(
        model,
        usage.prompt_tokens as u64,
        usage.completion_tokens as u64,
    ))
}

pub(super) fn observe_stream_call_info(
    cell: &Arc<Mutex<Option<LLMCallInfo>>>,
    response_model: &str,
    cost_model: &str,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) {
    if let Ok(mut guard) = cell.lock() {
        observe_stream_call_info_value(&mut guard, response_model, cost_model, usage);
    }
}

pub(super) fn observe_stream_call_info_value(
    info: &mut Option<LLMCallInfo>,
    response_model: &str,
    cost_model: &str,
    usage: Option<&anyllm_translate::openai::ChatUsage>,
) {
    let info = info.get_or_insert_with(LLMCallInfo::default);
    if info.response_model.is_none() {
        info.response_model = Some(response_model.to_string());
    }
    if info.estimated_cost_usd.is_none() {
        info.estimated_cost_usd = estimate_cost_usd(Some(cost_model), usage);
    }
}
