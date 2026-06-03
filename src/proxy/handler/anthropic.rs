use crate::clients::base::{LLMClient, LLMUsageDetails};
use crate::context::manager::ContextManager;
use crate::guardrails::{FinalResponseScorer, ToolCallScorer};
use crate::tool_output::{ToolOutputCompressionConfig, ToolOutputCompressionState};
use crate::tool_policy::ToolCallPolicyConfig;
use anyllm_translate::anthropic::MessageCreateRequest;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

use super::request_contract::FORGE_EXTENSION_FIELD;
use super::response_shape::anthropic_events_stream;
use super::{
    handle_chat_completions_impl, AnthropicHandlerError, AnthropicHandlerResult, HandlerError,
    HandlerResult,
};

fn chat_error_to_anthropic(error: HandlerError) -> AnthropicHandlerError {
    match error {
        HandlerError::BadRequest(message) => AnthropicHandlerError::BadRequest(message),
        HandlerError::Upstream(message) => AnthropicHandlerError::Upstream(message),
    }
}

/// Handle /v1/messages by translating Anthropic input through the guarded
/// OpenAI-compatible handler, then translating the response back to Anthropic.
#[allow(clippy::too_many_arguments)]
pub async fn handle_anthropic_messages<C: LLMClient + 'static>(
    body: &MessageCreateRequest,
    raw_body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
) -> Result<AnthropicHandlerResult, AnthropicHandlerError> {
    handle_anthropic_messages_with_scorer(
        body,
        raw_body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        None,
    )
    .await
}

/// Handle /v1/messages with an optional shadow classifier scorer.
#[allow(clippy::too_many_arguments)]
pub async fn handle_anthropic_messages_with_scorer<C: LLMClient + 'static>(
    body: &MessageCreateRequest,
    raw_body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
) -> Result<AnthropicHandlerResult, AnthropicHandlerError> {
    handle_anthropic_messages_with_scorers(
        body,
        raw_body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        scorer,
        None,
    )
    .await
}

/// Handle /v1/messages with optional tool-call and final-response scorers.
#[allow(clippy::too_many_arguments)]
pub async fn handle_anthropic_messages_with_scorers<C: LLMClient + 'static>(
    body: &MessageCreateRequest,
    raw_body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
) -> Result<AnthropicHandlerResult, AnthropicHandlerError> {
    handle_anthropic_messages_with_scorers_and_tool_controls(
        body,
        raw_body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        scorer,
        final_response_scorer,
        ToolOutputCompressionConfig::disabled(),
        None,
        ToolCallPolicyConfig::disabled(),
    )
    .await
}

/// Handle /v1/messages with optional scorers and tool-output compression.
#[allow(clippy::too_many_arguments)]
pub async fn handle_anthropic_messages_with_scorers_and_tool_output_compression<
    C: LLMClient + 'static,
>(
    body: &MessageCreateRequest,
    raw_body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    default_tool_output_compression: ToolOutputCompressionConfig,
    tool_output_state: Option<Arc<ToolOutputCompressionState>>,
) -> Result<AnthropicHandlerResult, AnthropicHandlerError> {
    handle_anthropic_messages_with_scorers_and_tool_controls(
        body,
        raw_body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        scorer,
        final_response_scorer,
        default_tool_output_compression,
        tool_output_state,
        ToolCallPolicyConfig::disabled(),
    )
    .await
}

/// Handle /v1/messages with optional scorers, tool-output compression, and tool-call policy.
#[allow(clippy::too_many_arguments)]
pub async fn handle_anthropic_messages_with_scorers_and_tool_controls<C: LLMClient + 'static>(
    body: &MessageCreateRequest,
    raw_body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    default_tool_output_compression: ToolOutputCompressionConfig,
    tool_output_state: Option<Arc<ToolOutputCompressionState>>,
    default_tool_call_policy: ToolCallPolicyConfig,
) -> Result<AnthropicHandlerResult, AnthropicHandlerError> {
    handle_anthropic_messages_with_scorers_tool_controls_and_headers(
        body,
        raw_body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        scorer,
        final_response_scorer,
        default_tool_output_compression,
        tool_output_state,
        default_tool_call_policy,
        None,
    )
    .await
}

/// Handle /v1/messages with optional scorers, tool-output compression, tool-call policy,
/// and safe Anthropic header passthrough.
#[allow(clippy::too_many_arguments)]
pub async fn handle_anthropic_messages_with_scorers_tool_controls_and_headers<
    C: LLMClient + 'static,
>(
    body: &MessageCreateRequest,
    raw_body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    default_tool_output_compression: ToolOutputCompressionConfig,
    tool_output_state: Option<Arc<ToolOutputCompressionState>>,
    default_tool_call_policy: ToolCallPolicyConfig,
    anthropic_headers: Option<Vec<(String, String)>>,
) -> Result<AnthropicHandlerResult, AnthropicHandlerError> {
    let config = anyllm_translate::TranslationConfig::default();
    let openai_req = anyllm_translate::translate_request(body, &config)
        .map_err(|e| AnthropicHandlerError::BadRequest(e.to_string()))?;
    let mut openai_value = serde_json::to_value(&openai_req)
        .map_err(|e| AnthropicHandlerError::Internal(e.to_string()))?;
    copy_forge_extension_if_missing(raw_body, &mut openai_value);
    if let (Some(max_tokens), Some(obj)) =
        (raw_body.get("max_tokens"), openai_value.as_object_mut())
    {
        obj.insert("max_tokens".to_string(), max_tokens.clone());
        obj.remove("max_completion_tokens");
    }

    match handle_chat_completions_impl(
        &openai_value,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        Some(raw_body.clone()),
        anthropic_headers,
        scorer,
        final_response_scorer,
        default_tool_output_compression,
        tool_output_state,
        default_tool_call_policy,
    )
    .await
    .map_err(chat_error_to_anthropic)?
    {
        HandlerResult::Response(openai_resp) => {
            let response: anyllm_translate::openai::ChatCompletionResponse =
                serde_json::from_value(openai_resp)
                    .map_err(|e| AnthropicHandlerError::Internal(e.to_string()))?;
            let anthropic_resp = anyllm_translate::translate_response(&response, &body.model);
            let mut value = serde_json::to_value(anthropic_resp)
                .map_err(|e| AnthropicHandlerError::Internal(e.to_string()))?;
            apply_anthropic_usage_details(&mut value, client.last_usage_details().as_ref());
            Ok(AnthropicHandlerResult::Response(value))
        }
        HandlerResult::StreamBody(openai_events) => Ok(AnthropicHandlerResult::StreamBody(
            anthropic_events_stream(openai_events, body.model.clone()),
        )),
        HandlerResult::AnthropicResponse(value) => Ok(AnthropicHandlerResult::Response(value)),
        HandlerResult::AnthropicStreamBody(events) => {
            Ok(AnthropicHandlerResult::StreamBody(events))
        }
    }
}

fn copy_forge_extension_if_missing(raw_body: &Value, openai_value: &mut Value) {
    let Some(forge) = raw_body.get(FORGE_EXTENSION_FIELD) else {
        return;
    };
    let Some(obj) = openai_value.as_object_mut() else {
        return;
    };
    obj.entry(FORGE_EXTENSION_FIELD.to_string())
        .or_insert_with(|| forge.clone());
}

fn apply_anthropic_usage_details(value: &mut Value, details: Option<&LLMUsageDetails>) {
    let Some(details) = details else {
        return;
    };
    let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) else {
        return;
    };
    if let Some(read) = details.cache_read_input_tokens {
        usage.insert("cache_read_input_tokens".to_string(), json!(read));
    }
    if let Some(created) = details.cache_creation_input_tokens {
        usage.insert("cache_creation_input_tokens".to_string(), json!(created));
    }
    if let Some(thinking) = details.anthropic_thinking_output_tokens {
        usage.insert(
            "output_tokens_details".to_string(),
            json!({"thinking_tokens": thinking}),
        );
    }
}
