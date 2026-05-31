//! Request handler: bridges HTTP layer and inference with guardrails.
//!
//! handle_chat_completions is the main entry point for /v1/chat/completions.
//! It converts inbound OpenAI messages, runs inference with validation/retry,
//! then strips respond() calls from output.

use crate::clients::base::{LLMClient, LLMRequestOptions, LLMResponse, TextResponse, ToolCall};
use crate::context::manager::ContextManager;
use crate::error::StreamError;
use crate::guardrails::{FinalResponseScorer, StepEnforcer, ToolCallScorer};
use crate::proxy::{
    extract_passthrough, extract_sampling, openai_to_messages, strip_respond_calls,
    OpenAiMessageError,
};
use crate::tool_output::{ToolOutputCompressionConfig, ToolOutputCompressionState};
use crate::tool_policy::{
    evaluate_tool_call_policy, ToolCallPolicyConfig, ToolCallPolicyRequestState,
};
use crate::tools::respond::RESPOND_TOOL_NAME;
use anyllm_translate::anthropic::streaming::StreamEvent;
use futures_core::Stream;
use indexmap::IndexSet;
use serde_json::Value;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

mod anthropic;
mod classifier_log;
mod compression;
mod nudge;
mod passthrough;
mod prior_tool_results;
mod request_contract;
mod response_shape;
mod scoring;
mod tool_specs;

pub use anthropic::{
    handle_anthropic_messages, handle_anthropic_messages_with_scorer,
    handle_anthropic_messages_with_scorers,
    handle_anthropic_messages_with_scorers_and_tool_controls,
    handle_anthropic_messages_with_scorers_and_tool_output_compression,
};
use compression::{compress_proxy_tool_results, patch_anthropic_tool_results};
use nudge::{
    emit_proxy_classifier_nudge_or_error, emit_proxy_final_response_tool_nudge_or_error,
    emit_proxy_step_nudge_or_error, emit_proxy_tool_policy_nudge_or_error,
    emit_proxy_user_classifier_nudge_or_error, synthetic_respond_tool_call,
};
pub use passthrough::run_passthrough;
use prior_tool_results::record_completed_proxy_tool_results;
#[cfg(test)]
use request_contract::sanitize_guarded_anthropic_body;
use request_contract::{
    add_proxy_respond_tool_if_needed, extract_forge_bool_field, extract_proxy_step_contract,
    extract_stream_include_usage, extract_tool_call_policy_config,
    extract_tool_output_compression_config, sanitize_guarded_request_options,
    strip_forge_extension_from_body, validate_proxy_step_contract, FORGE_EXTENSION_FIELD,
    FORGE_REQUIRED_STEPS_FIELD, FORGE_RETURN_RAW_ON_GUARDRAIL_FAILURE_FIELD,
};
#[cfg(test)]
use response_shape::{collect_anthropic_events, collect_openai_events};
use response_shape::{text_content_result, text_response_result, tool_calls_result};
use scoring::{score_proxy_final_text, score_proxy_final_tool_calls, score_proxy_tool_calls};
pub use tool_specs::parse_tool_specs;

/// Initialize the optional bounded proxy classifier JSONL sink from environment.
pub fn init_proxy_classifier_log_sink_from_env() {
    classifier_log::init_proxy_classifier_log_sink_from_env();
}

/// Stream of OpenAI chat completion chunk objects.
pub type OpenAiEventStream = Pin<Box<dyn Stream<Item = Result<Value, StreamError>> + Send>>;

/// Stream of Anthropic Messages API SSE events.
pub type AnthropicEventStream =
    Pin<Box<dyn Stream<Item = Result<StreamEvent, StreamError>> + Send>>;

/// Result of handling a chat completion request.
pub enum HandlerResult {
    /// Non-streaming: single OpenAI response object.
    Response(Value),
    /// Streaming: OpenAI response chunk objects.
    StreamBody(OpenAiEventStream),
}

const PROXY_STEP_INDEX: i64 = 0;

impl fmt::Debug for HandlerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Response(value) => f.debug_tuple("Response").field(value).finish(),
            Self::StreamBody(_) => f.write_str("StreamBody(<openai event stream>)"),
        }
    }
}

/// Error class for OpenAI-compatible chat completion request handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerError {
    /// The request is invalid or malformed.
    BadRequest(String),
    /// Backend or guarded inference failed after the request was accepted.
    Upstream(String),
}

impl HandlerError {
    /// Returns the underlying error message.
    pub fn message(&self) -> &str {
        match self {
            Self::BadRequest(message) | Self::Upstream(message) => message,
        }
    }
}

impl fmt::Display for HandlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for HandlerError {}

impl From<OpenAiMessageError> for HandlerError {
    fn from(error: OpenAiMessageError) -> Self {
        Self::BadRequest(error.to_string())
    }
}

/// Result of handling an Anthropic Messages API request.
pub enum AnthropicHandlerResult {
    /// Non-streaming: single Anthropic message response object.
    Response(Value),
    /// Streaming: Anthropic SSE events.
    StreamBody(AnthropicEventStream),
}

impl fmt::Debug for AnthropicHandlerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Response(value) => f.debug_tuple("Response").field(value).finish(),
            Self::StreamBody(_) => f.write_str("StreamBody(<anthropic event stream>)"),
        }
    }
}

/// Error class for Anthropic request handling.
#[derive(Debug)]
pub enum AnthropicHandlerError {
    /// The request is invalid or malformed.
    BadRequest(String),
    /// An error occurred in the upstream OpenAI/inference handler.
    Upstream(String),
    /// An internal server or serialization error occurred.
    Internal(String),
}

impl AnthropicHandlerError {
    /// Returns the underlying error message.
    pub fn message(&self) -> &str {
        match self {
            Self::BadRequest(message) | Self::Upstream(message) | Self::Internal(message) => {
                message
            }
        }
    }
}

/// Main handler for /v1/chat/completions.
///
/// When no tools are present, passes through to backend directly (no guardrails).
/// When tools are present, conditionally injects Forge's reserved respond tool,
/// runs inference with validation/retry, then strips respond() calls from output.
///
/// Sampling fields are extracted per-request and passed as a dict (or None);
/// they never persist on the client between calls.
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat_completions<C: LLMClient + 'static>(
    body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
) -> Result<HandlerResult, HandlerError> {
    handle_chat_completions_with_scorer(
        body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        None,
    )
    .await
}

/// Main handler with an optional shadow classifier scorer.
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat_completions_with_scorer<C: LLMClient + 'static>(
    body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
) -> Result<HandlerResult, HandlerError> {
    handle_chat_completions_with_scorers(
        body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        scorer,
        None,
    )
    .await
}

/// Main handler with optional tool-call and final-response scorers.
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat_completions_with_scorers<C: LLMClient + 'static>(
    body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
) -> Result<HandlerResult, HandlerError> {
    handle_chat_completions_with_scorers_and_tool_controls(
        body,
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

/// Main handler with optional scorers and tool-output compression.
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat_completions_with_scorers_and_tool_output_compression<
    C: LLMClient + 'static,
>(
    body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    default_tool_output_compression: ToolOutputCompressionConfig,
    tool_output_state: Option<Arc<ToolOutputCompressionState>>,
) -> Result<HandlerResult, HandlerError> {
    handle_chat_completions_with_scorers_and_tool_controls(
        body,
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

/// Main handler with optional scorers, tool-output compression, and tool-call policy.
#[allow(clippy::too_many_arguments)]
pub async fn handle_chat_completions_with_scorers_and_tool_controls<C: LLMClient + 'static>(
    body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    default_tool_output_compression: ToolOutputCompressionConfig,
    tool_output_state: Option<Arc<ToolOutputCompressionState>>,
    default_tool_call_policy: ToolCallPolicyConfig,
) -> Result<HandlerResult, HandlerError> {
    handle_chat_completions_impl(
        body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        None,
        scorer,
        final_response_scorer,
        default_tool_output_compression,
        tool_output_state,
        default_tool_call_policy,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn handle_chat_completions_impl<C: LLMClient + 'static>(
    body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    inbound_anthropic_body: Option<Value>,
    scorer: Option<Arc<dyn ToolCallScorer>>,
    final_response_scorer: Option<Arc<dyn FinalResponseScorer>>,
    default_tool_output_compression: ToolOutputCompressionConfig,
    tool_output_state: Option<Arc<ToolOutputCompressionState>>,
    default_tool_call_policy: ToolCallPolicyConfig,
) -> Result<HandlerResult, HandlerError> {
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| HandlerError::BadRequest("missing or invalid messages field".to_string()))?;

    let model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    let stream = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let stream_include_usage = extract_stream_include_usage(body)?;

    let tools_raw = match body.get("tools") {
        Some(Value::Array(tools)) => tools.clone(),
        Some(_) => {
            return Err(HandlerError::BadRequest(
                "tools must be an array".to_string(),
            ));
        }
        None => Vec::new(),
    };
    let step_contract = extract_proxy_step_contract(body)?;
    let return_raw_on_guardrail_failure =
        extract_forge_bool_field(body, FORGE_RETURN_RAW_ON_GUARDRAIL_FAILURE_FIELD)?;
    let tool_output_compression =
        extract_tool_output_compression_config(body, &default_tool_output_compression)?;
    let tool_call_policy = extract_tool_call_policy_config(body, &default_tool_call_policy)?;

    let sampling = extract_sampling(body);
    let passthrough = extract_passthrough(body);
    let mut request_options = LLMRequestOptions {
        sampling,
        passthrough,
        inbound_anthropic_body: inbound_anthropic_body
            .map(strip_forge_extension_from_body)
            .map(Arc::new),
        initial_openai_messages: None,
    };

    // Convert inbound OpenAI messages to internal format.
    let mut internal_msgs = openai_to_messages(messages)?;
    let tool_output_updates = compress_proxy_tool_results(
        &mut internal_msgs,
        &tool_output_compression,
        tool_output_state.as_deref(),
    );
    if !tool_output_updates.is_empty() {
        if let Some(body) = request_options.inbound_anthropic_body.take() {
            let mut patched = body.as_ref().clone();
            if patch_anthropic_tool_results(&mut patched, &tool_output_updates) {
                request_options.inbound_anthropic_body = Some(Arc::new(patched));
            } else {
                tracing::warn!(
                    "failed to patch compressed tool outputs into Anthropic request body; falling back to rebuilt body which may discard custom metadata or cache_control flags"
                );
            }
        }
    }
    if request_options.inbound_anthropic_body.is_some() {
        request_options.initial_openai_messages = Some(Arc::from(
            crate::core::inference::fold_and_serialize(
                &internal_msgs,
                client.api_format().as_str(),
            )
            .into_boxed_slice(),
        ));
    }

    // If no tools, pass through directly.
    if tools_raw.is_empty() {
        if let Some(contract) = step_contract.as_ref() {
            if !contract.required_steps.is_empty() {
                return Err(HandlerError::BadRequest(format!(
                    "{FORGE_EXTENSION_FIELD}.{FORGE_REQUIRED_STEPS_FIELD} requires tools"
                )));
            }
        }
        let api_format = client.api_format().as_str();
        let serialized = crate::core::inference::fold_and_serialize(&internal_msgs, api_format);
        return run_passthrough(
            client,
            &serialized,
            None,
            request_options,
            model_name,
            stream,
            stream_include_usage,
        )
        .await
        .map_err(HandlerError::Upstream);
    }

    // Parse client tools strictly, then add Forge's reserved terminal tool only
    // when the request has not declared a real terminal tool.
    let mut tool_specs = parse_tool_specs(&tools_raw)?;
    let respond_injected =
        add_proxy_respond_tool_if_needed(&mut tool_specs, step_contract.as_ref());

    let tool_names: IndexSet<String> = tool_specs.iter().map(|s| s.name.clone()).collect();
    let step_contract = validate_proxy_step_contract(step_contract, &tool_names, respond_injected)?;
    let request_options =
        sanitize_guarded_request_options(request_options, step_contract.as_ref())?;
    let validator = crate::guardrails::ResponseValidator::from_tool_specs(
        tool_specs.clone(),
        rescue_enabled,
        None,
    );
    let mut error_tracker = crate::guardrails::ErrorTracker::new(max_retries, 2);
    let mut tool_call_counter = 0;
    let mut step_enforcer = step_contract.map(|contract| {
        let mut enforcer = StepEnforcer::new(
            contract.required_steps,
            contract.terminal_tools.into_iter().collect(),
            None,
            3,
            2,
        );
        record_completed_proxy_tool_results(messages, &internal_msgs, &mut enforcer);
        enforcer
    });

    let mut accepted_usage = None;
    let mut accepted_usage_details = None;
    let mut tool_call_policy_state = ToolCallPolicyRequestState::new();
    let response = loop {
        let step_hint = step_enforcer
            .as_ref()
            .map(StepEnforcer::summary_hint)
            .unwrap_or_default();
        let inference_result = crate::core::inference::run_inference_with_options_shared_context(
            &mut internal_msgs,
            client.as_ref(),
            context_manager.as_ref(),
            &validator,
            &mut error_tracker,
            &tool_specs,
            &mut tool_call_counter,
            PROXY_STEP_INDEX,
            &step_hint,
            Some(max_retries + 1),
            stream,
            None,
            request_options.clone(),
        )
        .await;

        let result = match inference_result {
            Ok(Some(result)) => result,
            Ok(None) => break LLMResponse::Text(TextResponse::new("")),
            Err(crate::error::ForgeError::ToolCall(err)) => {
                if !return_raw_on_guardrail_failure {
                    return Err(HandlerError::Upstream(format!(
                        "model failed guarded tool-call validation after retries: {}",
                        err
                    )));
                }
                let raw = err.raw_response.unwrap_or_default();
                let usage = client.last_usage();
                let usage_details = client.last_usage_details();
                return Ok(text_content_result(
                    &raw,
                    model_name,
                    stream,
                    stream_include_usage,
                    usage.as_ref(),
                    usage_details.as_ref(),
                ));
            }
            Err(err) => return Err(HandlerError::Upstream(err.to_string())),
        };

        tool_call_counter = result.tool_call_counter;

        let result_usage = result.usage;
        let result_usage_details = result.usage_details;
        let response = result.response;
        let Some(enforcer) = step_enforcer.as_mut() else {
            match response {
                LLMResponse::ToolCalls(tool_calls) => {
                    if let Some(nudge) = evaluate_tool_call_policy(
                        &tool_calls,
                        &tool_specs,
                        &tool_call_policy,
                        &mut tool_call_policy_state,
                    ) {
                        emit_proxy_tool_policy_nudge_or_error(
                            &mut error_tracker,
                            tool_calls,
                            &mut internal_msgs,
                            &mut tool_call_counter,
                            &nudge.content,
                        )
                        .map_err(HandlerError::Upstream)?;
                        continue;
                    }
                    if let Some(nudge) = score_proxy_tool_calls(
                        scorer.clone(),
                        &internal_msgs,
                        &tool_calls,
                        None,
                        &tool_specs,
                    )
                    .await
                    {
                        emit_proxy_classifier_nudge_or_error(
                            &mut error_tracker,
                            tool_calls,
                            &mut internal_msgs,
                            &mut tool_call_counter,
                            &nudge,
                        )
                        .map_err(HandlerError::Upstream)?;
                        continue;
                    }
                    if let Some(nudge) = score_proxy_final_tool_calls(
                        final_response_scorer.clone(),
                        &internal_msgs,
                        &tool_calls,
                        None,
                        &tool_specs,
                    )
                    .await
                    {
                        emit_proxy_final_response_tool_nudge_or_error(
                            &mut error_tracker,
                            tool_calls,
                            &mut internal_msgs,
                            &mut tool_call_counter,
                            &nudge,
                        )
                        .map_err(HandlerError::Upstream)?;
                        continue;
                    }
                    accepted_usage = result_usage;
                    accepted_usage_details = result_usage_details;
                    break LLMResponse::ToolCalls(tool_calls);
                }
                LLMResponse::Text(text) => {
                    if let Some(nudge) = score_proxy_final_text(
                        final_response_scorer.clone(),
                        &internal_msgs,
                        &text.content,
                        None,
                        &tool_specs,
                    )
                    .await
                    {
                        emit_proxy_user_classifier_nudge_or_error(
                            &mut error_tracker,
                            &mut internal_msgs,
                            &nudge,
                        )
                        .map_err(HandlerError::Upstream)?;
                        continue;
                    }
                    accepted_usage = result_usage;
                    accepted_usage_details = result_usage_details;
                    break LLMResponse::Text(text);
                }
            }
        };

        match response {
            LLMResponse::ToolCalls(tool_calls) => {
                if !enforcer.is_satisfied() {
                    let step_check = enforcer.check(&tool_calls);
                    if step_check.needs_nudge {
                        emit_proxy_step_nudge_or_error(
                            enforcer,
                            step_check,
                            tool_calls,
                            &mut internal_msgs,
                            &mut tool_call_counter,
                        )
                        .map_err(HandlerError::Upstream)?;
                        continue;
                    }
                }

                if let Some(nudge) = evaluate_tool_call_policy(
                    &tool_calls,
                    &tool_specs,
                    &tool_call_policy,
                    &mut tool_call_policy_state,
                ) {
                    emit_proxy_tool_policy_nudge_or_error(
                        &mut error_tracker,
                        tool_calls,
                        &mut internal_msgs,
                        &mut tool_call_counter,
                        &nudge.content,
                    )
                    .map_err(HandlerError::Upstream)?;
                    continue;
                }
                if let Some(nudge) = score_proxy_tool_calls(
                    scorer.clone(),
                    &internal_msgs,
                    &tool_calls,
                    Some(enforcer),
                    &tool_specs,
                )
                .await
                {
                    emit_proxy_classifier_nudge_or_error(
                        &mut error_tracker,
                        tool_calls,
                        &mut internal_msgs,
                        &mut tool_call_counter,
                        &nudge,
                    )
                    .map_err(HandlerError::Upstream)?;
                    continue;
                }
                if let Some(nudge) = score_proxy_final_tool_calls(
                    final_response_scorer.clone(),
                    &internal_msgs,
                    &tool_calls,
                    Some(enforcer),
                    &tool_specs,
                )
                .await
                {
                    emit_proxy_final_response_tool_nudge_or_error(
                        &mut error_tracker,
                        tool_calls,
                        &mut internal_msgs,
                        &mut tool_call_counter,
                        &nudge,
                    )
                    .map_err(HandlerError::Upstream)?;
                    continue;
                }
                accepted_usage = result_usage;
                accepted_usage_details = result_usage_details;
                break LLMResponse::ToolCalls(tool_calls);
            }
            LLMResponse::Text(text) => {
                if !enforcer.is_satisfied() {
                    let tool_calls = vec![synthetic_respond_tool_call(&text)];
                    let step_check = enforcer.check(&tool_calls);
                    if step_check.needs_nudge {
                        emit_proxy_step_nudge_or_error(
                            enforcer,
                            step_check,
                            tool_calls,
                            &mut internal_msgs,
                            &mut tool_call_counter,
                        )
                        .map_err(HandlerError::Upstream)?;
                        continue;
                    }
                }

                if let Some(nudge) = score_proxy_final_text(
                    final_response_scorer.clone(),
                    &internal_msgs,
                    &text.content,
                    Some(enforcer),
                    &tool_specs,
                )
                .await
                {
                    emit_proxy_user_classifier_nudge_or_error(
                        &mut error_tracker,
                        &mut internal_msgs,
                        &nudge,
                    )
                    .map_err(HandlerError::Upstream)?;
                    continue;
                }
                accepted_usage = result_usage;
                accepted_usage_details = result_usage_details;
                break LLMResponse::Text(text);
            }
        }
    };

    let usage = accepted_usage;
    let usage_details = accepted_usage_details;

    let handler_result = match response {
        LLMResponse::Text(ref text) => text_response_result(
            text,
            model_name,
            stream,
            stream_include_usage,
            usage.as_ref(),
            usage_details.as_ref(),
        ),
        LLMResponse::ToolCalls(ref calls) => {
            let (real_calls, respond_text) = strip_respond_calls(calls);

            if real_calls.is_empty() {
                // Pure respond: convert to text.
                let text = respond_text.unwrap_or_default();
                text_content_result(
                    &text,
                    model_name,
                    stream,
                    stream_include_usage,
                    usage.as_ref(),
                    usage_details.as_ref(),
                )
            } else {
                // Real tool calls: return only those calls and drop respond.
                tool_calls_result(
                    &real_calls,
                    model_name,
                    stream,
                    stream_include_usage,
                    usage.as_ref(),
                    usage_details.as_ref(),
                )
            }
        }
    };

    Ok(handler_result)
}

/// Remove respond() calls, keeping only real tool calls.
pub fn filter_respond(calls: &[ToolCall]) -> Vec<ToolCall> {
    calls
        .iter()
        .filter(|c| c.tool != RESPOND_TOOL_NAME)
        .cloned()
        .collect()
}

/// Convert LLM response to OpenAI format (streaming or non-streaming).
pub fn process_response(response: &LLMResponse, model_name: &str, stream: bool) -> HandlerResult {
    match response {
        LLMResponse::ToolCalls(calls) => {
            tool_calls_result(calls, model_name, stream, false, None, None)
        }
        LLMResponse::Text(text) => {
            text_response_result(text, model_name, stream, false, None, None)
        }
    }
}

#[cfg(test)]
mod tests;
