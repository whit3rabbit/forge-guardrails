//! Core inference: compaction, folding, validation, and retries.
//! fold_and_serialize converts internal messages to API wire format.

use crate::clients::base::{
    ChunkType, LLMCallInfo, LLMClient, LLMRequestOptions, LLMResponse, LLMResponseEnvelope,
    LLMUsageDetails, StreamChunk, TokenUsage,
};
use crate::context::manager::ContextManager;
use crate::core::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use crate::core::tool_spec::ToolSpec;
use crate::error::{ForgeError, StreamError, ToolCallError};
use crate::guardrails::{ErrorTracker, ResponseValidator};
use futures_util::StreamExt;
use serde_json::Value;

/// Tool call ID prefix for monotonic counter formatting.
/// Matches Python: f"call_{tool_call_counter:09d}"
const TOOL_CALL_ID_PREFIX: &str = "call_";
const TOOL_CALL_ID_WIDTH: usize = 9;

/// Result of a single inference call.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// The response payload received from the LLM client.
    pub response: LLMResponse,
    /// Token usage reported by the backend for the accepted attempt.
    pub usage: Option<TokenUsage>,
    /// Provider-specific token/cache details for the accepted attempt.
    pub usage_details: Option<LLMUsageDetails>,
    /// Provider-routing and accounting metadata for the accepted attempt.
    pub call_info: Option<LLMCallInfo>,
    /// Any new messages generated during the inference process (e.g. nudges).
    pub new_messages: Vec<Message>,
    /// The running count of tool calls processed.
    pub tool_call_counter: i64,
    /// The number of attempts taken to produce a valid response.
    pub attempts: i32,
}

/// Format a tool call ID from a monotonic counter.
pub fn format_tool_call_id(counter: i64) -> String {
    format!(
        "{}{:0>width$}",
        TOOL_CALL_ID_PREFIX,
        counter,
        width = TOOL_CALL_ID_WIDTH
    )
}

mod context;
mod fold;

#[cfg(test)]
mod tests;

use context::ContextAccess;
pub use fold::fold_and_serialize;

/// Callback type for streaming chunks.
pub type OnChunkFn = Box<dyn Fn(&StreamChunk) + Send + Sync>;

/// Core inference function.
///
/// Takes a mutable message list (compaction modifies in-place), an LLM client,
/// context manager, response validator, error tracker, tool specs, and
/// configuration. Returns InferenceResult on success, None if max_attempts
/// is exhausted, or an error for backend, stream, or retry-budget failures.
#[allow(clippy::too_many_arguments)]
pub async fn run_inference<C: LLMClient>(
    messages: &mut Vec<Message>,
    client: &C,
    context_manager: &mut ContextManager,
    validator: &ResponseValidator,
    error_tracker: &mut ErrorTracker,
    tool_specs: &[ToolSpec],
    tool_call_counter: &mut i64,
    step_index: i64,
    step_hint: &str,
    max_attempts: Option<i32>,
    stream: bool,
    on_chunk: Option<&OnChunkFn>,
    sampling: Option<&serde_json::Map<String, Value>>,
) -> Result<Option<InferenceResult>, ForgeError> {
    let options = LLMRequestOptions::from_sampling(sampling.cloned());
    run_inference_with_options(
        messages,
        client,
        context_manager,
        validator,
        error_tracker,
        tool_specs,
        tool_call_counter,
        step_index,
        step_hint,
        max_attempts,
        stream,
        on_chunk,
        options,
    )
    .await
}

/// Core inference function with full request options.
#[allow(clippy::too_many_arguments)]
pub async fn run_inference_with_options<C: LLMClient>(
    messages: &mut Vec<Message>,
    client: &C,
    context_manager: &mut ContextManager,
    validator: &ResponseValidator,
    error_tracker: &mut ErrorTracker,
    tool_specs: &[ToolSpec],
    tool_call_counter: &mut i64,
    step_index: i64,
    step_hint: &str,
    max_attempts: Option<i32>,
    stream: bool,
    on_chunk: Option<&OnChunkFn>,
    options: LLMRequestOptions,
) -> Result<Option<InferenceResult>, ForgeError> {
    run_inference_with_options_inner(
        messages,
        client,
        ContextAccess::Direct(context_manager),
        validator,
        error_tracker,
        tool_specs,
        tool_call_counter,
        step_index,
        step_hint,
        max_attempts,
        stream,
        on_chunk,
        options,
    )
    .await
}

/// Core inference function that scopes a shared context lock to local mutations.
#[allow(clippy::too_many_arguments)]
pub async fn run_inference_shared_context<C: LLMClient>(
    messages: &mut Vec<Message>,
    client: &C,
    context_manager: &tokio::sync::Mutex<ContextManager>,
    validator: &ResponseValidator,
    error_tracker: &mut ErrorTracker,
    tool_specs: &[ToolSpec],
    tool_call_counter: &mut i64,
    step_index: i64,
    step_hint: &str,
    max_attempts: Option<i32>,
    stream: bool,
    on_chunk: Option<&OnChunkFn>,
    sampling: Option<&serde_json::Map<String, Value>>,
) -> Result<Option<InferenceResult>, ForgeError> {
    let options = LLMRequestOptions::from_sampling(sampling.cloned());
    run_inference_with_options_shared_context(
        messages,
        client,
        context_manager,
        validator,
        error_tracker,
        tool_specs,
        tool_call_counter,
        step_index,
        step_hint,
        max_attempts,
        stream,
        on_chunk,
        options,
    )
    .await
}

/// Core inference with full request options and scoped shared-context locking.
#[allow(clippy::too_many_arguments)]
pub async fn run_inference_with_options_shared_context<C: LLMClient>(
    messages: &mut Vec<Message>,
    client: &C,
    context_manager: &tokio::sync::Mutex<ContextManager>,
    validator: &ResponseValidator,
    error_tracker: &mut ErrorTracker,
    tool_specs: &[ToolSpec],
    tool_call_counter: &mut i64,
    step_index: i64,
    step_hint: &str,
    max_attempts: Option<i32>,
    stream: bool,
    on_chunk: Option<&OnChunkFn>,
    options: LLMRequestOptions,
) -> Result<Option<InferenceResult>, ForgeError> {
    run_inference_with_options_inner(
        messages,
        client,
        ContextAccess::Shared(context_manager),
        validator,
        error_tracker,
        tool_specs,
        tool_call_counter,
        step_index,
        step_hint,
        max_attempts,
        stream,
        on_chunk,
        options,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_inference_with_options_inner<C: LLMClient>(
    messages: &mut Vec<Message>,
    client: &C,
    mut context_manager: ContextAccess<'_>,
    validator: &ResponseValidator,
    error_tracker: &mut ErrorTracker,
    tool_specs: &[ToolSpec],
    tool_call_counter: &mut i64,
    step_index: i64,
    step_hint: &str,
    max_attempts: Option<i32>,
    stream: bool,
    on_chunk: Option<&OnChunkFn>,
    options: LLMRequestOptions,
) -> Result<Option<InferenceResult>, ForgeError> {
    let mut new_messages: Vec<Message> = Vec::new();
    let mut attempts = 0;
    let retry_limit = error_tracker.max_retries().saturating_add(1);
    let max = std::cmp::min(retry_limit, max_attempts.unwrap_or(i32::MAX));
    let api_format = client.api_format().as_str();
    let tools_opt = if tool_specs.is_empty() {
        None
    } else {
        Some(tool_specs.to_vec())
    };
    let mut next_options = options;

    while attempts < max {
        attempts += 1;
        let mut request_options = next_options.clone();

        // Compact context.
        let compacted = context_manager
            .maybe_compact(messages, step_index, Some(step_hint))
            .await;
        if let Some(new_msgs) = compacted {
            messages.clear();
            messages.extend(new_msgs);
            request_options.inbound_anthropic_body = None;
            request_options.initial_openai_messages = None;
        }

        // Check context thresholds and inject transient warning.
        let transient_warning = context_manager.check_thresholds(messages).await;
        if transient_warning.is_some() {
            request_options.inbound_anthropic_body = None;
            request_options.initial_openai_messages = None;
        }

        // Fold and serialize.
        let mut wire = fold_and_serialize(messages, api_format);

        // Inject transient context warning as a user message (not persisted).
        if let Some(ref warning) = transient_warning {
            let warning_msg = Message::new(
                MessageRole::User,
                warning.as_str(),
                MessageMeta::new(MessageType::ContextWarning),
            );
            wire.push(warning_msg.serialize(api_format));
            // Also emit to new_messages so on_message consumers see it (Python parity).
            new_messages.push(warning_msg);
        }

        next_options.inbound_anthropic_body = None;
        next_options.initial_openai_messages = None;

        // Send to LLM.
        let envelope = if stream {
            let mut stream = client
                .send_stream_with_options(wire, tools_opt.clone(), request_options)
                .await
                .map_err(ForgeError::from)?;
            let mut final_response: Option<LLMResponse> = None;
            let mut final_usage = None;
            let mut final_usage_details = None;
            let mut final_call_info = None;
            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result.map_err(ForgeError::from)?;
                if let Some(ref cb) = on_chunk {
                    cb(&chunk);
                }
                if chunk.chunk_type == ChunkType::Final {
                    final_usage = chunk.usage;
                    final_usage_details = chunk.usage_details;
                    final_call_info = chunk.call_info;
                    final_response = chunk.response;
                }
            }
            let response = final_response.ok_or_else(|| {
                ForgeError::Stream(StreamError::new(
                    "Stream ended without FINAL chunk - the client adapter may be malformed or the connection was interrupted",
                ))
            })?;
            LLMResponseEnvelope::from_response(response).with_metadata(
                final_usage.or_else(|| client.last_usage()),
                final_usage_details.or_else(|| client.last_usage_details()),
                final_call_info.or_else(|| client.last_call_info()),
            )
        } else {
            client
                .send_envelope_with_options(wire, tools_opt.clone(), request_options)
                .await
                .map_err(ForgeError::from)?
        };
        let LLMResponseEnvelope {
            response,
            usage,
            usage_details,
            call_info,
        } = envelope;

        // Sync token count: prefer real usage from client, fall back to heuristic.
        let observed_tokens = if let Some(usage) = usage.as_ref() {
            usage.total_tokens
        } else {
            estimate_tokens_from_response(&response)
        };
        context_manager.update_token_count(observed_tokens).await;

        // Validate response.
        let validation = validator.validate(&response);

        if validation.needs_retry {
            error_tracker.record_retry();

            if error_tracker.retries_exhausted() {
                let raw = response_to_raw_string(&response).unwrap_or_default();
                return Err(ForgeError::ToolCall(
                    ToolCallError::new(format!(
                        "Retries exhausted after {} consecutive failed attempts",
                        error_tracker.max_retries()
                    ))
                    .with_raw_response(raw),
                ));
            }

            // Emit retry messages based on response type.
            let nudge_content = validation
                .nudge
                .as_ref()
                .map(|n| n.content.clone())
                .unwrap_or_default();

            match &response {
                LLMResponse::Text(text) => {
                    // Bare text: emit assistant message + user nudge.
                    let assistant_msg = Message::new(
                        MessageRole::Assistant,
                        &text.content,
                        MessageMeta::new(MessageType::TextResponse).with_step_index(step_index),
                    );
                    messages.push(assistant_msg.clone());
                    new_messages.push(assistant_msg);

                    let nudge_msg = Message::new(
                        MessageRole::User,
                        &nudge_content,
                        MessageMeta::new(MessageType::RetryNudge).with_step_index(step_index),
                    );
                    messages.push(nudge_msg.clone());
                    new_messages.push(nudge_msg);
                }
                LLMResponse::ToolCalls(calls) => {
                    if calls.is_empty() {
                        let nudge_msg = Message::new(
                            MessageRole::User,
                            &nudge_content,
                            MessageMeta::new(MessageType::RetryNudge).with_step_index(step_index),
                        );
                        messages.push(nudge_msg.clone());
                        new_messages.push(nudge_msg);
                        continue;
                    }

                    // Unknown tool: emit reasoning (if present), tool_call, error results.
                    let mut tool_call_infos = Vec::new();
                    for tc in calls {
                        if let Some(ref reasoning) = tc.reasoning {
                            let reasoning_msg = Message::new(
                                MessageRole::Assistant,
                                reasoning.as_str(),
                                MessageMeta::new(MessageType::Reasoning)
                                    .with_step_index(step_index),
                            );
                            messages.push(reasoning_msg.clone());
                            new_messages.push(reasoning_msg);
                        }
                        let call_id = format_tool_call_id(*tool_call_counter);
                        *tool_call_counter += 1;
                        let info = ToolCallInfo::new(&tc.tool, Some(tc.args.clone()), &call_id);
                        tool_call_infos.push(info);
                    }
                    let tool_call_msg = Message::new(
                        MessageRole::Assistant,
                        "",
                        MessageMeta::new(MessageType::ToolCall).with_step_index(step_index),
                    )
                    .with_tool_calls(tool_call_infos.clone());
                    messages.push(tool_call_msg.clone());
                    new_messages.push(tool_call_msg);

                    let error_prefix = validation
                        .nudge
                        .as_ref()
                        .map(|nudge| match nudge.kind.as_str() {
                            "unknown_tool" => "[UnknownTool]",
                            "invalid_arguments" => "[InvalidArguments]",
                            _ => "[Guardrail]",
                        })
                        .unwrap_or("[Guardrail]");

                    for info in &tool_call_infos {
                        let error_content = format!("{} {}", error_prefix, nudge_content);
                        let result_msg = Message::new(
                            MessageRole::Tool,
                            &error_content,
                            MessageMeta::new(MessageType::RetryNudge).with_step_index(step_index),
                        )
                        .with_tool_name(&info.name)
                        .with_tool_call_id(&info.call_id);
                        messages.push(result_msg.clone());
                        new_messages.push(result_msg);
                    }
                }
            }
            continue;
        }

        // Valid response — reset retry budget (Python parity: error_tracker.reset_retries()).
        error_tracker.reset_retries();
        let mut tool_calls = validation.tool_calls.unwrap_or_default();
        for call in &mut tool_calls {
            call.id = None;
        }

        return Ok(Some(InferenceResult {
            response: LLMResponse::ToolCalls(tool_calls),
            usage,
            usage_details,
            call_info,
            new_messages,
            tool_call_counter: *tool_call_counter,
            attempts,
        }));
    }

    Ok(None)
}

/// Rough token estimate from a response for syncing the context manager.
fn estimate_tokens_from_response(response: &LLMResponse) -> i64 {
    match response {
        LLMResponse::Text(t) => (t.content.len() as i64) / 4,
        LLMResponse::ToolCalls(calls) => {
            let total: usize = calls
                .iter()
                .map(|c| {
                    c.tool.len()
                        + c.args.values().map(|v| v.to_string().len()).sum::<usize>()
                        + c.reasoning.as_ref().map(|r| r.len()).unwrap_or(0)
                })
                .sum();
            (total as i64) / 4
        }
    }
}

/// Extract a raw string from a response for error reporting.
#[allow(dead_code)]
pub(crate) fn response_to_raw_string(response: &LLMResponse) -> Option<String> {
    match response {
        LLMResponse::Text(t) => Some(t.content.clone()),
        LLMResponse::ToolCalls(calls) => {
            let s: Vec<String> = calls
                .iter()
                .map(|c| {
                    format!(
                        "{}({})",
                        c.tool,
                        serde_json::to_string(&c.args).unwrap_or_default()
                    )
                })
                .collect();
            Some(s.join(", "))
        }
    }
}
