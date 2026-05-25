//! Request handler: bridges HTTP layer and inference with guardrails.
//!
//! handle_chat_completions is the main entry point for /v1/chat/completions.
//! It converts inbound OpenAI messages, runs inference with validation/retry,
//! then strips respond() calls from output.

use crate::clients::base::{
    ChunkType, LLMClient, LLMRequestOptions, LLMResponse, LLMUsageDetails, TextResponse,
    TokenUsage, ToolCall,
};
use crate::context::manager::ContextManager;
use crate::core::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use crate::core::tool_spec::ToolSpec;
use crate::error::StreamError;
use crate::guardrails::{StepEnforcer, StepPrerequisite};
use crate::proxy::{
    extract_passthrough, extract_sampling, has_respond_tool, openai_to_messages,
    respond_tool_openai, strip_respond_calls, text_response_to_openai_with_usage_details,
    tool_calls_to_openai_with_usage_details,
};
use crate::tools::respond::RESPOND_TOOL_NAME;
use anyllm_translate::anthropic::streaming::StreamEvent;
use anyllm_translate::anthropic::MessageCreateRequest;
use futures_core::Stream;
use futures_util::StreamExt;
use indexmap::{IndexMap, IndexSet};
use serde_json::{json, Value};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

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

const FORGE_REQUIRED_STEPS_FIELD: &str = "forge_required_steps";
const FORGE_TERMINAL_TOOLS_FIELD: &str = "forge_terminal_tools";
const FORGE_TOOL_PREREQUISITES_FIELD: &str = "forge_tool_prerequisites";
const PROXY_STEP_INDEX: i64 = 0;

#[derive(Debug, Clone)]
struct ProxyWorkflowContract {
    required_steps: Vec<String>,
    terminal_tools: IndexSet<String>,
    tool_prerequisites: Option<IndexMap<String, Vec<StepPrerequisite>>>,
}

impl fmt::Debug for HandlerResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Response(value) => f.debug_tuple("Response").field(value).finish(),
            Self::StreamBody(_) => f.write_str("StreamBody(<openai event stream>)"),
        }
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
    let config = anyllm_translate::TranslationConfig::default();
    let openai_req = anyllm_translate::translate_request(body, &config)
        .map_err(|e| AnthropicHandlerError::BadRequest(e.to_string()))?;
    let mut openai_value = serde_json::to_value(&openai_req)
        .map_err(|e| AnthropicHandlerError::Internal(e.to_string()))?;
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
    )
    .await
    .map_err(AnthropicHandlerError::Upstream)?
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
    }
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
}

fn parse_proxy_workflow_contract(body: &Value) -> Result<Option<ProxyWorkflowContract>, String> {
    let has_contract_field = body.get(FORGE_REQUIRED_STEPS_FIELD).is_some()
        || body.get(FORGE_TERMINAL_TOOLS_FIELD).is_some()
        || body.get(FORGE_TOOL_PREREQUISITES_FIELD).is_some();
    if !has_contract_field {
        return Ok(None);
    }

    let required_steps =
        parse_string_array_field(body, FORGE_REQUIRED_STEPS_FIELD)?.unwrap_or_default();
    let terminal_tools = parse_string_array_field(body, FORGE_TERMINAL_TOOLS_FIELD)?
        .unwrap_or_else(|| vec![RESPOND_TOOL_NAME.to_string()]);
    let mut terminal_set: IndexSet<String> = terminal_tools.into_iter().collect();
    if terminal_set.is_empty() {
        terminal_set.insert(RESPOND_TOOL_NAME.to_string());
    }
    let tool_prerequisites =
        parse_tool_prerequisites_field(body.get(FORGE_TOOL_PREREQUISITES_FIELD))?;

    Ok(Some(ProxyWorkflowContract {
        required_steps,
        terminal_tools: terminal_set,
        tool_prerequisites,
    }))
}

fn parse_string_array_field(body: &Value, field: &str) -> Result<Option<Vec<String>>, String> {
    let Some(value) = body.get(field) else {
        return Ok(None);
    };
    let Some(items) = value.as_array() else {
        return Err(format!("{field} must be an array of strings"));
    };
    let mut strings = Vec::with_capacity(items.len());
    for item in items {
        let Some(s) = item.as_str() else {
            return Err(format!("{field} must be an array of strings"));
        };
        strings.push(s.to_string());
    }
    Ok(Some(strings))
}

fn parse_tool_prerequisites_field(
    value: Option<&Value>,
) -> Result<Option<IndexMap<String, Vec<StepPrerequisite>>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Some(obj) = value.as_object() else {
        return Err(format!(
            "{FORGE_TOOL_PREREQUISITES_FIELD} must be an object keyed by tool name"
        ));
    };

    let mut parsed = IndexMap::new();
    for (tool_name, raw_prereqs) in obj {
        let Some(items) = raw_prereqs.as_array() else {
            return Err(format!(
                "{FORGE_TOOL_PREREQUISITES_FIELD}.{tool_name} must be an array"
            ));
        };
        let mut prereqs = Vec::with_capacity(items.len());
        for item in items {
            if let Some(name) = item.as_str() {
                prereqs.push(StepPrerequisite::NameOnly(name.to_string()));
                continue;
            }
            let Some(prereq_obj) = item.as_object() else {
                return Err(format!(
                    "{FORGE_TOOL_PREREQUISITES_FIELD}.{tool_name} entries must be strings or objects"
                ));
            };
            let prereq_tool = prereq_obj
                .get("tool")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!(
                        "{FORGE_TOOL_PREREQUISITES_FIELD}.{tool_name} object entries require tool"
                    )
                })?;
            if let Some(match_arg) = prereq_obj.get("match_arg").and_then(Value::as_str) {
                prereqs.push(StepPrerequisite::ArgMatched {
                    tool: prereq_tool.to_string(),
                    match_arg: match_arg.to_string(),
                });
            } else {
                prereqs.push(StepPrerequisite::NameOnly(prereq_tool.to_string()));
            }
        }
        parsed.insert(tool_name.clone(), prereqs);
    }

    if parsed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parsed))
    }
}

/// Main handler for /v1/chat/completions.
///
/// When no tools are present, passes through to backend directly (no guardrails).
/// When tools are present, injects a respond tool if not already provided,
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
) -> Result<HandlerResult, String> {
    handle_chat_completions_impl(
        body,
        client,
        context_manager,
        max_retries,
        rescue_enabled,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn handle_chat_completions_impl<C: LLMClient + 'static>(
    body: &Value,
    client: &Arc<C>,
    context_manager: &Arc<Mutex<ContextManager>>,
    max_retries: i32,
    rescue_enabled: bool,
    inbound_anthropic_body: Option<Value>,
) -> Result<HandlerResult, String> {
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("missing or invalid messages field")?;

    let model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown");

    let stream = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let tools_raw = body
        .get("tools")
        .and_then(|t| t.as_array())
        .cloned()
        .unwrap_or_default();
    let workflow_contract = parse_proxy_workflow_contract(body)?;

    let sampling = extract_sampling(body);
    let passthrough = extract_passthrough(body);
    let request_options = LLMRequestOptions {
        sampling,
        passthrough,
        inbound_anthropic_body,
    };

    // Convert inbound OpenAI messages to internal format.
    let mut internal_msgs = openai_to_messages(messages);

    // If no tools, pass through directly.
    if tools_raw.is_empty() {
        let api_format = client.api_format().as_str();
        let serialized = crate::core::inference::fold_and_serialize(&internal_msgs, api_format);
        return run_passthrough(
            client,
            &serialized,
            None,
            request_options,
            model_name,
            stream,
        )
        .await;
    }

    // Tools present: inject respond tool if needed.
    let mut tools_with_respond = tools_raw.clone();
    if !has_respond_tool(&tools_with_respond) {
        tools_with_respond.push(respond_tool_openai());
    }

    // Parse tools into ToolSpec for the client.
    let tool_specs = parse_tool_specs(&tools_with_respond);

    let tool_names: Vec<String> = tool_specs.iter().map(|s| s.name.clone()).collect();
    let validator = crate::guardrails::ResponseValidator::new(tool_names, rescue_enabled, None);
    let mut error_tracker = crate::guardrails::ErrorTracker::new(max_retries, 2);
    let mut tool_call_counter = 0;
    let mut step_enforcer = workflow_contract.map(|contract| {
        let mut enforcer = StepEnforcer::new(
            contract.required_steps,
            contract.terminal_tools,
            contract.tool_prerequisites,
            3,
            2,
        );
        record_completed_proxy_tool_results(&internal_msgs, &mut enforcer);
        enforcer
    });

    let mut ctx = context_manager.lock().await;

    let response = loop {
        let step_hint = step_enforcer
            .as_ref()
            .map(StepEnforcer::summary_hint)
            .unwrap_or_default();
        let inference_result = crate::core::inference::run_inference_with_options(
            &mut internal_msgs,
            client.as_ref(),
            &mut ctx,
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
                let raw = err.raw_response.unwrap_or_default();
                let usage = client.last_usage();
                let usage_details = client.last_usage_details();
                drop(ctx);
                return Ok(text_content_result(
                    &raw,
                    model_name,
                    stream,
                    usage.as_ref(),
                    usage_details.as_ref(),
                ));
            }
            Err(err) => return Err(err.to_string()),
        };

        tool_call_counter = result.tool_call_counter;

        let tool_calls = match result.response {
            LLMResponse::ToolCalls(tool_calls) => tool_calls,
            other => break other,
        };

        let Some(enforcer) = step_enforcer.as_mut() else {
            break LLMResponse::ToolCalls(tool_calls);
        };

        let step_check = enforcer.check(&tool_calls);
        if step_check.needs_nudge {
            if enforcer.premature_exhausted() {
                return Err(format!(
                    "step enforcement exhausted after {} premature terminal tool attempts; pending required steps: {}",
                    enforcer.premature_attempts(),
                    enforcer.pending().join(", ")
                ));
            }
            let nudge = step_check.nudge.expect("step nudge required");
            let calls = emit_proxy_assistant_tool_calls(
                tool_calls,
                &mut internal_msgs,
                &mut tool_call_counter,
                PROXY_STEP_INDEX,
            );
            emit_proxy_guardrail_nudge_results(
                &calls,
                &mut internal_msgs,
                PROXY_STEP_INDEX,
                MessageType::StepNudge,
                "[StepEnforcementError]",
                &nudge.content,
            );
            continue;
        }

        let prereq_check = enforcer.check_prerequisites(&tool_calls);
        if prereq_check.needs_nudge {
            if enforcer.prereq_exhausted() {
                return Err(format!(
                    "prerequisite enforcement exhausted after {} violations",
                    enforcer.prereq_violations()
                ));
            }
            let nudge = prereq_check.nudge.expect("prerequisite nudge required");
            let calls = emit_proxy_assistant_tool_calls(
                tool_calls,
                &mut internal_msgs,
                &mut tool_call_counter,
                PROXY_STEP_INDEX,
            );
            emit_proxy_guardrail_nudge_results(
                &calls,
                &mut internal_msgs,
                PROXY_STEP_INDEX,
                MessageType::PrerequisiteNudge,
                "[PrerequisiteError]",
                &nudge.content,
            );
            continue;
        }

        break LLMResponse::ToolCalls(tool_calls);
    };

    drop(ctx);
    let usage = client.last_usage();
    let usage_details = client.last_usage_details();

    let handler_result = match response {
        LLMResponse::Text(ref text) => text_response_result(
            text,
            model_name,
            stream,
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
                    usage.as_ref(),
                    usage_details.as_ref(),
                )
            } else {
                // Real tool calls: return only those calls and drop respond.
                tool_calls_result(
                    &real_calls,
                    model_name,
                    stream,
                    usage.as_ref(),
                    usage_details.as_ref(),
                )
            }
        }
    };

    Ok(handler_result)
}

fn record_completed_proxy_tool_results(messages: &[Message], enforcer: &mut StepEnforcer) {
    let mut completed_call_ids = IndexSet::new();
    for message in messages {
        if message.role == MessageRole::Tool {
            if let Some(call_id) = &message.tool_call_id {
                completed_call_ids.insert(call_id.clone());
            }
        }
    }

    for message in messages {
        let Some(tool_calls) = &message.tool_calls else {
            continue;
        };
        for call in tool_calls {
            if completed_call_ids.contains(call.call_id.as_str()) {
                enforcer.record(&call.name, call.args.as_ref());
            }
        }
    }
}

fn emit_proxy_assistant_tool_calls(
    mut calls: Vec<ToolCall>,
    messages: &mut Vec<Message>,
    tool_call_counter: &mut i64,
    step_index: i64,
) -> Vec<ToolCall> {
    if let Some(reasoning) = calls.first().and_then(|tc| tc.reasoning.as_ref()) {
        messages.push(Message::new(
            MessageRole::Assistant,
            reasoning.as_str(),
            MessageMeta::new(MessageType::Reasoning).with_step_index(step_index),
        ));
    }

    let mut infos = Vec::with_capacity(calls.len());
    for tc in &mut calls {
        let call_id = tc.id.clone().unwrap_or_else(|| {
            let id = crate::core::inference::format_tool_call_id(*tool_call_counter);
            *tool_call_counter += 1;
            id
        });
        tc.id = Some(call_id.clone());
        infos.push(ToolCallInfo::new(&tc.tool, Some(tc.args.clone()), call_id));
    }

    messages.push(
        Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall).with_step_index(step_index),
        )
        .with_tool_calls(infos),
    );
    calls
}

fn emit_proxy_guardrail_nudge_results(
    calls: &[ToolCall],
    messages: &mut Vec<Message>,
    step_index: i64,
    msg_type: MessageType,
    prefix: &str,
    nudge_content: &str,
) {
    for tc in calls {
        let call_id = tc.id.as_deref().unwrap_or_default();
        let error_content = format!("{prefix} {nudge_content}");
        messages.push(
            Message::new(
                MessageRole::Tool,
                error_content,
                MessageMeta::new(msg_type).with_step_index(step_index),
            )
            .with_tool_name(&tc.tool)
            .with_tool_call_id(call_id),
        );
    }
}

/// Helper to forward requests to LLM Client without tools or guardrails (passthrough).
pub async fn run_passthrough<C: LLMClient + 'static>(
    client: &Arc<C>,
    serialized: &[Value],
    _tools: Option<Vec<ToolSpec>>,
    options: LLMRequestOptions,
    model_name: &str,
    stream: bool,
) -> Result<HandlerResult, String> {
    if stream {
        return run_passthrough_stream(client, serialized, options, model_name).await;
    }

    let response = client
        .send_with_options(serialized.to_vec(), None, options)
        .await
        .map_err(|e| e.to_string())?;
    let usage = client.last_usage();
    let usage_details = client.last_usage_details();

    match response {
        LLMResponse::Text(text) => Ok(text_response_result(
            &text,
            model_name,
            stream,
            usage.as_ref(),
            usage_details.as_ref(),
        )),
        LLMResponse::ToolCalls(_) => {
            // No-tools passthrough must not expose unexpected backend tool calls.
            Ok(text_content_result(
                "",
                model_name,
                stream,
                usage.as_ref(),
                usage_details.as_ref(),
            ))
        }
    }
}

async fn run_passthrough_stream<C: LLMClient + 'static>(
    client: &Arc<C>,
    serialized: &[Value],
    options: LLMRequestOptions,
    model_name: &str,
) -> Result<HandlerResult, String> {
    let mut backend_stream = client
        .send_stream_with_options(serialized.to_vec(), None, options)
        .await
        .map_err(|e| e.to_string())?;
    let client = client.clone();
    let model_name = model_name.to_string();
    let stream = async_stream::stream! {
        let completion_id = crate::proxy::proxy::openai_stream_completion_id();
        let mut emitted_text = false;

        while let Some(chunk_result) = backend_stream.next().await {
            let chunk = match chunk_result {
                Ok(chunk) => chunk,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            match chunk.chunk_type {
                ChunkType::TextDelta => {
                    if !chunk.content.is_empty() {
                        yield Ok(crate::proxy::proxy::text_delta_sse_event(
                            &completion_id,
                            &model_name,
                            &chunk.content,
                            !emitted_text,
                            None,
                        ));
                        emitted_text = true;
                    }
                }
                ChunkType::Final => {
                    if !emitted_text {
                        let content = match chunk.response {
                            Some(LLMResponse::Text(text)) => text.content,
                            _ => String::new(),
                        };
                        yield Ok(crate::proxy::proxy::text_delta_sse_event(
                            &completion_id,
                            &model_name,
                            &content,
                            true,
                            None,
                        ));
                    }
                    let usage = client.last_usage();
                    let usage_details = client.last_usage_details();
                    let usage_json = usage.as_ref().map(|u| {
                        crate::proxy::proxy::usage_to_openai_json_with_details(
                            Some(u),
                            usage_details.as_ref(),
                        )
                    });
                    yield Ok(crate::proxy::proxy::final_sse_event(
                        &completion_id,
                        &model_name,
                        "stop",
                        usage_json.as_ref(),
                    ));
                    return;
                }
                ChunkType::ToolCallDelta | ChunkType::Retry => {}
            }
        }

        yield Err(StreamError::default());
    };

    Ok(HandlerResult::StreamBody(Box::pin(stream)))
}

/// Convert a final text object while preserving the requested response shape.
fn text_response_result(
    text: &TextResponse,
    model_name: &str,
    stream: bool,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> HandlerResult {
    if stream {
        HandlerResult::StreamBody(text_events_stream(
            text.content.clone(),
            model_name.to_string(),
            usage.cloned(),
            usage_details.cloned(),
        ))
    } else {
        HandlerResult::Response(text_response_to_openai_with_usage_details(
            text,
            model_name,
            usage,
            usage_details,
        ))
    }
}

fn text_content_result(
    content: &str,
    model_name: &str,
    stream: bool,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> HandlerResult {
    if stream {
        HandlerResult::StreamBody(text_events_stream(
            content.to_string(),
            model_name.to_string(),
            usage.cloned(),
            usage_details.cloned(),
        ))
    } else {
        let text = TextResponse::new(content);
        HandlerResult::Response(text_response_to_openai_with_usage_details(
            &text,
            model_name,
            usage,
            usage_details,
        ))
    }
}

fn tool_calls_result(
    calls: &[ToolCall],
    model_name: &str,
    stream: bool,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> HandlerResult {
    if calls.is_empty() {
        text_content_result("", model_name, stream, usage, usage_details)
    } else if stream {
        HandlerResult::StreamBody(tool_call_events_stream(
            calls.to_vec(),
            model_name.to_string(),
            usage.cloned(),
            usage_details.cloned(),
        ))
    } else {
        HandlerResult::Response(tool_calls_to_openai_with_usage_details(
            calls,
            model_name,
            usage,
            usage_details,
        ))
    }
}

fn text_events_stream(
    content: String,
    model_name: String,
    usage: Option<TokenUsage>,
    usage_details: Option<LLMUsageDetails>,
) -> OpenAiEventStream {
    Box::pin(async_stream::stream! {
        for event in crate::proxy::proxy::text_to_sse_event_iter_with_usage_details(
            &content,
            &model_name,
            0,
            usage.as_ref(),
            usage_details.as_ref(),
        ) {
            yield Ok(event);
        }
    })
}

fn tool_call_events_stream(
    calls: Vec<ToolCall>,
    model_name: String,
    usage: Option<TokenUsage>,
    usage_details: Option<LLMUsageDetails>,
) -> OpenAiEventStream {
    Box::pin(async_stream::stream! {
        for event in crate::proxy::proxy::tool_calls_to_sse_event_iter_with_usage_details(
            &calls,
            &model_name,
            usage.as_ref(),
            usage_details.as_ref(),
        ) {
            yield Ok(event);
        }
    })
}

#[cfg(test)]
async fn collect_openai_events(mut stream: OpenAiEventStream) -> Result<Vec<Value>, StreamError> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
}

fn anthropic_events_stream(
    openai_events: OpenAiEventStream,
    model: String,
) -> AnthropicEventStream {
    Box::pin(async_stream::stream! {
        let mut openai_events = openai_events;
        let mut translator = anyllm_translate::new_stream_translator(model);

        while let Some(event) = openai_events.next().await {
            let event = match event {
                Ok(event) => event,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };
            let chunk: anyllm_translate::openai::ChatCompletionChunk = match serde_json::from_value(event) {
                Ok(chunk) => chunk,
                Err(err) => {
                    yield Err(StreamError::new(err.to_string()));
                    return;
                }
            };
            for event in translator.process_chunk(&chunk) {
                yield Ok(event);
            }
        }

        for event in translator.finish() {
            yield Ok(event);
        }
    })
}

#[cfg(test)]
async fn collect_anthropic_events(
    mut stream: AnthropicEventStream,
) -> Result<Vec<StreamEvent>, StreamError> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
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
        LLMResponse::ToolCalls(calls) => tool_calls_result(calls, model_name, stream, None, None),
        LLMResponse::Text(text) => text_response_result(text, model_name, stream, None, None),
    }
}

/// Parse OpenAI-format tool definitions into ToolSpec objects.
pub fn parse_tool_specs(tools: &[Value]) -> Vec<ToolSpec> {
    let mut specs = Vec::new();
    for tool in tools {
        if let Some(func) = tool.get("function") {
            let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let description = func
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let schema = func
                .get("parameters")
                .cloned()
                .unwrap_or(json!({"type": "object", "properties": {}}));

            if let Ok(mut spec) = ToolSpec::from_json_schema(name, description, &schema) {
                spec.json_schema = Some(schema);
                specs.push(spec);
            }
        }
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::base::{
        ApiFormat, ChunkStream, LLMRequestOptions, LLMUsageDetails, SamplingParams, StreamChunk,
        TokenUsage, ToolCall,
    };
    use crate::clients::AnthropicClient;
    use anyllm_translate::anthropic::MessageCreateRequest;
    use indexmap::IndexMap;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn filter_respond_removes_respond() {
        let calls = vec![
            ToolCall::new("respond", {
                let mut m = IndexMap::new();
                m.insert("message".into(), json!("hi"));
                m
            }),
            ToolCall::new("search", IndexMap::new()),
        ];
        let filtered = filter_respond(&calls);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].tool, "search");
    }

    #[test]
    fn filter_respond_keeps_all_real() {
        let calls = vec![
            ToolCall::new("search", IndexMap::new()),
            ToolCall::new("read", IndexMap::new()),
        ];
        let filtered = filter_respond(&calls);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn process_response_text_non_streaming() {
        let resp = LLMResponse::Text(TextResponse::new("hello"));
        let result = process_response(&resp, "model", false);
        match result {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "hello");
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Response"),
        }
    }

    async fn collect_stream_events(result: HandlerResult) -> Vec<Value> {
        match result {
            HandlerResult::StreamBody(stream) => collect_openai_events(stream).await.unwrap(),
            other => panic!("expected StreamBody, got {other:?}"),
        }
    }

    fn stream_from_response(response: LLMResponse) -> ChunkStream {
        Box::pin(futures_util::stream::iter(vec![Ok(StreamChunk::new(
            ChunkType::Final,
        )
        .with_response(response))]))
    }

    #[tokio::test]
    async fn process_response_text_streaming() {
        let resp = LLMResponse::Text(TextResponse::new("hello"));
        let result = process_response(&resp, "model", true);
        let events = collect_stream_events(result).await;
        assert!(!events.is_empty());
        let last = events.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn process_response_tool_calls_non_streaming() {
        let calls = vec![ToolCall::new("search", IndexMap::new())];
        let resp = LLMResponse::ToolCalls(calls);
        let result = process_response(&resp, "model", false);
        match result {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
            }
            _ => panic!("expected Response"),
        }
    }

    #[test]
    fn process_response_empty_tool_calls() {
        let resp = LLMResponse::ToolCalls(vec![]);
        let result = process_response(&resp, "model", false);
        match result {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "");
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Response"),
        }
    }

    #[test]
    fn parse_tool_specs_basic() {
        let schema = json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        });
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search things",
                "parameters": schema.clone()
            }
        })];
        let specs = parse_tool_specs(&tools);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "search");
        assert_eq!(specs[0].get_json_schema(), schema);
    }

    #[test]
    fn parse_tool_specs_empty() {
        let specs = parse_tool_specs(&[]);
        assert!(specs.is_empty());
    }

    #[test]
    fn extract_sampling_from_body() {
        let body = json!({
            "messages": [],
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42
        });
        let s = extract_sampling(&body).unwrap();
        assert_eq!(s["temperature"], 0.7);
        assert_eq!(s["seed"], 42);
    }

    #[test]
    fn extract_sampling_no_sampling_fields() {
        let body = json!({"messages": []});
        assert!(extract_sampling(&body).is_none());
    }

    // Integration-style tests for handle_chat_completions with a mock client.
    struct MockTextClient;

    impl LLMClient for MockTextClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            Ok(LLMResponse::Text(TextResponse::new("mock response")))
        }
        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
                "mock response",
            ))))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    struct MockOptionsClient {
        last_options: std::sync::Mutex<Option<LLMRequestOptions>>,
        usage: Option<TokenUsage>,
        usage_details: Option<LLMUsageDetails>,
    }

    struct MockStreamingOptionsClient {
        send_calls: AtomicUsize,
        stream_calls: AtomicUsize,
    }

    impl MockStreamingOptionsClient {
        fn new() -> Self {
            Self {
                send_calls: AtomicUsize::new(0),
                stream_calls: AtomicUsize::new(0),
            }
        }
    }

    impl LLMClient for MockStreamingOptionsClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }

        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            self.send_calls.fetch_add(1, Ordering::SeqCst);
            Ok(LLMResponse::Text(TextResponse::new("non-stream")))
        }

        async fn send_with_options(
            &self,
            messages: Vec<Value>,
            tools: Option<Vec<ToolSpec>>,
            options: LLMRequestOptions,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            self.send_calls.fetch_add(1, Ordering::SeqCst);
            self.send(messages, tools, options.sampling).await
        }

        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            Err(crate::error::StreamError::new("use stream_with_options"))
        }

        async fn send_stream_with_options(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _options: LLMRequestOptions,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            self.stream_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(StreamChunk::new(ChunkType::TextDelta).with_content("first")),
                Ok(StreamChunk::new(ChunkType::Final)
                    .with_response(LLMResponse::Text(TextResponse::new("first")))),
            ])))
        }

        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    impl MockOptionsClient {
        fn new(usage: Option<TokenUsage>) -> Self {
            Self {
                last_options: std::sync::Mutex::new(None),
                usage,
                usage_details: None,
            }
        }

        fn new_with_details(
            usage: Option<TokenUsage>,
            usage_details: Option<LLMUsageDetails>,
        ) -> Self {
            Self {
                last_options: std::sync::Mutex::new(None),
                usage,
                usage_details,
            }
        }
    }

    impl LLMClient for MockOptionsClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }

        fn last_usage(&self) -> Option<TokenUsage> {
            self.usage.clone()
        }

        fn last_usage_details(&self) -> Option<LLMUsageDetails> {
            self.usage_details.clone()
        }

        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            Ok(LLMResponse::Text(TextResponse::new("options response")))
        }

        async fn send_with_options(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            options: LLMRequestOptions,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            *self.last_options.lock().unwrap() = Some(options);
            Ok(LLMResponse::Text(TextResponse::new("options response")))
        }

        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
                "options response",
            ))))
        }

        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    struct MockToolCallClient;

    impl LLMClient for MockToolCallClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            let mut args = IndexMap::new();
            args.insert("message".into(), json!("responded text"));
            Ok(LLMResponse::ToolCalls(vec![ToolCall::new("respond", args)]))
        }
        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            let mut args = IndexMap::new();
            args.insert("message".into(), json!("responded text"));
            Ok(stream_from_response(LLMResponse::ToolCalls(vec![
                ToolCall::new("respond", args),
            ])))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    struct MockPassthroughToolCallClient;

    impl LLMClient for MockPassthroughToolCallClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            Ok(LLMResponse::ToolCalls(vec![ToolCall::new(
                "search",
                IndexMap::new(),
            )]))
        }
        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            Ok(stream_from_response(LLMResponse::ToolCalls(vec![
                ToolCall::new("search", IndexMap::new()),
            ])))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    fn dummy_ctx() -> ContextManager {
        ContextManager::new(
            Box::new(crate::context::strategies::NoCompact),
            4096,
            None,
            None,
            None,
        )
    }

    #[tokio::test]
    async fn handle_no_tools_passthrough() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false
        });
        let client = Arc::new(MockTextClient);
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "mock response");
            }
            _ => panic!("expected Response"),
        }
    }

    #[tokio::test]
    async fn handle_no_tools_forwards_passthrough_options_and_usage() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "request-model",
            "stream": false,
            "max_tokens": 128,
            "stop": ["done"],
            "tool_choice": {"type": "function", "function": {"name": "search"}},
            "response_format": {"type": "json_object"},
            "forge_required_steps": ["search"],
            "forge_terminal_tools": ["respond"],
            "forge_tool_prerequisites": {"respond": ["search"]},
            "temperature": 0.7
        });
        let client = Arc::new(MockOptionsClient::new(Some(TokenUsage::new(11, 5, 16))));
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;

        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "options response");
                assert_eq!(v["usage"]["prompt_tokens"], 11);
                assert_eq!(v["usage"]["completion_tokens"], 5);
                assert_eq!(v["usage"]["total_tokens"], 16);
            }
            _ => panic!("expected Response"),
        }

        let options = client
            .last_options
            .lock()
            .unwrap()
            .clone()
            .expect("options recorded");
        let passthrough = options.passthrough.expect("passthrough");
        assert_eq!(passthrough["model"], "request-model");
        assert_eq!(passthrough["max_tokens"], 128);
        assert_eq!(passthrough["stop"], json!(["done"]));
        assert_eq!(
            passthrough["tool_choice"],
            json!({"type": "function", "function": {"name": "search"}})
        );
        assert_eq!(
            passthrough["response_format"],
            json!({"type": "json_object"})
        );
        assert!(!passthrough.contains_key("messages"));
        assert!(!passthrough.contains_key("stream"));
        assert!(!passthrough.contains_key("temperature"));
        assert!(!passthrough.contains_key("forge_required_steps"));
        assert!(!passthrough.contains_key("forge_terminal_tools"));
        assert!(!passthrough.contains_key("forge_tool_prerequisites"));
        assert!(options.inbound_anthropic_body.is_none());
    }

    #[tokio::test]
    async fn handle_no_tools_emits_cache_usage_details() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "request-model",
            "stream": false
        });
        let details = LLMUsageDetails {
            cached_prompt_tokens: Some(8),
            prompt_cache_hit_tokens: Some(8),
            prompt_cache_miss_tokens: Some(3),
            cache_miss_prompt_tokens: Some(3),
            ..Default::default()
        };
        let client = Arc::new(MockOptionsClient::new_with_details(
            Some(TokenUsage::new(11, 5, 16)),
            Some(details),
        ));
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;

        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["usage"]["prompt_tokens"], 11);
                assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 8);
                assert_eq!(v["usage"]["prompt_cache_hit_tokens"], 8);
                assert_eq!(v["usage"]["prompt_cache_miss_tokens"], 3);
            }
            _ => panic!("expected Response"),
        }
    }

    struct MockRespondOptionsClient {
        last_options: std::sync::Mutex<Option<LLMRequestOptions>>,
    }

    impl MockRespondOptionsClient {
        fn new() -> Self {
            Self {
                last_options: std::sync::Mutex::new(None),
            }
        }
    }

    impl LLMClient for MockRespondOptionsClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }

        async fn send(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            let mut args = IndexMap::new();
            args.insert("message".into(), json!("done"));
            Ok(LLMResponse::ToolCalls(vec![ToolCall::new("respond", args)]))
        }

        async fn send_with_options(
            &self,
            messages: Vec<Value>,
            tools: Option<Vec<ToolSpec>>,
            options: LLMRequestOptions,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            *self.last_options.lock().unwrap() = Some(options.clone());
            self.send(messages, tools, options.sampling).await
        }

        async fn send_stream(
            &self,
            _messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
                "done",
            ))))
        }

        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    #[tokio::test]
    async fn handle_tools_forwards_prompt_cache_passthrough_fields() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "request-model",
            "stream": false,
            "prompt_cache_key": "tenant-a-tools-v1",
            "prompt_cache_retention": "24h",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }]
        });
        let client = Arc::new(MockRespondOptionsClient::new());
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true)
            .await
            .expect("handler");

        match result {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "done");
            }
            _ => panic!("expected Response"),
        }

        let options = client
            .last_options
            .lock()
            .unwrap()
            .clone()
            .expect("options recorded");
        let passthrough = options.passthrough.expect("passthrough");
        assert_eq!(passthrough["prompt_cache_key"], "tenant-a-tools-v1");
        assert_eq!(passthrough["prompt_cache_retention"], "24h");
    }

    #[tokio::test]
    async fn handle_no_tools_streaming_uses_stream_client() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "request-model",
            "stream": true,
            "temperature": 0.7
        });
        let client = Arc::new(MockStreamingOptionsClient::new());
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true)
            .await
            .expect("handler result");

        assert_eq!(client.send_calls.load(Ordering::SeqCst), 0);
        assert_eq!(client.stream_calls.load(Ordering::SeqCst), 1);

        let events = collect_stream_events(result).await;
        assert_eq!(events[0]["choices"][0]["delta"]["content"], "first");
        assert_eq!(
            events.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    }

    #[tokio::test]
    async fn anthropic_no_tools_streaming_uses_stream_client() {
        let raw = json!({
            "model": "claude-3",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
        let client = Arc::new(MockStreamingOptionsClient::new());
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true)
            .await
            .expect("handler result");

        assert_eq!(client.send_calls.load(Ordering::SeqCst), 0);
        assert_eq!(client.stream_calls.load(Ordering::SeqCst), 1);

        let events = match result {
            AnthropicHandlerResult::StreamBody(stream) => {
                collect_anthropic_events(stream).await.expect("events")
            }
            other => panic!("expected StreamBody, got {other:?}"),
        };
        let body = crate::proxy::server::format_anthropic_sse_body(events.as_slice());
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("first"));
        assert!(!body.contains("[DONE]"));
    }

    #[tokio::test]
    async fn anthropic_messages_translates_nonzero_usage() {
        let raw = json!({
            "model": "claude-3",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
        let client = Arc::new(MockOptionsClient::new(Some(TokenUsage::new(13, 7, 20))));
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true).await;

        match result.unwrap() {
            AnthropicHandlerResult::Response(v) => {
                assert_eq!(v["content"][0]["text"], "options response");
                assert_eq!(v["usage"]["input_tokens"], 13);
                assert_eq!(v["usage"]["output_tokens"], 7);
            }
            _ => panic!("expected Response"),
        }

        let options = client
            .last_options
            .lock()
            .unwrap()
            .clone()
            .expect("options recorded");
        assert_eq!(options.inbound_anthropic_body, Some(raw));
    }

    #[tokio::test]
    async fn anthropic_messages_includes_cache_usage_details() {
        let raw = json!({
            "model": "claude-3",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
        let details = LLMUsageDetails {
            cached_prompt_tokens: Some(13),
            cache_creation_prompt_tokens: Some(5),
            cache_read_input_tokens: Some(13),
            cache_creation_input_tokens: Some(5),
            ..Default::default()
        };
        let client = Arc::new(MockOptionsClient::new_with_details(
            Some(TokenUsage::new(20, 7, 27)),
            Some(details),
        ));
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true).await;

        match result.unwrap() {
            AnthropicHandlerResult::Response(v) => {
                assert_eq!(v["usage"]["input_tokens"], 20);
                assert_eq!(v["usage"]["output_tokens"], 7);
                assert_eq!(v["usage"]["cache_read_input_tokens"], 13);
                assert_eq!(v["usage"]["cache_creation_input_tokens"], 5);
            }
            _ => panic!("expected Response"),
        }
    }

    #[tokio::test]
    async fn anthropic_messages_clean_path_preserves_cache_control_to_backend() {
        let mut server = mockito::Server::new_async().await;
        let raw = json!({
            "model": "claude-3",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"}
                }]
            }]
        });
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::Json(raw.clone()))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1}
                })
                .to_string(),
            )
            .create_async()
            .await;
        let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
        let client = Arc::new(
            AnthropicClient::new("fallback-model", Some("test-key".to_string()))
                .with_base_url(server.url())
                .with_timeout(5.0),
        );
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true).await;

        match result.unwrap() {
            AnthropicHandlerResult::Response(v) => {
                assert_eq!(v["content"][0]["text"], "ok");
                assert_eq!(v["usage"]["input_tokens"], 3);
                assert_eq!(v["usage"]["output_tokens"], 1);
            }
            _ => panic!("expected Response"),
        }
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn anthropic_messages_streaming_preserves_cache_control_to_backend() {
        let mut server = mockito::Server::new_async().await;
        let raw = json!({
            "model": "claude-3",
            "max_tokens": 64,
            "stream": true,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "hi",
                    "cache_control": {"type": "ephemeral"}
                }]
            }]
        });
        let sse = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-3\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let mock = server
            .mock("POST", "/messages")
            .match_body(mockito::Matcher::Json(raw.clone()))
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(sse)
            .create_async()
            .await;
        let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
        let client = Arc::new(
            AnthropicClient::new("fallback-model", Some("test-key".to_string()))
                .with_base_url(server.url())
                .with_timeout(5.0),
        );
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 3, true)
            .await
            .expect("handler result");

        let events = match result {
            AnthropicHandlerResult::StreamBody(stream) => {
                collect_anthropic_events(stream).await.expect("events")
            }
            other => panic!("expected StreamBody, got {other:?}"),
        };
        let body = crate::proxy::server::format_anthropic_sse_body(events.as_slice());
        assert!(body.contains("ok"));
        assert!(!body.contains("[DONE]"));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn handle_no_tools_tool_calls_become_empty_text() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false
        });
        let client = Arc::new(MockPassthroughToolCallClient);
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "");
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Response"),
        }
    }

    #[tokio::test]
    async fn handle_no_tools_tool_calls_become_empty_text_streaming() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": true
        });
        let client = Arc::new(MockPassthroughToolCallClient);
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
        let events = collect_stream_events(result.unwrap()).await;
        assert_eq!(events[0]["choices"][0]["delta"]["content"], "");
        assert_eq!(
            events.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    }

    #[tokio::test]
    async fn handle_tools_respond_stripped() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false,
            "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        });
        let client = Arc::new(MockToolCallClient);
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "responded text");
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Response"),
        }
    }

    struct MockWorkflowContractClient {
        responses: Vec<LLMResponse>,
        calls: std::sync::Mutex<usize>,
        sent_messages: std::sync::Mutex<Vec<Vec<Value>>>,
    }

    impl MockWorkflowContractClient {
        fn new(responses: Vec<LLMResponse>) -> Self {
            Self {
                responses,
                calls: std::sync::Mutex::new(0),
                sent_messages: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn sent_messages(&self) -> Vec<Vec<Value>> {
            self.sent_messages.lock().unwrap().clone()
        }
    }

    impl LLMClient for MockWorkflowContractClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }

        async fn send(
            &self,
            messages: Vec<Value>,
            _tools: Option<Vec<ToolSpec>>,
            _sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            self.sent_messages.lock().unwrap().push(messages);
            let mut calls = self.calls.lock().unwrap();
            let response = self
                .responses
                .get(*calls)
                .or_else(|| self.responses.last())
                .cloned()
                .unwrap_or_else(|| LLMResponse::Text(TextResponse::new("")));
            *calls += 1;
            Ok(response)
        }

        async fn send_with_options(
            &self,
            messages: Vec<Value>,
            tools: Option<Vec<ToolSpec>>,
            options: LLMRequestOptions,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            self.send(messages, tools, options.sampling).await
        }

        async fn send_stream(
            &self,
            messages: Vec<Value>,
            tools: Option<Vec<ToolSpec>>,
            sampling: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            let response = self
                .send(messages, tools, sampling)
                .await
                .map_err(|err| crate::error::StreamError::new(err.to_string()))?;
            Ok(stream_from_response(response))
        }

        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    fn legacy_list_accounts_tool() -> Value {
        json!({
            "type": "function",
            "function": {
                "name": "legacy_list_accounts",
                "description": "List available accounts",
                "parameters": {"type": "object", "properties": {}}
            }
        })
    }

    #[tokio::test]
    async fn proxy_required_steps_block_premature_respond() {
        let mut respond_args = IndexMap::new();
        respond_args.insert("message".into(), json!("too soon"));
        let responses = vec![
            LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
            LLMResponse::ToolCalls(vec![ToolCall::new("legacy_list_accounts", IndexMap::new())]),
        ];
        let client = Arc::new(MockWorkflowContractClient::new(responses));
        let body = json!({
            "messages": [{"role": "user", "content": "audit account"}],
            "model": "test-model",
            "stream": false,
            "tools": [legacy_list_accounts_tool()],
            "forge_required_steps": ["legacy_list_accounts"],
            "forge_terminal_tools": ["respond"]
        });
        let ctx = Arc::new(Mutex::new(dummy_ctx()));

        let result = handle_chat_completions(&body, &client, &ctx, 3, true)
            .await
            .expect("handler result");

        match result {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
                let calls = v["choices"][0]["message"]["tool_calls"]
                    .as_array()
                    .expect("tool calls");
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0]["function"]["name"], json!("legacy_list_accounts"));
            }
            _ => panic!("expected Response"),
        }

        let sent = client.sent_messages();
        assert_eq!(sent.len(), 2);
        let second_wire = serde_json::to_string(&sent[1]).expect("wire json");
        assert!(second_wire.contains("[StepEnforcementError]"));
        assert!(second_wire.contains("legacy_list_accounts"));
    }

    #[tokio::test]
    async fn proxy_required_steps_use_prior_tool_result_history() {
        let mut respond_args = IndexMap::new();
        respond_args.insert("message".into(), json!("done"));
        let client = Arc::new(MockWorkflowContractClient::new(vec![
            LLMResponse::ToolCalls(vec![ToolCall::new("respond", respond_args)]),
        ]));
        let body = json!({
            "messages": [
                {"role": "user", "content": "audit account"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_list",
                        "type": "function",
                        "function": {"name": "legacy_list_accounts", "arguments": "{}"}
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_list",
                    "name": "legacy_list_accounts",
                    "content": "ACC-12345"
                }
            ],
            "model": "test-model",
            "stream": false,
            "tools": [legacy_list_accounts_tool()],
            "forge_required_steps": ["legacy_list_accounts"],
            "forge_terminal_tools": ["respond"]
        });
        let ctx = Arc::new(Mutex::new(dummy_ctx()));

        let result = handle_chat_completions(&body, &client, &ctx, 3, true)
            .await
            .expect("handler result");

        match result {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
                assert_eq!(v["choices"][0]["message"]["content"], "done");
            }
            _ => panic!("expected Response"),
        }
        let wire = serde_json::to_string(&client.sent_messages()).expect("wire json");
        assert!(!wire.contains("[StepEnforcementError]"));
    }

    struct MockAlwaysTextClient;
    impl LLMClient for MockAlwaysTextClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            Ok(LLMResponse::Text(TextResponse::new("always text")))
        }
        async fn send_stream(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
                "always text",
            ))))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    #[tokio::test]
    async fn handle_retries_exhausted_returns_text() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false,
            "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        });
        let client = Arc::new(MockAlwaysTextClient);
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 2, true).await;
        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "always text");
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Response"),
        }
    }

    struct MockTextSequenceClient {
        responses: Vec<String>,
        calls: std::sync::Mutex<usize>,
    }

    impl MockTextSequenceClient {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.into_iter().map(str::to_string).collect(),
                calls: std::sync::Mutex::new(0),
            }
        }
    }

    impl LLMClient for MockTextSequenceClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            let mut calls = self.calls.lock().unwrap();
            let content = self
                .responses
                .get(*calls)
                .or_else(|| self.responses.last())
                .cloned()
                .unwrap_or_default();
            *calls += 1;
            Ok(LLMResponse::Text(TextResponse::new(content)))
        }
        async fn send_stream(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            let mut calls = self.calls.lock().unwrap();
            let content = self
                .responses
                .get(*calls)
                .or_else(|| self.responses.last())
                .cloned()
                .unwrap_or_default();
            *calls += 1;
            Ok(stream_from_response(LLMResponse::Text(TextResponse::new(
                content,
            ))))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    #[tokio::test]
    async fn handle_retries_exhausted_returns_raw_response() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false,
            "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        });
        let client = Arc::new(MockTextSequenceClient::new(vec!["first bad", "raw final"]));
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 1, true).await;
        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["message"]["content"], "raw final");
                assert_eq!(v["choices"][0]["finish_reason"], "stop");
            }
            _ => panic!("expected Response"),
        }
    }

    #[tokio::test]
    async fn handle_retries_exhausted_returns_raw_response_streaming() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        });
        let client = Arc::new(MockTextSequenceClient::new(vec!["first bad", "raw final"]));
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 1, true).await;
        let events = collect_stream_events(result.unwrap()).await;
        assert_eq!(events[0]["choices"][0]["delta"]["content"], "raw final");
        assert_eq!(
            events.last().unwrap()["choices"][0]["finish_reason"],
            "stop"
        );
    }

    struct MockMixedToolClient;
    impl LLMClient for MockMixedToolClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            let mut respond_args = IndexMap::new();
            respond_args.insert("message".into(), json!("ignored text"));
            let mut search_args = IndexMap::new();
            search_args.insert("query".into(), json!("test"));
            Ok(LLMResponse::ToolCalls(vec![
                ToolCall::new("respond", respond_args),
                ToolCall::new("search", search_args),
            ]))
        }
        async fn send_stream(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            let mut respond_args = IndexMap::new();
            respond_args.insert("message".into(), json!("ignored text"));
            let mut search_args = IndexMap::new();
            search_args.insert("query".into(), json!("test"));
            Ok(stream_from_response(LLMResponse::ToolCalls(vec![
                ToolCall::new("respond", respond_args),
                ToolCall::new("search", search_args),
            ])))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    struct MockGuardedStreamingClient {
        stream_calls: AtomicUsize,
    }

    impl MockGuardedStreamingClient {
        fn new() -> Self {
            Self {
                stream_calls: AtomicUsize::new(0),
            }
        }
    }

    impl LLMClient for MockGuardedStreamingClient {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }

        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            Err(crate::error::BackendError::new(
                500,
                "send should not be used",
            ))
        }

        async fn send_stream(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            let call = self.stream_calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                Ok(Box::pin(futures_util::stream::iter(vec![
                    Ok(StreamChunk::new(ChunkType::ToolCallDelta).with_content("leaky-bogus")),
                    Ok(
                        StreamChunk::new(ChunkType::Final).with_response(LLMResponse::ToolCalls(
                            vec![ToolCall::new("bogus", IndexMap::new())],
                        )),
                    ),
                ])))
            } else {
                let mut args = IndexMap::new();
                args.insert("q".into(), json!("safe"));
                Ok(stream_from_response(LLMResponse::ToolCalls(vec![
                    ToolCall::new("search", args),
                ])))
            }
        }

        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    #[tokio::test]
    async fn guarded_streaming_holds_invalid_tool_chunks_until_validated() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
        });
        let client = Arc::new(MockGuardedStreamingClient::new());
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 2, true)
            .await
            .expect("handler result");

        assert_eq!(client.stream_calls.load(Ordering::SeqCst), 2);
        let events = collect_stream_events(result).await;
        let body = serde_json::to_string(&events).unwrap();
        assert!(!body.contains("leaky-bogus"));
        assert!(!body.contains("bogus"));
        assert!(body.contains("search"));
        assert_eq!(
            events.last().unwrap()["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    #[tokio::test]
    async fn anthropic_guarded_streaming_holds_invalid_tool_chunks_until_validated() {
        let raw = json!({
            "model": "claude-3",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "tools": [{
                "name": "search",
                "description": "Search",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}}
                }
            }]
        });
        let body: MessageCreateRequest = serde_json::from_value(raw.clone()).expect("request");
        let client = Arc::new(MockGuardedStreamingClient::new());
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_anthropic_messages(&body, &raw, &client, &ctx, 2, true)
            .await
            .expect("handler result");

        assert_eq!(client.stream_calls.load(Ordering::SeqCst), 2);
        let events = match result {
            AnthropicHandlerResult::StreamBody(stream) => {
                collect_anthropic_events(stream).await.expect("events")
            }
            other => panic!("expected StreamBody, got {other:?}"),
        };
        let body = crate::proxy::server::format_anthropic_sse_body(events.as_slice());
        assert!(!body.contains("leaky-bogus"));
        assert!(!body.contains("bogus"));
        assert!(body.contains("search"));
    }

    #[tokio::test]
    async fn handle_mixed_tools_drops_respond() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "stream": false,
            "tools": [
                {"type": "function", "function": {"name": "search", "description": "s", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "respond", "description": "r", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}}
            ]
        });
        let client = Arc::new(MockMixedToolClient);
        let ctx = Arc::new(Mutex::new(dummy_ctx()));
        let result = handle_chat_completions(&body, &client, &ctx, 3, true).await;
        match result.unwrap() {
            HandlerResult::Response(v) => {
                assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
                let tcs = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
                assert_eq!(tcs.len(), 1);
                assert_eq!(tcs[0]["function"]["name"], "search");
            }
            _ => panic!("expected Response"),
        }
    }

    struct MockSamplingTracker {
        last_sampling: std::sync::Mutex<Option<SamplingParams>>,
    }
    impl MockSamplingTracker {
        fn new() -> Self {
            Self {
                last_sampling: std::sync::Mutex::new(None),
            }
        }
    }
    impl LLMClient for MockSamplingTracker {
        fn api_format(&self) -> ApiFormat {
            ApiFormat::OpenAI
        }
        async fn send(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            sampling: Option<SamplingParams>,
        ) -> Result<LLMResponse, crate::error::BackendError> {
            *self.last_sampling.lock().unwrap() = sampling;
            Ok(LLMResponse::Text(TextResponse::new("ok")))
        }
        async fn send_stream(
            &self,
            _m: Vec<Value>,
            _t: Option<Vec<ToolSpec>>,
            _s: Option<SamplingParams>,
        ) -> Result<ChunkStream, crate::error::StreamError> {
            Err(crate::error::StreamError::new("not implemented"))
        }
        async fn get_context_length(
            &self,
        ) -> Result<Option<i64>, crate::error::ContextDiscoveryError> {
            Ok(Some(4096))
        }
    }

    #[tokio::test]
    async fn sampling_per_call_no_mutation() {
        let client = Arc::new(MockSamplingTracker::new());
        let ctx = Arc::new(Mutex::new(dummy_ctx()));

        // First call with sampling.
        let body1 = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test", "temperature": 0.7
        });
        handle_chat_completions(&body1, &client, &ctx, 0, true)
            .await
            .unwrap();
        let s1 = client.last_sampling.lock().unwrap().clone();
        assert_eq!(
            s1.as_ref().and_then(|m| m.get("temperature")),
            Some(&json!(0.7))
        );

        // Second call without sampling: should be None, not persisted from call 1.
        let body2 = json!({"messages": [{"role": "user", "content": "hi"}], "model": "test"});
        handle_chat_completions(&body2, &client, &ctx, 0, true)
            .await
            .unwrap();
        let s2 = client.last_sampling.lock().unwrap().clone();
        assert!(s2.is_none());
    }
}
