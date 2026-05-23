//! Core inference: compaction, folding, validation, and retries.
//! fold_and_serialize converts internal messages to API wire format.

use crate::client::LLMClient;
use crate::context::ContextManager;
use crate::error::StreamError;
use crate::guardrails::{ErrorTracker, ResponseValidator};
use crate::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use crate::streaming::{ChunkType, LLMResponse, StreamChunk};
use crate::tool_spec::ToolSpec;
use futures_util::StreamExt;
use serde_json::Value;

/// Tool call ID prefix for monotonic counter formatting.
const TOOL_CALL_ID_PREFIX: &str = "tc_";
const TOOL_CALL_ID_WIDTH: usize = 4;

/// Result of a single inference call.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub response: LLMResponse,
    pub new_messages: Vec<Message>,
    pub tool_call_counter: i64,
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

/// Convert internal messages to API-format values, folding reasoning into
/// the following tool-call message's content field.
///
/// Reasoning messages immediately preceding a tool-call message are merged
/// into the tool-call message's content. Orphaned reasoning (no following
/// tool-call) is emitted as a standalone assistant message.
pub fn fold_and_serialize(messages: &[Message], api_format: &str) -> Vec<Value> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        if msg.metadata.msg_type == MessageType::Reasoning {
            // Check if next message is a tool call.
            let next_is_tool_call = messages
                .get(i + 1)
                .map(|m| m.metadata.msg_type == MessageType::ToolCall)
                .unwrap_or(false);
            if next_is_tool_call {
                // Fold: merge reasoning content into the next message.
                let reasoning_content = msg.content.clone();
                let tool_msg = &messages[i + 1];
                let mut merged = tool_msg.clone();
                merged.content = if tool_msg.content.is_empty() {
                    reasoning_content
                } else {
                    format!("{}\n{}", reasoning_content, tool_msg.content)
                };
                result.push(merged.serialize(api_format));
                i += 2;
                continue;
            } else {
                // Orphaned reasoning: emit as standalone assistant message.
                let orphan = Message::new(
                    MessageRole::Assistant,
                    &msg.content,
                    MessageMeta::new(MessageType::Reasoning),
                );
                result.push(orphan.serialize(api_format));
                i += 1;
                continue;
            }
        }
        result.push(msg.serialize(api_format));
        i += 1;
    }
    result
}
/// Callback type for streaming chunks.
pub type OnChunkFn = Box<dyn Fn(&StreamChunk) + Send + Sync>;

/// Core inference function.
///
/// Takes a mutable message list (compaction modifies in-place), an LLM client,
/// context manager, response validator, error tracker, tool specs, and
/// configuration. Returns InferenceResult on success, or None if max_attempts
/// is exhausted.
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
) -> Option<InferenceResult> {
    let mut new_messages: Vec<Message> = Vec::new();
    let mut attempts = 0;
    let max = max_attempts.unwrap_or(i32::MAX);
    let api_format = client.api_format().as_str();
    let tools_opt = if tool_specs.is_empty() {
        None
    } else {
        Some(tool_specs.to_vec())
    };

    while attempts < max {
        attempts += 1;

        // Compact context.
        let compacted = context_manager.maybe_compact(messages, step_index, Some(step_hint));
        if let std::borrow::Cow::Owned(ref new_msgs) = compacted {
            messages.clear();
            messages.extend(new_msgs.iter().cloned());
        }

        // Check context thresholds and inject transient warning.
        let transient_warning = context_manager.check_thresholds(messages);

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
        }

        let sampling_owned = sampling.cloned();

        // Send to LLM.
        let response = if stream {
            match client
                .send_stream(wire, tools_opt.clone(), sampling_owned)
                .await
            {
                Ok(mut stream) => {
                    let mut final_response: Option<LLMResponse> = None;
                    while let Some(chunk_result) = stream.next().await {
                        match chunk_result {
                            Ok(chunk) => {
                                if let Some(ref cb) = on_chunk {
                                    cb(&chunk);
                                }
                                if chunk.chunk_type == ChunkType::Final {
                                    final_response = chunk.response.clone();
                                }
                            }
                            Err(_) => {
                                final_response = None;
                                break;
                            }
                        }
                    }
                    match final_response {
                        Some(resp) => resp,
                        None => {
                            // Stream ended without final chunk.
                            error_tracker.record_retry();
                            let err_msg = Message::new(
                                MessageRole::User,
                                StreamError::default().to_string(),
                                MessageMeta::new(MessageType::RetryNudge),
                            );
                            messages.push(err_msg.clone());
                            new_messages.push(err_msg);
                            continue;
                        }
                    }
                }
                Err(_) => {
                    error_tracker.record_retry();
                    continue;
                }
            }
        } else {
            match client.send(wire, tools_opt.clone(), sampling_owned).await {
                Ok(resp) => resp,
                Err(_) => {
                    error_tracker.record_retry();
                    continue;
                }
            }
        };

        // Sync token count.
        context_manager.update_token_count(estimate_tokens_from_response(&response));

        // Validate response.
        let validation = validator.validate(&response);

        if validation.needs_retry {
            error_tracker.record_retry();

            if error_tracker.retries_exhausted() {
                return None;
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
                        MessageMeta::new(MessageType::TextResponse),
                    );
                    messages.push(assistant_msg.clone());
                    new_messages.push(assistant_msg);

                    let nudge_msg = Message::new(
                        MessageRole::User,
                        &nudge_content,
                        MessageMeta::new(MessageType::RetryNudge),
                    );
                    messages.push(nudge_msg.clone());
                    new_messages.push(nudge_msg);
                }
                LLMResponse::ToolCalls(calls) => {
                    // Unknown tool: emit reasoning (if present), tool_call, error results.
                    let mut tool_call_infos = Vec::new();
                    for tc in calls {
                        if let Some(ref reasoning) = tc.reasoning {
                            let reasoning_msg = Message::new(
                                MessageRole::Assistant,
                                reasoning.as_str(),
                                MessageMeta::new(MessageType::Reasoning),
                            );
                            messages.push(reasoning_msg.clone());
                            new_messages.push(reasoning_msg);
                        }
                        let call_id = if let Some(ref id) = tc.id {
                            id.clone()
                        } else {
                            *tool_call_counter += 1;
                            format_tool_call_id(*tool_call_counter)
                        };
                        let info = ToolCallInfo::new(&tc.tool, Some(tc.args.clone()), &call_id);
                        tool_call_infos.push(info);
                    }
                    let tool_call_msg = Message::new(
                        MessageRole::Assistant,
                        "",
                        MessageMeta::new(MessageType::ToolCall),
                    )
                    .with_tool_calls(tool_call_infos.clone());
                    messages.push(tool_call_msg.clone());
                    new_messages.push(tool_call_msg);

                    // Error results with prefix tag.
                    for info in &tool_call_infos {
                        let error_content = format!("[TOOL_ERROR] {}", nudge_content);
                        let result_msg = Message::new(
                            MessageRole::Tool,
                            &error_content,
                            MessageMeta::new(MessageType::ToolResult),
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

        // Valid response.
        let tool_calls = validation.tool_calls.unwrap_or_default();

        if tool_calls.is_empty() {
            // Validated text response (should not happen normally but handle it).
            return Some(InferenceResult {
                response,
                new_messages,
                tool_call_counter: *tool_call_counter,
                attempts,
            });
        }

        // Build reasoning and tool call messages.
        for tc in &tool_calls {
            if let Some(ref reasoning) = tc.reasoning {
                let reasoning_msg = Message::new(
                    MessageRole::Assistant,
                    reasoning.as_str(),
                    MessageMeta::new(MessageType::Reasoning).with_step_index(step_index),
                );
                messages.push(reasoning_msg.clone());
                new_messages.push(reasoning_msg);
            }
        }

        let mut tool_calls = tool_calls;
        let mut infos = Vec::new();
        for tc in &mut tool_calls {
            let cid = if let Some(ref id) = tc.id {
                id.clone()
            } else {
                *tool_call_counter += 1;
                format_tool_call_id(*tool_call_counter)
            };
            tc.id = Some(cid.clone());
            infos.push(ToolCallInfo::new(&tc.tool, Some(tc.args.clone()), cid));
        }

        let tool_call_msg = Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall).with_step_index(step_index),
        )
        .with_tool_calls(infos);
        messages.push(tool_call_msg.clone());
        new_messages.push(tool_call_msg);

        return Some(InferenceResult {
            response: LLMResponse::ToolCalls(tool_calls),
            new_messages,
            tool_call_counter: *tool_call_counter,
            attempts,
        });
    }

    // Max attempts exhausted.
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    #[test]
    fn tool_call_id_format() {
        assert_eq!(format_tool_call_id(0), "tc_0000");
        assert_eq!(format_tool_call_id(1), "tc_0001");
        assert_eq!(format_tool_call_id(42), "tc_0042");
        assert_eq!(format_tool_call_id(9999), "tc_9999");
    }

    #[test]
    fn fold_and_serialize_basic_message() {
        let msg = Message::new(
            MessageRole::User,
            "hello",
            MessageMeta::new(MessageType::UserInput),
        );
        let result = fold_and_serialize(&[msg], "openai");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "hello");
    }

    #[test]
    fn fold_and_serialize_reasoning_folded_into_tool_call() {
        let reasoning = Message::new(
            MessageRole::Assistant,
            "thinking...",
            MessageMeta::new(MessageType::Reasoning),
        );
        let tool_call = Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "search",
            Some(IndexMap::new()),
            "tc_0001",
        )]);
        let result = fold_and_serialize(&[reasoning, tool_call], "openai");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["content"], "thinking...");
        assert!(result[0]["tool_calls"].is_array());
    }

    #[test]
    fn fold_and_serialize_orphaned_reasoning() {
        let reasoning = Message::new(
            MessageRole::Assistant,
            "thinking...",
            MessageMeta::new(MessageType::Reasoning),
        );
        let user = Message::new(
            MessageRole::User,
            "hello",
            MessageMeta::new(MessageType::UserInput),
        );
        let result = fold_and_serialize(&[reasoning, user], "openai");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "thinking...");
        assert_eq!(result[1]["content"], "hello");
    }

    #[test]
    fn fold_and_serialize_text_not_folded() {
        let text = Message::new(
            MessageRole::Assistant,
            "some text",
            MessageMeta::new(MessageType::TextResponse),
        );
        let tool_call = Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "search",
            Some(IndexMap::new()),
            "tc_0001",
        )]);
        let result = fold_and_serialize(&[text, tool_call], "openai");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn fold_and_serialize_reasoning_with_tool_call_content() {
        let reasoning = Message::new(
            MessageRole::Assistant,
            "let me think",
            MessageMeta::new(MessageType::Reasoning),
        );
        let tool_call = Message::new(
            MessageRole::Assistant,
            "existing content",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "search",
            Some(IndexMap::new()),
            "tc_0001",
        )]);
        let result = fold_and_serialize(&[reasoning, tool_call], "openai");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["content"], "let me think\nexisting content");
    }

    #[test]
    fn inference_result_fields() {
        let result = InferenceResult {
            response: LLMResponse::Text(crate::streaming::TextResponse::new("hi")),
            new_messages: vec![],
            tool_call_counter: 5,
            attempts: 1,
        };
        assert_eq!(result.tool_call_counter, 5);
        assert_eq!(result.attempts, 1);
    }
}
