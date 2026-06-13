use crate::clients::base::{TextResponse, ToolCall};
use crate::core::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};
use crate::guardrails::{ErrorTracker, StepCheck, StepEnforcer};
use crate::tools::respond::RESPOND_TOOL_NAME;
use indexmap::IndexMap;
use serde_json::Value;
use std::collections::HashSet;

use super::telemetry::capture_guardrail_exhausted;

const PROXY_STEP_INDEX: i64 = 0;

/// Synthesizes a mock tool call to the respond tool, wrapping the final response text.
pub fn synthetic_respond_tool_call(text: &TextResponse) -> ToolCall {
    let mut args = IndexMap::new();
    args.insert("message".to_string(), Value::String(text.content.clone()));
    ToolCall::new(RESPOND_TOOL_NAME, args)
}

/// Emits a step nudge or returns an error if the premature attempts budget is exhausted.
pub fn emit_proxy_step_nudge_or_error(
    enforcer: &StepEnforcer,
    step_check: StepCheck,
    tool_calls: Vec<ToolCall>,
    messages: &mut Vec<Message>,
    tool_call_counter: &mut i64,
) -> Result<(), String> {
    if enforcer.premature_exhausted() {
        let pending = enforcer.pending();
        capture_guardrail_exhausted(
            "step_enforcement_exhausted",
            &tool_calls,
            &pending,
            Some(enforcer.premature_attempts()),
            None,
            None,
        );
        return Err(format!(
            "step enforcement exhausted after {} premature terminal tool attempts; pending required steps: {}",
            enforcer.premature_attempts(),
            pending.join(", ")
        ));
    }
    let nudge = step_check.nudge.expect("step nudge required");
    let calls =
        emit_proxy_assistant_tool_calls(tool_calls, messages, tool_call_counter, PROXY_STEP_INDEX);
    emit_proxy_guardrail_nudge_results(
        &calls,
        messages,
        PROXY_STEP_INDEX,
        MessageType::StepNudge,
        "[StepEnforcementError]",
        &nudge.content,
    );
    Ok(())
}

/// Emits a classifier nudge or returns an error if the retries budget is exhausted.
pub fn emit_proxy_classifier_nudge_or_error(
    error_tracker: &mut ErrorTracker,
    tool_calls: Vec<ToolCall>,
    messages: &mut Vec<Message>,
    tool_call_counter: &mut i64,
    nudge_content: &str,
) -> Result<(), String> {
    error_tracker.record_retry();
    if error_tracker.retries_exhausted() {
        capture_guardrail_exhausted(
            "classifier_objections_exhausted",
            &tool_calls,
            &[],
            Some(error_tracker.consecutive_retries()),
            Some(error_tracker.max_retries()),
            None,
        );
        return Err(format!(
            "classifier objections exhausted after {} retries",
            error_tracker.max_retries()
        ));
    }
    let calls =
        emit_proxy_assistant_tool_calls(tool_calls, messages, tool_call_counter, PROXY_STEP_INDEX);
    emit_proxy_guardrail_nudge_results(
        &calls,
        messages,
        PROXY_STEP_INDEX,
        MessageType::RetryNudge,
        "[ClassifierNudge]",
        nudge_content,
    );
    Ok(())
}

/// Emits a tool policy nudge or returns an error if the retries budget is exhausted.
pub fn emit_proxy_tool_policy_nudge_or_error(
    error_tracker: &mut ErrorTracker,
    tool_calls: Vec<ToolCall>,
    messages: &mut Vec<Message>,
    tool_call_counter: &mut i64,
    nudge_content: &str,
) -> Result<(), String> {
    error_tracker.record_retry();
    if error_tracker.retries_exhausted() {
        capture_guardrail_exhausted(
            "tool_call_policy_exhausted",
            &tool_calls,
            &[],
            Some(error_tracker.consecutive_retries()),
            Some(error_tracker.max_retries()),
            None,
        );
        return Err(format!(
            "tool-call policy objections exhausted after {} retries",
            error_tracker.max_retries()
        ));
    }
    let calls =
        emit_proxy_assistant_tool_calls(tool_calls, messages, tool_call_counter, PROXY_STEP_INDEX);
    emit_proxy_guardrail_nudge_results(
        &calls,
        messages,
        PROXY_STEP_INDEX,
        MessageType::RetryNudge,
        "[ToolCallPolicyNudge]",
        nudge_content,
    );
    Ok(())
}

/// Emits a final response nudge or returns an error if the retries budget is exhausted.
pub fn emit_proxy_final_response_tool_nudge_or_error(
    error_tracker: &mut ErrorTracker,
    tool_calls: Vec<ToolCall>,
    messages: &mut Vec<Message>,
    tool_call_counter: &mut i64,
    nudge_content: &str,
) -> Result<(), String> {
    error_tracker.record_retry();
    if error_tracker.retries_exhausted() {
        capture_guardrail_exhausted(
            "final_response_classifier_exhausted",
            &tool_calls,
            &[],
            Some(error_tracker.consecutive_retries()),
            Some(error_tracker.max_retries()),
            None,
        );
        return Err(format!(
            "final-response classifier objections exhausted after {} retries",
            error_tracker.max_retries()
        ));
    }
    let calls =
        emit_proxy_assistant_tool_calls(tool_calls, messages, tool_call_counter, PROXY_STEP_INDEX);
    emit_proxy_guardrail_nudge_results(
        &calls,
        messages,
        PROXY_STEP_INDEX,
        MessageType::RetryNudge,
        "[FinalResponseNudge]",
        nudge_content,
    );
    Ok(())
}

/// Emits a classifier nudge to the user or returns an error if retries are exhausted.
pub fn emit_proxy_user_classifier_nudge_or_error(
    error_tracker: &mut ErrorTracker,
    messages: &mut Vec<Message>,
    nudge_content: &str,
) -> Result<(), String> {
    error_tracker.record_retry();
    if error_tracker.retries_exhausted() {
        capture_guardrail_exhausted(
            "final_response_classifier_exhausted",
            &[],
            &[],
            Some(error_tracker.consecutive_retries()),
            Some(error_tracker.max_retries()),
            None,
        );
        return Err(format!(
            "final-response classifier objections exhausted after {} retries",
            error_tracker.max_retries()
        ));
    }
    messages.push(Message::new(
        MessageRole::User,
        format!("[FinalResponseNudge] {nudge_content}"),
        MessageMeta::new(MessageType::RetryNudge).with_step_index(PROXY_STEP_INDEX),
    ));
    Ok(())
}

/// Emits the assistant's reasoning and tool calls into history with unique IDs.
pub fn emit_proxy_assistant_tool_calls(
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
    let mut seen_call_ids = HashSet::new();
    for tc in &mut calls {
        let call_id = if let Some(id) = tc.id.as_deref().filter(|id| !id.is_empty()) {
            if seen_call_ids.insert(id.to_string()) {
                id.to_string()
            } else {
                next_unique_proxy_tool_call_id(tool_call_counter, &mut seen_call_ids)
            }
        } else {
            next_unique_proxy_tool_call_id(tool_call_counter, &mut seen_call_ids)
        };
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

fn next_unique_proxy_tool_call_id(
    tool_call_counter: &mut i64,
    seen_call_ids: &mut HashSet<String>,
) -> String {
    loop {
        let id = crate::core::inference::format_tool_call_id(*tool_call_counter);
        *tool_call_counter += 1;
        if seen_call_ids.insert(id.clone()) {
            return id;
        }
    }
}

/// Emits guardrail nudge results into the message history for each tool call.
pub fn emit_proxy_guardrail_nudge_results(
    calls: &[ToolCall],
    messages: &mut Vec<Message>,
    step_index: i64,
    msg_type: MessageType,
    prefix: &str,
    nudge_content: &str,
) {
    let error_content = format!("{prefix} {nudge_content}");
    for tc in calls {
        let call_id = tc.id.as_deref().unwrap_or_default();
        messages.push(
            Message::new(
                MessageRole::Tool,
                error_content.as_str(),
                MessageMeta::new(msg_type).with_step_index(step_index),
            )
            .with_tool_name(&tc.tool)
            .with_tool_call_id(call_id),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    #[test]
    fn emit_proxy_assistant_tool_calls_rewrites_duplicate_ids_in_history() {
        let mut messages = Vec::new();
        let mut counter = 0;
        let calls = vec![
            ToolCall::new("read", IndexMap::new()).with_id("dup"),
            ToolCall::new("write", IndexMap::new()).with_id("dup"),
            ToolCall::new("search", IndexMap::new()),
        ];

        let calls = emit_proxy_assistant_tool_calls(calls, &mut messages, &mut counter, 0);
        let ids = calls
            .iter()
            .map(|call| call.id.as_deref().unwrap())
            .collect::<Vec<_>>();

        assert_eq!(ids[0], "dup");
        assert_ne!(ids[1], "dup");
        assert_ne!(ids[2], "dup");
        assert_ne!(ids[1], ids[2]);
        assert_eq!(counter, 2);

        let emitted_ids = messages[0]
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|call| call.call_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(emitted_ids, ids);
    }
}
