use crate::core::message::{Message, MessageRole, ToolCallInfo};
use crate::guardrails::StepEnforcer;
use indexmap::IndexMap;
use serde_json::Value;

use super::request_contract::FORGE_EXTENSION_FIELD;

const FORGE_TOOL_STATUS_FIELD: &str = "tool_status";
const FORGE_TOOL_STATUS_OK: &str = "ok";

pub(super) fn record_completed_proxy_tool_results(
    raw_messages: &[Value],
    messages: &[Message],
    enforcer: &mut StepEnforcer,
) {
    let mut pending_tool_calls: IndexMap<String, ToolCallInfo> = IndexMap::new();
    let raw_tool_statuses = proxy_tool_statuses_by_call_id(raw_messages);
    for message in messages {
        match message.role {
            MessageRole::Assistant => {
                let Some(tool_calls) = &message.tool_calls else {
                    continue;
                };
                for call in tool_calls {
                    pending_tool_calls.insert(call.call_id.clone(), call.clone());
                }
            }
            MessageRole::Tool => {
                let Some(call_id) = &message.tool_call_id else {
                    continue;
                };
                let raw_status = raw_tool_statuses.get(call_id.as_str()).copied().flatten();
                if !proxy_tool_result_succeeded(raw_status, &message.content) {
                    continue;
                }
                if let Some(call) = pending_tool_calls.get(call_id) {
                    enforcer.record(&call.name, call.args.as_ref());
                }
            }
            _ => {}
        }
    }
}

fn proxy_tool_statuses_by_call_id(raw_messages: &[Value]) -> IndexMap<&str, Option<&str>> {
    let mut statuses = IndexMap::new();
    for raw in raw_messages {
        if raw.get("role").and_then(Value::as_str) != Some("tool") {
            continue;
        }
        let Some(call_id) = raw.get("tool_call_id").and_then(Value::as_str) else {
            continue;
        };
        let status = raw
            .get(FORGE_EXTENSION_FIELD)
            .and_then(Value::as_object)
            .and_then(|forge| forge.get(FORGE_TOOL_STATUS_FIELD))
            .and_then(Value::as_str);
        statuses.insert(call_id, status);
    }
    statuses
}

fn proxy_tool_result_succeeded(raw_status: Option<&str>, content: &str) -> bool {
    if let Some(status) = raw_status {
        return status == FORGE_TOOL_STATUS_OK;
    }

    !has_explicit_proxy_tool_error_prefix(content)
}

fn has_explicit_proxy_tool_error_prefix(content: &str) -> bool {
    let normalized = content.trim_start().to_ascii_lowercase();
    normalized.starts_with("[toolerror]")
        || normalized.starts_with("[toolresolutionerror]")
        || normalized.starts_with("[toolexecutionerror]")
        || normalized.starts_with("[tool_error]")
}
