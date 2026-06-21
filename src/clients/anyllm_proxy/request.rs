use serde_json::{json, Value};

use crate::clients::base::{format_tool, LLMRequestOptions};
use crate::clients::openai_compat;
use crate::core::tool_spec::ToolSpec;

pub(super) fn build_openai_request_body(
    model: &str,
    messages: Vec<Value>,
    tools: Option<Vec<ToolSpec>>,
    options: LLMRequestOptions,
    stream: bool,
) -> Value {
    let mut messages = messages;
    let normalization = openai_compat::normalize_openai_message_tool_call_ids(&mut messages);
    // Tool-call ids are rewritten on every request, so a remap on its own is
    // routine and logged at debug. A duplicate, missing, or orphan id is a real
    // transcript anomaly worth a warning.
    if normalization.has_anomaly() {
        tracing::warn!(
            duplicate_tool_call_ids = normalization.duplicate_tool_call_ids,
            missing_tool_call_ids = normalization.missing_tool_call_ids,
            orphan_tool_results = normalization.orphan_tool_results,
            remapped_tool_results = normalization.remapped_tool_results,
            "normalized outbound OpenAI tool call ids with anomalies"
        );
    } else if normalization.changed() {
        tracing::debug!(
            remapped_tool_results = normalization.remapped_tool_results,
            "normalized outbound OpenAI tool call ids"
        );
    }

    let mut body = Value::Object(options.passthrough.unwrap_or_default());
    if let Some(obj) = body.as_object_mut() {
        obj.entry("model".to_string())
            .or_insert_with(|| json!(model));
        obj.insert("messages".to_string(), Value::Array(messages));
        obj.insert("stream".to_string(), Value::Bool(stream));
    }

    if let Some(tool_specs) = tools {
        if !tool_specs.is_empty() {
            let formatted: Vec<Value> = tool_specs.iter().map(format_tool).collect();
            if let Some(obj) = body.as_object_mut() {
                obj.insert("tools".to_string(), Value::Array(formatted));
            }
        }
    }

    if let Some(params) = options.sampling {
        if let Some(obj) = body.as_object_mut() {
            for (key, value) in params {
                obj.insert(key, value);
            }
        }
    }

    body
}

pub(super) fn normalize_chat_completions_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if trimmed.ends_with("/v1/chat/completions") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/chat/completions")
    } else {
        format!("{trimmed}/v1/chat/completions")
    }
}
