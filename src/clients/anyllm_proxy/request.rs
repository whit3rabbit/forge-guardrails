use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{json, Value};

use crate::clients::base::{format_tool, LLMRequestOptions};
use crate::core::tool_spec::ToolSpec;

pub(super) fn build_openai_request_body(
    model: &str,
    messages: Vec<Value>,
    tools: Option<Vec<ToolSpec>>,
    options: LLMRequestOptions,
    stream: bool,
) -> Value {
    let mut messages = messages;
    let normalization = normalize_openai_message_tool_call_ids(&mut messages);
    if normalization.changed() {
        tracing::warn!(
            duplicate_tool_call_ids = normalization.duplicate_tool_call_ids,
            missing_tool_call_ids = normalization.missing_tool_call_ids,
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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ToolCallIdNormalization {
    duplicate_tool_call_ids: usize,
    missing_tool_call_ids: usize,
    remapped_tool_results: usize,
}

impl ToolCallIdNormalization {
    fn changed(self) -> bool {
        self.duplicate_tool_call_ids > 0
            || self.missing_tool_call_ids > 0
            || self.remapped_tool_results > 0
    }
}

fn normalize_openai_message_tool_call_ids(messages: &mut [Value]) -> ToolCallIdNormalization {
    let mut seen_call_ids = HashSet::new();
    let mut pending_tool_call_ids: HashMap<String, VecDeque<String>> = HashMap::new();
    let mut generated_counter = 0;
    let mut stats = ToolCallIdNormalization::default();

    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if role == "assistant" {
            if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    let raw_id = call
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .map(str::to_string);
                    let had_raw_id = raw_id.is_some();
                    let was_duplicate = raw_id
                        .as_deref()
                        .is_some_and(|id| seen_call_ids.contains(id));
                    let normalized_id = unique_or_generated_call_id(
                        raw_id.as_deref(),
                        &mut seen_call_ids,
                        &mut generated_counter,
                    );
                    if was_duplicate {
                        stats.duplicate_tool_call_ids += 1;
                    } else if !had_raw_id {
                        stats.missing_tool_call_ids += 1;
                    }
                    if let Some(raw_id) = raw_id {
                        pending_tool_call_ids
                            .entry(raw_id)
                            .or_default()
                            .push_back(normalized_id.clone());
                    }
                    if let Some(obj) = call.as_object_mut() {
                        obj.insert("id".to_string(), Value::String(normalized_id));
                    }
                }
            }
        }

        if role == "tool" {
            let raw_id = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .map(str::to_string);
            if let Some(raw_id) = raw_id {
                if let Some(normalized_id) = pending_tool_call_ids
                    .get_mut(&raw_id)
                    .and_then(VecDeque::pop_front)
                {
                    if normalized_id != raw_id {
                        stats.remapped_tool_results += 1;
                    }
                    if let Some(obj) = message.as_object_mut() {
                        obj.insert("tool_call_id".to_string(), Value::String(normalized_id));
                    }
                }
            }
        }
    }
    stats
}

fn unique_or_generated_call_id(
    id: Option<&str>,
    seen_call_ids: &mut HashSet<String>,
    generated_counter: &mut usize,
) -> String {
    if let Some(id) = id {
        if seen_call_ids.insert(id.to_string()) {
            return id.to_string();
        }
    }

    loop {
        let id = format!("call_forge_{generated_counter:09}");
        *generated_counter += 1;
        if seen_call_ids.insert(id.clone()) {
            return id;
        }
    }
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
