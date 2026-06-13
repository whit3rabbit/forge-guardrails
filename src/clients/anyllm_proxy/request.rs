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

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ToolCallIdNormalization {
    duplicate_tool_call_ids: usize,
    missing_tool_call_ids: usize,
    remapped_tool_results: usize,
    /// Tool-result messages whose `tool_call_id` matched no assistant tool call
    /// in the transcript. We still rewrite these to a Mistral-safe id, but an
    /// unpaired tool result signals an upstream transcript problem (e.g. a
    /// dropped tool-call message).
    orphan_tool_results: usize,
}

impl ToolCallIdNormalization {
    fn changed(self) -> bool {
        self.duplicate_tool_call_ids > 0
            || self.missing_tool_call_ids > 0
            || self.remapped_tool_results > 0
            || self.orphan_tool_results > 0
    }

    fn has_anomaly(self) -> bool {
        self.duplicate_tool_call_ids > 0
            || self.missing_tool_call_ids > 0
            || self.orphan_tool_results > 0
    }
}

/// Rewrite every outbound tool-call id to a 9-character alphanumeric form.
///
/// Mistral (including via OpenRouter) rejects any `tool_call_id` that does not
/// match `^[a-zA-Z0-9]{9}$`: exactly 9 characters, alphanumeric only, with no
/// `call_` prefix or underscore. Forge's internal ids are `call_000000000`
/// (14 chars, underscore), which Mistral strips/truncates to a colliding
/// `call00000`, surfacing as "Duplicate tool call id" (code 3230). OpenAI and
/// llama.cpp treat ids as opaque, so a 9-digit numeric id is accepted by every
/// OpenAI-compatible upstream. We rewrite unconditionally rather than sniffing
/// the model name, because OpenRouter routing (e.g. `openrouter/free`) hides
/// which provider ultimately serves the request.
///
/// Assistant `tool_calls[].id` and the matching `role:"tool"` `tool_call_id`
/// are remapped to the same value so tool-call/tool-result pairing is preserved.
/// A tool-result whose `tool_call_id` matches no assistant tool call (e.g. its
/// tool-call message was dropped) is still rewritten to a fresh Mistral-safe id
/// and counted as an orphan, so no `call_*` id can leak to the upstream.
fn normalize_openai_message_tool_call_ids(messages: &mut [Value]) -> ToolCallIdNormalization {
    let mut seen_normalized: HashSet<String> = HashSet::new();
    let mut seen_raw: HashSet<String> = HashSet::new();
    let mut pending_tool_call_ids: HashMap<String, VecDeque<String>> = HashMap::new();
    let mut counter: usize = 0;
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
                    match raw_id.as_deref() {
                        Some(id) => {
                            if !seen_raw.insert(id.to_string()) {
                                stats.duplicate_tool_call_ids += 1;
                            }
                        }
                        None => stats.missing_tool_call_ids += 1,
                    }
                    let normalized_id = next_numeric_call_id(&mut counter, &mut seen_normalized);
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
                let normalized_id = match pending_tool_call_ids
                    .get_mut(&raw_id)
                    .and_then(VecDeque::pop_front)
                {
                    Some(paired_id) => {
                        if paired_id != raw_id {
                            stats.remapped_tool_results += 1;
                        }
                        paired_id
                    }
                    // No matching assistant tool call: rewrite anyway so a
                    // `call_*` id can never reach a strict upstream like Mistral.
                    None => {
                        stats.orphan_tool_results += 1;
                        next_numeric_call_id(&mut counter, &mut seen_normalized)
                    }
                };
                if let Some(obj) = message.as_object_mut() {
                    obj.insert("tool_call_id".to_string(), Value::String(normalized_id));
                }
            }
        }
    }
    stats
}

/// Next unused 9-digit numeric id (`000000000`, `000000001`, ...). All digits,
/// length 9, so it satisfies Mistral's `^[a-zA-Z0-9]{9}$`. The `seen` guard is a
/// defensive check; the monotonic counter already guarantees uniqueness for any
/// realistic transcript.
fn next_numeric_call_id(counter: &mut usize, seen: &mut HashSet<String>) -> String {
    loop {
        let id = format!("{counter:09}");
        *counter += 1;
        if seen.insert(id.clone()) {
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
