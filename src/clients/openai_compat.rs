//! Compatibility helpers for OpenAI-shaped local model servers.

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::{Map, Value};

pub(crate) fn tool_call_name(tool_call: &Value) -> Option<&str> {
    tool_call
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| tool_call.get("name").and_then(Value::as_str))
}

pub(crate) fn tool_call_arguments(tool_call: &Value) -> Option<&Value> {
    tool_call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .or_else(|| tool_call.get("arguments"))
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ToolCallIdNormalization {
    pub(crate) duplicate_tool_call_ids: usize,
    pub(crate) missing_tool_call_ids: usize,
    pub(crate) remapped_tool_results: usize,
    pub(crate) orphan_tool_results: usize,
}

impl ToolCallIdNormalization {
    pub(crate) fn changed(self) -> bool {
        self.duplicate_tool_call_ids > 0
            || self.missing_tool_call_ids > 0
            || self.remapped_tool_results > 0
            || self.orphan_tool_results > 0
    }

    pub(crate) fn has_anomaly(self) -> bool {
        self.duplicate_tool_call_ids > 0
            || self.missing_tool_call_ids > 0
            || self.orphan_tool_results > 0
    }
}

/// Rewrite every outbound tool-call id to a 9-character alphanumeric form.
///
/// Mistral-compatible upstreams reject Forge's internal `call_000000000`
/// identifiers. Rewriting all outbound OpenAI-compatible transcripts keeps
/// assistant tool calls paired with their tool result messages while avoiding
/// model/provider-specific ID rules.
pub(crate) fn normalize_openai_message_tool_call_ids(
    messages: &mut [Value],
) -> ToolCallIdNormalization {
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

fn next_numeric_call_id(counter: &mut usize, seen: &mut HashSet<String>) -> String {
    loop {
        let id = format!("{counter:09}");
        *counter += 1;
        if seen.insert(id.clone()) {
            return id;
        }
    }
}

pub(crate) fn normalize_openai_response_tool_calls(value: &mut Value) {
    normalize_message_tool_calls(value, "message", true);
    normalize_message_tool_calls(value, "delta", false);
}

fn normalize_message_tool_calls(
    value: &mut Value,
    message_key: &str,
    default_missing_arguments: bool,
) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        let Some(tool_calls) = choice
            .get_mut(message_key)
            .and_then(|message| message.get_mut("tool_calls"))
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        for tool_call in tool_calls {
            normalize_tool_call(tool_call, default_missing_arguments);
        }
    }
}

fn normalize_tool_call(tool_call: &mut Value, default_missing_arguments: bool) {
    let Some(obj) = tool_call.as_object_mut() else {
        return;
    };

    if obj.get("function").and_then(Value::as_object).is_some() {
        if let Some(function) = obj.get_mut("function").and_then(Value::as_object_mut) {
            normalize_function_arguments(function);
        }
        ensure_function_type(obj);
        return;
    }

    let name = obj.get("name").cloned();
    let arguments = obj.get("arguments").cloned();
    if name.is_none() && arguments.is_none() {
        return;
    }
    let mut function = Map::new();
    if let Some(name) = name {
        function.insert("name".to_string(), name);
    }
    if let Some(arguments) = arguments {
        function.insert("arguments".to_string(), normalized_arguments(arguments));
    } else if default_missing_arguments {
        function.insert("arguments".to_string(), Value::String("{}".to_string()));
    }
    ensure_function_type(obj);
    obj.insert("function".to_string(), Value::Object(function));
}

fn ensure_function_type(tool_call: &mut Map<String, Value>) {
    tool_call
        .entry("type".to_string())
        .or_insert_with(|| Value::String("function".to_string()));
}

fn normalize_function_arguments(function: &mut Map<String, Value>) {
    if let Some(arguments) = function.remove("arguments") {
        function.insert("arguments".to_string(), normalized_arguments(arguments));
    }
}

fn normalized_arguments(arguments: Value) -> Value {
    match arguments {
        Value::String(_) => arguments,
        Value::Null => Value::String("{}".to_string()),
        other => Value::String(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn tool_call_parts_prefer_nested_openai_fields() {
        let call = json!({
            "name": "top",
            "arguments": {"x": 1},
            "function": {"name": "nested", "arguments": "{\"x\":2}"}
        });

        assert_eq!(tool_call_name(&call), Some("nested"));
        assert_eq!(
            tool_call_arguments(&call).and_then(Value::as_str),
            Some("{\"x\":2}")
        );
    }

    #[test]
    fn tool_call_parts_accept_llama_cpp_top_level_fields() {
        let call = json!({"name": "run", "arguments": {"x": 1}});

        assert_eq!(tool_call_name(&call), Some("run"));
        assert_eq!(tool_call_arguments(&call), Some(&json!({"x": 1})));
    }

    #[test]
    fn response_normalization_converts_llama_cpp_top_level_tool_call() {
        let mut response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "name": "run",
                        "arguments": {"x": 1}
                    }]
                }
            }]
        });

        normalize_openai_response_tool_calls(&mut response);

        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"],
            json!({"name": "run", "arguments": "{\"x\":1}"})
        );
    }

    #[test]
    fn response_normalization_adds_missing_function_type() {
        let mut response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "name": "run",
                        "arguments": {"x": 1}
                    }]
                }
            }]
        });

        normalize_openai_response_tool_calls(&mut response);

        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["type"],
            json!("function")
        );
    }

    #[test]
    fn message_id_normalization_rewrites_forge_ids_and_preserves_pairing() {
        let mut messages = vec![
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_000000000",
                    "type": "function",
                    "function": {"name": "run", "arguments": "{}"}
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "call_000000000",
                "name": "run",
                "content": "ok"
            }),
        ];

        let stats = normalize_openai_message_tool_call_ids(&mut messages);
        let id = messages[0]["tool_calls"][0]["id"].as_str().unwrap();

        assert!(id.len() == 9 && id.chars().all(|c| c.is_ascii_alphanumeric()));
        assert_eq!(messages[1]["tool_call_id"].as_str(), Some(id));
        assert!(stats.changed());
        assert!(!stats.has_anomaly());
    }

    #[test]
    fn response_normalization_converts_stream_delta_tool_call() {
        let mut response = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "name": "run",
                        "arguments": "{\"x\""
                    }]
                }
            }]
        });

        normalize_openai_response_tool_calls(&mut response);

        assert_eq!(
            response["choices"][0]["delta"]["tool_calls"][0]["function"],
            json!({"name": "run", "arguments": "{\"x\""})
        );
    }

    #[test]
    fn response_normalization_converts_stream_delta_arguments_without_name() {
        let mut response = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "arguments": ":1}"
                    }]
                }
            }]
        });

        normalize_openai_response_tool_calls(&mut response);

        assert_eq!(
            response["choices"][0]["delta"]["tool_calls"][0]["function"],
            json!({"arguments": ":1}"})
        );
    }

    #[test]
    fn response_normalization_keeps_stream_delta_name_only_without_arguments() {
        let mut response = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "name": "run"
                    }]
                }
            }]
        });

        normalize_openai_response_tool_calls(&mut response);

        assert_eq!(
            response["choices"][0]["delta"]["tool_calls"][0]["function"],
            json!({"name": "run"})
        );
    }
}
