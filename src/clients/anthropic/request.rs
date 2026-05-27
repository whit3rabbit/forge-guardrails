use serde_json::Value;

use super::{convert, AnthropicClient};
use crate::clients::base::LLMRequestOptions;
use crate::core::tool_spec::ToolSpec;

impl AnthropicClient {
    pub(super) fn build_body_with_options(
        &self,
        messages: Vec<Value>,
        tools: Option<Vec<ToolSpec>>,
        options: LLMRequestOptions,
        stream: bool,
    ) -> Value {
        if let Some(mut body) = options.inbound_anthropic_body {
            if let Some(obj) = body.as_object_mut() {
                if stream {
                    obj.insert("stream".to_string(), Value::Bool(true));
                } else {
                    obj.remove("stream");
                }
                obj.entry("model".to_string())
                    .or_insert_with(|| Value::String(self.model.clone()));
                patch_raw_anthropic_transcript(
                    obj,
                    &messages,
                    options.initial_openai_messages.as_deref(),
                );
                if let Some(tool_specs) = tools.as_deref().filter(|tools| !tools.is_empty()) {
                    merge_guarded_tools_into_raw_body(obj, tool_specs);
                }
            }
            return body;
        }

        let (_system, mut body) = convert::build_request_body(
            &self.model,
            &messages,
            self.max_tokens,
            tools.as_deref(),
            self.tool_choice.as_deref(),
        );
        apply_rebuilt_anthropic_passthrough(options.passthrough.as_ref(), &mut body);
        if stream {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("stream".to_string(), Value::Bool(true));
            }
        }
        body
    }
}

fn patch_raw_anthropic_transcript(
    obj: &mut serde_json::Map<String, Value>,
    messages: &[Value],
    initial_openai_messages: Option<&[Value]>,
) {
    let Some(initial) = initial_openai_messages else {
        return;
    };
    if messages.len() >= initial.len() && &messages[..initial.len()] == initial {
        let tail = &messages[initial.len()..];
        if tail.is_empty() || append_anthropic_retry_tail(obj, tail) {
            return;
        }
    }

    let (system, converted_messages) = convert::convert_messages(messages);
    match system {
        Some(system) => {
            obj.insert("system".to_string(), system);
        }
        None => {
            obj.remove("system");
        }
    }
    obj.insert("messages".to_string(), Value::Array(converted_messages));
}

fn append_anthropic_retry_tail(
    obj: &mut serde_json::Map<String, Value>,
    tail_messages: &[Value],
) -> bool {
    let (tail_system, converted_tail) = convert::convert_messages(tail_messages);
    if tail_system.is_some() {
        return false;
    }
    if converted_tail.is_empty() {
        return true;
    }
    match obj.get_mut("messages").and_then(Value::as_array_mut) {
        Some(existing) => existing.extend(converted_tail),
        None => {
            obj.insert("messages".to_string(), Value::Array(converted_tail));
        }
    }
    true
}

fn merge_guarded_tools_into_raw_body(obj: &mut serde_json::Map<String, Value>, tools: &[ToolSpec]) {
    let converted = convert::convert_tools(tools);
    let existing_tools = obj
        .entry("tools".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    let Some(existing_tools) = existing_tools.as_array_mut() else {
        obj.insert("tools".to_string(), Value::Array(converted));
        return;
    };

    for tool in converted {
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        let already_present = existing_tools.iter().any(|existing| {
            existing
                .get("name")
                .and_then(Value::as_str)
                .is_some_and(|existing_name| existing_name == name)
        });
        if !already_present {
            existing_tools.push(tool);
        }
    }
}

fn apply_rebuilt_anthropic_passthrough(
    passthrough: Option<&serde_json::Map<String, Value>>,
    body: &mut Value,
) {
    let Some(passthrough) = passthrough else {
        return;
    };
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    if let Some(model) = passthrough.get("model").and_then(Value::as_str) {
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }
    if let Some(max_tokens) = passthrough
        .get("max_completion_tokens")
        .or_else(|| passthrough.get("max_tokens"))
    {
        obj.insert("max_tokens".to_string(), max_tokens.clone());
    }
    if let Some(stop) = passthrough.get("stop") {
        obj.insert("stop_sequences".to_string(), stop.clone());
    }
    if let Some(tool_choice) = passthrough.get("tool_choice") {
        if let Some(mapped) = openai_tool_choice_to_anthropic(tool_choice) {
            obj.insert("tool_choice".to_string(), mapped);
        }
    }
    if let Some(user) = passthrough.get("user").and_then(Value::as_str) {
        obj.insert("metadata".to_string(), serde_json::json!({"user_id": user}));
    }
}

fn openai_tool_choice_to_anthropic(value: &Value) -> Option<Value> {
    match value {
        Value::String(choice) if choice == "required" => Some(serde_json::json!({"type": "any"})),
        Value::String(choice) if choice == "auto" || choice == "none" => {
            Some(serde_json::json!({"type": choice}))
        }
        Value::Object(obj) => obj
            .get("function")
            .and_then(|func| func.get("name"))
            .and_then(Value::as_str)
            .map(|name| serde_json::json!({"type": "tool", "name": name})),
        _ => None,
    }
}
