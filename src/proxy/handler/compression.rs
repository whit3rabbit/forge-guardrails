use crate::core::message::{Message, MessageRole, MessageType, ToolCallInfo};
use crate::tool_output::{
    compress_tool_output, ToolOutputCompressionConfig, ToolOutputCompressionState,
};
use indexmap::IndexMap;
use serde_json::Value;

/// An update to a tool call output due to compression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutputCompressionUpdate {
    pub tool_call_id: Option<String>,
    pub output: String,
}

/// Compresses the tool result message content in place using the given config and state.
pub fn compress_proxy_tool_results(
    messages: &mut [Message],
    config: &ToolOutputCompressionConfig,
    state: Option<&ToolOutputCompressionState>,
) -> Vec<ToolOutputCompressionUpdate> {
    if !config.enabled() {
        return Vec::new();
    }

    let mut pending_tool_calls: IndexMap<String, ToolCallInfo> = IndexMap::new();
    let mut updates = Vec::new();
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
                if message.metadata.msg_type != MessageType::ToolResult {
                    continue;
                }
                let call = message
                    .tool_call_id
                    .as_deref()
                    .and_then(|call_id| pending_tool_calls.get(call_id));
                let tool_name = call
                    .map(|call| call.name.as_str())
                    .or(message.tool_name.as_deref())
                    .unwrap_or("generic");
                let args = call.and_then(|call| call.args.as_ref());
                let result = compress_tool_output(tool_name, args, &message.content, config, state);
                if result.output != message.content {
                    tracing::info!(
                        target: "forge.tool_output",
                        tool = %result.canonical_tool,
                        family = %result.family,
                        mode = %result.mode,
                        before_tokens = result.before_tokens,
                        after_tokens = result.after_tokens,
                        saved_tokens = result.saved_tokens,
                        saved_pct = result.saved_pct,
                        redacted = result.redacted,
                        capped = result.capped,
                        deduped = result.deduped,
                        "compressed proxy tool output"
                    );
                    updates.push(ToolOutputCompressionUpdate {
                        tool_call_id: message.tool_call_id.clone(),
                        output: result.output.clone(),
                    });
                    message.content = result.output;
                }
            }
            _ => {}
        }
    }
    updates
}

/// Patches the Anthropic request JSON body with the compressed tool outputs.
pub fn patch_anthropic_tool_results(
    body: &mut Value,
    updates: &[ToolOutputCompressionUpdate],
) -> bool {
    let mut pending = IndexMap::new();
    for update in updates {
        let Some(tool_call_id) = update
            .tool_call_id
            .as_deref()
            .filter(|tool_call_id| !tool_call_id.is_empty())
        else {
            return false;
        };
        pending.insert(tool_call_id.to_string(), update.output.clone());
    }

    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };
    for message in messages {
        let Some(content) = message.get_mut("content") else {
            continue;
        };
        if !patch_anthropic_content_blocks(content, &mut pending) {
            return false;
        }
    }

    pending.is_empty()
}

fn patch_anthropic_content_blocks(
    content: &mut Value,
    pending: &mut IndexMap<String, String>,
) -> bool {
    let Value::Array(blocks) = content else {
        return true;
    };

    for block in blocks {
        let Some(obj) = block.as_object_mut() else {
            continue;
        };
        if obj.get("type").and_then(Value::as_str) != Some("tool_result") {
            continue;
        }
        let Some(tool_use_id) = obj
            .get("tool_use_id")
            .and_then(Value::as_str)
            .map(str::to_string)
        else {
            continue;
        };
        let Some(output) = pending.get(&tool_use_id).cloned() else {
            continue;
        };
        if !patch_anthropic_tool_result_content(obj, &output) {
            return false;
        }
        pending.shift_remove(&tool_use_id);
    }

    true
}

fn patch_anthropic_tool_result_content(
    obj: &mut serde_json::Map<String, Value>,
    output: &str,
) -> bool {
    let output = raw_anthropic_tool_result_output(obj, output);
    match obj.get_mut("content") {
        Some(Value::String(content)) => {
            *content = output;
            true
        }
        Some(Value::Null) | None => {
            obj.insert("content".to_string(), Value::String(output));
            true
        }
        Some(Value::Array(blocks)) => patch_single_anthropic_tool_result_text_block(blocks, output),
        _ => false,
    }
}

fn raw_anthropic_tool_result_output(obj: &serde_json::Map<String, Value>, output: &str) -> String {
    if obj.get("is_error").and_then(Value::as_bool) == Some(true) {
        output.strip_prefix("Error: ").unwrap_or(output).to_string()
    } else {
        output.to_string()
    }
}

fn patch_single_anthropic_tool_result_text_block(blocks: &mut [Value], output: String) -> bool {
    let [block] = blocks else {
        return false;
    };
    let Some(obj) = block.as_object_mut() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) != Some("text") {
        return false;
    }
    if !obj.get("text").is_some_and(Value::is_string) {
        return false;
    }
    obj.insert("text".to_string(), Value::String(output));
    true
}
