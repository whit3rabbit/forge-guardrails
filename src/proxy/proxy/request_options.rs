use serde_json::{Map, Value};

use crate::clients::base::ToolCall;
use crate::tools::respond::RESPOND_TOOL_NAME;

/// Build a respond ToolSpec in OpenAI wire format for injection.
pub fn respond_tool_openai() -> Value {
    let spec = crate::tools::respond::respond_spec();
    crate::clients::base::format_tool(&spec)
}

/// Check if a list of tool specs already contains the respond tool.
pub fn has_respond_tool(tools: &[Value]) -> bool {
    tools.iter().any(|t| {
        t.get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            == Some(RESPOND_TOOL_NAME)
    })
}

/// Extract per-request sampling parameters from a request body.
/// Returns a Map of recognized fields, or None if none found.
pub fn extract_sampling(body: &Value) -> Option<Map<String, Value>> {
    let mut map = Map::new();
    let recognized = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "seed",
        "chat_template_kwargs",
    ];

    for key in &recognized {
        if let Some(v) = body.get(key) {
            map.insert((*key).to_string(), v.clone());
        }
    }

    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Extract non-forge-owned request fields for client passthrough.
pub fn extract_passthrough(body: &Value) -> Option<Map<String, Value>> {
    let obj = body.as_object()?;
    let mut map = Map::new();
    let forge_owned = ["messages", "tools", "stream", "system", "_forge"];
    let sampling_fields = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "seed",
        "chat_template_kwargs",
    ];

    for (key, value) in obj {
        if forge_owned.contains(&key.as_str()) || sampling_fields.contains(&key.as_str()) {
            continue;
        }
        map.insert(key.clone(), value.clone());
    }

    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

/// Strip respond() tool calls from a response, returning the remaining
/// tool calls and/or extracted respond text.
pub fn strip_respond_calls(calls: &[ToolCall]) -> (Vec<ToolCall>, Option<String>) {
    let mut respond_text = None;
    let mut real_calls = Vec::new();

    for tc in calls {
        if tc.tool == RESPOND_TOOL_NAME {
            let msg = tc
                .args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if respond_text.is_none() {
                respond_text = Some(msg.to_string());
            }
        } else {
            real_calls.push(tc.clone());
        }
    }

    (real_calls, respond_text)
}
