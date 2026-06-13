use indexmap::IndexMap;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::core::message::{Message, MessageMeta, MessageRole, MessageType, ToolCallInfo};

/// Error returned when OpenAI-format messages cannot be converted safely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAiMessageError {
    message: String,
}

impl OpenAiMessageError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the validation error message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for OpenAiMessageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OpenAiMessageError {}

/// Convert OpenAI-format chat messages to internal Message objects.
///
/// Handles system, user, assistant (with optional tool_calls), and tool roles.
/// List content blocks are joined with newlines. Null/empty content becomes
/// empty string. Empty tool_calls list is treated as a text response.
pub fn openai_to_messages(input: &[Value]) -> Result<Vec<Message>, OpenAiMessageError> {
    let mut messages = Vec::new();
    let mut pending_tool_call_ids: HashMap<String, VecDeque<String>> = HashMap::new();
    for (message_index, item) in input.iter().enumerate() {
        let content = extract_content(item, message_index)?;
        let role = parse_message_role(item, message_index)?;
        let msg_type = match role {
            MessageRole::System => MessageType::SystemPrompt,
            MessageRole::User => MessageType::UserInput,
            MessageRole::Assistant => MessageType::TextResponse,
            MessageRole::Tool => MessageType::ToolResult,
        };

        let mut msg = Message::new(role, content, MessageMeta::new(msg_type));

        // Handle tool_calls on assistant messages.
        if role == MessageRole::Assistant {
            if let Some(tcs) = item.get("tool_calls").and_then(|t| t.as_array()) {
                if !tcs.is_empty() {
                    let mut infos = Vec::new();
                    let mut seen_call_ids = HashSet::new();
                    for (tool_call_index, tc) in tcs.iter().enumerate() {
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|n| n.as_str())
                            .unwrap_or("");
                        let args_raw = tc.get("function").and_then(|f| f.get("arguments"));
                        let args = parse_args_value(args_raw, message_index, tool_call_index)?;
                        let raw_call_id = tc
                            .get("id")
                            .and_then(|i| i.as_str())
                            .filter(|id| !id.is_empty())
                            .map(str::to_string);
                        let call_id = super::id::unique_or_generated_call_id(
                            raw_call_id.as_deref(),
                            &mut seen_call_ids,
                        );
                        if let Some(raw_call_id) = raw_call_id {
                            pending_tool_call_ids
                                .entry(raw_call_id)
                                .or_default()
                                .push_back(call_id.clone());
                        }
                        infos.push(ToolCallInfo::new(name, Some(args), call_id));
                    }
                    msg = msg.with_tool_calls(infos);
                }
            }
        }

        // Handle tool result fields.
        if role == MessageRole::Tool {
            if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                msg = msg.with_tool_name(name);
            }
            if let Some(id) = item.get("tool_call_id").and_then(|i| i.as_str()) {
                let id = pending_tool_call_ids
                    .get_mut(id)
                    .and_then(VecDeque::pop_front)
                    .unwrap_or_else(|| id.to_string());
                msg = msg.with_tool_call_id(id);
            }
        }

        messages.push(msg);
    }
    Ok(messages)
}

fn parse_message_role(
    item: &Value,
    message_index: usize,
) -> Result<MessageRole, OpenAiMessageError> {
    let Some(role_value) = item.get("role") else {
        return Err(OpenAiMessageError::new(format!(
            "messages[{message_index}].role is required"
        )));
    };
    let Some(role) = role_value.as_str() else {
        return Err(OpenAiMessageError::new(format!(
            "messages[{message_index}].role must be a string"
        )));
    };
    match role {
        "system" => Ok(MessageRole::System),
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "tool" => Ok(MessageRole::Tool),
        _ => Err(OpenAiMessageError::new(format!(
            "messages[{message_index}].role must be one of system, user, assistant, tool"
        ))),
    }
}

/// Extract text content from an OpenAI message without silently dropping parts.
fn extract_content(item: &Value, message_index: usize) -> Result<String, OpenAiMessageError> {
    match item.get("content") {
        None => Ok(String::new()),
        Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut texts = Vec::new();
            for (part_index, part) in parts.iter().enumerate() {
                match part {
                    Value::String(s) => texts.push(s.clone()),
                    Value::Object(obj) => {
                        let part_type = obj.get("type").and_then(Value::as_str).ok_or_else(|| {
                            OpenAiMessageError::new(format!(
                                "messages[{message_index}].content[{part_index}].type must be a string"
                            ))
                        })?;
                        if part_type != "text" {
                            return Err(OpenAiMessageError::new(format!(
                                "messages[{message_index}].content[{part_index}] type '{part_type}' is unsupported; only text content parts are supported"
                            )));
                        }
                        let text = obj.get("text").and_then(Value::as_str).ok_or_else(|| {
                            OpenAiMessageError::new(format!(
                                "messages[{message_index}].content[{part_index}].text must be a string"
                            ))
                        })?;
                        texts.push(text.to_string());
                    }
                    _ => {
                        return Err(OpenAiMessageError::new(format!(
                            "messages[{message_index}].content[{part_index}] must be a string or text content part"
                        )));
                    }
                }
            }
            Ok(texts.join("\n"))
        }
        Some(_) => Err(OpenAiMessageError::new(format!(
            "messages[{message_index}].content must be a string, null, or array of text parts"
        ))),
    }
}

/// Parse arguments from various JSON shapes.
fn parse_args_value(
    args_raw: Option<&Value>,
    message_index: usize,
    tool_call_index: usize,
) -> Result<IndexMap<String, Value>, OpenAiMessageError> {
    match args_raw {
        None => Ok(IndexMap::new()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(obj)) => Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            Ok(_) => Err(OpenAiMessageError::new(format!(
                "messages[{message_index}].tool_calls[{tool_call_index}].function.arguments must decode to a JSON object"
            ))),
            Err(err) => Err(OpenAiMessageError::new(format!(
                "messages[{message_index}].tool_calls[{tool_call_index}].function.arguments must be valid JSON: {err}"
            ))),
        },
        Some(Value::Object(obj)) => Ok(obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        Some(_) => Err(OpenAiMessageError::new(format!(
            "messages[{message_index}].tool_calls[{tool_call_index}].function.arguments must be an object or JSON object string"
        ))),
    }
}
