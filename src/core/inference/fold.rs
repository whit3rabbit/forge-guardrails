use crate::core::message::{Message, MessageMeta, MessageRole, MessageType};
use serde_json::Value;

/// Convert internal messages to API-format values, folding reasoning into
/// the following tool-call message's content field.
///
/// Reasoning messages immediately preceding a tool-call message are merged
/// into the tool-call message's content. Orphaned reasoning (no following
/// tool-call) is emitted as a standalone assistant message.
pub fn fold_and_serialize(messages: &[Message], api_format: &str) -> Vec<Value> {
    let mut result = Vec::new();
    let mut pending_reasoning: Option<String> = None;

    for msg in messages {
        if msg.metadata.msg_type == MessageType::Reasoning {
            pending_reasoning = Some(msg.content.clone());
            continue;
        }

        if let Some(reasoning_content) = pending_reasoning.take() {
            if msg.metadata.msg_type == MessageType::ToolCall {
                let mut merged = msg.clone();
                merged.content = reasoning_content;
                result.push(merged.serialize(api_format));
            } else {
                let orphan = Message::new(
                    MessageRole::Assistant,
                    &reasoning_content,
                    MessageMeta::new(MessageType::Reasoning),
                );
                result.push(orphan.serialize(api_format));
                result.push(msg.serialize(api_format));
            }
            continue;
        }

        result.push(msg.serialize(api_format));
    }

    if let Some(reasoning_content) = pending_reasoning {
        let orphan = Message::new(
            MessageRole::Assistant,
            &reasoning_content,
            MessageMeta::new(MessageType::Reasoning),
        );
        result.push(orphan.serialize(api_format));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::message::ToolCallInfo;
    use indexmap::IndexMap;

    #[test]
    fn fold_and_serialize_basic_message() {
        let msg = Message::new(
            MessageRole::User,
            "hello",
            MessageMeta::new(MessageType::UserInput),
        );
        let result = fold_and_serialize(&[msg], "openai");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["role"], "user");
        assert_eq!(result[0]["content"], "hello");
    }

    #[test]
    fn fold_and_serialize_reasoning_folded_into_tool_call() {
        let reasoning = Message::new(
            MessageRole::Assistant,
            "thinking...",
            MessageMeta::new(MessageType::Reasoning),
        );
        let tool_call = Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "search",
            Some(IndexMap::new()),
            "tc_0001",
        )]);
        let result = fold_and_serialize(&[reasoning, tool_call], "openai");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["content"], "thinking...");
        assert!(result[0]["tool_calls"].is_array());
    }

    #[test]
    fn fold_and_serialize_orphaned_reasoning() {
        let reasoning = Message::new(
            MessageRole::Assistant,
            "thinking...",
            MessageMeta::new(MessageType::Reasoning),
        );
        let user = Message::new(
            MessageRole::User,
            "hello",
            MessageMeta::new(MessageType::UserInput),
        );
        let result = fold_and_serialize(&[reasoning, user], "openai");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0]["content"], "thinking...");
        assert_eq!(result[1]["content"], "hello");
    }

    #[test]
    fn fold_and_serialize_text_not_folded() {
        let text = Message::new(
            MessageRole::Assistant,
            "some text",
            MessageMeta::new(MessageType::TextResponse),
        );
        let tool_call = Message::new(
            MessageRole::Assistant,
            "",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "search",
            Some(IndexMap::new()),
            "tc_0001",
        )]);
        let result = fold_and_serialize(&[text, tool_call], "openai");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn fold_and_serialize_reasoning_with_tool_call_content() {
        let reasoning = Message::new(
            MessageRole::Assistant,
            "let me think",
            MessageMeta::new(MessageType::Reasoning),
        );
        let tool_call = Message::new(
            MessageRole::Assistant,
            "existing content",
            MessageMeta::new(MessageType::ToolCall),
        )
        .with_tool_calls(vec![ToolCallInfo::new(
            "search",
            Some(IndexMap::new()),
            "tc_0001",
        )]);
        let result = fold_and_serialize(&[reasoning, tool_call], "openai");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["content"], "let me think");
    }
}
