use indexmap::IndexMap;
use serde_json::Value;

use crate::clients::base::{LLMResponse, TextResponse, ToolCall};

pub(super) fn parse_openai_response(
    response: anyllm_translate::openai::ChatCompletionResponse,
) -> LLMResponse {
    let Some(choice) = response.choices.into_iter().next() else {
        return LLMResponse::Text(TextResponse::new(""));
    };

    let message = choice.message;
    if let Some(tool_calls) = message.tool_calls {
        if !tool_calls.is_empty() {
            let reasoning = message.reasoning_content.clone();
            let calls = tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, tc)| {
                    let mut call =
                        ToolCall::new(tc.function.name, parse_args_string(&tc.function.arguments))
                            .with_id(tc.id);
                    if index == 0 {
                        if let Some(ref text) = reasoning {
                            call = call.with_reasoning(text);
                        }
                    }
                    call
                })
                .collect();
            return LLMResponse::ToolCalls(calls);
        }
    }

    LLMResponse::Text(TextResponse::new(content_to_string(message.content)))
}

fn content_to_string(content: Option<anyllm_translate::openai::ChatContent>) -> String {
    match content {
        Some(anyllm_translate::openai::ChatContent::Text(text)) => text,
        Some(anyllm_translate::openai::ChatContent::Parts(parts)) => parts
            .into_iter()
            .filter_map(|part| match part {
                anyllm_translate::openai::ChatContentPart::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    }
}

pub(super) fn parse_args_string(args: &str) -> IndexMap<String, Value> {
    match serde_json::from_str::<Value>(args) {
        Ok(Value::Object(obj)) => obj.into_iter().collect(),
        _ => IndexMap::new(),
    }
}

pub(super) fn final_stream_response(
    accumulated_text: &str,
    accumulated_reasoning: &str,
    accumulated_tools: &[(String, String, String)],
) -> LLMResponse {
    let calls: Vec<ToolCall> = accumulated_tools
        .iter()
        .filter(|(_, name, _)| !name.is_empty())
        .enumerate()
        .map(|(index, (id, name, args))| {
            let mut call = ToolCall::new(name.clone(), parse_args_string(args)).with_id(id.clone());
            if index == 0 && !accumulated_reasoning.is_empty() {
                call = call.with_reasoning(accumulated_reasoning.to_string());
            }
            call
        })
        .collect();

    if calls.is_empty() {
        LLMResponse::Text(TextResponse::new(accumulated_text.to_string()))
    } else {
        LLMResponse::ToolCalls(calls)
    }
}
