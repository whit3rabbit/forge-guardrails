use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::usage::{anthropic_usage_details, usage_i64};
use crate::clients::base::{ChunkType, LLMUsageDetails, StreamChunk, TokenUsage};
use crate::error::StreamError;

struct AnthropicToolBlock {
    id: Option<String>,
    name: String,
    args_json: String,
}

#[derive(Default)]
pub(super) struct AnthropicStreamState {
    accumulated_text: String,
    tool_blocks: Vec<AnthropicToolBlock>,
    current_tool_idx: Option<usize>,
    usage_input: i64,
    usage_output: i64,
    usage_cache_creation: Option<i64>,
    usage_cache_read: Option<i64>,
}

pub(super) fn process_anthropic_sse_line(
    line: &str,
    state: &mut AnthropicStreamState,
    last_usage: &Arc<Mutex<Option<TokenUsage>>>,
    last_usage_details: &Arc<Mutex<Option<LLMUsageDetails>>>,
) -> Result<Option<StreamChunk>, StreamError> {
    let Some(data) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let data = data.trim_start();
    if data == "[DONE]" {
        return Ok(None);
    }
    let evt: Value = serde_json::from_str(data)
        .map_err(|err| StreamError::new(format!("Malformed Anthropic SSE data: {err}")))?;

    match evt.get("type").and_then(|t| t.as_str()).unwrap_or("") {
        "content_block_start" => {
            if let Some(block) = evt
                .get("content_block")
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
            {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string);
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                state.tool_blocks.push(AnthropicToolBlock {
                    id,
                    name,
                    args_json: String::new(),
                });
                state.current_tool_idx = Some(state.tool_blocks.len() - 1);
            }
        }
        "content_block_delta" => {
            if let Some(delta) = evt.get("delta") {
                match delta.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            state.accumulated_text.push_str(text);
                            return Ok(Some(
                                StreamChunk::new(ChunkType::TextDelta).with_content(text),
                            ));
                        }
                    }
                    "input_json_delta" => {
                        if let Some(idx) = state.current_tool_idx {
                            if let Some(partial) = delta.get("partial_json").and_then(Value::as_str)
                            {
                                if let Some(block) = state.tool_blocks.get_mut(idx) {
                                    block.args_json.push_str(partial);
                                }
                                return Ok(Some(
                                    StreamChunk::new(ChunkType::ToolCallDelta)
                                        .with_content(partial),
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        "content_block_stop" => {
            state.current_tool_idx = None;
        }
        "message_delta" => {
            if let Some(usage) = evt.get("usage") {
                state.usage_input = usage
                    .get("input_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(state.usage_input);
                state.usage_output = usage
                    .get("output_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(state.usage_output);
                state.usage_cache_creation = usage_i64(Some(usage), "cache_creation_input_tokens")
                    .or(state.usage_cache_creation);
                state.usage_cache_read =
                    usage_i64(Some(usage), "cache_read_input_tokens").or(state.usage_cache_read);
            }
        }
        "message_start" => {
            if let Some(usage) = evt.get("message").and_then(|msg| msg.get("usage")) {
                state.usage_input = usage
                    .get("input_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                state.usage_output = usage
                    .get("output_tokens")
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                state.usage_cache_creation = usage_i64(Some(usage), "cache_creation_input_tokens");
                state.usage_cache_read = usage_i64(Some(usage), "cache_read_input_tokens");
            }
        }
        "message_stop" => {
            let prompt_total = state.usage_input
                + state.usage_cache_creation.unwrap_or(0)
                + state.usage_cache_read.unwrap_or(0);
            if let Ok(mut guard) = last_usage.lock() {
                *guard = Some(TokenUsage::new(
                    prompt_total,
                    state.usage_output,
                    prompt_total + state.usage_output,
                ));
            }
            if let Ok(mut guard) = last_usage_details.lock() {
                *guard =
                    anthropic_usage_details(state.usage_cache_creation, state.usage_cache_read);
            }
            let final_resp = if !state.tool_blocks.is_empty() {
                let reasoning = if state.accumulated_text.is_empty() {
                    None
                } else {
                    Some(state.accumulated_text.clone())
                };
                let calls: Vec<crate::clients::base::ToolCall> = state
                    .tool_blocks
                    .iter()
                    .enumerate()
                    .map(|(i, block)| {
                        let args = serde_json::from_str::<Value>(&block.args_json)
                            .ok()
                            .and_then(|v| v.as_object().cloned())
                            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            .unwrap_or_default();
                        let mut call =
                            crate::clients::base::ToolCall::new(block.name.clone(), args);
                        if let Some(id) = block.id.as_ref() {
                            call = call.with_id(id.clone());
                        }
                        if i == 0 {
                            if let Some(ref r) = reasoning {
                                call = call.with_reasoning(r);
                            }
                        }
                        call
                    })
                    .collect();
                crate::clients::base::LLMResponse::ToolCalls(calls)
            } else {
                crate::clients::base::LLMResponse::Text(crate::clients::base::TextResponse::new(
                    &state.accumulated_text,
                ))
            };
            return Ok(Some(
                StreamChunk::new(ChunkType::Final).with_response(final_resp),
            ));
        }
        _ => {}
    }

    Ok(None)
}
