use serde_json::{json, Map, Value};

use crate::clients::base::{LLMUsageDetails, TextResponse, TokenUsage, ToolCall};

use super::id::{generate_call_id, generate_completion_id};

pub(crate) fn text_delta_sse_event(
    completion_id: &str,
    model: &str,
    content: &str,
    include_role: bool,
    usage: Option<&Value>,
) -> Value {
    let mut delta = Map::new();
    if include_role {
        delta.insert("role".into(), json!("assistant"));
    }
    delta.insert("content".into(), json!(content));

    let mut event = json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": Value::Object(delta),
            "finish_reason": Value::Null
        }]
    });
    if let Some(usage_value) = usage {
        event["usage"] = usage_value.clone();
    }
    event
}

pub(crate) fn final_sse_event(
    completion_id: &str,
    model: &str,
    finish_reason: &str,
    usage: Option<&Value>,
) -> Value {
    let mut event = json!({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }]
    });
    if let Some(usage_value) = usage {
        event["usage"] = usage_value.clone();
    }
    event
}

/// Convert internal tool calls to OpenAI chat completion response format.
///
/// Includes reasoning text as message content (or None if no reasoning).
/// Generates unique call and completion IDs.
pub fn tool_calls_to_openai(calls: &[ToolCall], model: &str) -> Value {
    tool_calls_to_openai_with_usage(calls, model, None)
}

/// Convert internal tool calls to OpenAI chat completion response format with usage.
pub fn tool_calls_to_openai_with_usage(
    calls: &[ToolCall],
    model: &str,
    usage: Option<&TokenUsage>,
) -> Value {
    tool_calls_to_openai_with_usage_details(calls, model, usage, None)
}

pub(crate) fn tool_calls_to_openai_with_usage_details(
    calls: &[ToolCall],
    model: &str,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> Value {
    let completion_id = generate_completion_id();
    let mut tool_calls_out = Vec::new();

    let reasoning = calls.first().and_then(|c| c.reasoning.clone());
    let content: Option<String> = reasoning;

    for (i, tc) in calls.iter().enumerate() {
        let call_id = tc
            .id
            .as_deref()
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(generate_call_id);
        let args_json = serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());
        let mut func_map = Map::new();
        func_map.insert("name".into(), json!(tc.tool));
        func_map.insert("arguments".into(), json!(args_json));

        let mut tc_map = Map::new();
        tc_map.insert("index".into(), json!(i));
        tc_map.insert("id".into(), json!(call_id));
        tc_map.insert("type".into(), json!("function"));
        tc_map.insert("function".into(), Value::Object(func_map));
        tool_calls_out.push(Value::Object(tc_map));
    }

    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert(
        "content".into(),
        match content {
            Some(c) => json!(c),
            None => Value::Null,
        },
    );
    message.insert("tool_calls".into(), json!(tool_calls_out));

    let mut choice = Map::new();
    choice.insert("index".into(), json!(0));
    choice.insert("message".into(), Value::Object(message));
    choice.insert("finish_reason".into(), json!("tool_calls"));

    json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [Value::Object(choice)],
        "usage": usage_to_openai_json_with_details(usage, usage_details)
    })
}

/// Convert internal TextResponse to OpenAI chat completion response format.
pub fn text_response_to_openai(response: &TextResponse, model: &str) -> Value {
    text_response_to_openai_with_usage(response, model, None)
}

/// Convert internal TextResponse to OpenAI chat completion response format with usage.
pub fn text_response_to_openai_with_usage(
    response: &TextResponse,
    model: &str,
    usage: Option<&TokenUsage>,
) -> Value {
    text_response_to_openai_with_usage_details(response, model, usage, None)
}

pub(crate) fn text_response_to_openai_with_usage_details(
    response: &TextResponse,
    model: &str,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> Value {
    let completion_id = generate_completion_id();

    json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.content},
            "finish_reason": "stop"
        }],
        "usage": usage_to_openai_json_with_details(usage, usage_details)
    })
}

/// Convert tool calls to SSE event chunks for streaming.
///
/// Events: reasoning as content delta first, then tool call deltas with
/// indexed entries, then a final chunk with finish_reason=tool_calls.
/// All events share the same completion ID.
pub fn tool_calls_to_sse_events(calls: &[ToolCall], model: &str) -> Vec<Value> {
    tool_calls_to_sse_events_with_usage(calls, model, None)
}

/// Convert tool calls to SSE event chunks with usage.
pub fn tool_calls_to_sse_events_with_usage(
    calls: &[ToolCall],
    model: &str,
    usage: Option<&TokenUsage>,
) -> Vec<Value> {
    tool_calls_to_sse_event_iter_with_usage(calls, model, usage).collect()
}

/// Convert text to SSE event chunks for streaming.
///
/// First chunk includes role=assistant. Text is split by chunk_size if >0.
/// Final chunk has finish_reason=stop. All events share the same completion ID.
pub fn text_to_sse_events(text: &str, model: &str, chunk_size: usize) -> Vec<Value> {
    text_to_sse_events_with_usage(text, model, chunk_size, None)
}

/// Convert text to SSE event chunks with usage.
pub fn text_to_sse_events_with_usage(
    text: &str,
    model: &str,
    chunk_size: usize,
    usage: Option<&TokenUsage>,
) -> Vec<Value> {
    text_to_sse_event_iter_with_usage(text, model, chunk_size, usage).collect()
}

pub(crate) fn text_to_sse_event_iter_with_usage<'a>(
    text: &'a str,
    model: &'a str,
    chunk_size: usize,
    usage: Option<&'a TokenUsage>,
) -> impl Iterator<Item = Value> + 'a {
    text_to_sse_event_iter_with_usage_details(text, model, chunk_size, usage, None)
}

pub(crate) fn text_to_sse_event_iter_with_usage_details<'a>(
    text: &'a str,
    model: &'a str,
    chunk_size: usize,
    usage: Option<&'a TokenUsage>,
    usage_details: Option<&'a LLMUsageDetails>,
) -> impl Iterator<Item = Value> + 'a {
    let completion_id = generate_completion_id();
    let usage_json = usage.map(|u| usage_to_openai_json_with_details(Some(u), usage_details));
    let mut chunks = text_chunks(text, chunk_size);
    let mut emitted_content = false;
    let mut emitted_final = false;

    std::iter::from_fn(move || {
        if let Some(chunk) = chunks.next() {
            let include_role = !emitted_content;
            emitted_content = true;
            return Some(text_delta_sse_event(
                &completion_id,
                model,
                chunk,
                include_role,
                None,
            ));
        }

        if emitted_final {
            return None;
        }
        emitted_final = true;
        Some(final_sse_event(
            &completion_id,
            model,
            "stop",
            usage_json.as_ref(),
        ))
    })
}

pub(crate) fn tool_calls_to_sse_event_iter_with_usage<'a>(
    calls: &'a [ToolCall],
    model: &'a str,
    usage: Option<&'a TokenUsage>,
) -> impl Iterator<Item = Value> + 'a {
    tool_calls_to_sse_event_iter_with_usage_details(calls, model, usage, None)
}

pub(crate) fn tool_calls_to_sse_event_iter_with_usage_details<'a>(
    calls: &'a [ToolCall],
    model: &'a str,
    usage: Option<&'a TokenUsage>,
    usage_details: Option<&'a LLMUsageDetails>,
) -> impl Iterator<Item = Value> + 'a {
    let completion_id = generate_completion_id();
    let usage_json = usage.map(|u| usage_to_openai_json_with_details(Some(u), usage_details));
    let reasoning = calls.first().and_then(|c| c.reasoning.clone());
    let mut step = 0;

    std::iter::from_fn(move || {
        if step == 0 {
            step = 1;
            if let Some(ref r) = reasoning {
                return Some(text_delta_sse_event(&completion_id, model, r, true, None));
            }
        }

        if step == 1 {
            step = 2;
            let mut tc_deltas = Vec::new();
            for (i, tc) in calls.iter().enumerate() {
                let call_id = tc
                    .id
                    .as_deref()
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(generate_call_id);
                let args_json =
                    serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string());
                let mut func_map = Map::new();
                func_map.insert("name".into(), json!(tc.tool));
                func_map.insert("arguments".into(), json!(args_json));

                let mut tc_map = Map::new();
                tc_map.insert("index".into(), json!(i));
                tc_map.insert("id".into(), json!(call_id));
                tc_map.insert("type".into(), json!("function"));
                tc_map.insert("function".into(), Value::Object(func_map));
                tc_deltas.push(Value::Object(tc_map));
            }

            let mut delta = Map::new();
            if reasoning.is_none() {
                delta.insert("role".into(), json!("assistant"));
            }
            delta.insert("tool_calls".into(), json!(tc_deltas));

            let event = json!({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": Value::Object(delta),
                    "finish_reason": Value::Null
                }]
            });
            return Some(event);
        }

        if step == 2 {
            step = 3;
            return Some(final_sse_event(
                &completion_id,
                model,
                "tool_calls",
                usage_json.as_ref(),
            ));
        }

        None
    })
}

struct TextChunks<'a> {
    text: &'a str,
    chunk_size: usize,
    offset: usize,
    emitted_empty: bool,
    emitted_full: bool,
}

fn text_chunks(text: &str, chunk_size: usize) -> TextChunks<'_> {
    TextChunks {
        text,
        chunk_size,
        offset: 0,
        emitted_empty: false,
        emitted_full: false,
    }
}

impl<'a> Iterator for TextChunks<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chunk_size == 0 {
            if self.emitted_full {
                return None;
            }
            self.emitted_full = true;
            return Some(self.text);
        }

        if self.text.is_empty() {
            if self.emitted_empty {
                return None;
            }
            self.emitted_empty = true;
            return Some("");
        }

        if self.offset >= self.text.len() {
            return None;
        }

        let start = self.offset;
        let mut end = self.text.len();
        for (count, (idx, _)) in self.text[start..].char_indices().enumerate() {
            if count == self.chunk_size {
                end = start + idx;
                break;
            }
        }
        self.offset = end;
        Some(&self.text[start..end])
    }
}

pub(crate) fn usage_to_openai_json_with_details(
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> Value {
    let usage = usage.cloned().unwrap_or_else(TokenUsage::empty);
    let mut value = json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    });

    if let Some(details) = usage_details {
        if let Some(cached) = details.cached_prompt_tokens {
            value["prompt_tokens_details"] = json!({"cached_tokens": cached});
        }
        if let Some(hit) = details.prompt_cache_hit_tokens {
            value["prompt_cache_hit_tokens"] = json!(hit);
        }
        if let Some(miss) = details.prompt_cache_miss_tokens {
            value["prompt_cache_miss_tokens"] = json!(miss);
        }
        if let Some(thinking) = details.anthropic_thinking_output_tokens {
            value["completion_tokens_details"] = json!({"reasoning_tokens": thinking});
        }
    }

    value
}
