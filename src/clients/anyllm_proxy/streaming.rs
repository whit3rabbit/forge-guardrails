use std::sync::{Arc, Mutex};

use futures_util::StreamExt;

use super::call_info::{observe_stream_call_info, observe_stream_call_info_value};
use super::response::final_stream_response;
use super::usage::{
    record_usage_cell, record_usage_details_cell, token_usage_from_openai_usage,
    usage_details_from_openai_usage,
};
use crate::clients::base::{ChunkType, LLMCallInfo, LLMUsageDetails, StreamChunk, TokenUsage};
use crate::clients::openai_compat;
use crate::error::StreamError;

pub(super) const MAX_STREAM_TOOL_CALLS: usize = 128;

pub(super) fn checked_stream_tool_call_index(
    index: u32,
    accumulated_len: usize,
) -> Result<usize, StreamError> {
    let Ok(index) = usize::try_from(index) else {
        return Err(StreamError::new("tool call index cannot fit in usize"));
    };
    if index >= MAX_STREAM_TOOL_CALLS {
        return Err(StreamError::new(format!(
            "tool call index {} exceeds limit {}",
            index, MAX_STREAM_TOOL_CALLS
        )));
    }
    if index > accumulated_len {
        return Err(StreamError::new(format!(
            "non-contiguous tool call index {} in stream chunk",
            index
        )));
    }
    Ok(index)
}

pub(super) fn parse_openai_sse(
    resp: reqwest::Response,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_usage_details: Arc<Mutex<Option<LLMUsageDetails>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
    initial_call_info: Option<LLMCallInfo>,
    default_cost_model: Option<String>,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    let byte_stream = resp.bytes_stream();
    async_stream::stream! {
        let mut inner = Box::pin(byte_stream);
        let mut line_buf = String::new();
        let mut accumulated_text = String::new();
        let mut accumulated_reasoning = String::new();
        let mut accumulated_tools: Vec<(String, String, String)> = Vec::new();
        let mut stream_usage = None;
        let mut stream_usage_details = None;
        let mut stream_call_info = initial_call_info;

        loop {
            let raw = if let Some(raw) = take_sse_line(&mut line_buf) {
                raw
            } else {
                match inner.next().await {
                    Some(Ok(bytes)) => {
                        line_buf.push_str(&String::from_utf8_lossy(&bytes));
                        continue;
                    }
                    Some(Err(e)) => {
                        yield Err(StreamError::new(e.to_string()));
                        return;
                    }
                    None if line_buf.is_empty() => break,
                    None => {
                        let raw = std::mem::take(&mut line_buf);
                        raw.trim_end_matches('\r').to_string()
                    }
                }
            };

            let Some(data) = sse_data_value(&raw) else {
                continue;
            };
            if data == "[DONE]" {
                let final_response = final_stream_response(
                    &accumulated_text,
                    &accumulated_reasoning,
                    &accumulated_tools,
                );
                yield Ok(StreamChunk::new(ChunkType::Final)
                    .with_response(final_response)
                    .with_metadata(
                        stream_usage.clone(),
                        stream_usage_details.clone(),
                        stream_call_info.clone(),
                    ));
                return;
            }

            let mut evt_value: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            openai_compat::normalize_openai_response_tool_calls(&mut evt_value);
            let evt: anyllm_translate::openai::ChatCompletionChunk = match serde_json::from_value(evt_value) {
                Ok(v) => v,
                Err(_) => continue,
            };

            if evt.usage.is_some() {
                stream_usage = Some(token_usage_from_openai_usage(evt.usage.as_ref()));
                stream_usage_details = usage_details_from_openai_usage(evt.usage.as_ref());
            }
            record_usage_cell(&last_usage, evt.usage.as_ref());
            record_usage_details_cell(&last_usage_details, stream_usage_details.clone());
            let cost_model = default_cost_model.as_deref().unwrap_or(&evt.model);
            observe_stream_call_info_value(
                &mut stream_call_info,
                &evt.model,
                cost_model,
                evt.usage.as_ref(),
            );
            observe_stream_call_info(
                &last_call_info,
                &evt.model,
                cost_model,
                evt.usage.as_ref(),
            );

            for choice in evt.choices {
                if let Some(reasoning) = choice.delta.reasoning_content {
                    accumulated_reasoning.push_str(&reasoning);
                }
                if let Some(content) = choice.delta.content {
                    accumulated_text.push_str(&content);
                    yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(content));
                }
                if let Some(tool_calls) = choice.delta.tool_calls {
                    for tc in tool_calls {
                        let index = match checked_stream_tool_call_index(
                            tc.index,
                            accumulated_tools.len(),
                        ) {
                            Ok(index) => index,
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        };
                        if index == accumulated_tools.len() {
                            accumulated_tools.push((String::new(), String::new(), String::new()));
                        }
                        if let Some(id) = tc.id {
                            if !id.is_empty() {
                                accumulated_tools[index].0 = id;
                            }
                        }
                        if let Some(function) = tc.function {
                            if let Some(name) = function.name {
                                if !name.is_empty() {
                                    accumulated_tools[index].1 = name;
                                }
                            }
                            if let Some(args) = function.arguments {
                                accumulated_tools[index].2.push_str(&args);
                            }
                        }
                    }
                }
            }
        }

        let final_response = final_stream_response(
            &accumulated_text,
            &accumulated_reasoning,
            &accumulated_tools,
        );
        yield Ok(StreamChunk::new(ChunkType::Final)
            .with_response(final_response)
            .with_metadata(
                stream_usage.clone(),
                stream_usage_details.clone(),
                stream_call_info.clone(),
            ));
    }
}

pub(super) fn take_sse_line(line_buf: &mut String) -> Option<String> {
    let newline_pos = line_buf.find('\n')?;
    let mut raw: String = line_buf.drain(..=newline_pos).collect();
    raw.pop();
    while raw.ends_with('\r') {
        raw.pop();
    }
    Some(raw)
}

pub(super) fn sse_data_value(raw: &str) -> Option<&str> {
    let value = raw.strip_prefix("data:")?;
    Some(value.trim_start())
}

pub(super) fn parse_openai_chunks(
    chunks: ::anyllm_proxy::runtime::ChatCompletionChunkStream,
    last_usage: Arc<Mutex<Option<TokenUsage>>>,
    last_usage_details: Arc<Mutex<Option<LLMUsageDetails>>>,
    last_call_info: Arc<Mutex<Option<LLMCallInfo>>>,
    initial_call_info: Option<LLMCallInfo>,
    cost_model: Option<String>,
) -> impl futures_core::Stream<Item = Result<StreamChunk, StreamError>> + Send {
    async_stream::stream! {
        let mut inner = chunks;
        let mut accumulated_text = String::new();
        let mut accumulated_reasoning = String::new();
        let mut accumulated_tools: Vec<(String, String, String)> = Vec::new();
        let mut stream_usage = None;
        let mut stream_usage_details = None;
        let mut stream_call_info = initial_call_info;

        while let Some(chunk) = inner.next().await {
            let evt = match chunk {
                Ok(evt) => evt,
                Err(e) => {
                    yield Err(StreamError::new(e.to_string()));
                    return;
                }
            };

            if evt.usage.is_some() {
                stream_usage = Some(token_usage_from_openai_usage(evt.usage.as_ref()));
                stream_usage_details = usage_details_from_openai_usage(evt.usage.as_ref());
            }
            record_usage_cell(&last_usage, evt.usage.as_ref());
            record_usage_details_cell(&last_usage_details, stream_usage_details.clone());
            let pricing_model = cost_model.as_deref().unwrap_or(&evt.model);
            observe_stream_call_info_value(
                &mut stream_call_info,
                &evt.model,
                pricing_model,
                evt.usage.as_ref(),
            );
            observe_stream_call_info(
                &last_call_info,
                &evt.model,
                pricing_model,
                evt.usage.as_ref(),
            );

            for choice in evt.choices {
                if let Some(reasoning) = choice.delta.reasoning_content {
                    accumulated_reasoning.push_str(&reasoning);
                }
                if let Some(content) = choice.delta.content {
                    accumulated_text.push_str(&content);
                    yield Ok(StreamChunk::new(ChunkType::TextDelta).with_content(content));
                }
                if let Some(tool_calls) = choice.delta.tool_calls {
                    for tc in tool_calls {
                        let index = match checked_stream_tool_call_index(
                            tc.index,
                            accumulated_tools.len(),
                        ) {
                            Ok(index) => index,
                            Err(e) => {
                                yield Err(e);
                                return;
                            }
                        };
                        if index == accumulated_tools.len() {
                            accumulated_tools.push((String::new(), String::new(), String::new()));
                        }
                        if let Some(id) = tc.id {
                            if !id.is_empty() {
                                accumulated_tools[index].0 = id;
                            }
                        }
                        if let Some(function) = tc.function {
                            if let Some(name) = function.name {
                                if !name.is_empty() {
                                    accumulated_tools[index].1 = name;
                                }
                            }
                            if let Some(args) = function.arguments {
                                accumulated_tools[index].2.push_str(&args);
                            }
                        }
                    }
                }
            }
        }

        let final_response = final_stream_response(
            &accumulated_text,
            &accumulated_reasoning,
            &accumulated_tools,
        );
        yield Ok(StreamChunk::new(ChunkType::Final)
            .with_response(final_response)
            .with_metadata(
                stream_usage.clone(),
                stream_usage_details.clone(),
                stream_call_info.clone(),
            ));
    }
}
