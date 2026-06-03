use std::sync::Arc;

use futures_util::StreamExt;
use serde_json::Value;

use crate::clients::base::{ChunkType, LLMClient, LLMRequestOptions, LLMResponse};
use crate::core::tool_spec::ToolSpec;
use crate::error::StreamError;

use super::response_shape::text_response_result;
use super::HandlerResult;

fn openai_event_to_anthropic_events(
    event: Value,
    translator: &mut anyllm_translate::mapping::streaming_map::StreamingTranslator,
) -> Result<Vec<anyllm_translate::anthropic::streaming::StreamEvent>, StreamError> {
    let chunk: anyllm_translate::openai::ChatCompletionChunk =
        serde_json::from_value(event).map_err(|err| StreamError::new(err.to_string()))?;
    Ok(translator.process_chunk(&chunk))
}

/// Forward requests to LLM Client without tools or guardrails.
pub async fn run_passthrough<C: LLMClient + 'static>(
    client: &Arc<C>,
    serialized: &[Value],
    _tools: Option<Vec<ToolSpec>>,
    options: LLMRequestOptions,
    model_name: &str,
    stream: bool,
    stream_include_usage: bool,
) -> Result<HandlerResult, String> {
    if stream {
        if options.preserve_provider_response {
            return run_passthrough_stream_preserving_anthropic(
                client,
                serialized,
                options,
                model_name,
                stream_include_usage,
            )
            .await;
        }
        return run_passthrough_stream(
            client,
            serialized,
            options,
            model_name,
            stream_include_usage,
        )
        .await;
    }

    let envelope = client
        .send_envelope_with_options(serialized.to_vec(), None, options)
        .await
        .map_err(|e| e.to_string())?;
    let usage = envelope.usage;
    let usage_details = envelope.usage_details;
    let provider_response = envelope.provider_response;

    match envelope.response {
        LLMResponse::Text(text) => {
            if let Some(value) = provider_response {
                Ok(HandlerResult::AnthropicResponse(value))
            } else {
                Ok(text_response_result(
                    &text,
                    model_name,
                    stream,
                    stream_include_usage,
                    usage.as_ref(),
                    usage_details.as_ref(),
                ))
            }
        }
        LLMResponse::ToolCalls(_) => {
            Err("backend returned tool calls for request without tools".to_string())
        }
    }
}

async fn run_passthrough_stream_preserving_anthropic<C: LLMClient + 'static>(
    client: &Arc<C>,
    serialized: &[Value],
    options: LLMRequestOptions,
    model_name: &str,
    include_usage: bool,
) -> Result<HandlerResult, String> {
    let backend_stream = client
        .send_stream_with_options(serialized.to_vec(), None, options)
        .await
        .map_err(|e| e.to_string())?;
    let client = client.clone();
    let model_name = model_name.to_string();
    let stream = async_stream::stream! {
        let completion_id = crate::proxy::proxy::openai_stream_completion_id();
        let mut emitted_text = false;
        let mut provider_seen = false;
        let mut backend_stream = backend_stream;
        let mut translator = anyllm_translate::new_stream_translator(model_name.clone());

        while let Some(chunk_result) = backend_stream.next().await {
            let chunk = match chunk_result {
                Ok(chunk) => chunk,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            match chunk.chunk_type {
                ChunkType::ProviderEvent => {
                    provider_seen = true;
                    if let Some(event) = chunk.provider_event {
                        match serde_json::from_value(event) {
                            Ok(event) => yield Ok(event),
                            Err(err) => {
                                yield Err(StreamError::new(err.to_string()));
                                return;
                            }
                        }
                    }
                }
                ChunkType::TextDelta if provider_seen => {}
                ChunkType::ToolCallDelta if provider_seen => {}
                ChunkType::TextDelta => {
                    if !chunk.content.is_empty() {
                        let events = match openai_event_to_anthropic_events(
                            crate::proxy::proxy::text_delta_sse_event(
                                &completion_id,
                                &model_name,
                                &chunk.content,
                                !emitted_text,
                                None,
                            ),
                            &mut translator,
                        ) {
                            Ok(events) => events,
                            Err(err) => {
                                yield Err(err);
                                return;
                            }
                        };
                        for event in events {
                            yield Ok(event);
                        }
                        emitted_text = true;
                    }
                }
                ChunkType::Final if provider_seen => return,
                ChunkType::Final => {
                    if !emitted_text {
                        let content = match chunk.response {
                            Some(LLMResponse::Text(text)) => text.content,
                            Some(LLMResponse::ToolCalls(_)) => {
                                yield Err(StreamError::new(
                                    "backend returned tool calls for request without tools",
                                ));
                                return;
                            }
                            None => String::new(),
                        };
                        let events = match openai_event_to_anthropic_events(
                            crate::proxy::proxy::text_delta_sse_event(
                                &completion_id,
                                &model_name,
                                &content,
                                true,
                                None,
                            ),
                            &mut translator,
                        ) {
                            Ok(events) => events,
                            Err(err) => {
                                yield Err(err);
                                return;
                            }
                        };
                        for event in events {
                            yield Ok(event);
                        }
                    }
                    let usage_json = if include_usage {
                        let usage = chunk.usage.or_else(|| client.last_usage());
                        let usage_details =
                            chunk.usage_details.or_else(|| client.last_usage_details());
                        usage.as_ref().map(|u| {
                            crate::proxy::proxy::usage_to_openai_json_with_details(
                                Some(u),
                                usage_details.as_ref(),
                            )
                        })
                    } else {
                        None
                    };
                    let events = match openai_event_to_anthropic_events(
                        crate::proxy::proxy::final_sse_event(
                            &completion_id,
                            &model_name,
                            "stop",
                            usage_json.as_ref(),
                        ),
                        &mut translator,
                    ) {
                        Ok(events) => events,
                        Err(err) => {
                            yield Err(err);
                            return;
                        }
                    };
                    for event in events {
                        yield Ok(event);
                    }
                    for event in translator.finish() {
                        yield Ok(event);
                    }
                    return;
                }
                ChunkType::ToolCallDelta => {
                    yield Err(StreamError::new(
                        "backend streamed tool calls for request without tools",
                    ));
                    return;
                }
                ChunkType::Retry => {}
            }
        }

        yield Err(StreamError::default());
    };

    Ok(HandlerResult::AnthropicStreamBody(Box::pin(stream)))
}

async fn run_passthrough_stream<C: LLMClient + 'static>(
    client: &Arc<C>,
    serialized: &[Value],
    options: LLMRequestOptions,
    model_name: &str,
    include_usage: bool,
) -> Result<HandlerResult, String> {
    let mut backend_stream = client
        .send_stream_with_options(serialized.to_vec(), None, options)
        .await
        .map_err(|e| e.to_string())?;
    let client = client.clone();
    let model_name = model_name.to_string();
    let stream = async_stream::stream! {
        let completion_id = crate::proxy::proxy::openai_stream_completion_id();
        let mut emitted_text = false;

        while let Some(chunk_result) = backend_stream.next().await {
            let chunk = match chunk_result {
                Ok(chunk) => chunk,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };

            match chunk.chunk_type {
                ChunkType::TextDelta => {
                    if !chunk.content.is_empty() {
                        yield Ok(crate::proxy::proxy::text_delta_sse_event(
                            &completion_id,
                            &model_name,
                            &chunk.content,
                            !emitted_text,
                            None,
                        ));
                        emitted_text = true;
                    }
                }
                ChunkType::Final => {
                    let final_usage = chunk.usage;
                    let final_usage_details = chunk.usage_details;
                    if !emitted_text {
                        let content = match chunk.response {
                            Some(LLMResponse::Text(text)) => text.content,
                            Some(LLMResponse::ToolCalls(_)) => {
                                yield Err(StreamError::new(
                                    "backend returned tool calls for request without tools",
                                ));
                                return;
                            }
                            None => String::new(),
                        };
                        yield Ok(crate::proxy::proxy::text_delta_sse_event(
                            &completion_id,
                            &model_name,
                            &content,
                            true,
                            None,
                        ));
                    }
                    let usage_json = if include_usage {
                        let usage = final_usage.or_else(|| client.last_usage());
                        let usage_details =
                            final_usage_details.or_else(|| client.last_usage_details());
                        usage.as_ref().map(|u| {
                            crate::proxy::proxy::usage_to_openai_json_with_details(
                                Some(u),
                                usage_details.as_ref(),
                            )
                        })
                    } else {
                        None
                    };
                    yield Ok(crate::proxy::proxy::final_sse_event(
                        &completion_id,
                        &model_name,
                        "stop",
                        usage_json.as_ref(),
                    ));
                    return;
                }
                ChunkType::ToolCallDelta => {
                    yield Err(StreamError::new(
                        "backend streamed tool calls for request without tools",
                    ));
                    return;
                }
                ChunkType::ProviderEvent => {}
                ChunkType::Retry => {}
            }
        }

        yield Err(StreamError::default());
    };

    Ok(HandlerResult::StreamBody(Box::pin(stream)))
}
