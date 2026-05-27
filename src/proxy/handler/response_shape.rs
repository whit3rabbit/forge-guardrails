use super::{AnthropicEventStream, HandlerResult, OpenAiEventStream};
use crate::clients::base::{LLMUsageDetails, TextResponse, TokenUsage, ToolCall};
use crate::error::StreamError;
use crate::proxy::{
    text_response_to_openai_with_usage_details, tool_calls_to_openai_with_usage_details,
};
#[cfg(test)]
use anyllm_translate::anthropic::streaming::StreamEvent;
use futures_util::StreamExt;
#[cfg(test)]
use serde_json::Value;

/// Convert a final text object while preserving the requested response shape.
pub(super) fn text_response_result(
    text: &TextResponse,
    model_name: &str,
    stream: bool,
    stream_include_usage: bool,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> HandlerResult {
    if stream {
        let (usage, usage_details) =
            cloned_stream_usage(stream_include_usage, usage, usage_details);
        HandlerResult::StreamBody(text_events_stream(
            text.content.clone(),
            model_name.to_string(),
            usage,
            usage_details,
        ))
    } else {
        HandlerResult::Response(text_response_to_openai_with_usage_details(
            text,
            model_name,
            usage,
            usage_details,
        ))
    }
}

pub(super) fn text_content_result(
    content: &str,
    model_name: &str,
    stream: bool,
    stream_include_usage: bool,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> HandlerResult {
    if stream {
        let (usage, usage_details) =
            cloned_stream_usage(stream_include_usage, usage, usage_details);
        HandlerResult::StreamBody(text_events_stream(
            content.to_string(),
            model_name.to_string(),
            usage,
            usage_details,
        ))
    } else {
        let text = TextResponse::new(content);
        HandlerResult::Response(text_response_to_openai_with_usage_details(
            &text,
            model_name,
            usage,
            usage_details,
        ))
    }
}

pub(super) fn tool_calls_result(
    calls: &[ToolCall],
    model_name: &str,
    stream: bool,
    stream_include_usage: bool,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> HandlerResult {
    if calls.is_empty() {
        text_content_result(
            "",
            model_name,
            stream,
            stream_include_usage,
            usage,
            usage_details,
        )
    } else if stream {
        let (usage, usage_details) =
            cloned_stream_usage(stream_include_usage, usage, usage_details);
        HandlerResult::StreamBody(tool_call_events_stream(
            calls.to_vec(),
            model_name.to_string(),
            usage,
            usage_details,
        ))
    } else {
        HandlerResult::Response(tool_calls_to_openai_with_usage_details(
            calls,
            model_name,
            usage,
            usage_details,
        ))
    }
}

fn cloned_stream_usage(
    include_usage: bool,
    usage: Option<&TokenUsage>,
    usage_details: Option<&LLMUsageDetails>,
) -> (Option<TokenUsage>, Option<LLMUsageDetails>) {
    if include_usage {
        (usage.cloned(), usage_details.cloned())
    } else {
        (None, None)
    }
}

fn text_events_stream(
    content: String,
    model_name: String,
    usage: Option<TokenUsage>,
    usage_details: Option<LLMUsageDetails>,
) -> OpenAiEventStream {
    Box::pin(async_stream::stream! {
        for event in crate::proxy::proxy::text_to_sse_event_iter_with_usage_details(
            &content,
            &model_name,
            0,
            usage.as_ref(),
            usage_details.as_ref(),
        ) {
            yield Ok(event);
        }
    })
}

fn tool_call_events_stream(
    calls: Vec<ToolCall>,
    model_name: String,
    usage: Option<TokenUsage>,
    usage_details: Option<LLMUsageDetails>,
) -> OpenAiEventStream {
    Box::pin(async_stream::stream! {
        for event in crate::proxy::proxy::tool_calls_to_sse_event_iter_with_usage_details(
            &calls,
            &model_name,
            usage.as_ref(),
            usage_details.as_ref(),
        ) {
            yield Ok(event);
        }
    })
}

#[cfg(test)]
pub(super) async fn collect_openai_events(
    mut stream: OpenAiEventStream,
) -> Result<Vec<Value>, StreamError> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
}

pub(super) fn anthropic_events_stream(
    openai_events: OpenAiEventStream,
    model: String,
) -> AnthropicEventStream {
    Box::pin(async_stream::stream! {
        let mut openai_events = openai_events;
        let mut translator = anyllm_translate::new_stream_translator(model);

        while let Some(event) = openai_events.next().await {
            let event = match event {
                Ok(event) => event,
                Err(err) => {
                    yield Err(err);
                    return;
                }
            };
            let chunk: anyllm_translate::openai::ChatCompletionChunk = match serde_json::from_value(event) {
                Ok(chunk) => chunk,
                Err(err) => {
                    yield Err(StreamError::new(err.to_string()));
                    return;
                }
            };
            for event in translator.process_chunk(&chunk) {
                yield Ok(event);
            }
        }

        for event in translator.finish() {
            yield Ok(event);
        }
    })
}

#[cfg(test)]
pub(super) async fn collect_anthropic_events(
    mut stream: AnthropicEventStream,
) -> Result<Vec<StreamEvent>, StreamError> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event?);
    }
    Ok(events)
}
