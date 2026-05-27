//! OpenAI-compatible proxy conversion functions.

mod id;
mod message_parse;
mod request_options;
mod response_format;

pub(crate) use id::openai_stream_completion_id;
pub use message_parse::{openai_to_messages, OpenAiMessageError};
pub use request_options::{
    extract_passthrough, extract_sampling, has_respond_tool, respond_tool_openai,
    strip_respond_calls,
};
pub(crate) use response_format::{
    final_sse_event, text_delta_sse_event, text_response_to_openai_with_usage_details,
    text_to_sse_event_iter_with_usage_details, tool_calls_to_openai_with_usage_details,
    tool_calls_to_sse_event_iter_with_usage_details, usage_to_openai_json_with_details,
};
pub use response_format::{
    text_response_to_openai, text_response_to_openai_with_usage, text_to_sse_events,
    text_to_sse_events_with_usage, tool_calls_to_openai, tool_calls_to_openai_with_usage,
    tool_calls_to_sse_events, tool_calls_to_sse_events_with_usage,
};

#[cfg(test)]
mod tests;
