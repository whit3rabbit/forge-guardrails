pub mod handler;
#[allow(clippy::module_inception)]
pub mod proxy;
mod response;
pub mod server;

pub use handler::{
    handle_anthropic_messages, handle_anthropic_messages_with_scorer,
    handle_anthropic_messages_with_scorers,
    handle_anthropic_messages_with_scorers_and_tool_controls,
    handle_anthropic_messages_with_scorers_and_tool_output_compression, handle_chat_completions,
    handle_chat_completions_with_scorer, handle_chat_completions_with_scorers,
    handle_chat_completions_with_scorers_and_tool_controls,
    handle_chat_completions_with_scorers_and_tool_output_compression,
    init_proxy_classifier_log_sink_from_env, AnthropicEventStream, AnthropicHandlerError,
    AnthropicHandlerResult, HandlerError, HandlerResult, OpenAiEventStream,
};
pub use proxy::{
    extract_passthrough, extract_sampling, has_respond_tool, openai_to_messages,
    respond_tool_openai, strip_respond_calls, text_response_to_openai,
    text_response_to_openai_with_usage, text_to_sse_events, text_to_sse_events_with_usage,
    tool_calls_to_openai, tool_calls_to_openai_with_usage, tool_calls_to_sse_events,
    tool_calls_to_sse_events_with_usage, OpenAiMessageError,
};
pub(crate) use proxy::{
    text_response_to_openai_with_usage_details, tool_calls_to_openai_with_usage_details,
};
pub use server::HTTPServer;
