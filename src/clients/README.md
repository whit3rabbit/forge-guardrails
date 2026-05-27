# LLM Clients Module

This module abstracts connections to various LLM provider backends through a unified [LLMClient](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/base.rs#L331) trait interface.

## File & Module Roles

- [base.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/base.rs): Establishes the core [LLMClient](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/base.rs#L331) trait and defines cross-backend datatypes for responses, token usage, cache telemetry, rate limiting, and request options.
- [sampling.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/sampling.rs): Manages recommended sampling configurations (temperature, top_p, top_k, min_p, etc.) for 69+ supported models based on their HuggingFace model cards.
- [anthropic/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/anthropic): Translate messages and thinking blocks to/from the Anthropic messages API.
- [llamafile/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/llamafile): Client interface for llama.cpp and llamafile local servers.
- [ollama/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/ollama): Client interface for local Ollama instances.
- [anyllm_proxy/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients/anyllm_proxy): Unified runtime client supporting sidecar process or in-process Axum routing translations.

## Common Architecture

All backend clients implement the `LLMClient` trait:

```rust
pub trait LLMClient: Send + Sync {
    fn api_format(&self) -> ApiFormat;
    async fn send(&self, messages: Vec<Value>, tools: Option<Vec<ToolSpec>>, sampling: Option<SamplingParams>) -> Result<LLMResponse, BackendError>;
    async fn send_stream(&self, messages: Vec<Value>, tools: Option<Vec<ToolSpec>>, sampling: Option<SamplingParams>) -> Result<ChunkStream, StreamError>;
    async fn get_context_length(&self) -> Result<Option<i64>, ContextDiscoveryError>;
}
```
