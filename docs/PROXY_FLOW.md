# Proxy Flow

This document describes the current request flow through the Forge proxy server.
It is a snapshot of how the code works now, not a target design.

This repo is Rust/Tokio. It does not use Python `asyncio`. The matching concerns
are Tokio async boundaries, scoped locks, shared clients, bounded stream parsing,
and avoiding repeated construction of expensive state.

## Current Request Flow

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Axum as forge-guardrails-proxy Axum routes
    participant Factory as ClientFactory
    participant Handler as proxy handler
    participant Inference as run_inference_with_options_shared_context
    participant LLMClient as Routed LLMClient
    participant Runtime as AnyLlmRuntimeClient
    participant Sidecar as AnyLlmProxyClient sidecar HTTP
    participant Backend as Provider or local backend
    participant Response as Response/SSE builders

    Client->>Axum: POST /v1/chat/completions or /v1/messages
    Axum->>Axum: enforce max body size and parse JSON
    Axum->>Axum: extract request model or default model
    Axum->>Factory: client_for_model(model)
    alt anyllm runtime mode
        Factory->>Runtime: for_model(model)
        Runtime-->>Factory: client sharing ChatCompletionRuntime
    else direct OpenAI or sidecar mode
        Factory->>Sidecar: new(model).with_base_url(...).with_http_client(shared)
        Sidecar-->>Factory: client using shared reqwest::Client
    else direct backend mode
        Factory->>LLMClient: construct backend-specific client
    end
    Factory-->>Axum: RoutedClient
    Axum->>Axum: create per-request ContextManager
    opt serialize_requests=true
        Axum->>Axum: acquire owned request mutex guard
    end

    alt POST /v1/messages
        Axum->>Handler: handle_anthropic_messages_with_scorers(raw, parsed)
        Handler->>Handler: translate Anthropic request to OpenAI shape
    else POST /v1/chat/completions
        Axum->>Handler: handle_chat_completions_with_scorers(body)
    end

    Handler->>Handler: openai_to_messages(messages)
    Handler->>Handler: extract sampling, passthrough, stream settings, _forge contract

    alt no tools
        Handler->>Handler: fold_and_serialize(messages)
        Handler->>LLMClient: send_with_options or send_stream_with_options(tools=None)
    else tools present
        Handler->>Handler: parse_tool_specs(tools)
        Handler->>Handler: inject reserved respond tool unless real terminal tools replace it
        Handler->>Handler: validate _forge required_steps and terminal_tools
        Handler->>Inference: run_inference_with_options_shared_context(...)
        loop validation, rescue, retry, scoring, step enforcement
            Inference->>Inference: maybe compact context and fold transcript
            Inference->>LLMClient: send_with_options or send_stream_with_options(tools)
            Inference->>Handler: validated LLMResponse or guardrail error
            Handler->>Handler: apply step/scorer nudges when needed
        end
    end

    alt RoutedClient is AnyLlmRuntimeClient
        LLMClient->>Runtime: build OpenAI request body
        Runtime->>Backend: ChatCompletionService complete or complete_stream
        Backend-->>Runtime: OpenAI response or chunk stream
        Runtime->>Runtime: parse_openai_response or parse_openai_chunks
    else RoutedClient is AnyLlmProxyClient
        LLMClient->>Sidecar: build OpenAI request body
        Sidecar->>Backend: HTTP POST /v1/chat/completions
        Backend-->>Sidecar: JSON response or SSE bytes
        Sidecar->>Sidecar: parse_openai_response or parse_openai_sse
    else other backend client
        LLMClient->>Backend: backend-specific request
        Backend-->>LLMClient: backend-specific response or stream
    end

    LLMClient-->>Handler: LLMResponse or StreamChunk final response
    Handler->>Handler: strip respond tool calls from client-visible output
    Handler->>Response: OpenAI response object or OpenAI SSE events
    opt original request was Anthropic
        Response->>Response: translate OpenAI response/events back to Anthropic
    end
    Response-->>Axum: JSON response or live SSE body
    opt serialize_requests=true and streaming
        Axum->>Response: keep owned mutex guard alive until SSE stream ends
    end
    Axum-->>Client: 200 response, 400 bad request, or 502 upstream error
```

## Path Notes

- The standalone binary entrypoint is `src/bin/forge-guardrails-proxy/main.rs`.
  It builds a current-thread Tokio runtime, then serves Axum routes from
  `src/bin/forge-guardrails-proxy/routes.rs`.
- Each request gets a `RoutedClient` from `ClientFactory::client_for_model`.
  Runtime mode reuses the same anyllm runtime service through
  `AnyLlmRuntimeClient::for_model`. Direct OpenAI and sidecar paths rebuild a
  light client wrapper per request, but reuse the shared `reqwest::Client`.
- The proxy handler is shared library code in `src/proxy/handler.rs`. It owns
  OpenAI-to-internal message conversion, `_forge` contract handling, `respond`
  injection, validation retries, scorer nudges, step enforcement, and final
  response shaping.
- Response and SSE formatting live in the library-private
  `src/proxy/response.rs`. The standalone binary's response module is a thin
  private wrapper over that same source, so CORS and SSE byte formatting stay
  aligned without making response builders public API.
- No-tools requests bypass guardrails through `run_passthrough`. Streaming
  passthrough is live from the backend.
- Tool-using guarded requests may accept `stream=true`, but the client-visible
  stream is emitted after guardrail validation resolves a complete response.
  This preserves validation correctness at the cost of token-by-token guarded
  streaming.
- Anthropic requests are translated to OpenAI request shape first, handled by
  the same guarded OpenAI path, then translated back to Anthropic response or
  Anthropic SSE events.
- Request body parsing and handler-to-HTTP status mapping are shared by the
  library server and standalone binary through a private helper. Route-specific
  orchestration, including binary model routing and per-request context
  construction, remains in each caller.

## Reuse And Drift Audit

| Area | Current state | Risk | Reuse direction |
| --- | --- | --- | --- |
| `src/proxy/response.rs` and `src/bin/forge-guardrails-proxy/response.rs` | Shared private response/SSE implementation; the binary module reuses the library source and exports only crate-local helpers. | Low drift risk. The tradeoff is a relative source include from the binary wrapper. | Keep this private unless there is an explicit public API decision. Continue testing both library server and standalone binary routes after SSE/header changes. |
| `parse_openai_sse` and `parse_openai_chunks` in `src/clients/anyllm_proxy/streaming.rs` | Sidecar SSE bytes and runtime chunk streams now feed the same `OpenAiStreamAccumulator` for text, reasoning, tool deltas, usage, and final response. | Lower drift risk. Source adapters can still diverge in line framing, `[DONE]`, and upstream error handling. | Keep bounds such as `MAX_STREAM_TOOL_CALLS`. Add parser tests before changing sparse-index, unterminated-line, or final-chunk behavior. |
| Library `HTTPServer` request handlers and binary Axum routes | Body size checks, JSON parsing, Anthropic typed parsing, and handler-to-HTTP status mapping are shared through a private helper. Route orchestration, request serialization, binary model routing, scorer wiring, and response/SSE construction stay local. | Lower drift risk. The remaining duplication is intentional where the two surfaces own different state. | Keep the helper private and behavior-only. Do not move per-request context, scorer, or client selection into shared state without a separate design. |
| `strip_respond_calls` and `filter_respond` | `filter_respond` delegates to `strip_respond_calls` and drops the optional text. | Low drift risk; `strip_respond_calls` remains the behavior-sensitive implementation. | Keep future respond semantics in `strip_respond_calls`. |
| Guarded handler scoring/nudge branches | Tool-call scoring, final-response tool scoring, final text scoring, and step enforcement have repeated nudge/error plumbing. | Duplication is visible, but the branch behavior is parity-sensitive. | Refactor only with focused proxy tests covering step nudges, classifier nudges, final response nudges, streaming, and `respond` filtering. |
| Per-request context manager in binary routes | `ContextManager::new(Box::new(NoCompact), ...)` is rebuilt for each request. | This is cheap and stateless by design, but it means proxy mode does not retain session memory. | Keep per-request context unless proxy session state is explicitly designed. Reusing this state would change behavior. |
| Sidecar/direct client wrapper construction | `AnyLlmProxyClient` and other direct clients are rebuilt per request around shared config and shared HTTP client. | Usually low cost, but repeated wrapper construction can obscure which state is meant to persist. | Runtime path already has `for_model`. A future sidecar `for_model` style builder could make reuse intent clearer without sharing last-call state. |

## Reusable Public Builders

Current public builders and conversion helpers that should be reused before
adding new construction paths:

- `AnyLlmRuntimeClient::from_runtime`
- `AnyLlmRuntimeClient::from_config`
- `AnyLlmRuntimeClient::from_multi_config`
- `AnyLlmRuntimeClient::from_multi_config_with_model_router`
- `AnyLlmRuntimeClient::with_context_length`
- `AnyLlmRuntimeClient::for_model`
- `AnyLlmProxyClient::new`
- `AnyLlmProxyClient::with_base_url`
- `AnyLlmProxyClient::with_api_key`
- `AnyLlmProxyClient::with_http_client`
- `AnyLlmProxyClient::with_context_length`
- `AnyLlmProxyClient::with_timeout`
- `openai_to_messages`
- `text_response_to_openai`
- `tool_calls_to_openai`
- `text_to_sse_events`
- `tool_calls_to_sse_events`
- `respond_tool_openai`
- `strip_respond_calls`

`build_openai_request_body` is already the right internal reuse point for
OpenAI request construction shared by `AnyLlmRuntimeClient` and
`AnyLlmProxyClient`. It is intentionally not public today.

## Refactor Guardrails

- Keep Forge in front of guarded traffic. Do not route guarded requests through
  anyllm HTTP handlers before Forge validates tool calls.
- Preserve Tokio async behavior. Avoid holding ordinary mutex guards across
  backend calls unless the guard is intentionally moved into a response stream
  to serialize the full request lifecycle.
- Keep stream accumulation bounded. Do not remove `MAX_STREAM_TOOL_CALLS` or
  non-contiguous index checks.
- Do not make public API changes for response builders, request builders, or
  proxy route helpers without explicit approval.
- For follow-up code changes, run at least:
  `cargo test proxy::handler`,
  `cargo test proxy::server`, and
  `cargo test --test anyllm_proxy_client_tests`.
