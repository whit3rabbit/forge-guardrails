# Proxy Server Module

This module exposes an HTTP server daemon providing OpenAI-compatible and Anthropic-compatible routing endpoints. It intercepts requests, runs them through the Forge guardrails pipeline, and streams responses back to clients.

## File Breakdown

- [server.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/server.rs): Manages the web server process lifecycle using the Axum framework, handling socket binding and connection setups.
- [handler.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/handler.rs): Contains route-specific controllers that map incoming JSON payloads into guarded workflow configurations and orchestrate the loop execution.
- [proxy.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/proxy.rs): Performs format conversion between OpenAI messaging shapes, Anthropic message structures, and the internal unified Forge message format.
- [response.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/response.rs): Declarations of serialization shapes representing SSE (Server-Sent Event) response frames.
- [mod.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/mod.rs): Direct module exports and interface mapping.
- [AGENTS.md](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/AGENTS.md): Strict development constraints for the proxy layer.
