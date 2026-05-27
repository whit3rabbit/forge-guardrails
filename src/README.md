# Src Directory Overview

This is the primary source directory for `forge-guardrails`, a Rust implementation of the Forge guarded LLM-agent workflow framework.

## Subdirectory Breakdown

The codebase is organized into several modules, each responsible for a distinct part of the guarded workflow loop:

- [core](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core): Core state machine and execution loop.
- [guardrails](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails): Step enforcers, schema validators, and neural/probabilistic model classifiers.
- [clients](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/clients): API wrappers and sampling configurations for Anthropic, Llamafile, Ollama, and AnyLLM.
- [context](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context): Budget managers, Apple Silicon/NVIDIA hardware probes, and transcript compaction strategies.
- [prompts](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/prompts): System nudge templates and parser utilities for JSON and XML tool extraction.
- [proxy](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy): OpenAI/Anthropic-compatible HTTP proxy layer.
- [server](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server): Backend daemon lifecycle manager.
- [tools](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools): Standard terminal commands, including the terminal `respond` tool.
- [bin](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/bin): Binary entrypoints for running local evaluations, proxy servers, and downloading model checkpoints.

## Top-Level Files

- [lib.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/lib.rs): Entry point for the crate library, exporting public API items and legacy re-exports for backwards compatibility.
- [error.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/error.rs): Unified error models across core execution, parsing, validation, and clients.
- [server.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server.rs): Re-exports lifecycle methods for the backend server runner.
