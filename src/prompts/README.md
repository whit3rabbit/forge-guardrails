# Prompts & Parsing Module

This module constructs system nudge templates used during tool retries and parses raw model output to extract structural tool call payloads.

## File Breakdown

- [nudges.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/prompts/nudges.rs): Contains helper functions to build retry/nudge prompts injected into chat logs. Includes:
  - `retry_nudge`: Retries a tool call that failed schema validation.
  - `step_nudge`: Retries blocked steps.
  - `unknown_tool_nudge`: Handles cases where the model invoked a non-existent tool.
  - `prerequisite_nudge`: Guides the model to satisfy prerequisite steps.
- [parse_strategies.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/prompts/parse_strategies.rs): Logic for extracting JSON/XML tool calls from assistant text. Designed to handle trailing markdown ticks, malformed characters, and multiple tool calls in a single block.
