# Core Module

This module contains the fundamental state machines, message types, and the multi-turn executor that orchestrates the guarded LLM-agent workflow loops.

## File Breakdown

- [runner.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/runner.rs): Implements [WorkflowRunner](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/runner.rs#L49), the main coordinator that drives agent loop turns, monitors iteration bounds, runs prerequisite validators, and executes guardrails.
- [inference.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/inference.rs): Handles request serialization, calls backend clients, folds reasoning text (extracting raw `<think>` blocks), and produces standardized [InferenceResult](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/inference.rs#L22) outputs.
- [message.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/message.rs): Declares the unified chat payload format [Message](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/message.rs#L83), identifying roles (`System`, `User`, `Assistant`, `Tool`) and storing metadata.
- [slot_worker.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/slot_worker.rs): Manages queues, concurrency controls, and slot limits for local backends (like llama-server) during high-throughput execution.
- [steps.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/steps.rs): Tracks step progress, completion state, and active iteration constraints.
- [tool_spec.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/tool_spec.rs): Structs for defining tools (`ToolSpec`) and parameters using JSON schema formats.
- [workflow.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/workflow.rs): Outlines targeted workflows, their step dependencies, and sequential prerequisites.
