# Context & Memory Module

This module is responsible for monitoring token consumption, warning about context window limits, and performing transcript compaction when approaching token budgets.

## File Breakdown

- [hardware.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context/hardware.rs): Probes system hardware configurations to determine native memory bounds. Supports:
  - Apple Silicon unified memory extraction via `system_profiler` plist parsing.
  - NVIDIA VRAM profiling via `nvidia-smi` parsing.
- [manager.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context/manager.rs): The central [ContextManager](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context/manager.rs#L44) which manages active token counts, tracks the history, triggers warnings, and executes compaction routines.
- [strategies.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context/strategies.rs): Contains concrete algorithms for compacting chat history:
  - `NoCompact`: A basic strategy that raises errors when the budget is exceeded.
  - `SlidingWindowCompact`: Drops oldest messages while preserving initial system setup messages.
  - `TieredCompact`: Multi-phased compaction that replaces detailed tool calls and responses with compacted summary logs to fit budgets.
- [mod.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context/mod.rs): Module exports and base helper utilities.
