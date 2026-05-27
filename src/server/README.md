# Backend Server Module

This module controls the lifecycle of local LLM backend engines (such as `llama-server`, `llamafile`, or `ollama`) using supervised child processes.

## File Breakdown

- [manager.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/manager.rs): Exposes the [ServerManager](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/manager.rs#L14) coordinator, managing active server health checks and execution lifecycles.
- [lifecycle.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/lifecycle.rs): Process spawning, stdout/stderr logging redirections, and process supervision logic. Reclaims background child engines when the parent process exits.
- [setup.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/setup.rs): High-level [setup_backend](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/setup.rs#L15) entrypoint used by CLI tools to start requested models automatically.
- [budget.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/budget.rs): Implements [BudgetMode](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/budget.rs#L13) configurations to calculate target memory and context boundaries depending on the system hardware.
- [args.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/args.rs): Spawning argument helpers for configuring backend processes.
- [runtime.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/runtime.rs): Structs representing target LLM backend endpoint addresses.
