# Standard Tools Module

This module defines built-in tools that can be registered in agent workflows and executed directly by the runtime engine.

## File Breakdown

- [respond.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools/respond.rs): Defines the terminal [respond_tool](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools/respond.rs#L68) and its JSON schema specifier. This is a special workflow-terminating tool that models call when they have completed the task and want to output a final answer to the user.
- [mod.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools/mod.rs): Direct exports for the built-in tools.
