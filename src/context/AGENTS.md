# Context & Compaction Development Guidelines

When modifying files under [src/context](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/context), adhere to the following safety constraints:

## Gotchas & Invariants

1. **System & Setup Message Preservation**:
   - Compaction strategies **MUST NOT** drop system setup messages (typically at index 0) or the initial user request that started the execution.
   - Any sliding window or summary algorithm must keep these anchors intact to prevent model drift.
2. **Paired Tool Call/Result Invariant**:
   - A tool call and its matching tool result must never be separated during compaction.
   - Dropping or summarizing a tool call without its corresponding tool result (or vice versa) produces a protocol-invalid transcript that will fail upstream backend validation.
   - Summarizations must treat tool call/result sequences as a unified block and replace them with a single consolidated summary text block.
3. **Graceful Hardware Fallbacks**:
   - Hardware profiling (`hardware.rs`) runs command-line utilities. Ensure that failures to execute `system_profiler` or `nvidia-smi` (due to operating system differences, permissions, or missing binaries) are caught and handled gracefully.
   - Fall back to standard defaults rather than panicking or failing build processes.

## Testing Target

When modifying context files, run:

```bash
cargo test --package forge-guardrails --lib context
```
