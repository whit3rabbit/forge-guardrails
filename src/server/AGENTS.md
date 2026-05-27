# Server Development Guidelines

When modifying files under [src/server](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server), maintain clean process supervision by adhering to these rules:

## Gotchas & Rules

1. **Orphan Process Prevention**:
   - The [ServerManager](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/manager.rs#L14) spawns local process engines (like `llama-server`).
   - You **MUST** ensure that these subprocesses are cleanly killed on runner exit, test panics, or SIGINT signals.
   - Any modifications to the lifecycle loop in [lifecycle.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/server/lifecycle.rs) must maintain the drop/kill implementation.
2. **Port Conflict Protection**:
   - Before binding or initiating a server command, verify that the target port (e.g., `8080` or `8081`) is available. If a server is already running on that port, handle it gracefully by raising a descriptive error rather than locking up.
3. **Signal Forwarding**:
   - Spawning scripts and wrappers must propagate termination signals (SIGINT/SIGTERM/SIGHUP) to the child process and block until it has cleanly exited, preventing locked port handles on rerun.

## Testing Target

When modifying server/lifecycle files, run:

```bash
cargo test --package forge-guardrails --lib server
```
