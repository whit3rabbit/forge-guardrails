# Binary Development Guidelines

When modifying binary entrypoints under [src/bin](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/bin), keep options parsing and downstream execution stable:

## Gotchas & Rules

1. **Deterministic CLI Configurations**:
   - Both `forge-eval` and `forge-guardrails-proxy` parse configuration options using `clap`.
   - Never change option names or default behaviors without verifying impact on local scripts (e.g. [scripts/run_local_eval.sh](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/scripts/run_local_eval.sh) and [scripts/start_llamaserver_proxy.sh](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/scripts/start_llamaserver_proxy.sh)).
   - Always verify that modifying sub-arguments maintains compatibility with upstream invocation arguments.
2. **Stable Model Checkpoint Downloader Defaults**:
   - Modifying download patterns in [download-classifier.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/bin/download-classifier.rs) must keep [DEFAULT_CLASSIFIER_REPO](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/guardrails/classifier_artifact.rs) and revision tags aligned with current production verifier weights. Do not hardcode custom repository details inside the downloader.

## Testing Target

Verify binary builds:

```bash
cargo check --bin download-classifier
cargo check --bin forge-eval
cargo check --bin forge-guardrails-proxy
```
