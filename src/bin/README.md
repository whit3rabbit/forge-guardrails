# Runnable Binaries Module

This directory contains executable entrypoints for run-time utilities, local evaluations, and proxy daemons.

## Sub-Binaries & File Roles

- [download-classifier.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/bin/download-classifier.rs): CLI utility to pull verifier checkpoints, ONNX classifier models, and calibration thresholds from the Hugging Face Hub.
- [forge-eval/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/bin/forge-eval): Local benchmark framework. Runs scenarios (`basic_2step`, `sequential_3step`, `error_recovery`, and optionally compaction chains) and measures accuracy, latency, and success rates.
- [forge-guardrails-proxy/](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/bin/forge-guardrails-proxy): The production proxy application daemon. Spawns endpoints implementing Anthropic/OpenAI specifications, translating models and checking safety pipelines.
