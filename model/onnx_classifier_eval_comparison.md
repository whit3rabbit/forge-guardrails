# ONNX Classifier Shadow Evaluation Comparison

## Abstract

This note compares two local Forge release evaluations for `Ministral-3-8B-Instruct-2512-Q8_0`: a baseline run without the ONNX classifier and a run with the ONNX classifier enabled in shadow mode. The observed score increased from `83.8%` to `84.2%`, a net gain of one correct row out of `260`. This is not sufficient evidence that ONNX improved behavior. The classifier was configured as `shadow`, so its predictions were telemetry and did not enforce, block, or alter tool calls. The most defensible interpretation is that the small score change reflects stochastic generation variance, while the classifier added latency and exposed false-positive risk.

Follow-up review: [`local_eval_findings_2026-05-30.md`](local_eval_findings_2026-05-30.md)
compares the later `release-onnx-enforce` and `release-onnx-final-shadow`
runs. It confirms that tool-call enforcement made `error_recovery*` worse and
that the final-response verifier remained shadow-only telemetry with material
proxy memory cost.

## Materials and Method

Artifacts reviewed:

- Baseline output directory: `target/local-eval/release-baseline`
- ONNX shadow output directory: `target/local-eval/release-onnx-shadow-2`
- Baseline oracle rows: `target/local-eval/release-baseline/python_oracle.jsonl`
- ONNX oracle rows: `target/local-eval/release-onnx-shadow-2/python_oracle.jsonl`
- Baseline summary: `target/local-eval/release-baseline/proxy_summary.txt`
- ONNX summary: `target/local-eval/release-onnx-shadow-2/proxy_summary.txt`
- ONNX smoke telemetry: `target/local-eval/release-onnx-shadow-2/rust_smoke.jsonl`
- ONNX thresholds: `target/classifier-artifacts/onnx/thresholds.json`

The two `python_oracle.jsonl` files were compared by `(scenario, run)` to identify outcome changes. The `proxy_summary.txt` files were used for aggregate accuracy, completeness, and failure-class counts. The ONNX smoke file was used only for classifier telemetry because the Python oracle rows do not include classifier score fields.

## Results

| Metric | Baseline | ONNX shadow | Difference |
|---|---:|---:|---:|
| Rows | 260 | 260 | 0 |
| Successful rows | 218 | 219 | +1 |
| Score / accuracy | 83.8% | 84.2% | +0.4 percentage points |
| Completeness | 260/260 | 260/260 | 0 |
| Efficiency | 100% | 100% | 0 |
| Average wasted calls | 0.0 | 0.0 | 0 |
| Average speed | 10.1s | 11.3s | +1.2s slower |
| Completed but inaccurate | 42 | 41 | -1 |
| Incomplete or protocol failures | 0 | 0 | 0 |

The aggregate gain is one net corrected row. The only scenario with an outcome delta was `data_gap_recovery_extended`:

| Scenario run | Baseline | ONNX shadow | Direction |
|---|---:|---:|---|
| `data_gap_recovery_extended`, run 1 | correct | incorrect | regression |
| `data_gap_recovery_extended`, run 5 | incorrect | correct | improvement |
| `data_gap_recovery_extended`, run 8 | incorrect | correct | improvement |

All persistent weak scenarios remained weak:

| Scenario | Baseline | ONNX shadow |
|---|---:|---:|
| `argument_transformation` | 0/10 | 0/10 |
| `argument_transformation_stateful` | 0/10 | 0/10 |
| `grounded_synthesis` | 0/10 | 0/10 |
| `grounded_synthesis_stateful` | 0/10 | 0/10 |

The ONNX run did not introduce incomplete workflows, missing required steps, or proxy contract mismatches. That is good for integration safety, but it is not evidence of accuracy benefit.

## Classifier Telemetry

The release oracle output does not include classifier score fields. The ONNX `rust_smoke.jsonl` file does include classifier telemetry for the smaller smoke suite. It recorded:

- `72` classifier calls.
- `52` `valid` predictions.
- `20` `wrong_tool_semantic` predictions.
- `16` non-`valid` predictions on rows that still succeeded.
- `4` non-`valid` predictions on rows that failed.

The most important qualitative observation is that the classifier marked successful `report` terminal calls and `summarize` summary calls as `wrong_tool_semantic`, often with high confidence. That is a false-positive pattern for guarded workflows because terminal and summary tools can be semantically correct only in workflow context.

This risk is partly mitigated by the current threshold artifact. In `target/classifier-artifacts/onnx/thresholds.json`, `wrong_tool_semantic` has:

- `advisory_min_confidence`: `1.01`
- `enforce_min_confidence`: `1.01`

Because classifier confidence is bounded by `1.0`, these thresholds intentionally prevent `wrong_tool_semantic` from taking advisory or enforcement action. This is correct for the present artifact. Lowering these thresholds without additional Forge-specific validation would likely block valid workflows.

## Discussion

The ONNX shadow run did not demonstrate a causal improvement. Shadow mode records predictions after deterministic guardrails allow a call, but it does not change the deterministic path. Therefore, the one-row improvement cannot be attributed to classifier intervention. The paired outcome changes also show bidirectional variance: one run regressed and two improved in the same scenario family.

The result is consistent with ordinary stochastic variation in local LLM generation. This is especially plausible because the changed rows were strict synthesis outputs in `data_gap_recovery_extended`, a scenario already documented as sensitive to string-literal and scorer requirements. The unchanged failure clusters in `argument_transformation*` and `grounded_synthesis*` show that ONNX did not address the model's main advanced-reasoning weaknesses.

The classifier is useful as instrumentation, not as an active guardrail. The telemetry surfaces where a semantic verifier disagrees with successful workflow behavior. That is valuable training data. It is not yet safe enough for enforcement.

## Answers

1. Can the model improve?

Yes. The model should improve on advanced reasoning and strict synthesis tasks, especially `argument_transformation*` and `grounded_synthesis*`. These remained at `0/10` with and without ONNX. The issue is not tool protocol completion, because completeness stayed at `100%`; the issue is accurate multi-step reasoning, argument transformation, and final synthesis under strict scoring.

2. Can the classifier improve?

Yes. The classifier needs more Forge-specific examples for terminal tools, summarization tools, successful guarded workflow completions, and state-dependent semantic validity. It also needs calibration work for `wrong_tool_semantic`, because the smoke telemetry shows high-confidence false positives on successful rows. Future evaluation should report classifier labels in the release oracle rows, not only in smoke output.

3. Did ONNX help enough to justify use beyond shadow telemetry?

No. The observed gain was `+0.4` percentage points, or one net row out of `260`, while average speed regressed from `10.1s` to `11.3s`. Since the classifier was in shadow mode, the score change is not causal evidence of classifier benefit. ONNX should remain shadow-only until a larger paired eval shows stable gains, lower false-positive rates, and clear benefit on the persistent weak scenarios.

## Conclusion

The ONNX classifier integration appears operationally safe in shadow mode for these runs: it did not reduce completeness, create protocol failures, or interfere with deterministic guardrails. It did not, however, provide demonstrated behavioral improvement. The current data supports continued telemetry collection and targeted classifier retraining, not advisory or enforcement promotion.
