# Local Eval Findings: ONNX Classifier and Final-Response Verifier

Date reviewed: 2026-05-30.

## Scope

This note compares the current local release eval artifacts:

- Tool-call classifier enforcement: `target/local-eval/release-onnx-enforce`
- Tool-call classifier shadow plus final-response verifier shadow:
  `target/local-eval/release-onnx-final-shadow`

Both runs used `Ministral-3-8B-Instruct-2512-Q8_0`, release suite,
`runs=10`, streaming, `budget_tokens=8192`, and the quantized ONNX artifacts.

This is not a clean A/B for final-response verifier benefit because the second
run changed both classifier mode and final-response verifier availability.
Resource deltas are directional unless repeated with matched seeds and modes.

## Aggregate Result

| Metric | Tool-call enforce | Tool-call shadow + final shadow | Direction |
|---|---:|---:|---|
| Rows | 260 | 260 | same |
| Score / accuracy | `75.8%` | `84.2%` | improved |
| Success | `197/260` | `219/260` | +22 rows |
| Completeness | `240/260` | `260/260` | +20 rows |
| Completed but inaccurate | `43/260` | `41/260` | -2 rows |
| Incomplete / protocol / tool-loop failures | `20/260` | `0/260` | fixed |

The final-shadow run passed the published comparison: local score `84.2%`
against published score `81.4%`, with `100%` completeness.

## Scenario Changes

The major difference is error recovery:

| Scenario | Tool-call enforce success | Tool-call shadow + final shadow success |
|---|---:|---:|
| `error_recovery` | `0/10` | `10/10` |
| `error_recovery_stateful` | `0/10` | `10/10` |
| `argument_transformation` | `0/10` | `1/10` |
| `argument_transformation_stateful` | `0/10` | `0/10` |
| `data_gap_recovery_extended` | `8/10` | `8/10` |
| `data_gap_recovery_extended_stateful` | `9/10` | `10/10` |
| `grounded_synthesis` | `0/10` | `0/10` |
| `grounded_synthesis_stateful` | `0/10` | `0/10` |

The enforcement run made behavior worse. It blocked or nudged recovery calls in
the exact scenario family that needs tool-error retry. Shadow mode removed that
damage, but it did not solve the persistent advanced-reasoning failures in
`argument_transformation*` or `grounded_synthesis*`.

## Classifier Telemetry

| Run | Kind | Events | Actions | Labels |
|---|---|---:|---|---|
| enforce | `tool_call` | 1782 | `allow=433`, `block=54`, `advisory_nudge=10`, `shadow_only=1285` | `valid=433`, `wrong_arguments_semantic=99`, `wrong_tool_semantic=1156`, `deterministic_invalid=88`, `tool_not_needed=6` |
| final-shadow | `tool_call` | 1629 | `allow=325`, `shadow_only=1304` | `valid=325`, `wrong_arguments_semantic=20`, `wrong_tool_semantic=1186`, `deterministic_invalid=97`, `tool_not_needed=1` |
| final-shadow | `final_response` | 302 | `shadow_only=302` | `failed_to_acknowledge_data_gap=302` |

The final-response verifier did not produce useful separation in this run. It
labeled every terminal response as `failed_to_acknowledge_data_gap` at roughly
`0.23` confidence, including passing responses. That confidence is close to
uninformative for a five-label classifier. Current thresholds prevented action,
which is correct.

## Recommended Threshold Policy

These are deployment-policy recommendations, not a claim that the downloaded
`thresholds.json` files already contain these values.

For the current quantized tool-call classifier, keep every non-valid label
action-disabled:

| Label | Current downloaded advisory / enforce | Recommended advisory / enforce | Reason |
|---|---:|---:|---|
| `valid` | `0.0 / 1.01` | `0.0 / 1.01` | allow-only label |
| `wrong_tool_semantic` | `1.01 / 1.01` | `1.01 / 1.01` | high-confidence false positives on terminal and summarize calls |
| `wrong_arguments_semantic` | `0.9 / 0.995` | `1.01 / 1.01` | false blocks on valid zero-padded numeric recovery calls |
| `tool_not_needed` | `0.8 / 0.95` | `1.01 / 1.01` | observed on a passing request; insufficient active-mode proof |
| `needs_clarification` | `0.8 / 0.95` | `1.01 / 1.01` | tiny held-out support; no active-mode proof |
| `deterministic_invalid` | `1.01 / 1.01` | `1.01 / 1.01` | deterministic-only by invariant |

For the final-response verifier, keep all non-valid labels action-disabled:

| Label | Current downloaded advisory / enforce | Recommended advisory / enforce | Reason |
|---|---:|---:|---|
| `valid_final_response` | `0.0 / 1.01` | `0.0 / 1.01` | allow-only label |
| `missing_tool_fact` | `0.9 / 0.995` | `1.01 / 1.01` | no useful release separation yet |
| `contradicts_tool_result` | `0.9 / 0.995` | `1.01 / 1.01` | no useful release separation yet |
| `unsupported_claim` | `0.9 / 0.995` | `1.01 / 1.01` | no useful release separation yet |
| `failed_to_acknowledge_data_gap` | `0.9 / 0.995` | `1.01 / 1.01` | every final response collapsed to this label at low confidence |

No confidence threshold below `1.0` can safely fix the current
`wrong_arguments_semantic` failure: valid `0010` calls were scored as wrong at
or above the existing `0.995` enforcement threshold.

## Numeric Argument Semantics

The `error_recovery` tool contract is precise:

- schema type: string
- description: zero-padded four-digit count
- runtime validation: length is exactly `4` and all characters are digits
- user request: fetch `10` records

For that contract, the structural range is `0000` through `9999`. For the
specific user request, the semantically correct value is the singleton `0010`.

| Candidate | Structural validity | Semantic validity for "Fetch 10 records" | Notes |
|---|---|---|---|
| `0010` | valid | valid | canonical fixed-width representation of 10 |
| `10` | invalid | invalid for this tool | numeric value matches, but fixed-width string contract fails |
| `000010` | invalid | invalid | over-padded width |
| `0000` | valid format | wrong value for this request | current runtime accepts the format; request semantics should reject it |
| `0001`, `0002`, `0004`, `0005` | valid format | wrong value for this request | observed as high-confidence wrong-argument negatives |

Observed release telemetry:

| Run | Candidate | Classifier behavior |
|---|---|---|
| enforce | `0010` | `40` `wrong_arguments_semantic` blocks, avg confidence `0.995551`; also `84` valid allows |
| enforce | `10` | `66` valid allows, avg confidence `0.996590` |
| final-shadow | `0010` | `20` `wrong_arguments_semantic` shadow predictions at confidence `0.995940`; also `10` valid allows |
| final-shadow | `10` | `20` valid allows at confidence `0.999394` |

Recommended training/eval range:

- Positive examples: for requested integer `N`, use `format!("{N:04}")` when
  `0 <= N <= 9999` and the tool explicitly allows zero; otherwise prefer
  `1 <= N <= 9999` for count-like fields.
- Boundary positives: `0000` only when zero is meaningful, `0001`, `0009`,
  `0010`, `0099`, `0100`, `0999`, `1000`, `9999`.
- Format negatives: unpadded strings like `10`, over-padded strings like
  `000010`, signed strings, decimals, comma-formatted numbers, whitespace, and
  integer JSON values when schema requires a string.
- Value negatives: correctly formatted but wrong values near the request,
  especially `N-1`, `N+1`, `0000` for positive count requests, and digit-shift
  confusions such as `0100` for `10`.

The classifier must learn both dimensions: fixed-width representation and
request-derived numeric value. Treating only one of those as semantic validity
causes the observed false blocks and false allows.

## Resource Impact

Proxy-side resource cost is material when the final-response verifier is loaded.

| Resource | Tool-call enforce | Tool-call shadow + final shadow | Delta |
|---|---:|---:|---:|
| Proxy CPU mean | `2.86%` | `3.57%` | `+0.71` points |
| Proxy CPU p95 | `10.90%` | `19.70%` | `+8.80` points |
| Proxy RSS mean | `416.82 MiB` | `906.51 MiB` | `+489.69 MiB` |
| Proxy RSS p95 | `603.73 MiB` | `1276.23 MiB` | `+672.50 MiB` |
| Proxy RSS max | `701.23 MiB` | `1346.62 MiB` | `+645.39 MiB` |

Classifier scoring latency:

| Run | Kind | Events | Avg | p95 | p99 | Max |
|---|---|---:|---:|---:|---:|---:|
| enforce | `tool_call` | 1782 | `46.601 ms` | `344.434 ms` | `938.511 ms` | `1476.474 ms` |
| final-shadow | `tool_call` | 1629 | `50.207 ms` | `378.812 ms` | `700.542 ms` | `1144.733 ms` |
| final-shadow | `final_response` | 302 | `94.876 ms` | `320.366 ms` | `375.993 ms` | `485.729 ms` |

Backend RSS also grew more in the final-shadow run, but that should not be
attributed to the verifier without a matched baseline because the backend is a
separate llama-server process and the run paths differed.

## Promotion Gate

Do not promote either verifier beyond shadow from this evidence.

Before advisory:

- regenerate or override thresholds so all non-valid labels are inactive;
- review release telemetry with sorted `top_k` probabilities, not just top
  label;
- add targeted zero-padded numeric fixtures and replay them through FP32 and
  quantized ONNX;
- rerun a no-classifier baseline, tool-call shadow, tool-call advisory, and
  final-response shadow matrix with resource sampling enabled.

Before enforcement:

- prove zero false blocks on known-valid tool calls, including `0010`;
- prove no regressions in `error_recovery*`, terminal-tool workflows, and
  summarize/report workflows;
- prove the final-response verifier predicts labels other than
  `failed_to_acknowledge_data_gap` on passing release responses;
- set explicit p95 RSS and latency budgets for local deployments.
