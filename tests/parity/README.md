# Python Golden Parity Tests

This directory holds Python-generated golden fixtures for Rust parity tests.
The goal is to keep behavior aligned with the checked-in Python reference
submodule in `forge/`, not to test broad Rust implementation details.

## Files

- `generate_fixtures.py` builds synthetic Python scenarios against the reference
  implementation.
- `fixtures/python_golden.json` is the checked-in output consumed by Rust tests.
- `generate_opentoken_tool_output_fixtures.py` checks out the pinned OpenToken
  commit, generates tool-output filter fixtures from its TypeScript source, and
  applies local safety overrides where Forge intentionally avoids upstream false
  summaries.
- `fixtures/opentoken_tool_output_filters.json` is the checked-in OpenToken
  output consumed by Rust tool-output tests.
- `../parity_tests.rs` builds equivalent Rust scenarios and compares against the
  golden JSON.

Normal Rust test runs do not invoke Python. They read the checked-in fixture.

## Current Result

Latest local verification during the parity tightening pass:

```text
cargo test --test parity_tests
24 passed, 0 failed

cargo test
full suite passed
```

Treat this as a local result, not a permanent guarantee. Re-run the commands
after any behavior change that touches workflow execution, prompts, compaction,
tool schemas, message serialization, or backend tool-call parsing.

## Fixture Flow

Regenerate fixtures only after an intentional Python-reference behavior change
or when adding a new parity scenario:

```bash
uv run --project forge python tests/parity/generate_fixtures.py
```

Then run the focused Rust parity suite:

```bash
cargo test --test parity_tests
```

Regenerate OpenToken tool-output fixtures after intentional filter-parity
changes:

```bash
python3 tests/parity/generate_opentoken_tool_output_fixtures.py
cargo test opentoken_filter_fixture_cases_match_expected_outputs --lib
```

For pre-commit verification, also run:

```bash
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Covered Matrix

The fixture file currently covers:

- `toolspec_schema_output`: Pydantic-style JSON Schema output with descriptions,
  defaults, enums, nested objects, arrays, `$defs`, and `$ref`.
- `format_tool_output`: OpenAI tool definition output for the same rich schema.
- `tool_prompt_text`: prompt-injected tool text.
- `unknown_tool_order`: available-tool ordering in unknown-tool nudges.
- `text_retry_history`: bare text response followed by retry nudge history.
- `unknown_tool_history`: unknown tool-call history with `[UnknownTool]`
  result content and generated call IDs.
- `llamafile_malformed_args`: malformed native tool arguments become
  `TextResponse`, not an empty argument map.
- `step_nudge_history`: premature terminal tool produces step nudge metadata.
- `prerequisite_nudge_history`: prerequisite violations produce prerequisite
  nudge metadata.
- `tool_resolution_soft_error_budget`: soft resolution errors do not exhaust the
  hard execution-error budget.
- `hard_tool_execution_error_budget`: hard tool failures exhaust the configured
  execution-error budget.
- `compaction_phases`: tiered compaction phases 1, 2, and 3.
- `fold_and_serialize_reasoning`: reasoning before tool calls, orphan reasoning,
  and consecutive reasoning.
- `max_iterations_pending_error`: max-iteration diagnostics report only pending
  required steps.
- Additional adapter and proxy parity cases for Anthropic conversion, Ollama
  thinking extraction, proxy sampling fields, proxy `respond` stripping,
  streaming-without-final errors, backend error propagation, retry budgets,
  pending-step state, Llamafile reasoning extraction, and tool-call ID generation.

## Assertion Rules

Parity assertions should be strict. Prefer exact comparisons of normalized JSON,
serialized message history, generated call IDs, error type strings, pending-step
lists, and message metadata.

For JSON schema and OpenAI tool output, compare both:

- the parsed JSON value, to catch semantic shape changes
- the Python-style `json.dumps(...)` string, to catch insertion-order drift that
  can affect model-facing prompts and provider payloads

Do not loosen the Rust assertion just because a failure looks cosmetic. Small
schema, prompt, nudge, or history changes can alter model behavior.

## Maintenance Notes

When a new fixture fails:

1. Confirm the Python fixture was regenerated from the current `forge/`
   submodule.
2. Inspect the corresponding Python implementation before changing Rust.
3. Fix only the narrow Rust behavior needed for parity.
4. Keep provider wire-format behavior in adapter tests when the Python baseline
   does not expose an equivalent path.

If Python behavior intentionally changes, regenerate the fixture and review the
JSON diff before updating Rust expectations.
