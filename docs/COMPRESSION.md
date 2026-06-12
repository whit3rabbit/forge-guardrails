# Tool Output Compression

Forge can compress prior tool-result messages before forwarding a proxy request
upstream. This is opt-in and disabled by default because it changes
model-visible tool output.

Compression only applies to prior tool results:

- OpenAI `role: "tool"` messages.
- Anthropic `tool_result` blocks after request translation.
- Internal `MessageRole::Tool` / `MessageType::ToolResult` content.

Forge does not compress tool calls, tool IDs, tool names, tool arguments, or
final responses. The tool-call/tool-result pairing invariant is preserved.

## Enable It

CLI:

```bash
forge-guardrails-proxy \
  --backend-url http://localhost:8080 \
  --tool-output-compression standard
```

Environment:

```bash
export FORGE_TOOL_OUTPUT_COMPRESSION=standard
```

Request override:

```json
{
  "model": "test-model",
  "messages": [],
  "_forge": {
    "tool_output_compression": "standard"
  }
}
```

The request-level `_forge.tool_output_compression` value overrides the process
default for that request.

## Modes

| Mode | Default | Behavior |
|---|---:|---|
| `disabled` | yes | No mutation. |
| `safe` | no | Redact secrets, strip ANSI, suppress binary output, cap oversized output. |
| `standard` | no | `safe` plus JSON/table cleanup, tool-family filters, repeated-line folding, whitespace cleanup. |
| `aggressive` | no | `standard` plus lossy log normalization, schema-table JSON-array conversion, and dictionary compression. |

Use `safe` when the main goal is to remove dangerous or noisy output while
keeping the transcript close to original. Use `standard` for normal proxy
operation when you want readable compression. Use `aggressive` only when the
caller explicitly accepts model-visible dictionary markers and lossy log
normalization.

## Aggressive Methods

Aggressive mode has a dictionary method selector:

| Method | Default | Behavior |
|---|---:|---|
| `lzw` | yes | Rolling-hash repeated-substring dictionary compression. |
| `repair` | no | RePair-style repeated token-sequence dictionary compression. |
| `auto` | no | Runs both bounded methods and keeps the smaller valid result. |

CLI:

```bash
forge-guardrails-proxy \
  --backend-url http://localhost:8080 \
  --tool-output-compression aggressive \
  --tool-output-compression-method auto
```

Environment:

```bash
export FORGE_TOOL_OUTPUT_COMPRESSION=aggressive
export FORGE_TOOL_OUTPUT_COMPRESSION_METHOD=auto
```

Request override:

```json
{
  "_forge": {
    "tool_output_compression": {
      "mode": "aggressive",
      "method": "repair"
    }
  }
}
```

The method selector only affects the aggressive dictionary stage. `safe` and
`standard` ignore it.

## Request Object

The object form inherits the process default and overrides only the fields
provided:

```json
{
  "_forge": {
    "tool_output_compression": {
      "mode": "standard",
      "session_id": "tenant-a-session-42",
      "dedup": true,
      "redact_secrets": true,
      "max_output_bytes": 65536
    }
  }
}
```

Fields:

| Field | Type | Purpose |
|---|---|---|
| `mode` | string | `disabled`, `safe`, `standard`, or `aggressive`. |
| `method` | string | `lzw`, `repair`, or `auto`; used only by aggressive mode. |
| `session_id` | string | Enables repeated-output dedup across requests for that session. |
| `dedup` | boolean | Enables or disables dedup. Requires `session_id` to take effect. |
| `redact_secrets` | boolean | Defaults to true. Redaction runs before other transforms. |
| `max_output_bytes` | positive integer | Caps retained safe output before standard/aggressive filters. Request overrides are capped at 1,048,576 bytes. |

Invalid mode, method, or field types return `400 Bad Request`.

Dedup is session-scoped and bounded. A duplicate marker is emitted only when
the current compressed output has the same canonical tool name, output byte
length, and output hash as a prior entry in the same `session_id`, and the
match comes from a different tool call id. A tool result re-sent in a later
request under the same tool call id keeps its content; without that rule,
full-history resends would degrade every tool result into a marker whose
referenced original is no longer visible upstream. Tool results without a
tool call id are never deduplicated. The marker references the original call:

```text
[Duplicate of call_abc123 (bash); see earlier result]
```

The later duplicate is collapsed, not the earlier one. Collapsing the earlier
message would rewrite an already-cached prompt prefix on every subsequent
request, defeating the purpose of upstream prompt-caching. Near-duplicate delta
encoding (storing only a diff from a base entry) was considered and rejected
because FIFO eviction can remove the base entry, which makes the delta
un-expandable and breaks determinism. A marker-format change busts the cache
prefix once for sessions that span the deploy; subsequent resends are
byte-stable with the new format.

## Memoization and prompt-cache stability

When a `session_id` is set, Forge memoizes the final compressed output for each
tool call id. On subsequent requests that resend the same tool result with the
same content and compression config, the stored bytes are returned without
recompression. This keeps prior messages byte-identical across full-history
resends, which preserves upstream prompt-cache prefixes (Anthropic
`cache_control`, OpenAI automatic caching) and skips redundant CPU work.

The memo is keyed by `(session_id, tool_call_id)`. A hit requires both the
input bytes and the compression config to match; a config change (mode, method,
redact_secrets, dedup, max_output_bytes) invalidates the entry and triggers a
fresh pass, setting `memo_changed = true` in telemetry as a cache-buster signal.
Outputs larger than 16 KB are not memoized. Per-session storage is bounded at
512 KB with FIFO eviction. The memo is process-local and clears on restart.

Memoization is enabled by default when `session_id` is present. Disable it for
a request with:

```json
{
  "_forge": {
    "tool_output_compression": {
      "mode": "standard",
      "session_id": "my-session",
      "memo": false
    }
  }
}
```

Telemetry fields `memo_reused` and `memo_changed` are emitted in every
compression JSONL event. Use `memo_changed` spikes to detect config or content
churn that busts the prompt-cache prefix on re-sends.

## Pipeline

When enabled, Forge applies transforms in this order:

1. Safe filters:
   redact secrets, strip ANSI, suppress binary output, and cap oversized
   output.
2. Standard filters:
   minify JSON, minimize table whitespace, route through tool-family filters,
   fold repeated lines, and normalize trailing whitespace.
3. Aggressive filters:
   normalize dynamic log noise, convert simple JSON arrays to tabular text,
   then run the selected dictionary method.
4. Dedup:
   if a `session_id` is set, repeated compressed output can be replaced with a
   bounded duplicate marker.

Standard and aggressive candidate transforms are kept only when they reduce the
heuristic token estimate without growing the byte size. The token estimate is
a structure-aware heuristic: word, digit, punctuation, and whitespace runs are
weighted separately so punctuation-heavy output (JSON, logs) is priced closer
to real BPE behavior than a flat character ratio. The same estimate feeds the
telemetry `before_tokens`/`after_tokens` fields and the live-eval
telemetry-estimate fallback, so threshold values calibrated against the old
flat chars/4 estimate may need small adjustments. Safe filters can still
replace dangerous or oversized content even when the replacement is not
smaller. Capped output keeps a head and tail around a short cap marker, so a
capped result can exceed `max_output_bytes` by the marker length. When the
plain head/tail split would drop a line carrying an error signal (`error:`,
`panicked at`, `Traceback`, `FAILED`, and similar), capping instead keeps a
third bounded window anchored at the first such line and reports the
`cap_error_window` strategy; windows that touch merge, so at most two cap
markers are emitted.

The aggressive JSON-array table transform is lossless for arrays of scalar
objects. It skips small inputs, nested values, and mixed arrays. Accepted output
uses a schema line followed by CSV-escaped rows:

```text
[3]{id:int,status:string,path:string}
1,passed,src/a.rs
2,failed,src/b.rs
3,passed,src/c.rs
```

## Dictionary Compression

Dictionary compression emits readable model-visible headers:

```text
[Forge LZW Dictionary]
<<F1:1>> = "error: repeated dependency resolution failure"

<<F1:1>> in crate alpha
```

```text
[Forge RePair Dictionary]
<<R1:1>> = "workspace crate alpha"

error in <<R1:1>>
```

The dictionary stages are bounded:

- Inputs over 50,000 bytes are skipped.
- Dictionary entries are capped at 20.
- Repeated entries require at least 3 occurrences.
- LZW entries containing newlines are skipped; RePair rules may contain up to 2 newlines.
- Compression must produce meaningful net savings.
- Per-entry savings below 16 bytes are skipped.
- Existing Forge dictionary output is not compressed again.

LZW uses rolling hashes over selected substring lengths and verifies hash hits
by exact string comparison before replacing anything. RePair tokenizes exact
text spans and replaces repeated adjacent token sequences while preserving a
round-trippable expansion in the dictionary.

## Security Notes

Redaction runs before size reduction and dictionary compression. That matters:
secret-looking values should not be moved into dictionary entries before they
are redacted.

Compression is not an access-control boundary. Treat compressed tool output as
still model-visible. Do not rely on it to hide data from an upstream model.

## Telemetry

Set `FORGE_TOOL_OUTPUT_COMPRESSION_LOG` to write bounded JSONL events for
compressed tool results:

```bash
FORGE_TOOL_OUTPUT_COMPRESSION_LOG=target/compression.jsonl \
  forge-guardrails-proxy \
    --backend-url http://localhost:8080 \
    --tool-output-compression standard
```

Each event includes strategy names, tool family, mode, token estimates, byte
and line counts, bounded request debug metadata when provided by the caller,
redaction/capping/dedup flags, and `dictionary_method` when an aggressive
dictionary output was accepted. It also includes non-cryptographic 64-bit
fingerprints for the original output, compressed output, and the
secret-redacted serialization of tool arguments so eval regressions can be
correlated without logging raw payloads or fingerprinting raw secrets. Strategy
totals are event-level and do not report marginal per-transform savings. It
does not include raw or compressed tool output. Optional bounds are
`FORGE_TOOL_OUTPUT_COMPRESSION_LOG_QUEUE_CAPACITY` and
`FORGE_TOOL_OUTPUT_COMPRESSION_LOG_MAX_EVENT_BYTES`.

## Verification

Run focused unit and proxy checks after compression changes:

```bash
cargo test tool_output --lib
cargo test proxy::handler
python -m unittest tests/test_eval_openai_proxy.py
cargo fmt --all --check
```

Use the live compression eval only when a local backend and model are available:

```bash
make eval-release-compression \
  TOOL_OUTPUT_COMPRESSION=standard \
  TOOL_OUTPUT_COMPRESSION_METHOD=lzw \
  COMPRESSION_MIN_INPUT_TOKEN_SAVINGS=1
```

The live eval runs the same suite once with compression disabled and once with
the selected mode, then compares Python-oracle JSONL rows for prompt-token
savings and behavior regressions. If the backend does not report prompt-token
usage, the comparator falls back to the compressed run's
`proxy_tool_output_compression_*.jsonl` estimated before/after token counts for
the savings threshold and labels the report as a telemetry estimate. Behavior
regression checks remain strict. The live eval should not replace
deterministic unit and proxy tests.

## Schema Compression

In addition to tool-output compression, Forge can minify tool schema
descriptions before forwarding a request upstream. This is a separate opt-in
surface that reduces the fixed per-request token cost from verbose tool
descriptions, not the variable cost from tool-result history.

### What is compressed

Only description strings inside tool schemas:

- The top-level `description` of each tool.
- Every `"description"` string inside each tool's JSON Schema (parameter
  descriptions), recursed up to depth 32.

Nothing else is touched: tool names, parameter names, types, `required` arrays,
and all other schema structure are never modified. System prompts and final
responses are excluded.

### Transforms applied (Minify mode)

- Trailing whitespace trimmed from each line.
- Consecutive blank lines collapsed to at most one blank line.
- Internal runs of spaces or tabs collapsed to a single space, outside fenced
  code blocks (content between ` ``` ` fences is passed through with only
  trailing whitespace trimmed).
- Empty `"description": ""` keys dropped entirely.

All transforms are deterministic and idempotent: applying minification twice
produces the same result as applying it once.

### Cache note

Tool schemas live in the cached prefix of the conversation. Enabling schema
compression busts that prefix once on deploy. After the first request with
minification enabled, the smaller prefix is stable across all subsequent
requests as long as the tool set and mode do not change.

### Configuration

CLI flag (overrides env):

```bash
forge-guardrails-proxy --schema-compression minify
```

Environment variable:

```bash
FORGE_SCHEMA_COMPRESSION=minify
```

Per-request override via `_forge.schema_compression`:

```json
{
  "_forge": { "schema_compression": "minify" },
  "tools": [...],
  "messages": [...]
}
```

Valid values: `disabled` (default), `minify`. An invalid value returns HTTP 400.

### Dual-path parity

Tool schemas flow through the proxy via two paths:

- **OpenAI path**: schemas are parsed into `ToolSpec` structs via
  `parse_tool_specs`, then `compress_tool_schemas` minifies in place.
- **Anthropic path**: the original Anthropic body is preserved as
  `inbound_anthropic_body`; `patch_anthropic_tool_schemas` walks
  `tools[*].description` and `tools[*].input_schema`.

Both paths call the same `minify_description` function, so the resulting
description strings are byte-identical.

## Package Notes

The current Rust implementation uses existing dependencies only. No OpenToken
runtime, Node, Bun, tokenizer, or compression crate is required for proxy
runtime compression.
