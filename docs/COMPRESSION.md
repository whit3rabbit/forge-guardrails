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
| `aggressive` | no | `standard` plus lossy log normalization, JSON-array tabular conversion, and dictionary compression. |

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

Standard and aggressive candidate transforms are kept only when they reduce
output size. Safe filters can still replace dangerous or oversized content even
when the replacement is not smaller.

## Dictionary Compression

Dictionary compression emits readable model-visible headers:

```text
[Forge LZW Dictionary]
<<FORGE_LZW_1_1>> = "error: repeated dependency resolution failure"

<<FORGE_LZW_1_1>> in crate alpha
```

```text
[Forge RePair Dictionary]
<<FORGE_REPAIR_1_1>> = "workspace crate alpha"

error in <<FORGE_REPAIR_1_1>>
```

The dictionary stages are bounded:

- Inputs over 50,000 bytes are skipped.
- Dictionary entries are capped at 20.
- Repeated entries require at least 3 occurrences.
- Entries containing newlines are skipped.
- Compression must produce meaningful net savings.
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

## Package Notes

The current Rust implementation uses existing dependencies only. No OpenToken
runtime, Node, Bun, tokenizer, or compression crate is required for proxy
runtime compression.
