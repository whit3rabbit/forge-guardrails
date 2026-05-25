# API Schema and Extensions

This document defines the schemas, payload contracts, and metadata extensions used by `forge-guardrails` to manage tool calling, workflow validation, and request routing.

---

## 1. The `_forge` Extension Schema

When using the Forge proxy, client requests to `/v1/chat/completions` (OpenAI format) or `/v1/messages` (Anthropic format) can include a private `_forge` field at the root of the JSON payload. This object configures the guardrail policies and required step checks enforced by the proxy during inference.

### Schema Definition (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ForgeExtension",
  "type": "object",
  "properties": {
    "required_steps": {
      "type": "array",
      "description": "An ordered or unordered list of tool names that the LLM must execute before completing the workflow.",
      "items": {
        "type": "string"
      },
      "default": []
    },
    "terminal_tools": {
      "type": "array",
      "description": "A list of tool names that, when called, indicate the workflow is finished. If empty, defaults to ['respond'].",
      "items": {
        "type": "string"
      },
      "default": ["respond"]
    }
  },
  "additionalProperties": false
}
```

### Struct Representation
The Rust implementation maps this contract via `ProxyStepContract` in [handler.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/handler.rs#L53-L57):

```rust
struct ProxyStepContract {
    required_steps: Vec<String>,
    terminal_tools: Vec<String>,
}
```

### Example OpenAI Request Payload with `_forge`

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": "Retrieve the current weather for Seattle."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Fetch the current weather details.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string" }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "_forge": {
    "required_steps": ["get_weather"],
    "terminal_tools": ["respond"]
  }
}
```

### Execution Lifecyle and Interception
1. **Extraction:** The proxy parses the `_forge` object during request ingestion.
2. **Validation:** The proxy validates that all tools in `required_steps` and `terminal_tools` exist in the request's `tools` array (with the exception of `respond`, which is auto-injected by the proxy).
3. **Stripping:** The `_forge` field is **stripped** from the payload before forwarding the request to the upstream LLM client to ensure provider compatibility.
4. **Enforcement:** The step enforcer monitors the ongoing message trajectory to ensure all `required_steps` are invoked before a `terminal_tool` is executed.

---

## 2. Tool Definition & Parameter Schema

Forge uses a clean, Pydantic-compatible JSON Schema representation for defining tools and their arguments.

### The `respond` Terminal Tool
The proxy automatically injects a reserved terminal tool named `respond` if one is not already provided by the client. This tool is defined in [respond.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/tools/respond.rs) and allows the LLM to output final textual content inside a structured tool call.

#### Response Tool Specification
```json
{
  "type": "function",
  "function": {
    "name": "respond",
    "description": "Synthesizes and delivers the final text response back to the user after checking that all required steps are satisfied.",
    "parameters": {
      "properties": {
        "message": {
          "description": "The final message content to deliver to the user.",
          "title": "Message",
          "type": "string"
        }
      },
      "required": ["message"],
      "title": "RespondParams",
      "type": "object"
    }
  }
}
```

### Parameter Models (`ParamModel`)
Tool parameters are parsed into a recursive tree structure defined by the `ParamModel` enum in [tool_spec.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/core/tool_spec.rs#L7-L69):

| Parameter Type | Rust Representation | Key Fields |
| :--- | :--- | :--- |
| **String** | `ParamModel::String` | `description`, `required`, `default`, `enum_values` |
| **Number** | `ParamModel::Number` | `description`, `required`, `default` |
| **Boolean** | `ParamModel::Boolean` | `description`, `required`, `default` |
| **Integer** | `ParamModel::Integer` | `description`, `required`, `default` |
| **Object** | `ParamModel::Object` | `description`, `required`, `properties` (recursive mapping) |
| **Array** | `ParamModel::Array` | `description`, `required`, `items` (boxed `ParamModel`) |
| **Unsupported** | `ParamModel::Unsupported` | `type_name` |

---

## 3. Sampling vs. Passthrough Parameter Schema

The proxy splits other top-level fields in the request body into sampling parameters and passthrough settings:

### Sampling Schema
Sampling options are extracted by the proxy to configure parameters that vary from call to call. They do not persist across requests.

The following fields are extracted in `extract_sampling` in [proxy.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/proxy.rs#L653-L679):
- `temperature`: Floating point number (e.g., `0.7`)
- `top_p`: Floating point number (e.g., `0.9`)
- `top_k`: Integer value
- `min_p`: Floating point number
- `repeat_penalty`: Floating point number
- `presence_penalty`: Floating point number
- `seed`: Integer value
- `chat_template_kwargs`: Object containing arbitrary provider template settings (e.g., `{"enable_thinking": true}`)

### Passthrough Schema
Any properties in the request body that are **not** forge-owned (i.e. not `messages`, `tools`, `stream`, `system`, or `_forge`) and **not** sampling parameters are gathered by `extract_passthrough` in [proxy.rs](file:///Users/whit3rabbit/Documents/GitHub/forge-rs/src/proxy/proxy.rs#L681-L709) and forwarded transparently to the LLM client.

Common passthrough fields include:
- `model`: String (e.g. `"gpt-4o-mini"`)
- `max_tokens` / `max_completion_tokens`: Integer limits
- `stop`: List of stop sequences
- `response_format`: JSON Schema/object settings
- `tool_choice`: String or tool choice specification

---

## 4. Message Trajectory Schema

The proxy receives external messages from client requests, converts them to Forge's internal message representation, and converts them back during response generation.

### Message Translation Map
The `openai_to_messages` function converts standard message structures into Forge's internal `Message` schema:

```rust
pub struct Message {
    pub role: MessageRole,                     // System, User, Assistant, Tool
    pub content: String,
    pub metadata: MessageMeta,
    pub tool_name: Option<String>,                 // Present if role == Tool
    pub tool_call_id: Option<String>,              // Correlation ID for tool results
    pub tool_calls: Option<Vec<ToolCallInfo>>,     // Tool calls initiated by Assistant
}
```

### Correlation IDs
To ensure tracking consistency across LLM turns, tool calls and tool results must remain paired using matching IDs:
- **Tool Call:** Represented by `ToolCallInfo` which includes a `call_id`.
- **Tool Result:** Has `role: "tool"` and contains a matching `tool_call_id` that maps back to the corresponding `call_id`.

If the client or LLM supplies empty tool call IDs, the proxy automatically generates fallback IDs (e.g., `call_1`, `call_2`) to maintain pairing logic.
