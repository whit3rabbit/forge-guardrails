use indexmap::IndexSet;
use serde_json::{json, Value};

use crate::core::tool_spec::ToolSpec;
use crate::tools::respond::RESPOND_TOOL_NAME;

use super::HandlerError;

/// Parse OpenAI-format tool definitions into ToolSpec objects.
pub fn parse_tool_specs(tools: &[Value]) -> Result<Vec<ToolSpec>, HandlerError> {
    let mut specs = Vec::new();
    let mut seen = IndexSet::new();
    for (index, tool) in tools.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some("function") {
            return Err(HandlerError::BadRequest(format!(
                "tools[{index}] must be a function tool"
            )));
        }
        let func = tool
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                HandlerError::BadRequest(format!("tools[{index}].function must be an object"))
            })?;
        let name = func.get("name").and_then(Value::as_str).ok_or_else(|| {
            HandlerError::BadRequest(format!("tools[{index}].function.name must be a string"))
        })?;
        if name.trim().is_empty() {
            return Err(HandlerError::BadRequest(format!(
                "tools[{index}].function.name must not be empty"
            )));
        }
        if name == RESPOND_TOOL_NAME {
            return Err(HandlerError::BadRequest(format!(
                "tool name '{RESPOND_TOOL_NAME}' is reserved by Forge; use a different tool name"
            )));
        }
        if !seen.insert(name.to_string()) {
            return Err(HandlerError::BadRequest(format!(
                "tools[{index}].function.name duplicates tool '{name}'"
            )));
        }

        let description = func
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        let schema = func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        let schema_obj = schema.as_object().ok_or_else(|| {
            HandlerError::BadRequest(format!(
                "tools[{index}] ('{name}') function.parameters must be an object schema"
            ))
        })?;
        if schema_obj.get("type").and_then(Value::as_str) != Some("object") {
            return Err(HandlerError::BadRequest(format!(
                "tools[{index}] ('{name}') function.parameters must have type 'object'"
            )));
        }

        let mut spec = ToolSpec::from_json_schema(name, description, &schema).map_err(|err| {
            HandlerError::BadRequest(format!("tools[{index}] ('{name}') invalid schema: {err}"))
        })?;
        spec.json_schema = Some(schema);
        specs.push(spec);
    }
    Ok(specs)
}
