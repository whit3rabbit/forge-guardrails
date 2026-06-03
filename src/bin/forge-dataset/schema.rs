use serde_json::{json, Map, Value};

pub(crate) const CAPTURE_SCHEMA_VERSION: &str = "forge-dataset-capture/v1";
pub(crate) const TRAINING_SCHEMA_VERSION: &str = "toolcall-verifier-training/v1";
pub(crate) const TRAINING_INPUT_SCHEMA_VERSION_V1: &str = "toolcall-verifier-input/v1";
pub(crate) const TRAINING_INPUT_SCHEMA_VERSION: &str = "toolcall-verifier-input/v2";
pub(crate) const ALLOWED_LABELS: &[&str] = &[
    "valid",
    "wrong_tool_semantic",
    "wrong_arguments_semantic",
    "tool_not_needed",
    "needs_clarification",
];
pub(crate) const TRAINING_LABELS: &[&str] = &[
    "valid",
    "wrong_tool_semantic",
    "wrong_arguments_semantic",
    "tool_not_needed",
    "needs_clarification",
    "deterministic_invalid",
];

pub(crate) fn is_allowed_label(label: &str) -> bool {
    ALLOWED_LABELS.contains(&label)
}

pub(crate) fn is_training_label(label: &str) -> bool {
    TRAINING_LABELS.contains(&label)
}

pub(crate) fn capture_candidate_call(name: &str, arguments: Value) -> Value {
    json!({
        "name": name,
        "arguments": arguments,
    })
}

pub(crate) fn validate_candidate_call(
    available_tools: &Value,
    candidate_call: &Value,
) -> Result<(), String> {
    let name = candidate_call
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "candidate_call.name must be a string".to_string())?;
    let arguments = candidate_call
        .get("arguments")
        .and_then(Value::as_object)
        .ok_or_else(|| "candidate_call.arguments must be an object".to_string())?;
    let tool = tool_by_name(available_tools, name)
        .ok_or_else(|| format!("candidate_call.name references unknown tool '{name}'"))?;
    validate_arguments_against_tool(tool, arguments)
}

pub(crate) fn tool_by_name<'a>(available_tools: &'a Value, name: &str) -> Option<&'a Value> {
    available_tools.as_array()?.iter().find(|tool| {
        tool.get("name")
            .and_then(Value::as_str)
            .is_some_and(|tool_name| tool_name == name)
    })
}

pub(crate) fn default_arguments_for_tool(tool: &Value) -> Value {
    let mut arguments = Map::new();
    let Some(properties) = parameters(tool)
        .and_then(|params| params.get("properties"))
        .and_then(Value::as_object)
    else {
        return Value::Object(arguments);
    };

    for (name, schema) in properties {
        arguments.insert(name.clone(), default_value_for_property(name, schema));
    }
    Value::Object(arguments)
}

pub(crate) fn mutated_arguments_for_tool(tool: &Value, original: &Value) -> Option<Value> {
    let mut arguments = original.as_object().cloned().unwrap_or_else(Map::new);
    if arguments.is_empty() {
        arguments = default_arguments_for_tool(tool)
            .as_object()
            .cloned()
            .unwrap_or_else(Map::new);
    }

    let (key, value) = arguments.iter_mut().next()?;
    *value = mutated_value(key, value);
    Some(Value::Object(arguments))
}

pub(crate) fn parse_json_object_from_text(text: &str) -> Result<Value, String> {
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(text.trim()) {
        return Ok(Value::Object(obj));
    }
    let start = text
        .find('{')
        .ok_or_else(|| "LLM response did not contain a JSON object".to_string())?;
    let end = text
        .rfind('}')
        .ok_or_else(|| "LLM response did not contain a complete JSON object".to_string())?;
    if end < start {
        return Err("LLM response JSON object bounds are invalid".to_string());
    }
    match serde_json::from_str::<Value>(&text[start..=end]) {
        Ok(Value::Object(obj)) => Ok(Value::Object(obj)),
        Ok(_) => Err("LLM response JSON was not an object".to_string()),
        Err(err) => Err(format!("failed to parse LLM JSON response: {err}")),
    }
}

fn validate_arguments_against_tool(
    tool: &Value,
    arguments: &Map<String, Value>,
) -> Result<(), String> {
    let params = parameters(tool).ok_or_else(|| "tool.parameters must be an object".to_string())?;
    let properties = params
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    if let Some(required) = params.get("required").and_then(Value::as_array) {
        for item in required {
            let Some(name) = item.as_str() else {
                continue;
            };
            if !arguments.contains_key(name) {
                return Err(format!(
                    "candidate_call.arguments missing required key '{name}'"
                ));
            }
        }
    }

    for (key, value) in arguments {
        let Some(schema) = properties.get(key) else {
            continue;
        };
        validate_value_type(key, value, schema)?;
    }
    Ok(())
}

fn parameters(tool: &Value) -> Option<&Value> {
    tool.get("parameters").filter(|value| value.is_object())
}

fn validate_value_type(key: &str, value: &Value, schema: &Value) -> Result<(), String> {
    let Some(kind) = schema.get("type").and_then(Value::as_str) else {
        return Ok(());
    };
    let valid = match kind {
        "string" => value.is_string(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "number" => value.as_f64().is_some(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => true,
    };
    if valid {
        Ok(())
    } else {
        Err(format!("candidate_call.arguments.{key} must be {kind}"))
    }
}

fn default_value_for_property(name: &str, schema: &Value) -> Value {
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => Value::String(default_string(name)),
        Some("integer") => json!(default_integer(name)),
        Some("number") => json!(default_integer(name) as f64),
        Some("boolean") => Value::Bool(true),
        Some("array") => Value::Array(default_array(name)),
        Some("object") => Value::Object(Map::new()),
        _ => Value::String(default_string(name)),
    }
}

fn default_string(name: &str) -> String {
    match name {
        "date" => "2026-06-05".to_string(),
        "path" => "docs/README.md".to_string(),
        "product_id" => "SKU-HEADPHONES-1".to_string(),
        "ticket_id" => "TCK-1001".to_string(),
        "slot_id" => "slot-001".to_string(),
        "hold_id" => "hold-001".to_string(),
        "count" => "0010".to_string(),
        "scenario" => "basic_2step".to_string(),
        "error_code" => "TOOL_CALL_REJECTED".to_string(),
        "summary" => "Forge workflow completed.".to_string(),
        "content" => "Fetched 0010 records.".to_string(),
        "query" => "project documentation".to_string(),
        "glob" => "docs/*.md".to_string(),
        "note" => "Reviewed by dataset stub.".to_string(),
        "reason" => "Requires human follow-up.".to_string(),
        _ => "test-value".to_string(),
    }
}

fn default_integer(name: &str) -> i64 {
    match name {
        "quantity" => 1,
        "duration_minutes" => 30,
        _ => 1,
    }
}

fn default_array(name: &str) -> Vec<Value> {
    match name {
        "product_ids" => vec![json!("SKU-HEADPHONES-1"), json!("SKU-DOCK-1")],
        _ => vec![Value::String("test-value".to_string())],
    }
}

fn mutated_value(name: &str, value: &Value) -> Value {
    match value {
        Value::String(_) => Value::String(format!("wrong-{name}-value")),
        Value::Number(_) => json!(9999),
        Value::Bool(current) => Value::Bool(!current),
        Value::Array(_) => Value::Array(vec![Value::String(format!("wrong-{name}-value"))]),
        Value::Object(_) | Value::Null => Value::String(format!("wrong-{name}-value")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Value {
        json!([
            {
                "name": "search_products",
                "description": "Search products.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            },
            {
                "name": "add_to_cart",
                "description": "Add product to cart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "quantity": {"type": "integer"}
                    },
                    "required": ["product_id", "quantity"]
                }
            }
        ])
    }

    #[test]
    fn validates_known_tool_and_required_arguments() {
        let candidate = json!({
            "name": "add_to_cart",
            "arguments": {"product_id": "SKU-HEADPHONES-1", "quantity": 1}
        });
        validate_candidate_call(&tools(), &candidate).expect("valid");
    }

    #[test]
    fn rejects_missing_required_argument() {
        let candidate = json!({
            "name": "add_to_cart",
            "arguments": {"product_id": "SKU-HEADPHONES-1"}
        });
        let err = validate_candidate_call(&tools(), &candidate).expect_err("invalid");
        assert!(err.contains("quantity"));
    }

    #[test]
    fn parses_json_object_from_wrapped_text() {
        let parsed =
            parse_json_object_from_text("Here is the JSON: {\"accepted\":true}").expect("json");
        assert_eq!(parsed["accepted"], true);
    }

    #[test]
    fn alternative_argument_helpers_preserve_schema_shape() {
        let available_tools = tools();
        let tool = tool_by_name(&available_tools, "add_to_cart").expect("tool");
        let defaults = default_arguments_for_tool(tool);
        validate_candidate_call(
            &available_tools,
            &json!({"name": "add_to_cart", "arguments": defaults}),
        )
        .expect("default args valid");
        let mutated = mutated_arguments_for_tool(
            tool,
            &json!({"product_id": "SKU-HEADPHONES-1", "quantity": 1}),
        )
        .expect("mutated");
        validate_candidate_call(
            &available_tools,
            &json!({"name": "add_to_cart", "arguments": mutated}),
        )
        .expect("mutated args still schema-valid");
    }
}
