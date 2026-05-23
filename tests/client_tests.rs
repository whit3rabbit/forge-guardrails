use forge_guardrails::{format_tool, TokenUsage, ToolSpec};
use serde_json::json;

fn make_simple_spec() -> ToolSpec {
    ToolSpec::from_json_schema(
        "search",
        "Search for information",
        &json!({
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }),
    )
    .expect("valid spec")
}

fn make_multi_param_spec() -> ToolSpec {
    ToolSpec::from_json_schema(
        "analyze",
        "Analyze data",
        &json!({
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input data"
                },
                "depth": {
                    "type": "integer",
                    "description": "Analysis depth"
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose output"
                }
            },
            "required": ["input"]
        }),
    )
    .expect("valid spec")
}

#[test]
fn token_usage_new() {
    let usage = TokenUsage::new(100, 50, 150);
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn token_usage_empty() {
    let usage = TokenUsage::empty();
    assert_eq!(usage.prompt_tokens, 0);
    assert_eq!(usage.completion_tokens, 0);
    assert_eq!(usage.total_tokens, 0);
}

#[test]
fn token_usage_equality() {
    let a = TokenUsage::new(10, 20, 30);
    let b = TokenUsage::new(10, 20, 30);
    assert_eq!(a, b);
}

#[test]
fn token_usage_inequality() {
    let a = TokenUsage::new(10, 20, 30);
    let b = TokenUsage::new(1, 2, 3);
    assert_ne!(a, b);
}

#[test]
fn format_tool_basic_structure() {
    let spec = make_simple_spec();
    let result = format_tool(&spec);

    // Top-level must have type="function"
    assert_eq!(result["type"], "function");

    // Must have function key
    let func = &result["function"];
    assert_eq!(func["name"], "search");
    assert_eq!(func["description"], "Search for information");

    // Parameters must be a JSON schema object
    let params = &func["parameters"];
    assert_eq!(params["type"], "object");
    assert!(params.get("properties").is_some());
}

#[test]
fn format_tool_has_required_fields() {
    let spec = make_simple_spec();
    let result = format_tool(&spec);
    let params = &result["function"]["parameters"];
    let required = params["required"].as_array().expect("required array");
    assert!(required.iter().any(|v| v == "query"));
}

#[test]
fn format_tool_multi_param() {
    let spec = make_multi_param_spec();
    let result = format_tool(&spec);

    let func = &result["function"];
    assert_eq!(func["name"], "analyze");

    let props = func["parameters"]["properties"].as_object().expect("props");
    assert!(props.contains_key("input"));
    assert!(props.contains_key("depth"));
    assert!(props.contains_key("verbose"));
}

#[test]
fn format_tool_is_openai_compatible() {
    let spec = make_simple_spec();
    let result = format_tool(&spec);

    // The output must be serializable to a valid JSON object
    let serialized = serde_json::to_string(&result).expect("serialize");
    let reparsed: serde_json::Value = serde_json::from_str(&serialized).expect("deserialize");
    assert_eq!(reparsed["type"], "function");
    assert_eq!(reparsed["function"]["name"], "search");
}

#[test]
fn format_tool_preserves_description() {
    let spec = make_simple_spec();
    let result = format_tool(&spec);
    assert_eq!(result["function"]["description"], "Search for information");
}
