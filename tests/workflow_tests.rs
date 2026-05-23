use forge_guardrails::error::ToolResolutionError;
use forge_guardrails::workflow::*;
use forge_guardrails::{ToolDef, ToolSpec, Workflow};
use indexmap::IndexMap;

fn dummy_callable(_args: Vec<String>) -> Result<String, ToolResolutionError> {
    Ok("result".to_string())
}

fn make_spec(name: &str) -> ToolSpec {
    let schema = serde_json::json!({
        "properties": {
            "input": {"type": "string"}
        },
        "required": ["input"]
    });
    ToolSpec::from_json_schema(name, "test tool", &schema).expect("valid spec")
}

fn make_tool_def(name: &str) -> ToolDef {
    ToolDef::new(make_spec(name), dummy_callable)
}

fn make_tools(names: &[&str]) -> IndexMap<String, ToolDef> {
    let mut map = IndexMap::new();
    for name in names {
        map.insert(name.to_string(), make_tool_def(name));
    }
    map
}

#[test]
fn valid_workflow_single_terminal() {
    let tools = make_tools(&["step_a", "finish"]);
    let wf = Workflow::new(
        "test",
        "desc",
        tools,
        vec!["step_a".to_string()],
        TerminalToolInput::Single("finish".to_string()),
        "template",
    );
    assert!(wf.is_ok());
    let wf = wf.expect("ok");
    assert!(wf.terminal_tools.contains("finish"));
    assert_eq!(wf.terminal_tools.len(), 1);
}

#[test]
fn valid_workflow_multiple_terminal() {
    let tools = make_tools(&["step_a", "finish1", "finish2"]);
    let wf = Workflow::new(
        "test",
        "desc",
        tools,
        vec!["step_a".to_string()],
        TerminalToolInput::Multiple(vec!["finish1".to_string(), "finish2".to_string()]),
        "template",
    );
    assert!(wf.is_ok());
    let wf = wf.expect("ok");
    assert!(wf.terminal_tools.contains("finish1"));
    assert!(wf.terminal_tools.contains("finish2"));
}

#[test]
fn error_key_name_mismatch() {
    let spec = make_spec("correct_name");
    let def = ToolDef::new(spec, dummy_callable);
    let mut tools = IndexMap::new();
    tools.insert("wrong_key".to_string(), def);
    let result = Workflow::new(
        "test",
        "desc",
        tools,
        vec![],
        TerminalToolInput::Single("t".to_string()),
        "template",
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("wrong_key"));
    assert!(err.contains("correct_name"));
}

#[test]
fn error_required_step_not_in_tools() {
    let tools = make_tools(&["step_a"]);
    let result = Workflow::new(
        "test",
        "desc",
        tools,
        vec!["nonexistent".to_string()],
        TerminalToolInput::Single("step_a".to_string()),
        "template",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("nonexistent"));
}

#[test]
fn error_terminal_not_in_tools() {
    let tools = make_tools(&["step_a"]);
    let result = Workflow::new(
        "test",
        "desc",
        tools,
        vec!["step_a".to_string()],
        TerminalToolInput::Single("nonexistent".to_string()),
        "template",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("nonexistent"));
}

#[test]
fn error_terminal_is_required_step() {
    let tools = make_tools(&["finish"]);
    let result = Workflow::new(
        "test",
        "desc",
        tools,
        vec!["finish".to_string()],
        TerminalToolInput::Single("finish".to_string()),
        "template",
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("cannot also be a required step"));
}

#[test]
fn error_prereq_not_in_tools() {
    let spec = make_spec("tool_a");
    let def = ToolDef::new(spec, dummy_callable)
        .with_prerequisites(vec![PrerequisiteSpec::NameOnly("nonexistent".to_string())]);
    let mut tools = IndexMap::new();
    tools.insert("tool_a".to_string(), def);
    let result = Workflow::new(
        "test",
        "desc",
        tools,
        vec![],
        TerminalToolInput::Single("tool_a".to_string()),
        "template",
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("nonexistent"));
}

#[test]
fn build_system_prompt_replaces_vars() {
    let tools = make_tools(&["finish"]);
    let wf = Workflow::new(
        "test",
        "desc",
        tools,
        vec![],
        TerminalToolInput::Single("finish".to_string()),
        "Hello {name}, welcome to {place}!",
    )
    .expect("ok");

    let mut vars = IndexMap::new();
    vars.insert("name".to_string(), "World".to_string());
    vars.insert("place".to_string(), "Rust".to_string());
    let result = wf.build_system_prompt(&vars);
    assert_eq!(result, "Hello World, welcome to Rust!");
}

#[test]
fn get_tool_specs_preserves_order() {
    let tools = make_tools(&["zebra", "alpha", "middle"]);
    let wf = Workflow::new(
        "test",
        "desc",
        tools,
        vec![],
        TerminalToolInput::Single("zebra".to_string()),
        "template",
    )
    .expect("ok");

    let specs = wf.get_tool_specs();
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["zebra", "alpha", "middle"]);
}

#[test]
fn get_callable_found() {
    let tools = make_tools(&["search", "finish"]);
    let wf = Workflow::new(
        "test",
        "desc",
        tools,
        vec!["search".to_string()],
        TerminalToolInput::Single("finish".to_string()),
        "template",
    )
    .expect("ok");

    assert!(wf.get_callable("search").is_ok());
}

#[test]
fn get_callable_not_found() {
    let tools = make_tools(&["finish"]);
    let wf = Workflow::new(
        "test",
        "desc",
        tools,
        vec![],
        TerminalToolInput::Single("finish".to_string()),
        "template",
    )
    .expect("ok");

    let result = wf.get_callable("nonexistent");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("nonexistent"));
}

#[test]
fn toolspec_from_json_schema() {
    let schema = serde_json::json!({
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer"}
        },
        "required": ["query"]
    });
    let spec = ToolSpec::from_json_schema("search", "search tool", &schema).expect("ok");
    assert_eq!(spec.name, "search");
    assert_eq!(spec.description, "search tool");
}

#[test]
fn toolspec_from_json_schema_nested_object() {
    let schema = serde_json::json!({
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"}
                },
                "required": ["key"]
            }
        },
        "required": ["config"]
    });
    let spec = ToolSpec::from_json_schema("tool", "nested tool", &schema).expect("ok");
    assert_eq!(spec.name, "tool");
}

#[test]
fn toolspec_from_json_schema_array() {
    let schema = serde_json::json!({
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["items"]
    });
    let spec = ToolSpec::from_json_schema("tool", "array tool", &schema).expect("ok");
    assert_eq!(spec.name, "tool");
}

#[test]
fn toolspec_from_json_schema_missing_properties() {
    let schema = serde_json::json!({});
    let result = ToolSpec::from_json_schema("tool", "test", &schema);
    assert!(result.is_err());
}

#[test]
fn toolspec_get_json_schema() {
    let schema = serde_json::json!({
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    });
    let spec = ToolSpec::from_json_schema("search", "search tool", &schema).expect("ok");
    let output = spec.get_json_schema();
    assert_eq!(output["type"], "object");
    assert!(output.get("properties").is_some());
    let props = output["properties"].as_object().expect("properties object");
    assert!(props.contains_key("query"));
}

#[test]
fn tool_def_name_property() {
    let spec = make_spec("my_tool");
    let def = ToolDef::new(spec, dummy_callable);
    assert_eq!(def.name(), "my_tool");
}

#[test]
fn valid_workflow_with_prerequisites() {
    let spec_a = make_spec("step_a");
    let spec_b = make_spec("step_b");
    let spec_finish = make_spec("finish");
    let def_a = ToolDef::new(spec_a, dummy_callable);
    let def_b = ToolDef::new(spec_b, dummy_callable)
        .with_prerequisites(vec![PrerequisiteSpec::NameOnly("step_a".to_string())]);
    let def_finish = ToolDef::new(spec_finish, dummy_callable);
    let mut tools = IndexMap::new();
    tools.insert("step_a".to_string(), def_a);
    tools.insert("step_b".to_string(), def_b);
    tools.insert("finish".to_string(), def_finish);

    let result = Workflow::new(
        "test",
        "desc",
        tools,
        vec!["step_a".to_string(), "step_b".to_string()],
        TerminalToolInput::Single("finish".to_string()),
        "template",
    );
    assert!(result.is_ok());
}
