//! Integration tests for prompts tests.

use forge_guardrails::{
    build_tool_prompt, extract_tool_call, rescue_tool_call, ParamModel, ToolSpec,
};
use indexmap::IndexMap;
use serde_json::Value;

fn make_tool_spec(name: &str, desc: &str, params: Vec<(&str, &str, bool)>) -> ToolSpec {
    let mut properties = IndexMap::new();
    for (pname, ptype, required) in params {
        if ptype == "string" {
            properties.insert(
                pname.to_string(),
                ParamModel::String {
                    description: Some(format!("The {} parameter", pname)),
                    required,
                    default: None,
                    enum_values: None,
                },
            );
        }
    }
    ToolSpec {
        name: name.to_string(),
        description: desc.to_string(),
        parameters: ParamModel::Object {
            description: None,
            required: true,
            properties,
        },
        json_schema: None,
    }
}

// -- build_tool_prompt tests --

#[test]
fn build_tool_prompt_single_tool() {
    let spec = make_tool_spec("search", "Search the web", vec![("query", "string", true)]);
    let result = build_tool_prompt(&[spec]);
    assert!(result.contains("search"));
    assert!(result.contains("Search the web"));
    assert!(result.contains("query"));
    assert!(result.contains("string"));
    assert!(result.contains("required"));
    assert!(result.contains("\"tool\""));
    assert!(result.contains("\"args\""));
}

#[test]
fn build_tool_prompt_matches_python_template() {
    let spec = make_tool_spec("search", "Search the web", vec![("query", "string", true)]);
    let result = build_tool_prompt(&[spec]);
    assert_eq!(
        result,
        concat!(
            "You have access to the following tools:\n",
            "\n",
            "## search\n",
            "Description: Search the web\n",
            "Parameters:\n",
            "  - query (string (required)): The query parameter\n",
            "\n",
            "To call a tool, respond with ONLY a JSON object in this exact format:\n",
            "{\"tool\": \"<tool_name>\", \"args\": {<arguments>}}\n",
            "\n",
            "Example:\n",
            "{\"tool\": \"search\", \"args\": {\"query\": \"<query>\"}}\n",
            "\n",
            "Respond with ONLY the JSON tool call. Do not include any other text."
        )
    );
}

#[test]
fn build_tool_prompt_two_tools() {
    let spec1 = make_tool_spec("search", "Search the web", vec![("query", "string", true)]);
    let spec2 = make_tool_spec("analyze", "Analyze data", vec![("data", "string", true)]);
    let result = build_tool_prompt(&[spec1, spec2]);
    assert!(result.contains("search"));
    assert!(result.contains("Search the web"));
    assert!(result.contains("analyze"));
    assert!(result.contains("Analyze data"));
}

#[test]
fn build_tool_prompt_optional_param() {
    let spec = make_tool_spec("search", "Search", vec![("query", "string", false)]);
    let result = build_tool_prompt(&[spec]);
    assert!(result.contains("optional"));
}

#[test]
fn build_tool_prompt_enum_param() {
    let mut properties = IndexMap::new();
    properties.insert(
        "mode".to_string(),
        ParamModel::String {
            description: Some("Search mode".to_string()),
            required: true,
            default: None,
            enum_values: Some(vec!["fast".to_string(), "deep".to_string()]),
        },
    );
    let spec = ToolSpec {
        name: "search".to_string(),
        description: "Search".to_string(),
        parameters: ParamModel::Object {
            description: None,
            required: true,
            properties,
        },
        json_schema: None,
    };
    let result = build_tool_prompt(&[spec]);
    assert!(result.contains("fast"));
    assert!(result.contains("deep"));
}

// -- extract_tool_call tests --

#[test]
fn extract_forge_format() {
    let text = r#"{"tool": "search", "args": {"query": "rust"}}"#;
    let result = extract_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
    assert_eq!(result[0].args["query"], Value::String("rust".to_string()));
}

#[test]
fn extract_code_fence_json() {
    let text = "```json\n{\"tool\": \"search\", \"args\": {\"query\": \"test\"}}\n```";
    let result = extract_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
}

#[test]
fn extract_code_fence_no_lang() {
    let text = "```\n{\"tool\": \"search\", \"args\": {\"query\": \"test\"}}\n```";
    let result = extract_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn extract_embedded_in_text() {
    let text =
        "Let me search for that.\n{\"tool\": \"search\", \"args\": {\"query\": \"test\"}}\nDone.";
    let result = extract_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn extract_unknown_tool_rejected() {
    let text = r#"{"tool": "hack", "args": {}}"#;
    let result = extract_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn extract_no_json() {
    let text = "I think the answer is 42";
    let result = extract_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn extract_malformed_json() {
    let text = r#"{"tool": "search", "args": {"query": "#;
    let result = extract_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn extract_no_tool_or_name_key() {
    let text = r#"{"type": "unknown", "data": {}}"#;
    let result = extract_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn extract_missing_args_defaults_empty() {
    let text = r#"{"tool": "search"}"#;
    let result = extract_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert!(result[0].args.is_empty());
}

#[test]
fn extract_nested_json_args() {
    let text = r#"{"tool": "search", "args": {"query": "test", "options": {"limit": 5}}}"#;
    let result = extract_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    let options = &result[0].args["options"];
    assert_eq!(options["limit"], Value::Number(5.into()));
}

#[test]
fn extract_openai_format() {
    let text = r#"{"name": "search", "arguments": {"query": "rust"}}"#;
    let result = extract_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
}

#[test]
fn extract_two_tool_calls() {
    let text = r#"{"tool": "search", "args": {"q": "a"}}{"tool": "analyze", "args": {"d": "b"}}"#;
    let result = extract_tool_call(text, &["search", "analyze"]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].tool, "search");
    assert_eq!(result[1].tool, "analyze");
}

#[test]
fn extract_tool_key_priority_over_name() {
    let text = r#"{"tool": "search", "name": "analyze", "args": {"q": "test"}}"#;
    let result = extract_tool_call(text, &["search", "analyze"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
}

// -- rescue_tool_call tests --

#[test]
fn rescue_json_in_text() {
    let text = "Here is my result.\n{\"tool\": \"search\", \"args\": {\"query\": \"test\"}}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
}

#[test]
fn rescue_rehearsal_syntax() {
    let text = "search[ARGS]{\"query\": \"rust\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
    assert_eq!(result[0].args["query"], Value::String("rust".to_string()));
}

#[test]
fn rescue_bracket_think_tag() {
    let text = "[THINK]Let me think about this[/THINK]\nsearch[ARGS]{\"query\": \"test\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn rescue_unicode_think_tag() {
    let text = "\u{200B}[THINK]thinking[/THINK]\u{200B}\nsearch[ARGS]{\"query\": \"test\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn rescue_lowercase_xml_think_tag() {
    let text = "<think >reason</think >{\"tool\": \"search\", \"args\": {\"query\": \"test\"}}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
}

#[test]
fn rescue_rehearsal_unknown_tool() {
    let text = "hack[ARGS]{\"query\": \"test\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn rescue_no_patterns() {
    let result = rescue_tool_call("just plain text", &["search"]);
    assert!(result.is_empty());
}

#[test]
fn rescue_rehearsal_malformed_json() {
    let text = "search[ARGS]{broken json}";
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn rescue_empty_string() {
    let result = rescue_tool_call("", &["search"]);
    assert!(result.is_empty());
}

#[test]
fn rescue_only_think_tags() {
    let result = rescue_tool_call("[THINK]just thinking[/THINK]", &["search"]);
    assert!(result.is_empty());
}

#[test]
fn rescue_json_wins_over_rehearsal() {
    let text = r#"{"tool": "search", "args": {"q": "json"}}search[ARGS]{"q": "rehearsal"}"#;
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].args["q"], Value::String("json".to_string()));
}

#[test]
fn rescue_json_wins_over_qwen() {
    let text = r#"<function=search><parameter=query>xml</parameter></function>{"tool": "search", "args": {"q": "json"}}"#;
    let result = rescue_tool_call(text, &["search"]);
    assert!(!result.is_empty());
}

// -- Qwen XML tests --

#[test]
fn qwen_xml_single_param() {
    let text = "<function=search><parameter=query>hello world</parameter></function>";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
    assert_eq!(
        result[0].args["query"],
        Value::String("hello world".to_string())
    );
}

#[test]
fn qwen_xml_multi_param() {
    let text = "<function=search><parameter=query>rust</parameter><parameter=mode>fast</parameter></function>";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].args["query"], Value::String("rust".to_string()));
    assert_eq!(result[0].args["mode"], Value::String("fast".to_string()));
}

#[test]
fn qwen_xml_missing_param_close_before_next_param() {
    let text = "<function=search><parameter=query>rust<parameter=mode>fast</parameter></function>";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].args["query"], Value::String("rust".to_string()));
    assert_eq!(result[0].args["mode"], Value::String("fast".to_string()));
}

#[test]
fn qwen_xml_missing_param_close_before_function_close() {
    let text = "<function=search><parameter=query>rust</function>";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].args["query"], Value::String("rust".to_string()));
}

#[test]
fn qwen_xml_multiline_value() {
    let text = "<function=write><parameter=content>\nline1\nline2\n</parameter></function>";
    let result = rescue_tool_call(text, &["write"]);
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].args["content"],
        Value::String("line1\nline2".to_string())
    );
}

#[test]
fn qwen_xml_two_functions() {
    let text = "<function=search><parameter=query>a</parameter></function><function=analyze><parameter=data>b</parameter></function>";
    let result = rescue_tool_call(text, &["search", "analyze"]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].tool, "search");
    assert_eq!(result[1].tool, "analyze");
}

#[test]
fn qwen_xml_unknown_tool() {
    let text = "<function=hack><parameter=q>x</parameter></function>";
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn qwen_xml_unclosed_tag() {
    let text = "<function=search><parameter=query>hello</parameter>";
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

// -- Mistral bracket-tag tests --

#[test]
fn mistral_no_whitespace() {
    let text = "[TOOL_CALLS]search{\"query\": \"rust\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].tool, "search");
}

#[test]
fn mistral_with_space() {
    let text = "[TOOL_CALLS]search {\"query\": \"rust\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn mistral_with_newline() {
    let text = "[TOOL_CALLS]search\n{\"query\": \"rust\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn mistral_braces_in_string() {
    let text = r#"[TOOL_CALLS]search{"query": "function() { return 1; }"}"#;
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].args["query"],
        Value::String("function() { return 1; }".to_string())
    );
}

#[test]
fn mistral_escaped_quote() {
    let text = r#"[TOOL_CALLS]search{"query": "say \"hello\""}"#;
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].args["query"],
        Value::String("say \"hello\"".to_string())
    );
}

#[test]
fn mistral_two_calls() {
    let text = r#"[TOOL_CALLS]search{"q": "a"}[TOOL_CALLS]analyze{"d": "b"}"#;
    let result = rescue_tool_call(text, &["search", "analyze"]);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].tool, "search");
    assert_eq!(result[1].tool, "analyze");
}

#[test]
fn mistral_unknown_tool() {
    let text = r#"[TOOL_CALLS]hack{"q": "a"}"#;
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn mistral_preceding_text() {
    let text = r#"Let me do this for you. [TOOL_CALLS]search{"q": "test"}"#;
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn mistral_unclosed_braces() {
    let text = "[TOOL_CALLS]search{\"q\": \"test\"";
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn mistral_json_wins_over_bracket() {
    let text = r#"{"tool": "search", "args": {"q": "json"}}[TOOL_CALLS]search{"q": "bracket"}"#;
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].args["q"], Value::String("json".to_string()));
}

#[test]
fn mistral_with_think_tags() {
    let text = "[THINK]thinking[/THINK][TOOL_CALLS]search{\"q\": \"test\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert_eq!(result.len(), 1);
}

#[test]
fn mistral_non_dict_rejected() {
    let text = "[TOOL_CALLS]search{\"q\"}";
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}

#[test]
fn mistral_array_rejected() {
    let text = "[TOOL_CALLS]search[1, 2, 3]";
    let result = rescue_tool_call(text, &["search"]);
    assert!(result.is_empty());
}
